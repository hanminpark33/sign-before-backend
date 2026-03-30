"""
사인전에 — Flask 백엔드
계약서 이미지를 Claude Vision API로 분석하는 중계 서버.

원칙:
- 이미지를 서버에 저장하지 않음 (메모리에서 즉시 처리 후 폐기)
- 분석 결과를 서버 DB에 저장하지 않음
- 국외이전 동의 확인 후에만 API 호출
"""

import os
import io
import re
import json
import base64
import anthropic
import pytesseract
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image, ImageFilter

# 루트 .env (공통 키) → 프로젝트 .env (고유 설정) 순서로 로드
_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(_root)
load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

MAX_IMAGE_BYTES = 3.5 * 1024 * 1024  # 3.5MB — Claude Vision 5MB 제한 이내, OCR 정확도 우선
MAX_DIMENSION = 1568                 # Claude Vision 최적 해상도 (한글 계약서 가독성 확보)

# 개인정보 패턴 (주민등록번호, 계좌번호)
RRN_PATTERN = re.compile(r'\d{6}\s*[-–]\s*\d{7}')
ACCOUNT_PATTERN = re.compile(r'\d{3,4}\s*[-–]\s*\d{4,6}\s*[-–]\s*\d{4,6}(?:\s*[-–]\s*\d{2,3})?')


def blur_sensitive_regions(img: Image.Image) -> Image.Image:
    """Tesseract OCR로 텍스트 위치를 감지하고, 주민번호/계좌번호 영역을 블러 처리.
    블러된 이미지를 반환. OCR 실패 시 원본 그대로 반환 (분석 중단 방지)."""
    try:
        # Tesseract OCR — 단어 단위 bounding box 추출 (한국어)
        ocr_data = pytesseract.image_to_data(
            img, lang='kor', output_type=pytesseract.Output.DICT
        )
    except Exception:
        # Tesseract 오류 시 원본 반환 (분석 흐름 유지)
        return img

    n = len(ocr_data['text'])
    if n == 0:
        return img

    # 연속 텍스트를 같은 줄(block+par+line) 기준으로 합쳐서 패턴 매칭
    lines = {}  # (block, par, line) -> [(text, left, top, width, height), ...]
    for i in range(n):
        text = ocr_data['text'][i].strip()
        if not text:
            continue
        key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
        lines.setdefault(key, []).append({
            'text': text,
            'left': ocr_data['left'][i],
            'top': ocr_data['top'][i],
            'width': ocr_data['width'][i],
            'height': ocr_data['height'][i],
        })

    # 블러할 영역 수집
    blur_regions = []

    for key, words in lines.items():
        line_text = ' '.join(w['text'] for w in words)

        for pattern in (RRN_PATTERN, ACCOUNT_PATTERN):
            for match in pattern.finditer(line_text):
                # 매칭된 텍스트가 포함된 단어들의 bounding box 합산
                char_pos = 0
                region_left = None
                region_top = None
                region_right = 0
                region_bottom = 0

                for w in words:
                    w_start = char_pos
                    w_end = char_pos + len(w['text'])
                    # 이 단어가 매칭 범위와 겹치는지 확인
                    if w_end > match.start() and w_start < match.end():
                        if region_left is None:
                            region_left = w['left']
                            region_top = w['top']
                        region_right = max(region_right, w['left'] + w['width'])
                        region_bottom = max(region_bottom, w['top'] + w['height'])
                    char_pos = w_end + 1  # +1 for space

                if region_left is not None:
                    # 여유 패딩 추가
                    pad = 4
                    blur_regions.append((
                        max(0, region_left - pad),
                        max(0, region_top - pad),
                        min(img.width, region_right + pad),
                        min(img.height, region_bottom + pad),
                    ))

    if not blur_regions:
        return img

    # 해당 영역만 강한 블러 적용
    for (x1, y1, x2, y2) in blur_regions:
        region = img.crop((x1, y1, x2, y2))
        # 블러 강도: 반복 적용으로 완전히 식별 불가하게
        for _ in range(5):
            region = region.filter(ImageFilter.GaussianBlur(radius=10))
        # 블러 위에 반투명 회색 오버레이 (이중 보호)
        overlay = Image.new('RGB', region.size, (128, 128, 128))
        region = Image.blend(region, overlay, alpha=0.5)
        img.paste(region, (x1, y1))

    return img


def normalize_image(image_base64: str) -> str:
    """이미지를 JPEG으로 정규화하고 Claude API 제한(5MB) 이내로 압축 후 base64 반환.
    PNG 등 비JPEG 포맷도 모두 JPEG으로 변환해 media_type 불일치 방지.
    전송 전 주민번호/계좌번호 영역을 블러 처리."""
    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # 개인정보 블러 처리 (해상도 축소 전, OCR 정확도 높은 상태에서 수행)
    img = blur_sensitive_regions(img)

    # 해상도 축소
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    # 이미 작으면 품질 92로 JPEG 변환 (한글 OCR 정확도를 위해 고품질 유지)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    if buf.tell() <= MAX_IMAGE_BYTES:
        return base64.b64encode(buf.getvalue()).decode()

    # JPEG 품질 조절로 목표 크기 맞추기
    for quality in (82, 70, 55):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.b64encode(buf.getvalue()).decode()

    # 최저 품질로 강제 압축
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=45)
    return base64.b64encode(buf.getvalue()).decode()


def load_prompt(contract_type: str) -> str:
    if not os.path.isdir(PROMPTS_DIR):
        raise FileNotFoundError(f"프롬프트 디렉토리 없음: {PROMPTS_DIR}")
    path = os.path.join(PROMPTS_DIR, f"{contract_type}.txt")
    if not os.path.exists(path):
        raise ValueError(f"지원하지 않는 계약서 유형: {contract_type}")
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"프롬프트 파일이 비어있음: {contract_type}")
    return content


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": CLAUDE_MODEL})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "요청 데이터가 없어요."}), 400

    # 국외이전 동의 확인
    if not data.get("consent_verified"):
        return jsonify({"error": "개인정보 국외이전 동의가 필요해요."}), 403

    # images_base64 (배열) 우선, 구버전 호환을 위해 image_base64 (단일)도 지원
    images_base64 = data.get("images_base64")
    if not images_base64:
        single = data.get("image_base64")
        if single:
            images_base64 = [single]

    contract_type = data.get("contract_type", "lease")

    if not images_base64 or len(images_base64) == 0:
        return jsonify({"error": "이미지 데이터가 없어요."}), 400

    if len(images_base64) > 20:
        return jsonify({"error": "이미지는 최대 20장까지 분석할 수 있어요."}), 400

    # base64 유효성 검사 + 이미지 정규화
    normalized = []
    for idx, img_b64 in enumerate(images_base64):
        try:
            base64.b64decode(img_b64, validate=True)
        except Exception:
            return jsonify({"error": f"{idx + 1}번째 이미지 형식을 확인해요."}), 400
        try:
            normalized.append(normalize_image(img_b64))
        except Exception:
            return jsonify({"error": f"{idx + 1}번째 이미지를 처리하지 못했어요. 다시 시도해요."}), 400

    # 프롬프트 로드
    try:
        system_prompt = load_prompt(contract_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Claude Vision API 호출 — 이미지 전체를 하나의 메시지에 담아 전송
    content = []
    for i, img_b64 in enumerate(normalized):
        if len(normalized) > 1:
            content.append({"type": "text", "text": f"[{i + 1}페이지]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64,
            },
        })
    content.append({
        "type": "text",
        "text": f"이 계약서({'총 ' + str(len(normalized)) + '페이지' if len(normalized) > 1 else ''})를 분석해주세요.",
    })

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIError as e:
        return jsonify({"error": "분석 중 오류가 발생했어요. 잠시 후 다시 시도해요."}), 502

    # 응답 파싱
    raw_text = response.content[0].text.strip()
    try:
        # { ... } 범위 직접 추출 — 코드블록/설명 텍스트 등 노이즈 제거
        start = raw_text.index('{')
        end = raw_text.rindex('}') + 1
        result = json.loads(raw_text[start:end])
    except (ValueError, json.JSONDecodeError):
        return jsonify({"error": "분석 결과를 처리하지 못했어요. 다시 시도해요."}), 500

    # 이미지 데이터는 응답에 포함하지 않음 (메모리에서 폐기됨)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") == "development"

    cert_path = os.environ.get("MTLS_CERT_PATH")
    key_path = os.environ.get("MTLS_KEY_PATH")

    if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
        # mTLS 인증서 있으면 SSL 적용
        ssl_context = (cert_path, key_path)
        print(f"[mTLS] 인증서 적용됨: {cert_path}")
        app.run(host="0.0.0.0", port=port, debug=debug, ssl_context=ssl_context)
    else:
        # 개발 환경: 인증서 없이 실행
        print("[mTLS] 인증서 없음 — HTTP로 실행 (개발 모드)")
        app.run(host="0.0.0.0", port=port, debug=debug)
