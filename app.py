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
import json
import base64
import anthropic
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

app = Flask(__name__)
CORS(app)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4MB (Claude 5MB 제한 여유분)
MAX_DIMENSION = 1920


def compress_image(image_base64: str) -> str:
    """이미지를 Claude API 제한(5MB) 이내로 압축 후 base64 반환."""
    raw = base64.b64decode(image_base64)
    if len(raw) <= MAX_IMAGE_BYTES:
        return image_base64  # 이미 작으면 그대로

    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # 해상도 축소
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    # JPEG 품질 조절로 목표 크기 맞추기
    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            return base64.b64encode(buf.getvalue()).decode()

    # 최저 품질로 강제 압축
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30)
    return base64.b64encode(buf.getvalue()).decode()


def load_prompt(contract_type: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{contract_type}.txt")
    if not os.path.exists(path):
        raise ValueError(f"지원하지 않는 계약서 유형: {contract_type}")
    with open(path, encoding="utf-8") as f:
        return f.read()


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

    image_base64 = data.get("image_base64")
    contract_type = data.get("contract_type", "lease")

    if not image_base64:
        return jsonify({"error": "이미지 데이터가 없어요."}), 400

    # base64 유효성 검사
    try:
        base64.b64decode(image_base64, validate=True)
    except Exception:
        return jsonify({"error": "이미지 형식을 확인해요."}), 400

    # 이미지 압축 (Claude API 5MB 제한 대응)
    try:
        image_base64 = compress_image(image_base64)
    except Exception:
        return jsonify({"error": "이미지를 처리하지 못했어요. 다시 시도해요."}), 400

    # 프롬프트 로드
    try:
        system_prompt = load_prompt(contract_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Claude Vision API 호출 (이미지는 메모리에서만 처리)
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "이 계약서를 분석해주세요.",
                        },
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        return jsonify({
            "error": "분석을 완료하지 못했어요. 사진을 더 밝고 선명하게 찍어 다시 시도해요."
        }), 502

    # 응답 파싱
    raw_text = response.content[0].text.strip()
    try:
        # 마크다운 코드블록 제거
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        return jsonify({
            "error": "분석 결과를 처리하지 못했어요. 다시 시도해요."
        }), 500

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
