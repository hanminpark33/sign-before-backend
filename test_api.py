"""
Flask 서버 테스트 스크립트
실행 전: python app.py 로 서버 먼저 켜두세요.
사용법: python test_api.py [이미지경로]
예시:  python test_api.py contract.jpg
"""

import sys
import json
import base64
import requests

SERVER_URL = "https://localhost:5001"


def test_health():
    r = requests.get(f"{SERVER_URL}/health", verify=False)
    print(f"[health] {r.json()}")


def test_analyze(image_path: str):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "image_base64": image_base64,
        "contract_type": "lease",
        "consent_verified": True,
    }

    print(f"[analyze] 이미지 전송 중... ({image_path})")
    r = requests.post(f"{SERVER_URL}/analyze", json=payload, timeout=60, verify=False)

    if r.status_code == 200:
        result = r.json()
        print("\n=== 분석 결과 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"[오류] {r.status_code}: {r.text}")


if __name__ == "__main__":
    test_health()

    if len(sys.argv) > 1:
        test_analyze(sys.argv[1])
    else:
        print("\n이미지 경로를 인수로 넣어주세요.")
        print("예시: python test_api.py contract.jpg")
