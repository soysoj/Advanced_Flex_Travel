import os
from openai import OpenAI
from dotenv import load_dotenv

# 현재 폴더에 있는 .env 파일 로드
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"1. API Key 확인: {api_key[:10]}... (로드 성공)")

try:
    # Upstage 가이드대로 설정
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1"
    )

    print("2. Solar Pro 2 모델에게 인사하는 중...")
    response = client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {"role": "user", "content": "Hi! Are you ready?"}
        ]
    )
    print("\n✅ 연결 성공! 응답 내용:")
    print(response.choices[0].message.content)

except Exception as e:
    print("\n❌ 실패... 에러 메시지:")
    print(e)