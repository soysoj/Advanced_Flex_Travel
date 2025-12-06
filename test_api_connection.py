#!/usr/bin/env python3
"""
Upstage API 연동 테스트 스크립트
flow_add_mpe.py의 Evaluatee 클래스를 사용하여 API 연결을 테스트합니다.
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# implement 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'implement'))

from agents.flow_add_mpe import Evaluatee
from agents.swarm.core import Swarm

def test_upstage_api():
    """Upstage API 연동 테스트"""
    
    print("=" * 60)
    print("Upstage API 연동 테스트 시작")
    print("=" * 60)
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하거나 환경 변수를 설정하세요.")
        return False
    
    print(f"✅ API Key 확인: {api_key[:10]}... (로드 성공)")
    print()
    
    # 테스트용 config 설정
    config = {
        "evaluatee": {
            "name": "test_evaluatee",
            "model": "solar-pro2",  # Upstage solar-pro2 모델
            "api_type": "openai",   # OpenAI 호환 API 사용
            "instructions": "You are a helpful assistant.",
            "client": {
                "api_key": api_key
            }
        }
    }
    
    try:
        print("1. Evaluatee 클래스 초기화 중...")
        evaluatee = Evaluatee(config=config)
        print(f"   ✅ 모델: {evaluatee.model}")
        print(f"   ✅ API 타입: {evaluatee.api_type}")
        print(f"   ✅ Client 타입: {type(evaluatee.client).__name__}")
        
        # base_url 확인
        if hasattr(evaluatee.client, 'base_url'):
            print(f"   ✅ Base URL: {evaluatee.client.base_url}")
        print()
        
        print("2. 간단한 API 호출 테스트 중...")
        swarm = Swarm()
        
        test_messages = [
            {"role": "user", "content": "안녕하세요! 간단히 인사만 해주세요."}
        ]
        
        response = swarm.run(
            agent=evaluatee,
            messages=test_messages,
            stream=False,
            debug=False
        )
        
        if response and response.messages:
            assistant_message = response.messages[-1]
            if isinstance(assistant_message, dict):
                content = assistant_message.get("content", "")
            else:
                content = str(assistant_message)
            
            print("   ✅ API 호출 성공!")
            print()
            print("   응답 내용:")
            print("   " + "-" * 50)
            print(f"   {content[:200]}")  # 처음 200자만 출력
            if len(content) > 200:
                print("   ...")
            print("   " + "-" * 50)
            print()
            
            print("=" * 60)
            print("✅ 모든 테스트 통과! Upstage API 연동이 정상적으로 작동합니다.")
            print("=" * 60)
            return True
        else:
            print("   ❌ 응답을 받지 못했습니다.")
            return False
            
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ 오류 발생!")
        print("=" * 60)
        print(f"오류 타입: {type(e).__name__}")
        print(f"오류 메시지: {str(e)}")
        print()
        import traceback
        print("상세 오류 정보:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_upstage_api()
    sys.exit(0 if success else 1)

