from openai import OpenAI
from path.config import settings
from typing import List, Literal

# Upstage 클라이언트
upstage = OpenAI(
    api_key=settings.UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# OpenAI 클라이언트
client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_embedding(text: str, language: Literal["ko", "en"] = "ko") -> List[float]:
    """
    텍스트에 대한 임베딩 생성
    
    Args:
        text: 임베딩할 텍스트
        language: 언어 설정 ("ko" 또는 "en")
        
    Returns:
        임베딩 벡터
    """
    try:
        text = text.replace("\n", " ")
        
        if language == "ko":
            response = upstage.embeddings.create(
                input=[text], 
                model="embedding-query"
            )
            print(f"한국어 임베딩 생성 성공")
            return response.data[0].embedding
        else:
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            print(f"영어 임베딩 생성 성공")
            return response.data[0].embedding
            
    except Exception as e:
        print(f"임베딩 생성 실패: {str(e)}")
        raise

def test_embedding():
    """임베딩 테스트"""
    # 테스트 텍스트
    korean_text = "안녕하세요. 이것은 한국어 테스트입니다."
    english_text = "Hello. This is an English test."
    
    # 한국어 임베딩 테스트
    ko_embedding = get_embedding(korean_text, "ko")
    print(f"한국어 임베딩 차원: {len(ko_embedding)}")
    
    # 영어 임베딩 테스트
    en_embedding = get_embedding(english_text, "en")
    print(f"영어 임베딩 차원: {len(en_embedding)}")

if __name__ == "__main__":
    test_embedding()