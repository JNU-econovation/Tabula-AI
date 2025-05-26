import tiktoken

from openai import OpenAI
from .config import settings
from typing import List, Literal

# Upstage 클라이언트
upstage = OpenAI(
    api_key=settings.UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# OpenAI 클라이언트
client = OpenAI(api_key=settings.OPENAI_API_KEY_J)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """문자열의 토큰 수를 계산"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 텍스트 임베딩 생성
def get_embedding(text: str, language: Literal["ko", "en"] = "ko") -> List[float]:
    """
    Args:
        text: 임베딩할 텍스트
        language: 언어 설정 ("ko" 또는 "en")
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