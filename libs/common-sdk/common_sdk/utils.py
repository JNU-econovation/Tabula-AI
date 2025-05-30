import tiktoken

from openai import OpenAI
from .config import settings
from typing import List, Literal

from common_sdk.exceptions import ExternalConnectionError
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()

# Upstage 클라이언트
upstage = OpenAI(
    api_key=settings.UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# OpenAI 클라이언트
client = OpenAI(api_key=settings.OPENAI_API_KEY_J)

# 토큰 수 확인
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    
    return num_tokens

# 텍스트 임베딩 생성
def get_embedding(text: str, language: Literal["ko", "en"] = "ko") -> List[float]:

    try:
        text = text.replace("\n", " ")
        
        if language == "ko":
            response = upstage.embeddings.create(
                input=[text], 
                model="embedding-query"
            )
            return response.data[0].embedding
        else:
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
            
    except Exception as e:
        logger.error(f"[get_embedding] Failed to create embedding: {str(e)}")
        raise ExternalConnectionError()