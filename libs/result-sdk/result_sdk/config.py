import os
from dotenv import load_dotenv
from pathlib import Path

# 경로 설정: result-sdk
RESULT_SDK_ROOT = os.path.dirname(__file__)

# result-sdk의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Settings:

    def __init__(self):

        # Pinecone 설정
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.INDEX_NAME = os.getenv("INDEX_NAME")

        # OpenAI 설정
        self.OPENAI_API_KEY_J = os.getenv("OPENAI_API_KEY_J")
        self.OPENAI_API_KEY_K = os.getenv("OPENAI_API_KEY_K")
        self.OPENAI_API_KEY_B = os.getenv("OPENAI_API_KEY_B")

settings = Settings()

# 인덱스 목록 확인
# from pinecone import Pinecone

# pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# indexes = pc.list_indexes()
# print([idx.name for idx in indexes.indexes])