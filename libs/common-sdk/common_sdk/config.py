import os

from dotenv import load_dotenv
from pathlib import Path

# 프로젝트 루트 디렉토리 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# common-sdk의 prompts 디렉토리 경로 설정
COMMON_SDK_ROOT = os.path.dirname(os.path.dirname(__file__))

# common-sdk .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Settings:

    # Swagger
    SWAGGER_USERNAME = os.getenv("SWAGGER_USERNAME")
    SWAGGER_PASSWORD = os.getenv("SWAGGER_PASSWORD")
    
    # API Key
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    OPENAI_API_KEY_J = os.getenv("OPENAI_API_KEY_J")
    OPENAI_API_KEY_K = os.getenv("OPENAI_API_KEY_K")
    OPENAI_API_KEY_B = os.getenv("OPENAI_API_KEY_B")

    # MongoDB 
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_DATABASE = os.getenv("MONGO_DATABASE", "tabula")

    @property
    def MONGO_URI(self) -> str:
        """MongoDB 연결 URI를 생성합니다."""
        if self.MONGO_USERNAME and self.MONGO_PASSWORD:
            return f"mongodb://{self.MONGO_USERNAME}:{self.MONGO_PASSWORD}@{self.MONGO_HOST}:{self.MONGO_PORT}"
        return f"mongodb://{self.MONGO_HOST}:{self.MONGO_PORT}"

    # AWS S3 
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    S3_BUCKET = os.getenv("S3_BUCKET")

    # Auth
    JWT_SECRET = os.getenv("JWT_SECRET")
    ALGORITHM = os.getenv("ALGORITHM")

    # Path setting
    LOG_PATH = os.getenv("LOG_PATH", os.path.join(PROJECT_ROOT, "logs"))
    LOG_PATH = Path(LOG_PATH)

    # SDK-specific log directory settings
    NOTE_SDK_LOG_PATH = os.getenv("NOTE_SDK_LOG_PATH", os.path.join(PROJECT_ROOT, "libs", "note-sdk", "logs"))
    NOTE_SDK_LOG_PATH = Path(NOTE_SDK_LOG_PATH)
    RESULT_SDK_LOG_PATH = os.getenv("RESULT_SDK_LOG_PATH", os.path.join(PROJECT_ROOT, "libs", "result-sdk", "logs"))
    RESULT_SDK_LOG_PATH = Path(RESULT_SDK_LOG_PATH)

    # Prompt Base Path
    PROMPT_BASE_PATH = os.getenv("PROMPT_BASE_PATH", os.path.join(COMMON_SDK_ROOT, "common_sdk", "prompts"))
    PROMPT_BASE_PATH = Path(PROMPT_BASE_PATH)

settings = Settings()

