import os

from dotenv import load_dotenv
from pathlib import Path

# 1. 환경 구분
env = os.getenv("ENV", "test")

env_file_mapping = {
    "test": ".env.test",
    "dev": ".env.dev",
    "prod": ".env.prod"
}

# 2. COMMON_SDK_ROOT 설정: common_sdk 
COMMON_SDK_ROOT = Path(__file__).resolve().parent

# 3. .env 파일 로드
env_file = env_file_mapping.get(env, ".env.test")
env_path = COMMON_SDK_ROOT / env_file
load_dotenv(env_path)

# 4. 프로젝트 루트 설정
env_project_root_mapping = {
    "test": COMMON_SDK_ROOT.parent.parent.parent,
    "dev": Path("/app"),
    "prod": Path("/app")
}
"""
test: Tabula-AI
dev: app
prod: app
"""
PROJECT_ROOT = env_project_root_mapping.get(env, COMMON_SDK_ROOT.parent.parent.parent)

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

    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME_KOR_DEN_CONTENTS = os.getenv("INDEX_NAME_KOR_DEN_CONTENTS")
    INDEX_NAME_ENG_DEN_CONTENTS = os.getenv("INDEX_NAME_ENG_DEN_CONTENTS")
    INDEX_NAME_KOR_SPA_CONTENTS = os.getenv("INDEX_NAME_KOR_SPA_CONTENTS")
    INDEX_NAME_ENG_SPA_CONTENTS = os.getenv("INDEX_NAME_ENG_SPA_CONTENTS")

    # MongoDB 
    MONGO_HOST = os.getenv("MONGO_HOST")
    MONGO_PORT = int(os.getenv("MONGO_PORT"))
    MONGO_USERNAME = os.getenv("MONGO_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_DATABASE = os.getenv("MONGO_DATABASE")

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
    PROMPT_BASE_PATH = os.getenv("PROMPT_BASE_PATH", os.path.join(COMMON_SDK_ROOT, "prompts"))
    PROMPT_BASE_PATH = Path(PROMPT_BASE_PATH)

settings = Settings()

