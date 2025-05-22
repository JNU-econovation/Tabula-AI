from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Settings:
    
    # API Key
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Pinecone
    INDEX_NAME_KOR_DEN_CONTENTS = os.getenv("INDEX_NAME_KOR_DEN_CONTENTS")
    INDEX_NAME_ENG_DEN_CONTENTS = os.getenv("INDEX_NAME_ENG_DEN_CONTENTS")
    
    INDEX_NAME_KOR_SPA_CONTENTS = os.getenv("INDEX_NAME_KOR_SPA_CONTENTS")
    INDEX_NAME_ENG_SPA_CONTENTS = os.getenv("INDEX_NAME_ENG_SPA_CONTENTS")

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
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
    S3_BUCKET = os.getenv("S3_BUCKET", "")


    # Path setting
    LOG_PATH = os.getenv("LOG_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"))


settings = Settings()

