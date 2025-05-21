from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Settings:
    
    UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    INDEX_NAME_KOR_DEN_CONTENTS = os.getenv("INDEX_NAME_KOR_DEN_CONTENTS")
    INDEX_NAME_ENG_DEN_CONTENTS = os.getenv("INDEX_NAME_ENG_DEN_CONTENTS")
    
    INDEX_NAME_KOR_SPA_CONTENTS = os.getenv("INDEX_NAME_KOR_SPA_CONTENTS")
    INDEX_NAME_ENG_SPA_CONTENTS = os.getenv("INDEX_NAME_ENG_SPA_CONTENTS")

    LOG_PATH = os.getenv("LOG_PATH")


settings = Settings()

