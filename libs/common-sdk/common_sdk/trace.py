import os
from .config import settings


def langsmith(project_name=None, set_enable=True):

    if set_enable:        
        langsmith_key = settings.LANGSMITH_API_KEY

        if langsmith_key.strip() == "":
            print(
                "LangChain/LangSmith API Key Not Found "
            )
            return

        # os.environ["LANGSMITH_ENDPOINT"] = (
        #     "https://api.smith.langchain.com"  # LangSmith API 엔드포인트
        # )
        # os.environ["LANGSMITH_TRACING"] = "true"  # true: 활성화
        # os.environ["LANGSMITH_PROJECT"] = project_name  # 프로젝트명
        print(f"LangSmith Trace Start\n[Project Name]\n{settings.LANGSMITH_PROJECT}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"  # false: 비활성화
        print("LangSmith Trace Disable")


# def env_variable(key, value):
#     os.environ[key] = value
