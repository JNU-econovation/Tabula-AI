import os
from path.config import settings


def langsmith(project_name=None, set_enable=True):

    if set_enable:        
        langsmith_key = settings.LANGSMITH_API_KEY

        if langsmith_key.strip() == "":
            print(
                "LangChain/LangSmith API Key가 설정되지 않았습니다. 참고: https://wikidocs.net/250954"
            )
            return

        # os.environ["LANGSMITH_ENDPOINT"] = (
        #     "https://api.smith.langchain.com"  # LangSmith API 엔드포인트
        # )
        # os.environ["LANGSMITH_TRACING"] = "true"  # true: 활성화
        # os.environ["LANGSMITH_PROJECT"] = project_name  # 프로젝트명
        print(f"LangSmith 추적을 시작합니다.\n[프로젝트명]\n{settings.LANGSMITH_PROJECT}")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"  # false: 비활성화
        print("LangSmith 추적을 하지 않습니다.")


# def env_variable(key, value):
#     os.environ[key] = value
