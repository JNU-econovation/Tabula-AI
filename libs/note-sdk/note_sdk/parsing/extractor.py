from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from openai import RateLimitError

from note_sdk.llm import MultiModal
from note_sdk.parsing.state import ParseState
from note_sdk.parsing.base import BaseNode
from note_sdk.parsing.preprocessing import IMAGE_TYPES
from common_sdk.config import settings as common_settings
from common_sdk.get_logger import get_logger
from common_sdk.sse import update_progress

logger = get_logger()

# 이미지 엔티티 추출 클래스
class PageElementsExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PageElementsExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]
        elements_by_page = dict()
        max_page = 0

        # 최대 페이지 번호 찾기
        for elem in elements:
            page_num = int(elem.page)
            max_page = max(max_page, page_num)
            if page_num not in elements_by_page:
                elements_by_page[page_num] = []
            if elem.category in IMAGE_TYPES:
                elements_by_page[page_num] = []
            elements_by_page[page_num].append(elem)

        texts_by_page = dict()
        images_by_page = dict()

        # 0부터 max_page까지 모든 페이지에 대해 빈 문자열 초기화
        for page_num in range(max_page + 1):
            texts_by_page[page_num] = ""
            images_by_page[page_num] = []

        # 실제 컨텐츠 채우기
        for page_num, elems in elements_by_page.items():
            for elem in elems:
                if elem.category in IMAGE_TYPES:
                    images_by_page[page_num].append(elem)
                else:
                    texts_by_page[page_num] += elem.content

        return {
            "texts_by_page": texts_by_page,
            "images_by_page": images_by_page,
        }


@chain
def image_entity_extractor(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=common_settings.OPENAI_API_KEY_J
    )

    system_prompt = """Extract key information and insights from an image based on the provided context.

Given the context related to the image, analyze and interpret the image to generate a structured output that includes a title, key details, entities, and hypothetical questions. Ensure that your response is coherent and follows the specified format.

# Steps

1. **Analyze the Context**: Understand the context provided in relation to the image. This will guide the interpretation and extraction process.
2. **Title Generation**: Create a concise and descriptive title that encapsulates the main theme or subject of the image.
3. **Details Extraction**: Identify and articulate key insights and details visible in the image.
4. **Entity Identification**: Recognize and list significant entities or objects present in the image.
5. **Hypothetical Questions**: Formulate relevant hypothetical questions that arise from the content of the image, encouraging deeper inquiry or reflection."""

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["context"]
        image_path = data_batch["image"]
        language = data_batch["language"]
        user_prompt_template = f"""# Output Format

- The output should be structured using the following tags:
- `<image>`: Wrap the entire output.
- `<title>`: Enclose the generated title.
- `<details>`: Include detailed insights extracted from the image.
- `<entities>`: List the identified entities.
- `<hypothetical_questions>`: Present the formulated hypothetical questions.
- Ensure all sections are filled appropriately, maintaining clarity and relevance to the context.
- The output must be written in {language}.

# Example

**Input**: 
Here is the context related to the image: 
{context}

**Output**:
<image>
<title>
The Rise of Artificial Intelligence in Modern Technology
</title>
<details>
The image depicts the integration of AI in various technological devices, highlighting advancements in automation and data processing.
</details>
<entities>
AI algorithms, robotics, smart devices
</entities>
<hypothetical_questions>
- How will AI continue to evolve in the next decade? 
- What are the ethical implications of AI in everyday life?
</hypothetical_questions>
</image>

# Notes

- Use the provided context to inform and enhance the extraction process.
- Ensure that the hypothetical questions are thought-provoking and relevant to the image's theme.
- Maintain clarity and coherence throughout the response.
- Be sure to include numerical values, proper nouns, terms, and teminologies."""
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    try:
        # 이미지 파일로 부터 질의
        answer = multimodal_llm.batch(
            image_paths, system_prompts, user_prompts, display_image=False
        )
        return answer
    except RateLimitError as e:
        logger.error(f"[ImageEntityExtractor] Rate limit exceeded: {str(e)}")
        update_progress(
            space_id=data_batches[0].get("space_id"),
            progress=-1,
            status={
                "status": "error",
                "error": {
                    "code": "FILE_413_2",
                    "reason": "파일(PDF)의 이미지 토큰값 허용 범위 초과입니다.",
                    "http_status": 413
                }
            }
        )
        return []  # 빈 리스트 반환
    except Exception as e:
        logger.error(f"[ImageEntityExtractor] Image processing error: {str(e)}")
        update_progress(
            space_id=data_batches[0].get("space_id"),
            progress=-1,
            status={
                "status": "error",
                "error": {
                    "code": "SSE_400_1",
                    "reason": "진행률 조회(SSE) 에러 발생",
                    "http_status": 400
                }
            }
        )
        return []  # 빈 리스트 반환


class ImageEntityExtractorNode(BaseNode):
    def __init__(self, verbose=False, language="ko", domain_type="academic", **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.name = "ImageEntityExtractorNode"
        self.language = language
        self.domain_type = domain_type

    def execute(self, state: ParseState) -> ParseState:
        # 모든 페이지의 이미지 요소들을 하나의 리스트로 합치기
        images_files = []
        for page_images in state["images_by_page"].values():
            images_files.extend(page_images)

        BATCH_SIZE = 10
        extracted_image_entities = []

        # 이미지를 배치 크기로 나누어 처리
        for i in range(0, len(images_files), BATCH_SIZE):
            batch = images_files[i : i + BATCH_SIZE]
            batch_data = []
            for image_element in batch:
                batch_data.append(
                    {
                        "image": image_element.image_filename,
                        "context": state["texts_by_page"][image_element.page],
                        "language": self.language,
                        "domain_type": self.domain_type
                    }
                )
            # 배치 단위로 처리
            batch_result = image_entity_extractor.invoke(batch_data)
            # Element 인스턴스로 변환하고 결과 추가
            for j, result in enumerate(batch_result):
                # Element 객체의 얕은 복사 대신 새로운 객체 생성
                element = batch[j].copy()  # 원본 Element 객체 사용
                element.entity = result
                extracted_image_entities.append(element)
        return {"extracted_image_entities": extracted_image_entities}