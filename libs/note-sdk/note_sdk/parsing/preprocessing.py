import os
import base64
from note_sdk.parsing.state import ParseState
from note_sdk.parsing.element import Element
from note_sdk.parsing.base import BaseNode
from common_sdk.get_logger import get_logger
from note_sdk.config import settings

logger = get_logger()

"""
문서 전처리 관련 기능
"""

IMAGE_TYPES = ["figure", "chart"]
TEXT_TYPES = ["text", "equation", "caption", "paragraph", "list", "index", "heading1"]


# 요소 생성 노드 클래스
class CreateElementsNode(BaseNode):
    def __init__(self, verbose=False, add_newline=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.add_newline = add_newline
        self.newline = "\n" if add_newline else ""

    def save_base64_image(self, base64_str, task_id, page_num, element_id):
        """base64 인코딩된 이미지를 파일로 저장하는 함수"""
        # settings.get_image_dir() 사용
        image_dir = settings.get_image_dir(task_id)
        os.makedirs(image_dir, exist_ok=True)
        
        # 이미지 파일명 생성
        img_filename = f"image_{page_num}_{element_id}.png"
        img_filepath = os.path.join(image_dir, img_filename)
        
        # base64 디코딩 및 이미지 저장
        img_data = base64.b64decode(base64_str)
        with open(img_filepath, "wb") as f:
            f.write(img_data)
        return img_filepath

    def execute(self, state: ParseState) -> ParseState:
        post_processed_elements = []
        task_id = state["task_id"]
        document_id = task_id

        for element in state["elements_from_parser"]:
            elem = None
            if element["category"] in ["footnote", "header", "footer"]:
                continue

            if element["category"] in ["equation"]:
                # (markdown only)
                # equation
                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )

            elif element["category"] in ["figure", "chart"]:
                # (markdown + image crop/image save)
                # figure, chart
                image_filename = self.save_base64_image(
                    element["base64_encoding"],
                    task_id,
                    element["page"],
                    element["id"],
                )

                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    base64_encoding=element["base64_encoding"],
                    image_filename=image_filename,
                    page=element["page"],
                    id=element["id"],
                    coordinates=element["coordinates"],
                )
            elif element["category"] in ["heading1"]:
                # (text w/ heading)
                # heading1
                elem = Element(
                    category=element["category"],
                    content=f'# {element["content"]["text"]}{self.newline}',
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )
            elif element["category"] in ["caption", "paragraph", "list", "index"]:
                # (text)
                # caption, paragraph, list
                elem = Element(
                    category=element["category"],
                    content=element["content"]["text"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )

            if elem is not None:
                post_processed_elements.append(elem)

        return {"elements": post_processed_elements}


# 엔티티 병합 노드
class MergeEntityNode(BaseNode):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]

        for elem in state["extracted_image_entities"]:
            for e in elements:
                if elem.id == e.id:
                    e.entity = elem.entity
                    break

        return {"elements": elements}


# 요소 재구성 노드
class ReconstructElementsNode(BaseNode):
    def __init__(self, verbose=False, use_relative_path=True):
        super().__init__(verbose)
        self.use_relative_path = use_relative_path

    def add_src_to_markdown(self, image_filename):
        """마크다운 이미지 문법에 src 경로를 추가하는 함수

        상대 경로를 사용하도록 설정된 경우 ./images/{category}/{basename} 형식의 상대 경로를 사용하고,
        그렇지 않은 경우 절대 경로를 사용합니다.
        """
        if not image_filename:
            return ""

        if not self.use_relative_path:
            # 절대 경로 사용
            abs_image_path = os.path.abspath(image_filename)
            image_md = f"![](file:///{abs_image_path})"
            return image_md

        # 상대 경로 사용
        # 파일 경로가 file:// 프로토콜을 사용하는 경우 제거
        if isinstance(image_filename, str) and image_filename.startswith("file:///"):
            image_filename = image_filename[7:]  # 'file:///' 제거
        elif isinstance(image_filename, str) and image_filename.startswith("file://"):
            image_filename = image_filename[6:]  # 'file://' 제거

        # 이미 상대 경로인 경우 (./images/ 또는 images/ 로 시작하는 경우)
        if isinstance(image_filename, str) and (
            image_filename.startswith("./images/")
            or image_filename.startswith("images/")
        ):
            return f"![]({image_filename})"

        # 파일명 추출
        basename = os.path.basename(image_filename)

        # 파일명으로부터 카테고리 추출
        parts = basename.split("_")
        if len(parts) >= 2:
            category_part = parts[1].lower()
            # 알려진 카테고리 중 하나인지 확인
            if category_part in ["figure", "chart"]:
                category = category_part
            else:
                # 기본값은 figure
                category = "figure"
        else:
            # 파일명에서 카테고리를 추출할 수 없는 경우 기본값 사용
            category = "figure"

        # 상대 경로 생성 (./images/{category}/{basename} 형식)
        rel_path = os.path.join(".", "images", category, basename)
        rel_path = rel_path.replace("\\", "/")  # 경로 구분자 통일

        return f"![]({rel_path})"

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]
        filepath = state["filepath"]
        # 파일 경로에서 basename 추출
        basename = os.path.basename(filepath)

        pages = sorted(list(state["texts_by_page"].keys()))
        max_page = pages[-1]

        reconstructed_elements = dict()
        for page_num in range(max_page + 1):
            reconstructed_elements[int(page_num)] = {
                "text": "",
                "image": [],
            }

        for elem in elements:
            if elem.category in IMAGE_TYPES:
                image_elem = {
                    "content": self.add_src_to_markdown(elem.image_filename)
                    + "\n\n"
                    + elem.entity,
                    "metadata": {
                        "image": self.add_src_to_markdown(elem.image_filename),
                        "entity": elem.entity,
                        "page": elem.page,
                        "source": basename,  # filepath 대신 basename 사용
                    },
                }
                reconstructed_elements[elem.page]["image"].append(image_elem)
            elif elem.category in TEXT_TYPES:
                reconstructed_elements[elem.page]["text"] += elem.content

        return {"reconstructed_elements": reconstructed_elements}