import os
import base64

from note_sdk.parsing.base import BaseNode
from note_sdk.parsing.state import ParseState
from note_sdk.config import settings
from common_sdk.get_logger import get_logger

# 로그 설정
logger = get_logger()


# 문서 추출 이미지 저장 클래스
class ExportImage(BaseNode):

    def __init__(self, verbose=False, use_relative_path=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.use_relative_path = use_relative_path

    def save_to_png(self, base64_encoding, task_id, page, index):
        # settings.get_image_dir() 사용
        image_dir = settings.get_image_dir(task_id)
        os.makedirs(image_dir, exist_ok=True)

        # 이미지 파일명 생성
        image_filename = f"image_{page}_{index}.png"
        image_path = os.path.join(image_dir, image_filename)

        # base64 디코딩 및 이미지 저장
        image_data = base64.b64decode(base64_encoding)
        with open(image_path, "wb") as f:
            f.write(image_data)

        return image_path

    def execute(self, state: ParseState):
        task_id = state["task_id"]

        for elem in state["elements_from_parser"]:
            if elem["category"] in ["figure", "chart"]:
                base64_encoding = elem.get("base64_encoding")
                if base64_encoding:
                    image_path = self.save_to_png(
                        base64_encoding,
                        task_id,
                        elem["page"],
                        elem["id"],
                    )
                    elem["png_filepath"] = image_path

        return {"elements_from_parser": state["elements_from_parser"]}
    

# 문서 내용 마크다운 형식으로 저장하는 클래스
class ExportMarkdown(BaseNode):

    def __init__(
        self,
        ignore_new_line_in_text=False,
        show_image=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(verbose=verbose, **kwargs)
        self.ignore_new_line_in_text = ignore_new_line_in_text
        self.show_image = show_image
        self.separator = "\n\n"

    def execute(self, state: ParseState):
        task_id = state["task_id"]
        document_id = os.path.splitext(os.path.basename(state["filepath"]))[0]

        md_path = settings.get_markdown_path(task_id, document_id)
        os.makedirs(md_path.parent, exist_ok=True)

        with open(md_path, "w", encoding="utf-8") as f:
            for elem in state["elements_from_parser"]:
                if elem["category"] in ["header", "footer", "footnote"]:
                    continue

                if elem["category"] in ["figure", "chart"]:
                    if self.show_image and "png_filepath" in elem:
                        f.write(f"![]({elem['png_filepath']}){self.separator}")

                elif elem["category"] in ["paragraph"]:
                    if self.ignore_new_line_in_text:
                        f.write(elem["content"]["markdown"].replace("\n", " ") + self.separator)
                    else:
                        f.write(elem["content"]["markdown"] + self.separator)
                else:
                    f.write(elem["content"]["markdown"] + self.separator)

        logger.info(f"Markdown file successfully created: {md_path}")
        return {"export": [str(md_path)]}
