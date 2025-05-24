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

    def save_to_png(self, base64_encoding, dirname, basename, category, page, index):
        # document_id 추출
        document_id = os.path.splitext(basename)[0]
        
        # settings.RESULT_DIR 사용
        image_dir = os.path.join(settings.RESULT_DIR, "images", document_id)
        os.makedirs(image_dir, exist_ok=True)

        # 이미지 파일명을 document_id_Page_X_Index_Y.png 형식으로 생성
        image_filename = f"{document_id}_Page_{page}_Index_{index}.png"
        image_path = os.path.join(image_dir, image_filename)

        # base64 디코딩 및 이미지 저장
        image_data = base64.b64decode(base64_encoding)
        with open(image_path, "wb") as f:
            f.write(image_data)

        # 상대 경로 반환 (md 파일 기준)
        return f"../images/{document_id}/{image_filename}"

    def execute(self, state: ParseState):
        filepath = state["filepath"]
        basename = os.path.basename(filepath)
        document_id = os.path.splitext(basename)[0]

        for elem in state["elements_from_parser"]:
            if elem["category"] in ["figure", "chart"]:
                base64_encoding = elem.get("base64_encoding")
                if base64_encoding:
                    image_path = self.save_to_png(
                        base64_encoding,
                        "result",
                        basename,
                        elem["category"],
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
        filepath = state["filepath"]
        basename = os.path.basename(filepath)
        document_id = os.path.splitext(basename)[0]

        md_dir = os.path.join(settings.RESULT_DIR, "md")
        os.makedirs(md_dir, exist_ok=True)

        md_filepath = os.path.join(md_dir, f"{document_id}.md")

        with open(md_filepath, "w", encoding="utf-8") as f:
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

        logger.info(f"Markdown file successfully created: {md_filepath}")
        return {"export": [md_filepath]}
