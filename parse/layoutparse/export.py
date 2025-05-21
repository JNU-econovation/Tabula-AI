import os
import base64
import pandas as pd
from bs4 import BeautifulSoup
from layoutparse.base import BaseNode
from layoutparse.state import ParseState

# 문서 파싱 결과를 다양한 형식(이미지, 마크다운, CSV)으로 내보내는 모듈입니다.
# 각 클래스는 특정 형식으로 변환하는 기능을 담당합니다.


class ExportImage(BaseNode):
    """문서에서 추출한 이미지를 PNG 파일로 저장하는 클래스입니다.

    base64로 인코딩된 이미지 데이터를 디코딩하여 PNG 파일로 저장합니다.
    저장된 이미지는 카테고리별로 분류되어 저장됩니다.
    """

    def __init__(self, verbose=False, use_relative_path=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.use_relative_path = use_relative_path

    def save_to_png(self, base64_encoding, dirname, basename, category, page, index):
        # document_id 추출
        document_id = os.path.splitext(basename)[0]
        
        # result/images/{document_id} 디렉토리 생성
        image_dir = os.path.join("result", "images", document_id)
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
            if elem["category"] in ["figure", "chart", "table"]:
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
    

class ExportMarkdown(BaseNode):
    """문서 내용을 마크다운 형식으로 변환하여 저장하는 클래스입니다.

    이미지는 로컬 파일 경로를 참조하는 방식으로 저장됩니다.
    테이블은 마크다운 테이블 문법으로 변환됩니다.
    텍스트의 줄바꿈 처리를 선택적으로 할 수 있습니다.
    """

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

        # result/md 디렉토리 생성
        md_dir = os.path.join("result", "md")
        os.makedirs(md_dir, exist_ok=True)

        md_filepath = os.path.join(md_dir, f"{document_id}.md")

        with open(md_filepath, "w", encoding="utf-8") as f:
            for elem in state["elements_from_parser"]:
                if elem["category"] in ["header", "footer", "footnote"]:
                    continue

                if elem["category"] in ["figure", "chart"]:
                    if self.show_image and "png_filepath" in elem:
                        f.write(f"![]({elem['png_filepath']}){self.separator}")

                elif elem["category"] in ["table"]:
                    if self.show_image and "png_filepath" in elem:
                        f.write(f"![]({elem['png_filepath']}){self.separator}")
                    f.write(elem["content"]["markdown"] + self.separator)

                elif elem["category"] in ["paragraph"]:
                    if self.ignore_new_line_in_text:
                        f.write(elem["content"]["markdown"].replace("\n", " ") + self.separator)
                    else:
                        f.write(elem["content"]["markdown"] + self.separator)
                else:
                    f.write(elem["content"]["markdown"] + self.separator)

        self.log(f"마크다운 파일이 성공적으로 생성되었습니다: {md_filepath}")
        return {"export": [md_filepath]}


class ExportTableCSV(BaseNode):
    """문서에서 추출한 테이블을 CSV 파일로 저장하는 클래스입니다.

    HTML 테이블을 pandas DataFrame으로 변환한 후 CSV 파일로 저장합니다.
    """

    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        filepath = state["filepath"]
        basename = os.path.basename(filepath)
        document_id = os.path.splitext(basename)[0]

        # result/tables/{document_id} 디렉토리 생성
        table_dir = os.path.join("result", "tables", document_id)
        os.makedirs(table_dir, exist_ok=True)

        for elem in state["elements_from_parser"]:
            if elem["category"] == "table":
                # HTML 테이블을 pandas DataFrame으로 변환
                soup = BeautifulSoup(elem["content"]["html"], "html.parser")
                df = pd.read_html(str(soup))[0]
                
                # CSV 파일 저장
                table_path = os.path.join(table_dir, f"table_{elem['id']}.csv")
                df.to_csv(table_path, index=False, encoding="utf-8-sig")
                
                self.log(f"CSV 파일이 성공적으로 생성되었습니다: {table_path}")

        return {"export": [table_dir]}



# class ExportHTML(BaseNode):
#     """문서 내용을 HTML 형식으로 변환하여 저장하는 클래스입니다.

#     이미지는 로컬 파일 경로를 참조하는 방식으로 저장됩니다.
#     또는 이미지가 포함된 경우 base64 인코딩을 통해 HTML 내에 직접 삽입할 수 있습니다.
#     텍스트의 줄바꿈 처리를 선택적으로 할 수 있습니다.
#     """

#     def __init__(
#         self,
#         ignore_new_line_in_text=False,
#         show_image=True,
#         verbose=False,
#         **kwargs,
#     ):
#         super().__init__(verbose=verbose, **kwargs)
#         self.ignore_new_line_in_text = ignore_new_line_in_text
#         self.show_image = show_image

#     def execute(self, state: ParseState):
#         filepath = state["filepath"]
#         basename = os.path.basename(filepath)
#         document_id = os.path.splitext(basename)[0]

#         # result/html 디렉토리 생성
#         html_dir = os.path.join("result", "html")
#         os.makedirs(html_dir, exist_ok=True)

#         html_filepath = os.path.join(html_dir, f"{document_id}.html")

#         with open(html_filepath, "w", encoding="utf-8") as f:
#             f.write('<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n')
#             f.write(f"<title>{document_id}</title>\n")
#             f.write("<style>\nimg { max-width: 100%; }\n</style>\n")
#             f.write("</head>\n<body>\n")

#             for elem in state["elements_from_parser"]:
#                 if elem["category"] in ["header", "footer", "footnote"]:
#                     continue

#                 if elem["category"] in ["figure", "chart"]:
#                     if self.show_image and "png_filepath" in elem:
#                         f.write(f'<img src="{elem["png_filepath"]}" alt="image" />\n')
#                     f.write(elem["content"]["html"] + "\n")

#                 elif elem["category"] in ["table"]:
#                     if self.show_image and "png_filepath" in elem:
#                         f.write(f'<img src="{elem["png_filepath"]}" alt="table" />\n')
#                     f.write(elem["content"]["html"] + "\n")

#                 else:
#                     if self.ignore_new_line_in_text:
#                         f.write(elem["content"]["html"].replace("<br>", " "))
#                     else:
#                         f.write(elem["content"]["html"])

#             f.write("\n</body>\n</html>")

#         self.log(f"HTML 파일이 성공적으로 생성되었습니다: {html_filepath}")
#         return {"export": [html_filepath]}
