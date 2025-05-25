import os
import time
import shutil
import requests
from note_sdk.parsing.base import BaseNode
from note_sdk.parsing.state import ParseState
from note_sdk.config import settings
from common_sdk.config import settings as common_settings
from common_sdk.get_logger import get_logger

logger = get_logger()

"""
Upstage API 호출 관련 기능
"""

DEFAULT_CONFIG = {
    "ocr": False,
    "coordinates": True,
    "output_formats": "['text', 'markdown']",
    "model": "document-parse",
    "base64_encoding": "['figure', 'chart']",
}

# 문서 파싱 노드
class DocumentParseNode(BaseNode):
    def __init__(self, use_ocr=False, verbose=False, output_dir="result", **kwargs):
        """
        param use_ocr: OCR 사용 여부
        param verbose: 상세 로그 출력 여부
        param output_dir: 출력 디렉토리 경로
        """
        super().__init__(verbose=verbose, **kwargs)
        self.api_key = common_settings.UPSTAGE_API_KEY
        if not self.api_key:
            logger.error("UPSTAGE_API_KEY is not set")
            raise ValueError()
        self.config = DEFAULT_CONFIG
        if use_ocr:
            self.config["ocr"] = True
        self.output_dir = output_dir
        self.temp_dir = None

    def upstage_layout_analysis(self, input_file, task_id):
        """
        Upstage의 Document Parse API를 호출하여 문서 분석을 수행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        try:
            # settings.get_temp_dir() 사용
            self.temp_dir = settings.get_temp_dir(task_id)
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.info(f"임시 디렉토리 생성: {self.temp_dir}")

            # API 요청 헤더 설정
            headers = {"Authorization": f"Bearer {self.api_key}"}

            # 분석할 PDF 파일 열기
            files = {"document": open(input_file, "rb")}

            # API 요청 보내기
            response = requests.post(
                "https://api.upstage.ai/v1/document-ai/document-parse",
                headers=headers,
                data=self.config,
                files=files,
            )

            # API 응답 처리 및 결과 저장
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                logger.error(f"API request failed. Status code: {response.status_code}")
                raise ValueError()
        finally:
            if 'files' in locals():
                files['document'].close()

    def parse_start_end_page(self, filepath):
        # 파일명에서 페이지 번호 추출 (예: WorldEnergyOutlook2024_0040_0049.pdf)
        filename = os.path.basename(filepath)
        # .pdf 확장자 제거
        name_without_ext = filename.rsplit(".", 1)[0]

        # 파일명 형식 검증
        try:
            # 파일명이 최소 9자 이상이어야 함
            if len(name_without_ext) < 9:
                return (-1, -1)

            # 마지막 9자리 추출 (예: 0040_0049)
            page_numbers = name_without_ext[-9:]

            # 형식이 ####_#### 인지 검증 (숫자4개_숫자4개)
            if not (
                page_numbers[4] == "_"
                and page_numbers[:4].isdigit()
                and page_numbers[5:].isdigit()
            ):
                return (-1, -1)

            # 시작 페이지와 끝 페이지 추출
            start_page = int(page_numbers[:4])
            end_page = int(page_numbers[5:])

            # 시작 페이지가 끝 페이지보다 크면 검증 실패
            if start_page > end_page:
                return (-1, -1)

            return (start_page, end_page)

        except (IndexError, ValueError):
            return (-1, -1)

    def execute(self, state: ParseState):
        if "task_id" not in state:
            raise ValueError("task_id is required in state")
        
        start_time = time.time()
        logger.info(f"Start Parsing: {state['working_filepath']}")

        try:
            filepath = state["working_filepath"]
            task_id = state["task_id"]
            parsed_json = self.upstage_layout_analysis(filepath, task_id)

            # 파일명에서 시작 페이지 추출
            start_page, _ = self.parse_start_end_page(filepath)
            page_offset = start_page - 1 if start_page != -1 else 0

            data = parsed_json

            # 페이지 번호와 ID 재계산
            for element in data["elements"]:
                element["page"] += page_offset

            metadata = {
                "api": data.pop("api"),
                "model": data.pop("model"),
                "usage": data.pop("usage"),
            }

            duration = time.time() - start_time
            logger.info(f"Finished Parsing in {duration:.2f} seconds")

            return {
                "metadata": [metadata],
                "raw_elements": [data["elements"]],
                "temp_dir": self.temp_dir,
                "task_id": task_id
            }
        except Exception as e:
            logger.error(f"파싱 중 오류 발생: {str(e)}")
            raise


class PostDocumentParseNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        elements_list = state["raw_elements"]
        id_counter = 0  # ID를 순차적으로 부여하기 위한 카운터
        post_processed_elements = []

        for elements in elements_list:
            for element in elements:
                elem = element.copy()
                # ID 순차적으로 부여
                elem["id"] = id_counter
                id_counter += 1

                post_processed_elements.append(elem)

        logger.info(f"Total Post-processed Elements: {id_counter}")

        pages_count = 0
        metadata = state["metadata"]

        for meta in metadata:
            for k, v in meta.items():
                if k == "usage":
                    pages_count += int(v["pages"])

        total_cost = pages_count * 0.01

        logger.info(f"Total Cost: ${total_cost:.2f}")

        # 임시 디렉토리 정리
        temp_root_dir = settings.get_temp_dir(state["filepath"])
        if os.path.exists(temp_root_dir):
            try:
                shutil.rmtree(temp_root_dir)
                logger.info(f"Temporary directory cleaned: {temp_root_dir}")
            except Exception as e:
                logger.error(f"Error cleaning temporary directory: {str(e)}")

        # 재정렬된 elements를 state에 업데이트
        return {
            "elements_from_parser": post_processed_elements,
            "total_cost": total_cost,
        }


class WorkingQueueNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        # filepath가 없는 경우 처리
        if "filepath" not in state:
            raise ValueError("filepath is required in state")

        working_filepath = state.get("working_filepath", None)
        
        if not working_filepath or working_filepath == "":
            if len(state["split_filepaths"]) > 0:
                working_filepath = state["split_filepaths"][0]
            else:
                working_filepath = "<<FINISHED>>"
        else:
            if working_filepath == "<<FINISHED>>":
                return {
                    "working_filepath": "<<FINISHED>>",
                    "filepath": state["filepath"]  # filepath 유지
                }

            current_index = state["split_filepaths"].index(working_filepath)
            if current_index + 1 < len(state["split_filepaths"]):
                working_filepath = state["split_filepaths"][current_index + 1]
            else:
                working_filepath = "<<FINISHED>>"

        return {
            "working_filepath": working_filepath,
            "filepath": state["filepath"]  # filepath 유지
        }


def continue_parse(state: ParseState):
    if state["working_filepath"] == "<<FINISHED>>":
        return False
    else:
        return True
