from typing import Dict
from dataclasses import dataclass
from copy import deepcopy

"""
문서 요소(Element) 데이터 클래스 정의
"""
@dataclass
class Element:
    category: str
    content: str = ""
    html: str = ""
    markdown: str = ""
    base64_encoding: str = None
    image_filename: str = None
    page: int = None
    id: int = None
    coordinates: list[Dict] = None
    entity: str = ""

    def copy(self):
        return deepcopy(self)
