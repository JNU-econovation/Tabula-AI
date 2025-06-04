from pydantic import BaseModel
from typing import List, Dict, Any, Optional, TypedDict


class WrongAnswer(BaseModel):
    """전체 채점 결과 모델"""
    id: List[int]
    wrong_answer: str
    feedback: str


class PageResult(BaseModel):
    """페이지별 채점 결과 모델"""
    page: int
    wrong_answers: List[WrongAnswer]


class EvaluationResponse(BaseModel):
    """전체 평가 응답 모델"""
    results: List[PageResult]


class GradingEntry(BaseModel):
    """채점 대상 항목 모델"""
    key: List[int]
    text: str
    page_number: int


class GradingResult(BaseModel):
    """개별 채점 결과 모델"""
    key: List[int]
    original_text: Optional[str]
    feedback: Optional[str]
    is_wrong: bool


class GradingConfig(BaseModel):
    """채점 설정 모델"""
    space_id: str
    prompt_template: str
    openai_api_keys: List[str]
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0
    max_tokens: int = 700
    lang_type: str = "ko"  # 언어 타입


class GraphState(TypedDict):
    """그래프 상태 타입"""
    space_id: str
    lang_type: str  # 언어 타입
    user_inputs: List[Any]
    prompt_template: str
    all_texts: List[str]
    pages: Dict[int, List[Any]]
    document_finder: Any
    wrong_answers_by_page: Dict[int, List[Dict[str, Any]]]
    page_results: List[tuple]
    final_results: List[PageResult]
    page_processes_pending: bool