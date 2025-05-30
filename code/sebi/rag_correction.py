from fastapi import APIRouter, HTTPException, File, Form
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import time
import os
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional, Annotated, TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_upstage import UpstageEmbeddings
from pinecone_text.sparse import BM25Encoder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 환경 변수 로드
load_dotenv()

router = APIRouter(
    tags=['AI']
)

# API 키 설정 (병렬 처리, 2개의 키 리스트로 관리)
OPENAI_API_KEY_1 = os.getenv("OPENAI_API_KEY_JUN")
OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_KI")
OPENAI_API_KEYS = [OPENAI_API_KEY_1, OPENAI_API_KEY_2]

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
CORRECTION_PROMPT_PATH = os.getenv("CORRECTION_PROMPT_PATH")

# Pinecone 연결 준비
pc = Pinecone(api_key=PINECONE_API_KEY)

# 데이터 모델 정의
# 세부 피드백 내용 모델
class WrongAnswer(BaseModel):
    id: List[int]
    wrong_answer: str
    feedback: str

# 페이지 별 피드백 저장 모델
class PageResult(BaseModel):
    page: int
    wrong_answers: List[WrongAnswer]

# 측정 시간 저장 모델
class EvaluationResponse(BaseModel):
    results: List[PageResult]
    correction_time: float
    pinecone_time: float  # Pinecone 처리 시간
    openai_time: float    # OpenAI API 처리 시간

# LangGraph 상태 모델 정의
class GraphState(TypedDict):
    document_id: str
    index_name: str
    user_inputs: List[Any]
    prompt_template: str
    all_texts: List[str]
    pages: Dict[int, List[Any]]
    bm25_encoder: Any
    wrong_answers_by_page: Dict[int, List[Dict[str, Any]]]
    pinecone_time: float
    openai_time: float
    start_time: float
    embedder: Any
    page_results: List[Tuple[int, List[WrongAnswer], float]]
    final_results: List[PageResult]
    correction_time: float
    page_processes_pending: bool

# 프롬프트 로드
def load_prompt():
    try:
        with open(CORRECTION_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prompt file not found")

# OpenAI 클라이언트를 가져오는 함수 (페이지 번호에 따라 다른 API 키 사용)
def get_openai_client(page_number: int):
    # 페이지 번호가 홀수면 첫 번째 키(인덱스 0), 짝수면 두 번째 키(인덱스 1) 사용
    key_index = 0 if page_number % 2 == 1 else 1
    return OpenAI(api_key=OPENAI_API_KEYS[key_index])

# LangGraph 노드
# 데이터 입력 노드 (사용자 입력 데이터를 처리, 초기 상태 생성)
def data_input_node(state: GraphState) -> GraphState:
    # API 키 유효성 검사
    if not all(OPENAI_API_KEYS):
        raise ValueError("OpenAI API 키가 올바르게 설정되지 않았습니다.")
    
    # 시작 시간 기록
    state["start_time"] = time.time()
    
    # 프롬프트 로드
    state["prompt_template"] = load_prompt()
    
    # 문자열 입력을 리스트로 변환
    if isinstance(state["user_inputs"], str):
        try:
            state["user_inputs"] = json.loads(state["user_inputs"])
        except json.JSONDecodeError:
            raise ValueError("입력된 데이터를 JSON으로 파싱할 수 없습니다.")
    
    # 모든 텍스트 수집 (BM25 인코더 초기화)
    all_texts = []
    for item in state["user_inputs"]:
        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            text = item[1][0] if item[1] else ""
            all_texts.append(text)
    
    state["all_texts"] = all_texts
    
    # 페이지별로 데이터 그룹화
    pages = {}
    for item in state["user_inputs"]:
        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            page_num = item[0][0]
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(item)
    
    state["pages"] = pages
    
    # 피드백 결과 및 시간 측정용 빈 컨테이너 초기화
    state["wrong_answers_by_page"] = {}
    state["pinecone_time"] = 0.0
    state["openai_time"] = 0.0
    state["page_results"] = []
    state["final_results"] = []
    state["page_processes_pending"] = False
    
    return state

# 초기화 노드 (임베딩 모델 및 인코더 초기화)
def initialization_node(state: GraphState) -> GraphState:
    # 임베딩 모델 초기화
    embedder = UpstageEmbeddings(
        model="solar-embedding-1-large-passage",
        upstage_api_key=UPSTAGE_API_KEY
    )
    state["embedder"] = embedder
    
    # BM25 인코더 초기화
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(state["all_texts"])
    state["bm25_encoder"] = bm25_encoder
    
    # 기본 상태 반환
    return state

# 하이브리드 검색 함수
def hybrid_search(
    query: str, 
    document_id: str, 
    index_name: str, 
    embedder: Any, 
    bm25_encoder: Any,
    top_k: int = 3, 
    alpha: float = 0.7
) -> Tuple[Any, float]:
    start_time = time.time()
    
    # 쿼리에 대한 dense 임베딩 생성
    dense_vector = embedder.embed_query(query)
    
    # 쿼리에 대한 sparse 벡터 생성
    sparse_vector = bm25_encoder.encode_queries([query])[0]
    
    # Pinecone 인덱스에 연결
    index = pc.Index(index_name)
    
    # document_id를 기준으로 필터링하는 필터 생성
    filter_dict = {"document_id": document_id}
    
    # 하이브리드 검색 수행
    results = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
        alpha=alpha  # dense와 sparse의 가중치 조절
    )
    
    # 실행 시간 측정
    execution_time = time.time() - start_time
    
    return results, execution_time

# 채점 함수
async def grade_entry(
    entry: Any, 
    document_id: str, 
    index_name: str, 
    prompt_template: str, 
    page_number: int,
    embedder: Any,
    bm25_encoder: Any
) -> Tuple[List[int], Optional[str], Optional[str], float, float]:
    
    key = entry[0]
    text = entry[1][0] if entry[1] else ""
    
    local_pinecone_time = 0.0
    local_openai_time = 0.0
    
    # 단어 1-2개로만 구성된 짧은 문장은 건너뛰기 (키워드는 채점에 포함하지 않기 위함)
    words = text.split()
    if len(words) <= 1:
        return key, None, None, 0.0, 0.0
    
    # Pinecone에서 document_id를 기준으로 관련 정답 문서 검색 (+ 시간 측정)
    pinecone_start_time = time.time()
    
    # 하이브리드 검색 수행
    results, search_time = hybrid_search(
        query=text, 
        document_id=document_id, 
        index_name=index_name,
        embedder=embedder,
        bm25_encoder=bm25_encoder
    )
    
    # 검색 결과에서 정답 텍스트 추출
    correct_text = ""
    if results and results.matches and len(results.matches) > 0:
        correct_text = results.matches[0].metadata.get("content", "")
    
    local_pinecone_time = search_time
    
    # 정답 문서를 찾지 못한 경우
    if not correct_text:
        return key, None, None, local_pinecone_time, 0.0
    
    # 프롬프트 구성 (포맷팅)
    prompt = prompt_template.format(
        reference_text=correct_text,
        user_text=text
    )
    
    # 페이지 번호에 따른 OpenAI 클라이언트 가져오기
    client = get_openai_client(page_number)
    
    # OpenAI API 호출 (+ 시간 측정)
    try:
        openai_start_time = time.time()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=700, # max token 지정
        )
        local_openai_time = time.time() - openai_start_time
        
        # 응답 파싱
        content = response.choices[0].message.content
        
        # JSON 추출
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                # 틀린 내용 확인
                is_wrong = result.get("is_wrong", False)
                
                # 틀린 내용이 있는 경우
                if is_wrong:
                    feedback = result.get("feedback") # OpenAI 응답에서 feedback 가져오기
                    if feedback:
                        # 원본 텍스트와 피드백 반환
                        return key, text, feedback, local_pinecone_time, local_openai_time
                    else:
                        return key, None, None, local_pinecone_time, local_openai_time
                else:
                    return key, None, None, local_pinecone_time, local_openai_time
            else:
                return key, None, None, local_pinecone_time, local_openai_time
        except Exception as e:
            return key, None, None, local_pinecone_time, local_openai_time
    except Exception as e:
        return key, None, None, local_pinecone_time, 0.0

# 페이지 처리 노드 (각 페이지를 처리하는 노드)
async def process_page_node(state: GraphState) -> GraphState:
    # 이미 모든 페이지가 처리 중이라면 건너뜀
    if state["page_processes_pending"]:
        return state
    
    state["page_processes_pending"] = True
    
    # 각 페이지를 병렬로 처리하는 태스크 생성
    page_tasks = []
    for page_number, page_entries in state["pages"].items():
        task = async_process_page(
            page_number=page_number, 
            page_entries=page_entries, 
            document_id=state["document_id"], 
            index_name=state["index_name"], 
            prompt_template=state["prompt_template"],
            embedder=state["embedder"],
            bm25_encoder=state["bm25_encoder"]
        )
        page_tasks.append(task)
    
    # 모든 페이지를 병렬로 처리
    print(f"총 {len(page_tasks)}개 페이지 병렬 처리 시작")
    state["page_results"] = await asyncio.gather(*page_tasks)
    
    return state

# 단일 페이지 처리 함수
async def async_process_page(
    page_number: int, 
    page_entries: List, 
    document_id: str, 
    index_name: str, 
    prompt_template: str,
    embedder: Any,
    bm25_encoder: Any
) -> Tuple[int, List[WrongAnswer], float]:
    
    print(f"[페이지 {page_number}] 채점 중... (API 키 {1 if page_number % 2 == 1 else 2} 사용)")
    
    # 페이지별 OpenAI 시간 측정
    page_openai_time = 0.0
    
    # 개별 항목 처리 작업 생성
    tasks = []
    for entry in page_entries:
        task = grade_entry(
            entry=entry, 
            document_id=document_id, 
            index_name=index_name, 
            prompt_template=prompt_template, 
            page_number=page_number,
            embedder=embedder,
            bm25_encoder=bm25_encoder
        )
        tasks.append(task)
    
    # 모든 항목 처리 작업 실행 및 결과 수집
    results = await asyncio.gather(*tasks)
    
    # 틀린 답변만 필터링 및 시간 누적
    wrong_answers = []
    for key, original_text, feedback, local_pinecone_time, local_openai_time in results:
        # 각 항목의 시간 추가 (페이지별로 별도 측정)
        page_openai_time += local_openai_time
        
        if original_text and feedback:
            wrong_answers.append(WrongAnswer(
                id=key,
                wrong_answer=original_text,
                feedback=feedback
            ))
    
    print(f"[페이지 {page_number}] 채점 완료: {len(wrong_answers)}개 틀림 (OpenAI 시간: {page_openai_time:.2f}초)")
    
    return page_number, wrong_answers, page_openai_time

# 결과 정리 노드 (최종 결과 형식 구성 및 반환)
def compile_results_node(state: GraphState) -> GraphState:
    # 결과를 페이지 번호순으로 정렬
    state["page_results"].sort(key=lambda x: x[0])
    
    # 최종 결과 구성
    graded_results = []
    total_openai_time = 0.0
    
    for page_number, wrong_answers, page_openai_time in state["page_results"]:
        if wrong_answers:  # 틀린 답변이 있는 페이지만 추가
            graded_results.append(PageResult(
                page=page_number,
                wrong_answers=wrong_answers
            ))
        # 페이지별 OpenAI 시간 누적
        total_openai_time += page_openai_time
    
    state["final_results"] = graded_results
    state["openai_time"] = total_openai_time
    
    # API 종료 시간 기록 및 총 실행 시간 계산
    end_time = time.time()
    state["correction_time"] = end_time - state["start_time"]
    
    print(f"총 실행 시간: {state['correction_time']:.2f}초")
    print(f"Pinecone 검색 시간: {state['pinecone_time']:.2f}초")
    print(f"OpenAI API 시간: {state['openai_time']:.2f}초")
    
    return state

# 다음 노드 결정 함수
def should_end(state: GraphState) -> str:
    # 페이지 처리가 시작되었고 결과가 있으면 결과 정리로, 아니면 계속 페이지 처리
    if state["page_processes_pending"] and state["page_results"]:
        return "compile_results"
    return "process_page"

# 그래프 생성 함수
def create_correction_graph():
    # 상태 그래프 생성
    graph = StateGraph(GraphState)
    
    # 노드 추가
    graph.add_node("data_input", data_input_node)
    graph.add_node("initialization", initialization_node)
    graph.add_node("process_page", process_page_node)
    graph.add_node("compile_results", compile_results_node)
    
    # 엣지 연결
    graph.add_edge("data_input", "initialization")
    graph.add_edge("initialization", "process_page")
    graph.add_conditional_edges(
        "process_page",
        should_end,
        {
            "process_page": "process_page",
            "compile_results": "compile_results"
        }
    )
    graph.add_edge("compile_results", END)
    
    # 시작 노드 설정
    graph.set_entry_point("data_input")
    
    # 그래프 컴파일
    return graph.compile()

# API 엔드포인트
@router.post("/correct", response_model=EvaluationResponse)
async def correct_texts(
    document_id: str = Form("5481b11f-ea69-4314-a922-2d1b99ce3c9d"),
    index_name: str = Form("tabular"),
    user_inputs: str = Form(...)
):
    try:
        # 그래프 생성
        correction_graph = create_correction_graph()
        
        # 초기 상태 설정
        initial_state: GraphState = {
            "document_id": document_id,
            "index_name": index_name,
            "user_inputs": user_inputs,
            "prompt_template": "",
            "all_texts": [],
            "pages": {},
            "bm25_encoder": None,
            "embedder": None,
            "wrong_answers_by_page": {},
            "pinecone_time": 0.0,
            "openai_time": 0.0,
            "start_time": 0.0,
            "page_results": [],
            "final_results": [],
            "correction_time": 0.0,
            "page_processes_pending": False
        }
        
        # 그래프 실행
        final_state = await correction_graph.ainvoke(initial_state)
        
        # 결과 반환
        return EvaluationResponse(
            results=final_state["final_results"],
            correction_time=final_state["correction_time"],
            pinecone_time=final_state["pinecone_time"],
            openai_time=final_state["openai_time"]
        )
    
    except Exception as e:
        # 예외 처리
        logging.error(f"Error in correction process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))