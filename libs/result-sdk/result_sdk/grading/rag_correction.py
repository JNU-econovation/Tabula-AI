import json
import asyncio
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict, Any, Tuple, TypedDict, Optional
from common_sdk.get_logger import get_logger
from common_sdk.prompt_loader import PromptLoader
from common_sdk.utils import get_embedding
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langgraph.graph import StateGraph, END
from common_sdk.config import settings

# 로거 설정
logger = get_logger()

# 프롬프트 로더 초기화
prompt_loader = PromptLoader()

# 프롬프트 키 설정
GRADING_PROMPT_KEY = "grading-prompt"

# API 키 설정
OPENAI_API_KEY_J = settings.OPENAI_API_KEY_J
OPENAI_API_KEY_K = settings.OPENAI_API_KEY_K
OPENAI_API_KEYS = [OPENAI_API_KEY_J, OPENAI_API_KEY_K]
PINECONE_API_KEY = settings.PINECONE_API_KEY
UPSTAGE_API_KEY = settings.UPSTAGE_API_KEY

# Pinecone 연결
pc = Pinecone(api_key=PINECONE_API_KEY)

# BM25 인코더 초기화
bm25_encoder = BM25Encoder()

# 데이터 모델 정의
class WrongAnswer(BaseModel):
    id: List[int]
    wrong_answer: str
    feedback: str

class PageResult(BaseModel):
    page: int
    wrong_answers: List[WrongAnswer]

class EvaluationResponse(BaseModel):
    results: List[PageResult]

class GraphState(TypedDict):
    document_id: str
    index_name: str
    user_inputs: List[Any]
    prompt_template: str
    all_texts: List[str]
    pages: Dict[int, List[Any]]
    bm25_encoder: Any
    wrong_answers_by_page: Dict[int, List[Dict[str, Any]]]
    embedder: Any
    page_results: List[Tuple[int, List[WrongAnswer]]]
    final_results: List[PageResult]
    page_processes_pending: bool

def get_openai_client(page_number: int) -> OpenAI:
    key_index = 0 if page_number % 2 == 1 else 1
    return OpenAI(api_key=OPENAI_API_KEYS[key_index])

def hybrid_search(
    query: str, 
    document_id: str, 
    index_name: str, 
    embedder: Any, 
    bm25_encoder: Any,
    top_k: int = 3, 
    alpha: float = 0.7
) -> Any:
    dense_vector = get_embedding(query)
    sparse_vector = bm25_encoder.encode_queries([query])[0]
    index = pc.Index(index_name)
    filter_dict = {"document_id": document_id}
    
    return index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
        alpha=alpha
    )

async def grade_entry(
    entry: Any, 
    document_id: str, 
    index_name: str, 
    prompt_template: str, 
    page_number: int,
    embedder: Any,
    bm25_encoder: Any
) -> Tuple[List[int], Optional[str], Optional[str]]:
    key = entry[0]
    text = entry[1][0] if entry[1] else ""
    
    words = text.split()
    if len(words) <= 1:
        return key, None, None
    
    results = hybrid_search(
        query=text, 
        document_id=document_id, 
        index_name=index_name,
        embedder=embedder,
        bm25_encoder=bm25_encoder
    )
    
    correct_text = ""
    if results and results.matches and len(results.matches) > 0:
        correct_text = results.matches[0].metadata.get("content", "")
    
    if not correct_text:
        return key, None, None
    
    prompt = prompt_template.format(
        reference_text=correct_text,
        user_text=text
    )
    
    try:
        client = get_openai_client(page_number)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            max_tokens=700,
        )
        
        content = response.choices[0].message.content
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                result = json.loads(json_content)
                
                is_wrong = result.get("is_wrong", False)
                
                if is_wrong:
                    feedback = result.get("feedback")
                    if feedback:
                        return key, text, feedback
                    else:
                        return key, None, None
                else:
                    return key, None, None
            else:
                return key, None, None
        except Exception as e:
            return key, None, None
    except Exception as e:
        return key, None, None

async def async_process_page(
    page_number: int, 
    page_entries: List, 
    document_id: str, 
    index_name: str, 
    prompt_template: str,
    embedder: Any,
    bm25_encoder: Any
) -> Tuple[int, List[WrongAnswer]]:
    logger.info(f"[페이지 {page_number}] 채점 중... (API 키 {1 if page_number % 2 == 1 else 2} 사용)")
    
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
    
    results = await asyncio.gather(*tasks)
    
    wrong_answers = []
    for key, original_text, feedback in results:
        if original_text and feedback:
            wrong_answers.append(WrongAnswer(
                id=key,
                wrong_answer=original_text,
                feedback=feedback
            ))
    
    logger.info(f"[페이지 {page_number}] 채점 완료: {len(wrong_answers)}개 틀림")
    
    return page_number, wrong_answers

def data_input_node(state: GraphState) -> GraphState:
    if not all(OPENAI_API_KEYS):
        raise ValueError("OpenAI API 키가 올바르게 설정되지 않았습니다.")
    
    prompt_data = prompt_loader.load_prompt(GRADING_PROMPT_KEY)
    state["prompt_template"] = prompt_data["template"]
    
    if isinstance(state["user_inputs"], str):
        try:
            state["user_inputs"] = json.loads(state["user_inputs"])
        except json.JSONDecodeError:
            raise ValueError("입력된 데이터를 JSON으로 파싱할 수 없습니다.")
    
    all_texts = []
    for item in state["user_inputs"]:
        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            text = item[1][0] if item[1] else ""
            all_texts.append(text)
    
    state["all_texts"] = all_texts
    
    pages = {}
    for item in state["user_inputs"]:
        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            page_num = item[0][0]
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(item)
    
    state["pages"] = pages
    state["wrong_answers_by_page"] = {}
    state["page_results"] = []
    state["final_results"] = []
    state["page_processes_pending"] = False
    
    return state

def initialization_node(state: GraphState) -> GraphState:
    # BM25 인코더 초기화
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(state["all_texts"])
    state["bm25_encoder"] = bm25_encoder
    
    return state

async def process_page_node(state: GraphState) -> GraphState:
    if state["page_processes_pending"]:
        return state
    
    state["page_processes_pending"] = True
    
    page_tasks = []
    for page_number, page_entries in state["pages"].items():
        task = async_process_page(
            page_number=page_number, 
            page_entries=page_entries, 
            document_id=state["document_id"], 
            index_name=state["index_name"], 
            prompt_template=state["prompt_template"],
            embedder=None,  # get_embedding은 함수이므로 embedder 파라미터는 None으로 설정
            bm25_encoder=state["bm25_encoder"]
        )
        page_tasks.append(task)
    
    logger.info(f"총 {len(page_tasks)}개 페이지 병렬 처리 시작")
    state["page_results"] = await asyncio.gather(*page_tasks)
    
    return state

def compile_results_node(state: GraphState) -> GraphState:
    state["page_results"].sort(key=lambda x: x[0])
    
    graded_results = []
    for page_number, wrong_answers in state["page_results"]:
        if wrong_answers:
            graded_results.append(PageResult(
                page=page_number,
                wrong_answers=wrong_answers
            ))
    
    state["final_results"] = graded_results
    
    return state

def should_end(state: GraphState) -> str:
    if state["page_processes_pending"] and state["page_results"]:
        return "compile_results"
    return "process_page"

def create_correction_graph():
    graph = StateGraph(GraphState)
    
    graph.add_node("data_input", data_input_node)
    graph.add_node("initialization", initialization_node)
    graph.add_node("process_page", process_page_node)
    graph.add_node("compile_results", compile_results_node)
    
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
    
    graph.set_entry_point("data_input")
    
    return graph.compile()

async def process_correction(
    document_id: str,
    index_name: str,
    user_inputs: str
) -> EvaluationResponse:
    try:
        correction_graph = create_correction_graph()
        
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
            "page_results": [],
            "final_results": [],
            "page_processes_pending": False
        }
        
        final_state = await correction_graph.ainvoke(initial_state)
        
        return EvaluationResponse(
            results=final_state["final_results"]
        )
    
    except Exception as e:
        logger.error(f"Error in correction process: {str(e)}")
        raise Exception(f"채점 처리 중 오류 발생: {str(e)}")