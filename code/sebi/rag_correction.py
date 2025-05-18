from fastapi import APIRouter, HTTPException, File, Form
from pydantic import BaseModel
from openai import OpenAI
import json
import time
import os
import asyncio
import logging
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_upstage import UpstageEmbeddings
from pinecone_text.sparse import BM25Encoder

# 환경 변수 로드
load_dotenv()

router = APIRouter(
    tags=['AI']
)

# API 키 설정 (병렬 처리를 위해 2개의 키 리스트로 관리)
OPENAI_API_KEY_1 = os.getenv("OPENAI_API_KEY_JUN")
OPENAI_API_KEY_2 = os.getenv("OPENAI_API_KEY_KI")
OPENAI_API_KEYS = [OPENAI_API_KEY_1, OPENAI_API_KEY_2]

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
PROMPT_PATH = os.getenv("PROMPT_PATH")

# Pinecone 연결 준비
pc = Pinecone(api_key=PINECONE_API_KEY)

# 임베딩 모델 초기화
embedder = UpstageEmbeddings(
    model="solar-embedding-1-large-passage",
    upstage_api_key=UPSTAGE_API_KEY
)

# BM25 인코더 초기화
bm25_encoder = BM25Encoder()

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

# 시간 측정을 위한 전역 변수
pinecone_total_time = 0.0
openai_total_time = 0.0

# 프롬프트 로드
def load_prompt():
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prompt file not found")

# 하이브리드 검색 함수 (dense + sparse, document_id 사용)
def hybrid_search(query: str, document_id: str, index_name: str, top_k: int = 3, alpha: float = 0.7):
    global pinecone_total_time
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
    
    # 실행 시간 측정 및 누적
    execution_time = time.time() - start_time
    pinecone_total_time += execution_time
    
    return results

# Pinecone에서 관련 정답 문서 가져오기 함수 (document_id 사용)
def get_correct_text(user_text: str, document_id: str, index_name: str) -> str:
    global pinecone_total_time
    start_time = time.time()
    
    # 하이브리드 검색 수행
    results = hybrid_search(user_text, document_id, index_name, top_k=3)
    
    result_text = ""
    if results and results.matches and len(results.matches) > 0:
        # 가장 관련성 높은 문서의 content 반환, 없으면 공백
        result_text = results.matches[0].metadata.get("content", "")
    
    # hybrid_search 이외의 작업 시간 누적
    execution_time = time.time() - start_time
    pinecone_total_time += execution_time - (time.time() - start_time)
    
    return result_text

# OpenAI 클라이언트를 가져오는 함수 (페이지 번호에 따라 다른 API 키 사용)
def get_openai_client(page_number: int):

    # 페이지 번호가 홀수면 첫 번째 키(인덱스 0), 짝수면 두 번째 키(인덱스 1) 사용
    key_index = 0 if page_number % 2 == 1 else 1
    return OpenAI(api_key=OPENAI_API_KEYS[key_index])

# 문장 채점 함수 (비동기, document_id 사용, 페이지 번호에 따른 API 키 선택)
async def grade_entry(user_input: dict, document_id: str, index_name: str, prompt_template: str, page_number: int) -> Tuple[List[int], Optional[str], Optional[str], float, float]:

    key = user_input[0]
    text = user_input[1][0] if user_input[1] else ""
    
    local_pinecone_time = 0.0
    local_openai_time = 0.0
    
    # 단어 1-2개로만 구성된 짧은 문장은 건너뛰기 (키워드는 채점에 포함하지 않기 위함)
    words = text.split()
    if len(words) <= 1:
        return key, None, None, 0.0, 0.0
    
    # Pinecone에서 document_id를 기준으로 관련 정답 문서 검색 (+ 시간 측정)
    pinecone_start_time = time.time()
    correct_text = get_correct_text(text, document_id, index_name)
    local_pinecone_time = time.time() - pinecone_start_time
    
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

# 단일 페이지를 처리하는 함수 (병렬로 처리)
async def process_page(page_number: int, page_entries: List, document_id: str, index_name: str, prompt_template: str) -> Tuple[int, List[WrongAnswer], float]:

    print(f"[페이지 {page_number}] 채점 중... (API 키 {1 if page_number % 2 == 1 else 2} 사용)")
    
    # 페이지별 OpenAI 시간 측정
    page_openai_time = 0.0
    
    # 개별 항목 처리 작업 생성 (비동기, document_id 사용)
    tasks = []
    for entry in page_entries:
        task = grade_entry(entry, document_id, index_name, prompt_template, page_number)
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

@router.post("/correct", response_model=EvaluationResponse)
async def correct_texts(
    document_id: str = Form(...),
    index_name: str = Form("tabular"),
    user_inputs: str = Form(...)
):
    # API 키 유효성 검사
    if not all(OPENAI_API_KEYS):
        raise HTTPException(status_code=500, detail="OpenAI API 키가 올바르게 설정되지 않았습니다.")
    
    # 전역 시간 변수 초기화
    global pinecone_total_time, openai_total_time
    pinecone_total_time = 0.0
    openai_total_time = 0.0
    
    # API 시작 시간 기록
    start_time = time.time()
    
    # 프롬프트 로드
    prompt_template = load_prompt()
    
    # 문자열을 리스트로 변환
    try:
        input_list = json.loads(user_inputs)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="입력된 데이터를 JSON으로 파싱할 수 없습니다.")
    
    # BM25 인코더 초기화을 위한 모든 텍스트 수집
    all_texts = []
    for item in input_list:
        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            text = item[1][0] if item[1] else ""
            all_texts.append(text)
    
    # BM25 인코더 초기화
    bm25_encoder.fit(all_texts)
    
    # 페이지별로 데이터 그룹화
    pages = {}
    for item in input_list:
        if len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
            page_num = item[0][0]
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(item)
    
    # 각 페이지를 병렬로 처리하는 태스크 생성
    page_tasks = []
    for page_number, page_entries in pages.items():
        task = process_page(page_number, page_entries, document_id, index_name, prompt_template)
        page_tasks.append(task)
    
    # 모든 페이지를 병렬로 처리
    print(f"총 {len(page_tasks)}개 페이지 병렬 처리 시작")
    page_results = await asyncio.gather(*page_tasks)
    
    # 결과를 페이지 번호순으로 정렬
    page_results.sort(key=lambda x: x[0])
    
    # 최종 결과 구성
    graded_results = []
    for page_number, wrong_answers, page_openai_time in page_results:
        if wrong_answers:  # 틀린 답변이 있는 페이지만 추가
            graded_results.append(PageResult(
                page=page_number,
                wrong_answers=wrong_answers
            ))
        # 페이지별 OpenAI 시간 누적
        openai_total_time += page_openai_time
    
    # API 종료 시간 기록 및 총 실행 시간 계산
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"총 실행 시간: {execution_time:.2f}초")
    print(f"Pinecone 검색 시간: {pinecone_total_time:.2f}초")
    print(f"OpenAI API 시간: {openai_total_time:.2f}초")
    
    # 실행 시간을 포함한 응답 반환
    return EvaluationResponse(
        results=graded_results,
        correction_time=execution_time,
        pinecone_time=pinecone_total_time,
        openai_time=openai_total_time
    )