# services로 이전
# from fastapi import APIRouter, HTTPException, File, Form, UploadFile
# from fastapi.responses import JSONResponse

import os
import json
from typing import Dict, List, Any, Optional, TypedDict
from openai import OpenAI
from common_sdk.get_logger import get_logger
import time
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from common_sdk.config import settings

# 로거 설정
logger = get_logger()


OPENAI_API_KEY = settings.OPENAI_API_KEY_B

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# LangGraph 상태 정의
class MissingAnalysisState(TypedDict):
    # 입력 데이터
    raw_keyword_data: Optional[Dict[str, Any]]  # 업로드된 키워드 파일 데이터
    raw_user_inputs: str  # 사용자 입력 원본
    
    # 처리된 데이터
    formatted_hierarchy: Optional[str]  # 포맷팅된 계층 구조
    extracted_user_content: Optional[str]  # 추출된 사용자 텍스트
    api_response: Optional[str]  # OpenAI API 응답
    
    # 최종 결과
    missing_items: List[str]  # 누락된 항목 목록
    
    # 메타데이터
    start_time: float  # 시작 시간
    processing_time: Optional[float]  # 총 처리 시간
    
    # 에러 처리
    error: Optional[str]  # 에러 메시지

# 프롬프트 로드
def load_prompt():
    try:
        with open(MISSING_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prompt file not found")

# 키워드 계층형 구조로 변경
def format_hierarchy_list(node: Dict[str, Any], level: int = 0) -> str:
    indent = "  " * level
    
    if level == 0:
        result = f"{node['name']}\n"
    else:
        # 레벨에 따라 다른 기호 사용
        prefix = "-" if level == 1 else ("*" if level == 2 else "+")
        result = f"{indent}{prefix} {node['name']}\n"
    
    if "children" in node and node["children"]:
        for child in node["children"]:
            result += format_hierarchy_list(child, level + 1)
    
    return result

# 사용자 입력에서 텍스트 내용만 추출
def extract_user_content(user_inputs: str) -> str:
    try:
        # 문자열을 JSON으로 파싱
        data = json.loads(user_inputs)
        
        # 텍스트 내용만 추출 (각 항목의 두 번째 요소의 첫 번째 요소)
        texts = []
        for item in data:
            if len(item) >= 2 and len(item[1]) >= 1:
                texts.append(item[1][0])
        
        # 텍스트를 줄바꿈으로 결합
        return "\n".join(texts)
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 원본 텍스트 반환
        return user_inputs

# LangGraph 노드
# 초기화 노드
async def initialize(state: MissingAnalysisState) -> MissingAnalysisState:
    """초기 설정 및 시작 시간 기록"""
    state["start_time"] = time.time()
    state["missing_items"] = []
    state["error"] = None
    
    # logger.info("Missing Analysis 초기화 완료")
    print("Missing Analysis 워크플로우 시작")
    
    return state

# 키워드 포맷팅 노드 (계층적 형태로 포맷팅)
async def formatting_keywords(state: MissingAnalysisState) -> MissingAnalysisState:
    try:
        # 키워드 데이터 처리
        raw_data = state["raw_keyword_data"]
        if not raw_data:
            raise ValueError("키워드 데이터가 없습니다.")
        
        # 계층 구조를 중첩 리스트 형식으로 포맷팅
        formatted_hierarchy = format_hierarchy_list(raw_data)
        state["formatted_hierarchy"] = formatted_hierarchy
        
        # 사용자 입력에서 텍스트 내용 추출
        raw_inputs = state["raw_user_inputs"]
        extracted_content = extract_user_content(raw_inputs)
        state["extracted_user_content"] = extracted_content
        
        # logger.info(f"키워드 포맷팅 완료: {len(formatted_hierarchy)}자")
        # logger.info(f"사용자 입력 추출 완료: {len(extracted_content)}자")
        print(f"키워드 포맷팅 및 사용자 입력 추출 완료")
        
        return state
        
    except Exception as e:
        error_msg = f"키워드 포맷팅 중 오류: {str(e)}"
        state["error"] = error_msg
        # logger.error(error_msg, exc_info=True)
        return state

# 결과 처리 노드 (OpenAI 요청 → 결과 구성)
async def process_results(state: MissingAnalysisState) -> MissingAnalysisState:
    try:
        formatted_hierarchy = state["formatted_hierarchy"]
        extracted_content = state["extracted_user_content"]
        
        if not formatted_hierarchy or not extracted_content:
            raise ValueError("필요한 데이터가 누락되었습니다.")
        
        # 프롬프트 템플릿 로드
        prompt_template = load_prompt()
        
        # 최종 프롬프트 구성
        final_prompt = prompt_template.format(
            keywords=formatted_hierarchy,
            user_content=extracted_content
        )
        
        print("OpenAI API 호출")
        
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "당신은 한국사 내용 평가 전문가입니다. 누락된 개념을 정확히 찾아내고 결과를 JSON 형식으로 반환해야 합니다."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}  # JSON 응답 강제
        )
        
        # 응답 내용 저장
        response_content = response.choices[0].message.content
        state["api_response"] = response_content
        
        # JSON 응답 파싱
        try:
            result = json.loads(response_content)
            
            # 결과 형식 확인 및 추출
            missing_items = []
            if isinstance(result, dict):
                for key in result:
                    if isinstance(result[key], list):
                        missing_items = result[key]
                        break
                else:
                    # 리스트를 찾지 못한 경우
                    missing_items = []
            elif isinstance(result, list):
                missing_items = result
            else:
                missing_items = []
                
        except json.JSONDecodeError:
            # JSON 파싱 실패 시, 줄바꿈으로 분리
            missing_items = [item.strip() for item in response_content.split("\n") if item.strip()]
        
        # 중복 제거 및 빈 문자열 제거
        unique_items = []
        for item in missing_items:
            if item and isinstance(item, str) and item not in unique_items:
                # 마침표 표준화
                if not item.endswith('.'):
                    item += '.'
                unique_items.append(item)
        
        state["missing_items"] = unique_items
        
        # logger.info("OpenAI API 호출 완료")
        # logger.info(f"결과 처리 완료: {len(unique_items)}개 누락 항목 발견")
        print(f"AI 분석 및 결과 처리 완료: {len(unique_items)}개 누락 항목 발견")
        
        return state
        
    except Exception as e:
        error_msg = f"결과 처리 중 오류: {str(e)}"
        state["error"] = error_msg
        # logger.error(error_msg, exc_info=True)
        return state

# 최종 컴파일 노드
async def compile_results(state: MissingAnalysisState) -> MissingAnalysisState:
    """분석 완료 및 메타데이터 정리"""
    # 총 처리 시간 계산
    total_time = time.time() - state["start_time"]
    state["processing_time"] = total_time
    
    missing_count = len(state["missing_items"])
    
    if state.get("error"):
        # logger.error(f"Missing Analysis 오류로 종료: {state['error']}")
        print(f"Missing Analysis 오류로 종료: {state['error']}")
    else:
        # logger.info(f"Missing Analysis 완료: 총 처리 시간 {total_time:.2f}초, {missing_count}개 항목")
        print(f"Missing Analysis 완료 - 처리 시간: {total_time:.2f}초, {missing_count}개 누락 항목")
    
    return state

# LangGraph 생성 함수
def create_missing_analysis_graph():
    workflow = StateGraph(MissingAnalysisState)
    
    # 노드 추가
    workflow.add_node("initialize", initialize)
    workflow.add_node("formatting_keywords", formatting_keywords)
    workflow.add_node("process_results", process_results)
    workflow.add_node("compile_results", compile_results)
    
    # 시작점 설정
    workflow.set_entry_point("initialize")
    
    # 엣지 추가
    workflow.add_edge("initialize", "formatting_keywords")
    workflow.add_edge("formatting_keywords", "process_results")
    workflow.add_edge("process_results", "compile_results")
    workflow.add_edge("compile_results", END)
    
    return workflow.compile()

# FastAPI 엔드포인트 (LangGraph 버전)
@router.post("/missing")
async def find_missing_langgraph(
    keyword: UploadFile = File(...),
    user_inputs: str = Form(...)
):
    try:
        # 파일 내용 읽기
        content = await keyword.read()
        keyword_data = json.loads(content)
        
        # 그래프 생성
        graph = create_missing_analysis_graph()
        
        # 초기 상태 설정
        initial_state = {
            "raw_keyword_data": keyword_data,
            "raw_user_inputs": user_inputs
        }
        
        # 그래프 실행
        print("Missing Analysis LangGraph 실행 시작")
        result = await graph.ainvoke(initial_state)
        
        # 에러가 있으면 HTTP 예외 발생
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 결과 반환
        return JSONResponse(
            content={"missing_answer": result["missing_items"]},
            status_code=200
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효하지 않은 JSON 파일입니다.")
    except HTTPException:
        raise
    except Exception as e:
        # logger.error(f"전체 처리 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")