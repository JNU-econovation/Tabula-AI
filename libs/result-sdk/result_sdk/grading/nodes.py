# grading/nodes.py

import json
import asyncio
from typing import List, Any, Tuple
from pinecone_text.sparse import BM25Encoder
from openai import OpenAI

from .models import GraphState, PageResult, WrongAnswer, GradingConfig
from ..retrieval import DocumentFinder, RetrievalConfig
from common_sdk import get_logger, PromptLoader

logger = get_logger()

class GradingNodes:
    """채점 기능의 LangGraph 노드들"""

    def __init__(self, config: GradingConfig):
        self.config = config
        self.prompt_loader = PromptLoader()

    def data_input_node(self, state: GraphState) -> GraphState:
        """데이터 입력 노드"""
        if not all(self.config.openai_api_keys):
            raise ValueError("OpenAI API 키가 올바르게 설정되지 않았습니다.")
        
        prompt_data = self.prompt_loader.load_prompt("grading-prompt")
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

    def initialization_node(self, state: GraphState) -> GraphState:
        """초기화 노드"""
        # BM25 인코더 초기화
        bm25_encoder = BM25Encoder()
        bm25_encoder.fit(state["all_texts"])
        state["bm25_encoder"] = bm25_encoder
        
        return state
    
    async def process_page_node(self, state: GraphState) -> GraphState:
        """페이지 처리 노드"""
        if state["page_processes_pending"]:
            return state
        
        state["page_processes_pending"] = True
        
        page_tasks = []
        for page_number, page_entries in state["pages"].items():
            task = self._async_process_page(
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
    
    def compile_results_node(self, state: GraphState) -> GraphState:
        """결과 컴파일 노드"""
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
    
    def should_end(self, state: GraphState) -> str:
        """종료 조건 판단"""
        if state["page_processes_pending"] and state["page_results"]:
            return "compile_results"
        return "process_page"
    
    def _get_openai_client(self, page_number: int) -> OpenAI:
        """페이지 번호에 따라 API 키 선택"""
        key_index = 0 if page_number % 2 == 1 else 1
        key_index = min(key_index, len(self.config.openai_api_keys) - 1)
        return OpenAI(api_key=self.config.openai_api_keys[key_index])
    
    async def _grade_entry(
            self,
            entry: Any,
            document_id: str,
            index_name: str,
            prompt_template: str,
            page_number: int,
            embedder: Any,
            bm25_encoder: Any
        ) -> Tuple[List[int], str, str]:
            """개별 항목 채점 (원본 grade_entry 함수)"""
            key = entry[0]
            text = entry[1][0] if entry[1] else ""
            
            words = text.split()
            if len(words) <= 1:
                return key, None, None
            
            # DocumentFinder 초기화 및 BM25 설정
            document_finder = DocumentFinder()
            document_finder.bm25_manager.encoder = bm25_encoder
            
            # 검색 설정
            retrieval_config = RetrievalConfig(
                document_id=document_id,
                index_name=index_name
            )
            
            # 참고 텍스트 검색
            correct_text = document_finder.find_reference_text(
                query=text,
                config=retrieval_config
            )
            
            if not correct_text:
                return key, None, None
            
            prompt = prompt_template.format(
                reference_text=correct_text,
                user_text=text
            )
            
            try:
                client = self._get_openai_client(page_number)
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=self.config.model_name,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"}
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
        
    async def _async_process_page(
        self,
        page_number: int,
        page_entries: List,
        document_id: str,
        index_name: str,
        prompt_template: str,
        embedder: Any,
        bm25_encoder: Any
    ) -> Tuple[int, List[WrongAnswer]]:
        """페이지별 비동기 처리 (원본 async_process_page 함수)"""
        logger.info(f"[페이지 {page_number}] 채점 중... (API 키 {1 if page_number % 2 == 1 else 2} 사용)")
        
        tasks = []
        for entry in page_entries:
            task = self._grade_entry(
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