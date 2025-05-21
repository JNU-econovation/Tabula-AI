import os
import sys
import time

# 현재 파일의 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from typing import List, Dict
from openai import OpenAI
from path.config import settings
from load.utils import get_embedding
from PIL import Image
import io
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml
from core.models import MultiModal, LLMs
import re
from langchain_openai import ChatOpenAI
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """문자열의 토큰 수를 계산"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class ObjectSummary:
    def __init__(self, image_base_path: str = "/Users/kyeong6/Desktop/test/parse/result"):
        """ObjectSummary 초기화"""
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.image_base_path = image_base_path
        
        # 프롬프트 로드
        self.image_system_prompt = self._load_prompt("layoutparse/prompts/IMAGE-SYSTEM-PROMPT.yaml")
        loaded_image_user_prompt = self._load_prompt("layoutparse/prompts/IMAGE-USER-PROMPT.yaml")
        # 사용자 프롬프트에 1-2줄 요약 및 한글 요약 지시 추가
        self.image_user_prompt = loaded_image_user_prompt + "\n\nSummarize the image in 1-2 sentences. Please provide the summary in Korean."
        
        # 테이블 관련 프롬프트 로드 및 모델 초기화 제거
        # self.table_system_prompt = self._load_prompt("layoutparse/prompts/TABLE-SYSTEM-PROMPT.yaml")
        # self.table_user_prompt = self._load_prompt("layoutparse/prompts/TABLE-USER-PROMPT.yaml")
        
        # 토큰 수 계산 및 출력
        self.system_tokens = num_tokens_from_string(self.image_system_prompt)
        self.user_tokens = num_tokens_from_string(self.image_user_prompt)
        self.total_prompt_tokens = self.system_tokens + self.user_tokens
        
        # LLM 인스턴스 생성
        llm_instance = ChatOpenAI(model_name=LLMs.GPT4.value, openai_api_key=settings.OPENAI_API_KEY)

        # MultiModal 모델 초기화
        self.image_model = MultiModal(
            model=llm_instance,
            system_prompt=self.image_system_prompt,
            user_prompt=self.image_user_prompt
        )
        
        # 테이블 모델 초기화 제거
        # self.table_model = MultiModal(
        #     model=llm_instance,
        #     system_prompt=self.table_system_prompt,
        #     user_prompt=self.table_user_prompt
        # )

        # 전체 토큰 수를 저장할 변수
        self.total_context_tokens = 0
        self.total_summary_tokens = 0

        # 이미지 처리 결과 저장을 위한 변수 추가
        self.processed_images = []

    def _load_prompt(self, prompt_path: str) -> str:
        """프롬프트 파일 로드"""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = yaml.safe_load(f)
                return prompt.get('prompt', '')
        except Exception as e:
            print(f"프롬프트 로드 실패: {str(e)}")
            return ""

    def _resize_image(self, image_path: str, max_size: int = 1024) -> bytes:
        """이미지 크기 조정"""
        try:
            with Image.open(image_path) as img:
                # 이미지 비율 유지하면서 리사이징
                ratio = min(max_size / max(img.size), 1.0)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 이미지를 메모리에 저장
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=85)
                return img_byte_arr.getvalue()
        except Exception as e:
            print(f"이미지 리사이징 실패: {str(e)}")
            raise

    def _extract_image_paths(self, content: str) -> List[Dict[str, str]]:
        """마크다운에서 이미지 경로와 컨텍스트 추출"""
        # 이미지 태그의 전체 문자열, alt 텍스트, 경로, 시작 및 끝 위치를 캡처하는 수정된 패턴
        pattern = r'(!\[(.*?)\]\((.*?)\))' 
        matches = re.finditer(pattern, content)
        results = []
        
        for match in matches:
            full_tag = match.group(1) # 전체 이미지 태그 (e.g., ![alt text](path/to/image.png))
            alt_text = match.group(2) # alt 텍스트
            path = match.group(3)     # 이미지 경로
            
            # 이미지 주변 컨텍스트 추출 (이전과 동일)
            context_start = max(0, match.start() - 200)
            context_end = min(len(content), match.end() + 200)
            context = content[context_start:context_end]
            
            results.append({
                "full_tag": full_tag, # 전체 태그 추가
                "alt_text": alt_text,
                "path": path,         # 원본 MD의 경로 유지
                "context": context,
                "position": match.start(), # 태그 시작 위치
                "end_position": match.end()  # 태그 끝 위치 추가
            })
        
        return results

    # def _extract_table_paths(self, content: str) -> List[Dict[str, str]]: # 테이블 경로 추출 메서드 제거
    #     """마크다운에서 테이블 경로와 컨텍스트 추출"""
    #     # 현재는 이미지와 동일한 패턴 사용
    #     return self._extract_image_paths(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_image(self, image_info: Dict[str, str]) -> Dict:
        """이미지 처리"""
        actual_image_path = ""
        try:
            # 실제 이미지 경로 조합
            relative_path = image_info["path"]
            if relative_path.startswith("../"):
                relative_path = relative_path[len("../"):]
            elif relative_path.startswith("./"):
                 relative_path = relative_path[len("./"):]
            actual_image_path = os.path.join(self.image_base_path, relative_path)
            
            # 이미지 리사이징
            resized_image = self._resize_image(actual_image_path)
            
            # 컨텍스트 토큰 수 계산 및 누적
            context_tokens = num_tokens_from_string(image_info['context'])
            self.total_context_tokens += context_tokens
            
            # 이미지 분석
            summary = self.image_model.invoke(
                actual_image_path,
                system_prompt=self.image_system_prompt,
                user_prompt=f"{self.image_user_prompt}\n\nImage Context in Markdown:\n{image_info['context']}"
            )
            
            # 요약 토큰 수 계산 및 누적
            summary_tokens = num_tokens_from_string(summary)
            self.total_summary_tokens += summary_tokens
            
            # 임베딩 생성
            embedding = get_embedding(summary, "ko")
            
            return {
                "type": "image",
                "full_tag": image_info["full_tag"],
                "path": image_info["path"],
                "actual_path": actual_image_path,
                "alt_text": image_info["alt_text"],
                "summary": summary,
                "embedding": embedding,
                "position": image_info["position"],
                "end_position": image_info["end_position"]
            }
            
        except Exception as e:
            error_path = actual_image_path if actual_image_path else image_info["path"]
            print(f"이미지 처리 실패 (경로: {error_path}): {str(e)}")
            raise

    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)) # 테이블 처리 메서드 제거
    # async def process_table(self, table_info: Dict[str, str], context: str) -> Dict:
    #     """테이블 처리"""
    #     actual_table_path = ""
    #     try:
    #         # 실제 테이블 이미지 경로 조합 (이미지와 동일한 로직 적용 가정)
    #         relative_path = table_info["path"]
    #         if relative_path.startswith("../"):
    #             relative_path = relative_path[len("../"):]
    #         elif relative_path.startswith("./"):
    #             relative_path = relative_path[len("./"):]
    #         actual_table_path = os.path.join(self.image_base_path, relative_path)

    #         # 테이블 분석
    #         summary = self.table_model.invoke(
    #             actual_table_path, # 수정된 경로 사용
    #             system_prompt=self.table_system_prompt,
    #             user_prompt=f"{self.table_user_prompt}\n\nContext: {context}"
    #         )
            
    #         # 임베딩 생성
    #         embedding = get_embedding(summary, "ko")
            
    #         return {
    #             "type": "table",
    #             "path": table_info["path"], # 원본 상대 경로 유지
    #             "actual_path": actual_table_path, # 실제 접근 경로
    #             "alt_text": table_info["alt_text"],
    #             "summary": summary,
    #             "embedding": embedding,
    #             "position": table_info["position"]
    #         }
            
    #     except Exception as e:
    #         error_path = actual_table_path if actual_table_path else table_info["path"]
    #         print(f"테이블 처리 실패 (경로: {error_path}): {str(e)}")
    #         raise

    def _insert_summaries(self, content: str, summaries: List[Dict]) -> str:
        """마크다운에 요약 내용 삽입"""
        # end_position 기준으로 역순 정렬 (뒤에서부터 삽입해야 인덱스 꼬임 방지)
        summaries.sort(key=lambda x: x["end_position"], reverse=True)
        
        for summary in summaries:
            # 이미지 태그 바로 다음 줄에 "요약: {내용}" 형식으로 삽입
            insert_text = f"\n{summary['summary']}" 
            content = content[:summary["end_position"]] + insert_text + content[summary["end_position"]:]
        
        return content

    async def process_markdown(self, content: str, document_id: str) -> Dict:
        """마크다운 문서 처리"""
        try:
            # 이미지 처리 결과 저장
            self.processed_images = []
            image_infos = self._extract_image_paths(content)
            
            print(f"\n=== 이미지 처리 시작 ===")
            print(f"발견된 이미지 수: {len(image_infos)}")
            
            # 배치 크기 설정 (한 번에 처리할 이미지 수)
            batch_size = 3
            for i in range(0, len(image_infos), batch_size):
                batch = image_infos[i:i + batch_size]
                print(f"\n배치 {i//batch_size + 1} 처리 중... ({i+1}-{min(i+batch_size, len(image_infos))}/{len(image_infos)})")
                
                # 배치 내 이미지 순차 처리
                for j, image_info in enumerate(batch):
                    print(f"이미지 {i+j+1}/{len(image_infos)} 처리 중...")
                    print(f"경로: {image_info['path']}")
                    
                    try:
                        result = await self.process_image(image_info)
                        self.processed_images.append(result)
                        print(f"요약: {result['summary'][:100]}...")
                        
                        # 이미지 처리 간 딜레이 (1초)
                        if j < len(batch) - 1:  # 마지막 이미지가 아닌 경우에만 딜레이
                            await asyncio.sleep(1)
                            
                    except Exception as e:
                        print(f"이미지 처리 실패: {str(e)}")
                        continue
                
                # 배치 간 딜레이 (3초)
                if i + batch_size < len(image_infos):
                    print("다음 배치 처리 전 대기 중...")
                    await asyncio.sleep(3)
            
            # 전체 토큰 수 출력
            print("\n=== 전체 토큰 사용량 ===")
            print(f"시스템 프롬프트: {self.system_tokens} 토큰")
            print(f"사용자 프롬프트: {self.user_tokens} 토큰")
            print(f"총 컨텍스트: {self.total_context_tokens} 토큰")
            print(f"총 요약: {self.total_summary_tokens} 토큰")
            print(f"총 토큰 수: {self.total_prompt_tokens + self.total_context_tokens + self.total_summary_tokens} 토큰")
            
            return {
                "document_id": document_id,
                "objects": self.processed_images,
                "status": "success"
            }
            
        except Exception as e:
            print(f"마크다운 처리 실패: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }

async def main():
    """메인 테스트 함수"""
    start_time = time.time() # 실행 시작 시간 기록
    
    # --- 경로 변수 설정 ---
    project_base_dir = "/Users/kyeong6/Desktop/test/parse" # 사용자의 프로젝트 기본 경로
    result_dir = os.path.join(project_base_dir, "result")
    
    # 원본 마크다운 파일 경로
    original_md_filename = "korean_diff.md"
    original_md_path = os.path.join(result_dir, "md", original_md_filename)

    # 요약이 추가된 새 마크다운 파일 경로
    output_md_filename = "korean_diff_mod.md"
    output_md_path = os.path.join(result_dir, "md", output_md_filename)

    # ObjectSummary 인스턴스 생성
    # image_base_path는 result 디렉토리로 설정하여, 
    # 마크다운 내의 "images/korean/..." 와 같은 상대 경로와 조합될 수 있도록 함
    summarizer = ObjectSummary(image_base_path=result_dir)

    # --- 1. 원본 마크다운 파일 읽기 ---
    if not os.path.exists(original_md_path):
        print(f"오류: 원본 마크다운 파일({original_md_filename})을 찾을 수 없습니다: {original_md_path}")
        return

    with open(original_md_path, "r", encoding="utf-8") as f_orig:
        original_md_content = f_orig.read()
    
    print(f"원본 마크다운 파일 '{original_md_filename}'을 읽었습니다.")

    # --- 2. 마크다운 내용 처리 (이미지 요약 및 원본에 삽입) ---
    print("이미지 요약 및 원본 내용에 삽입을 시작합니다...")
    # process_markdown은 내부적으로 _extract_image_paths와 _insert_summaries를 호출함
    # _insert_summaries가 수정되어 원본 content에 요약을 태그 바로 뒤에 삽입
    processed_data = await summarizer.process_markdown(original_md_content, original_md_filename)
    
    modified_md_content = original_md_content # 기본값으로 원본 설정
    if processed_data and processed_data.get("content"):
        modified_md_content = processed_data["content"]
        print(f"총 {len(processed_data.get('objects', []))}개의 이미지에 대한 요약이 처리되었습니다.")
    else:
        print("마크다운 처리 중 오류가 발생했거나, 처리된 내용이 없습니다.")
        # 오류 발생 시 원본 내용을 그대로 사용할지, 아니면 여기서 중단할지 결정할 수 있습니다.
        # 여기서는 원본 내용을 유지하도록 두겠습니다.

    # --- 3. 수정된 마크다운을 새 파일에 저장 --- 
    with open(output_md_path, "w", encoding="utf-8") as f_out:
        f_out.write(modified_md_content)
            
    print(f"'{output_md_filename}' 파일이 {output_md_path} 에 생성되었습니다.")

    end_time = time.time() # 실행 종료 시간 기록
    print(f"총 실행 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    asyncio.run(main())
