import os
import io
import re
import yaml
import asyncio
from typing import List, Dict
from openai import OpenAI
from common_sdk.config import settings as common_settings
from note_sdk.config import settings
from common_sdk.utils import get_embedding
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from note_sdk.llm import MultiModal, LLMs
from langchain_openai import ChatOpenAI
from common_sdk.utils import num_tokens_from_string
from common_sdk.get_logger import get_logger
from common_sdk.prompt_loader import PromptLoader

logger = get_logger()

# 이미지 요약 클래스
class ImageSummary:
    def __init__(self, task_id: str = None):
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=common_settings.OPENAI_API_KEY_J)

        # 경로 설정
        self.image_base_path = settings.get_image_dir(task_id)
        
        # 프롬프트 로더 초기화
        self.prompt_loader = PromptLoader()
        
        # 프롬프트 로드
        self.image_system_prompt = self.prompt_loader.load_prompt("image-system-prompt")["template"]
        self.image_user_prompt = self.prompt_loader.load_prompt("image-user-prompt")["template"]

        # 토큰 수 계산 및 출력
        self.system_tokens = num_tokens_from_string(self.image_system_prompt)
        self.user_tokens = num_tokens_from_string(self.image_user_prompt)
        self.total_prompt_tokens = self.system_tokens + self.user_tokens
        
        # LLM 인스턴스 생성
        llm_instance = ChatOpenAI(model_name=LLMs.GPT4.value, openai_api_key=common_settings.OPENAI_API_KEY_J)

        # MultiModal 모델 초기화
        self.image_model = MultiModal(
            model=llm_instance,
            system_prompt=self.image_system_prompt,
            user_prompt=self.image_user_prompt
        )

        # 전체 토큰 수를 저장할 변수
        self.total_context_tokens = 0
        self.total_summary_tokens = 0

        # 이미지 처리 결과 저장을 위한 변수 추가
        self.processed_images = []

    def resize_image(self, image_path: str, max_size: int = 1024) -> bytes:
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
            logger.error(f"Image resizing failed: {str(e)}")
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
            resized_image = self.resize_image(actual_image_path)
            
            # 컨텍스트 토큰 수 계산 및 누적
            context_tokens = num_tokens_from_string(image_info['context'])
            self.total_context_tokens += context_tokens
            
            # 이미지 분석
            summary = self.image_model.invoke(
                actual_image_path,
                system_prompt=self.image_system_prompt,
                user_prompt=self.image_user_prompt.format(
                    language="Korean",
                    context=image_info['context']
                )
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
            logger.error(f"Image processing failed (path: {error_path}): {str(e)}")
            raise


    async def process_markdown(self, content: str, document_id: str) -> Dict:
        """마크다운 문서 처리"""
        try:
            # 이미지 처리 결과 저장
            self.processed_images = []
            image_infos = self._extract_image_paths(content)
            
            logger.info(f"\n=== Start Image Processing ===")
            logger.info(f"Found {len(image_infos)} images")
            
            # 배치 크기 설정 (한 번에 처리할 이미지 수)
            batch_size = 3
            for i in range(0, len(image_infos), batch_size):
                batch = image_infos[i:i + batch_size]
                logger.info(f"\nBatch {i//batch_size + 1} processing... ({i+1}-{min(i+batch_size, len(image_infos))}/{len(image_infos)})")
                
                # 배치 내 이미지 순차 처리
                for j, image_info in enumerate(batch):
                    logger.info(f"Processing image {i+j+1}/{len(image_infos)}...")
                    
                    try:
                        result = await self.process_image(image_info)
                        self.processed_images.append(result)
                        
                        # 이미지 처리 간 딜레이 (1초)
                        if j < len(batch) - 1:  # 마지막 이미지가 아닌 경우에만 딜레이
                            await asyncio.sleep(1)
                            
                    except Exception as e:
                        logger.error(f"Image processing failed: {str(e)}")
                        continue
                
                # 배치 간 딜레이 (3초)
                if i + batch_size < len(image_infos):
                    logger.info("Waiting for next batch...")
                    await asyncio.sleep(3)
            
            # 전체 토큰 수 출력
            logger.info("\n=== 전체 토큰 사용량 ===")
            logger.info(f"시스템 프롬프트: {self.system_tokens} 토큰")
            logger.info(f"사용자 프롬프트: {self.user_tokens} 토큰")
            logger.info(f"총 컨텍스트: {self.total_context_tokens} 토큰")
            logger.info(f"총 요약: {self.total_summary_tokens} 토큰")
            logger.info(f"총 토큰 수: {self.total_prompt_tokens + self.total_context_tokens + self.total_summary_tokens} 토큰")
            
            return {
                "document_id": document_id,
                "objects": self.processed_images,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Markdown processing failed: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            }