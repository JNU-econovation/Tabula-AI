import os
import sys

# 현재 파일의 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


import re 
from common_sdk.config import settings
from typing import List, Dict, Any
import json
import time
import tiktoken
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """문자열의 토큰 수를 계산"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

class KeywordGuide:
    def __init__(self, domain: str, llm_model_name: str = "gpt-4.1-mini", temperature: float = 0):
        """
        KeywordGuide 초기화
        
        Args:
            domain: 도메인 타입
            llm_model_name: 사용할 LLM 모델 이름
            temperature: LLM의 temperature 값
        """
        self.domain_type = domain
        self.llm = ChatOpenAI(model=llm_model_name, temperature=temperature, openai_api_key=settings.OPENAI_API_KEY)
        print(f"\n=== 모델 정보 ===")
        print(f"사용 모델: {llm_model_name}")
        print(f"Temperature: {temperature}")

    def preprocess_markdown_content(self, content: str) -> str:
        """
        마크다운 내용에서 이미지 및 테이블 링크 제거
        
        Args:
            content: 원본 마크다운 내용
            
        Returns:
            이미지 및 테이블 링크가 제거된 마크다운 내용
        """
        # 이미지 링크 제거
        content = re.sub(r"!\[[^\]]*?\]\([^)]*?\)", "", content)
        
        # 테이블 CSV 링크 제거
        content = re.sub(r"\[[^\]]*?\]\([^)]*?\.csv\)", "", content)
        
        return content

    def extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """
        마크다운 내용에서 Heading 추출
        
        Args:
            content: 마크다운 내용
            
        Returns:
            추출된 Heading 리스트
        """
        headings = []
        for line in content.split('\n'):
            if line.startswith('#'):
                # Heading 레벨과 텍스트 분리
                level = len(line.split()[0])
                text = line.strip('# ')
                headings.append({
                    'level': level,
                    'text': text
                })
        return headings

    def call_llm(self, system_prompt_content: str, user_prompt_content: str) -> Any:
        """
        LLM 호출
        
        Args:
            system_prompt_content: 시스템 프롬프트 내용
            user_prompt_content: 사용자 프롬프트 내용
            
        Returns:
            LLM으로부터 받은 응답 (JSON 문자열을 파싱한 Python 객체 또는 문자열 리스트)
        """
        messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=user_prompt_content)
        ]
        
        # 토큰 수 계산
        system_tokens = num_tokens_from_string(system_prompt_content)
        user_tokens = num_tokens_from_string(user_prompt_content)
        total_tokens = system_tokens + user_tokens
        
        print(f"\n=== 토큰 사용량 ===")
        print(f"시스템 프롬프트: {system_tokens} 토큰")
        print(f"사용자 프롬프트: {user_tokens} 토큰")
        print(f"총 토큰 수: {total_tokens} 토큰")
        
        response_obj = None # API 응답 객체를 저장할 변수 초기화
        try:
            # LLM 호출
            response_obj = self.llm.invoke(messages)
            
            # --- LLM 응답 디버깅 시작 ---
            if response_obj and hasattr(response_obj, 'content'):
                raw_content = response_obj.content
                
                # 응답 토큰 수 계산
                response_tokens = num_tokens_from_string(raw_content)
                print(f"응답 토큰 수: {response_tokens} 토큰")
                print(f"총 처리 토큰 수: {total_tokens + response_tokens} 토큰")
                
                if not raw_content or not raw_content.strip():
                    print("Error: LLM response content is empty or consists only of whitespace.")
                    return None
                
                # 마크다운 코드 블록 제거 로직 추가
                cleaned_json_str = raw_content.strip()
                if cleaned_json_str.startswith("```json") and cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[len("```json"):].rstrip("`\n ")
                elif cleaned_json_str.startswith("```") and cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[len("```"):].rstrip("`\n ")

                # 여기서부터 JSON 파싱 시도
                return json.loads(cleaned_json_str)
            
            elif response_obj:
                print(f"Error: LLM response object does not have 'content' attribute or is None.")
                print(f"DEBUG: LLM response_obj type: {type(response_obj)}")
                print(f"DEBUG: LLM response_obj: {str(response_obj)}")
                return None
            else:
                print("Error: LLM call did not return a response object (response_obj is None).")
                return None
            
        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}. 파싱 시도한 내용은 위 DEBUG 출력을 참고하세요.")
            return None 
        except Exception as e:
            print(f"LLM 호출 또는 응답 처리 중 예기치 않은 오류 발생: {e}")
            if response_obj and hasattr(response_obj, 'content'):
                 print(f"DEBUG: (Exception context) LLM content: {response_obj.content}")
            elif response_obj:
                 print(f"DEBUG: (Exception context) LLM response object: {str(response_obj)}")
            return None

    def create_mindmap_prompt(self, content: str, headings: List[Dict[str, Any]]) -> str:
        """
        LLM을 위한 마인드맵 생성용 프롬프트 생성
        
        Args:
            content: 마크다운 내용
            headings: 추출된 헤딩 정보 리스트 (현재 프롬프트에서는 사용되지 않도록 수정)
            
        Returns:
            생성된 프롬프트
        """
        # 내용 전처리
        processed_content = self.preprocess_markdown_content(content)

        prompt = f"""
당신은 교육용 문서를 분석하여 체계적인 마인드맵 구조를 생성하는 AI입니다.

다음 마크다운 문서를 분석하여, 문서의 핵심 주제를 루트 개념으로 선정하고, 
최대 3계층 깊이까지의 마인드맵을 구성해주세요.

마인드맵 구성 시 다음 원칙을 따라주세요:
1. 루트 개념: 문서의 가장 핵심적인 주제
2. 1계층: 주요 대분류 (예: 시대, 영역, 주제)
3. 2계층: 세부 분류 (예: 하위 시대, 세부 영역)
4. 3계층: 구체적 개념 (예: 중요 사건, 핵심 용어, 특징)

각 계층별 구성 원칙:
- 1계층: 3-5개의 주요 대분류
- 2계층: 각 1계층당 2-4개의 세부 분류
- 3계층: 각 2계층당 2-3개의 구체적 개념

출력은 반드시 다음 JSON 형식이어야 합니다:
{{
  "name": "루트 개념 (문서의 핵심 주제)",
  "children": [
    {{
      "name": "주요 대분류 1 (1계층)",
      "children": [
        {{
          "name": "세부 분류 1-1 (2계층)",
          "children": [
            {{"name": "구체적 개념 1-1-1 (3계층)"}},
            {{"name": "구체적 개념 1-1-2 (3계층)"}}
          ]
        }},
        {{
          "name": "세부 분류 1-2 (2계층)",
          "children": [
            {{"name": "구체적 개념 1-2-1 (3계층)"}},
            {{"name": "구체적 개념 1-2-2 (3계층)"}}
          ]
        }}
      ]
    }},
    {{
      "name": "주요 대분류 2 (1계층)",
      "children": [
        {{
          "name": "세부 분류 2-1 (2계층)",
          "children": [
            {{"name": "구체적 개념 2-1-1 (3계층)"}},
            {{"name": "구체적 개념 2-1-2 (3계층)"}}
          ]
        }}
      ]
    }}
  ]
}}

주의사항:
1. 각 노드의 이름은 간결하고 명확하게 작성
2. 계층 간의 관계가 논리적으로 연결되도록 구성
3. 3계층 노드에는 'children' 속성을 포함하지 않음
4. 각 계층의 노드 수는 위에서 제시한 범위 내에서 유지
5. 교육적 가치가 있는 핵심 개념을 우선적으로 포함

다음은 분석할 문서 내용입니다:
---
{processed_content}
---

위 지시사항과 JSON 형식을 철저히 준수하여 응답해주세요.
        """
        return prompt

    def generate_mindmap_from_markdown(self, md_path: str) -> Dict[str, Any]:
        """
        마크다운 파일로부터 마인드맵 구조 생성
        
        Args:
            md_path: 마크다운 파일 경로
            
        Returns:
            추출된 마인드맵 구조 (Python 딕셔너리)
            오류 발생 시 빈 딕셔너리 반환
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Error: Markdown file not found at {md_path}")
            return {}
        
        # 헤딩 추출
        headings = self.extract_headings(content)

        # 사용자 프롬프트 생성 (헤딩 정보 포함)
        user_prompt = self.create_mindmap_prompt(content, headings)
        # 마인드맵 생성용 시스템 프롬프트
        system_prompt = "당신은 문서를 분석하여 JSON 형식의 마인드맵을 생성하는 정확하고 지시를 잘 따르는 AI 어시스턴트입니다."
        
        mindmap_data = self.call_llm(system_prompt_content=system_prompt, user_prompt_content=user_prompt)
        
        if isinstance(mindmap_data, dict):
            mindmap_data["document_id"] = Path(md_path).stem
            return mindmap_data
        else:
            print(f"Error: Failed to generate mindmap data or unexpected format received. Data: {mindmap_data}")
            return {}

def main():
    
    guide = KeywordGuide(domain="Korean history", llm_model_name="gpt-4.1-mini")
    
    # 사용할 마크다운 파일 경로
    md_file_path = "/Users/kyeong6/Desktop/test/parse/result/md/korean.md"
    output_mindmap_path = "/Users/kyeong6/Desktop/test/parse/result/keyword/output_mindmap_korean.json"

    if not os.path.exists(md_file_path):
        print(f"오류: 마크다운 파일이 존재하지 않습니다 - {md_file_path}")
        return

    print("\n--- 마인드맵 생성 테스트 시작 ---")
    start_time = time.time() # 시작 시간 기록

    mindmap_structure = guide.generate_mindmap_from_markdown(md_file_path)
    
    end_time = time.time() # 종료 시간 기록
    elapsed_time = end_time - start_time # 소요 시간 계산
    print(f"--- 마인드맵 생성 완료 (소요 시간: {elapsed_time:.2f}초) ---")

    if mindmap_structure:
        # print("생성된 마인드맵 구조:")
        # print(json.dumps(mindmap_structure, indent=2, ensure_ascii=False))
        
        with open(output_mindmap_path, 'w', encoding='utf-8') as f: 
            json.dump(mindmap_structure, f, indent=2, ensure_ascii=False)
        print(f"마인드맵 구조가 {output_mindmap_path} 파일로 저장되었습니다.")
    else:
        print("마인드맵 구조 생성 실패")

if __name__ == "__main__":
    main()
