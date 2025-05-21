#!/usr/bin/env python3
"""
TeddyNote Parser API - PDF 문서 파싱 서비스

이 스크립트는 FastAPI 기반 RESTful API로 PDF 문서 파싱 서비스를 제공합니다.
"""

import os
import uuid
import shutil
import time
import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Depends,
    status,
    Query,
    Form,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# main.py에서 파싱 기능 임포트
from layoutparse.parsing import parse_document, check_required_env_vars


# 작업 상태 정의
class JobStatus(str, Enum):
    PENDING = "pending"  # 작업 대기 중
    PROCESSING = "processing"  # 처리 중
    COMPLETED = "completed"  # 완료됨
    FAILED = "failed"  # 실패


# 작업 모델 정의
class Job(BaseModel):
    id: str = Field(..., description="작업 고유 ID")
    status: JobStatus = Field(..., description="작업 상태")
    created_at: float = Field(..., description="작업 생성 시간 (UNIX 타임스탬프)")
    completed_at: Optional[float] = Field(
        None, description="작업 완료 시간 (UNIX 타임스탬프)"
    )
    filename: str = Field(..., description="원본 파일명")
    language: str = Field(..., description="문서 언어")
    output_dir: str = Field(..., description="결과 저장 디렉토리")
    include_image: bool = Field(..., description="이미지 포함 여부")
    batch_size: int = Field(30, description="처리할 PDF 페이지의 배치 크기")
    test_page: Optional[int] = Field(
        None, description="처리할 최대 페이지 수 (처음부터 지정한 페이지까지만 처리)"
    )
    zip_filename: Optional[str] = Field(None, description="생성된 ZIP 파일 경로")
    error: Optional[str] = Field(None, description="에러 메시지 (실패시)")


# 작업 상태 응답 모델
class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    completed_at: Optional[float] = None
    filename: str
    batch_size: int
    test_page: Optional[int] = None
    zip_filename: Optional[str] = None
    error: Optional[str] = None


# 작업 목록 응답 모델
class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]


# 파일 업로드 응답 모델
class UploadResponse(BaseModel):
    job_id: str
    message: str
    status: JobStatus


# 작업 저장소
class JobStore:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.upload_dir = Path("uploads")
        self.result_dir = Path("result")

        # 디렉토리 생성
        self.upload_dir.mkdir(exist_ok=True)
        self.result_dir.mkdir(exist_ok=True)

    def create_job(
        self,
        filename: str,
        language: str,
        include_image: bool = True,
        batch_size: int = 30,
        test_page: Optional[int] = None,
    ) -> Job:
        """새 작업 생성"""
        job_id = str(uuid.uuid4())
        output_dir = str(self.result_dir / job_id)

        # 작업 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
            completed_at=None,
            filename=filename,
            language=language,
            output_dir=output_dir,
            include_image=include_image,
            batch_size=batch_size,
            test_page=test_page,
            zip_filename=None,
            error=None,
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """작업 ID로 작업 조회"""
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> Optional[Job]:
        """작업 상태 업데이트"""
        if job_id in self.jobs:
            for key, value in kwargs.items():
                if hasattr(self.jobs[job_id], key):
                    setattr(self.jobs[job_id], key, value)
            return self.jobs[job_id]
        return None

    def list_jobs(self) -> List[Job]:
        """모든 작업 목록 조회"""
        return list(self.jobs.values())

    def get_upload_path(self, job_id: str, filename: str) -> Path:
        """업로드 파일 경로 생성"""
        return self.upload_dir / f"{job_id}_{filename}"


# FastAPI 앱 생성
app = FastAPI(
    title="TeddyNote Parser API",
    description="PDF 문서 파싱 서비스 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 작업 저장소 인스턴스
job_store = JobStore()


# 의존성 주입 - 환경 변수 검증
def verify_api_keys():
    if "UPSTAGE_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="UPSTAGE_API_KEY 환경 변수가 설정되지 않았습니다.",
        )

    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.",
        )

    return True


# 의존성 주입 - 작업 존재 여부 확인
def get_job_or_404(job_id: str) -> Job:
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"작업 ID {job_id}를 찾을 수 없습니다.",
        )
    return job


# 비동기 파싱 작업 처리 함수
async def process_pdf(
    job_id: str,
    file_path: str,
    language: str,
    include_image: bool,
    batch_size: int = 30,
    test_page: Optional[int] = None,
):
    """
    비동기적으로 PDF 파싱 작업을 처리하는 함수

    Args:
        job_id: 작업 ID
        file_path: 파싱할 PDF 파일 경로
        language: 문서 언어
        include_image: 이미지 포함 여부
        batch_size: 처리할 PDF 페이지의 배치 크기 (기본값: 30)
        test_page: 처리할 최대 페이지 수 (기본값: None, 모든 페이지 처리)
    """
    # 작업 상태 업데이트
    job_store.update_job(job_id, status=JobStatus.PROCESSING)

    # 디버깅용 - 함수 시작 시 매개변수 출력
    print(f"[process_pdf] 함수 호출. 매개변수:")
    print(f"  - job_id: {job_id} (타입: {type(job_id)})")
    print(f"  - file_path: {file_path} (타입: {type(file_path)})")
    print(f"  - language: {language} (타입: {type(language)})")
    print(f"  - include_image: {include_image} (타입: {type(include_image)})")
    print(f"  - batch_size: {batch_size} (타입: {type(batch_size)})")
    print(f"  - test_page: {test_page} (타입: {type(test_page)})")

    try:
        # 매개변수 타입 확인 및 변환
        batch_size_int = int(batch_size) if batch_size is not None else 30
        test_page_int = int(test_page) if test_page is not None else None

        print(f"[process_pdf] 변환된 값:")
        print(f"  - batch_size_int: {batch_size_int} (타입: {type(batch_size_int)})")
        print(
            f"  - test_page_int: {test_page_int} (타입: {type(test_page_int) if test_page_int is not None else None})"
        )

        # 작업 정보 확인
        job = job_store.get_job(job_id)
        print(f"[process_pdf] 작업 객체 정보:")
        print(f"  - job.batch_size: {job.batch_size} (타입: {type(job.batch_size)})")
        print(
            f"  - job.test_page: {job.test_page} (타입: {type(job.test_page) if job.test_page is not None else None})"
        )

        # 파싱 작업은 CPU 집약적이므로 ThreadPoolExecutor를 통해 실행
        loop = asyncio.get_event_loop()

        print(f"[process_pdf] parse_document 함수 호출 전 파라미터:")
        print(f"  - filepath: {file_path}")
        print(f"  - output_dir: {job.output_dir}")
        print(f"  - language: {language}")
        print(f"  - include_image: {include_image}")
        print(f"  - batch_size: {batch_size_int}")
        print(f"  - test_page: {test_page_int}")

        # 파싱 함수를 별도 스레드에서 실행
        snapshot = await loop.run_in_executor(
            None,
            lambda: parse_document(
                filepath=file_path,
                output_dir=job.output_dir,
                language=language,
                include_image=include_image,
                create_zip=True,
                batch_size=batch_size_int,
                test_page=test_page_int,
                verbose=True,  # API에서는 항상 자세한 로그 출력
            ),
        )

        # ZIP 파일명 찾기 (가장 최근 생성된 파일)
        output_dir = job.output_dir
        zip_files = list(Path(".").glob(f"{output_dir}_*.zip"))
        zip_filename = None

        if zip_files:
            # 가장 최근 생성된 ZIP 파일 선택
            zip_filename = str(
                sorted(zip_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            )

        # 작업 상태 업데이트
        job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=time.time(),
            zip_filename=zip_filename,
        )

        print(f"[process_pdf] 작업 '{job_id}' 완료됨. ZIP 파일: {zip_filename}")

    except Exception as e:
        # 에러 발생 시 상태 업데이트
        error_msg = str(e)
        print(f"[process_pdf] 오류 발생: {error_msg}")
        job_store.update_job(
            job_id, status=JobStatus.FAILED, completed_at=time.time(), error=error_msg
        )


# 헬스 체크 엔드포인트
@app.get("/health", tags=["시스템"])
async def health_check():
    """API 서버의 상태를 확인합니다."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# PDF 파일 업로드 및 파싱 작업 요청
@app.post("/parse", response_model=UploadResponse, tags=["파싱"])
async def parse_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="파싱할 PDF 파일"),
    language: str = Form(
        "English", description="문서 언어 (English, Korean, Japanese, Chinese)"
    ),
    include_image: bool = Form(True, description="결과에 이미지 포함 여부"),
    batch_size: int = Form(
        30, description="처리할 PDF 페이지의 배치 크기 (한 번에 처리할 페이지 수)"
    ),
    test_page: Optional[int] = Form(
        None,
        description="처리할 최대 페이지 수 (처음부터 지정한 페이지까지만 처리, 지정하지 않으면 모든 페이지 처리)",
    ),
    _: bool = Depends(verify_api_keys),
):
    """
    PDF 파일을 업로드하고 파싱 작업을 시작합니다.

    작업은 비동기적으로 처리되며, 작업 ID가 반환됩니다.
    작업 상태는 `/status/{job_id}` 엔드포인트를 통해 확인할 수 있습니다.
    """

    # 디버깅용 - 요청 파라미터 출력
    print(
        f"[/parse] 요청 원본 값: language={language}, include_image={include_image}, batch_size={batch_size}, test_page={test_page}"
    )
    print(
        f"[/parse] 요청 값 타입: language={type(language)}, include_image={type(include_image)}, batch_size={type(batch_size)}, test_page={type(test_page)}"
    )

    # 값 유효성 확인 및 변환
    try:
        batch_size_int = int(batch_size)
        print(f"[/parse] batch_size 변환: {batch_size} -> {batch_size_int}")
    except (ValueError, TypeError):
        print(f"[/parse] batch_size 변환 실패, 기본값 30 사용")
        batch_size_int = 30

    test_page_int = None
    if test_page is not None:
        try:
            test_page_int = int(test_page)
            print(f"[/parse] test_page 변환: {test_page} -> {test_page_int}")
        except (ValueError, TypeError):
            print(f"[/parse] test_page 변환 실패, None 사용")

    # 파일 확장자 확인
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext != "pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF 파일만 업로드 가능합니다.",
        )

    # 새 작업 생성
    job = job_store.create_job(
        filename=file.filename,
        language=language,
        include_image=include_image,
        batch_size=batch_size_int,
        test_page=test_page_int,
    )

    # 업로드 파일 저장
    file_path = job_store.get_upload_path(job.id, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 값이 제대로 설정되었는지 확인
    print(
        f"[/parse] 작업 생성 후: job.id={job.id}, job.batch_size={job.batch_size}, job.test_page={job.test_page}"
    )

    # 비동기 작업 시작 - 직접 값을 전달
    background_tasks.add_task(
        process_pdf,
        job.id,
        str(file_path),
        language,
        include_image,
        batch_size_int,  # 정수로 명시적 변환
        test_page_int,  # 정수로 명시적 변환 또는 None
    )

    return {
        "job_id": job.id,
        "message": "PDF 파싱 작업이 시작되었습니다.",
        "status": JobStatus.PENDING,
    }


# 작업 상태 확인 엔드포인트
@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["작업"])
async def get_job_status(job: Job = Depends(get_job_or_404)):
    """작업 ID를 사용하여 작업 상태를 확인합니다."""
    return {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "filename": job.filename,
        "batch_size": job.batch_size,
        "test_page": job.test_page,
        "zip_filename": job.zip_filename,
        "error": job.error,
    }


# 결과 다운로드 엔드포인트
@app.get("/download/{job_id}", tags=["결과"])
async def download_result(job: Job = Depends(get_job_or_404)):
    """
    작업 ID를 사용하여 결과 파일을 다운로드합니다.

    작업이 완료된 경우에만 다운로드가 가능합니다.
    결과는 ZIP 파일 형태로 제공됩니다.
    """
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="작업이 아직 완료되지 않았습니다.",
        )

    if not job.zip_filename or not os.path.exists(job.zip_filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="결과 파일을 찾을 수 없습니다.",
        )

    return FileResponse(
        job.zip_filename,
        media_type="application/zip",
        filename=f"{job.filename.split('.')[0]}_result.zip",
    )


# 작업 목록 조회 엔드포인트
@app.get("/jobs", response_model=JobListResponse, tags=["작업"])
async def list_jobs():
    """모든 작업 목록을 조회합니다."""
    jobs = []
    for job in job_store.list_jobs():
        jobs.append(
            {
                "job_id": job.id,
                "status": job.status,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "filename": job.filename,
                "batch_size": job.batch_size,
                "test_page": job.test_page,
                "zip_filename": job.zip_filename,
                "error": job.error,
            }
        )
    return {"jobs": jobs}


# 서버 실행
if __name__ == "__main__":
    # 환경 변수 확인
    if not check_required_env_vars():
        print("필수 환경 변수가 설정되지 않았습니다. 서버를 종료합니다.")
        import sys

        sys.exit(1)

    # 서버 포트
    port = int(os.environ.get("PORT", 9996))

    # 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=port)
