import uvicorn
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from result_service.router import router as result_router

# result-service 로직 추가 필요 (swagger, exception handler)
from common_sdk.swagger import router as swagger_router
from common_sdk.exceptions import register_exception_handlers

# FastAPI Instance
app = FastAPI(
    title="Tabula Service",
    description="Result Service API",
    version="1.0.0",
    docs_url=None,  # 기본 Swagger UI 비활성화
    redoc_url=None  # 기본 ReDoc UI 비활성화
)

# CORS
origins = [
    "http://localhost:3000",
    "https://tabula.co.kr"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# root
@app.get("/")
async def root():
    return {"message": "Welcome to Tabula Result Service"}

# health check
@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# exception handler
register_exception_handlers(app)

# routers
app.include_router(result_router, prefix="/v1/ai/results")
app.include_router(swagger_router, prefix="/v1/ai/results/api")  # /api/docs, /api/redoc

# run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
