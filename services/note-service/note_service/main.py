import uvicorn

from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from note_service.router import router

app = FastAPI(
    title="Tabula Service",
    description="Note Service API",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    # Front Deploy URL
]
# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Tabula Note Service"}

# health check
@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


app.include_router(router, prefix="/v1/spaces")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)