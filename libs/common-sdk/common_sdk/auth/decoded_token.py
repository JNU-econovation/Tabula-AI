import base64

from fastapi import Depends, Header
from jose import jwt, ExpiredSignatureError

from common_sdk.config import settings
from common_sdk.exceptions import InvalidJWT, ExpiredJWT, EmptyJWT
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()


# 인증 환경변수 세팅
ALGORITHM = settings.ALGORITHM
JWT_SECRET = base64.urlsafe_b64decode(settings.JWT_SECRET)


# Bearer token 추출 및 디코딩
def get_token_from_header(authorization: str = Header(None)):
    if not authorization:
        logger.error("[get_token_from_header] Authorization header is missing")
        raise EmptyJWT()
    
    try:
        token = authorization.split("Bearer ")[1]
        return token
    except IndexError:
        logger.error("[get_token_from_header] Invalid token format")
        raise InvalidJWT()


# Token에서 member id 가져오기
def get_current_member(token: str = Depends(get_token_from_header)):
    
    if isinstance(token, dict):
        return token

    try:
        decoded_payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id: int = decoded_payload.get("sub")
        
        logger.info(f"[get_current_member] Successfully decoded token for user_id: {user_id}")
        return user_id

    except ExpiredSignatureError:
        logger.error(f"[get_current_member] Token expired for user_id: {decoded_payload.get('sub', 'unknown')}")
        raise ExpiredJWT()