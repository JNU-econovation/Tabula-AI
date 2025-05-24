import base64
import logging
from fastapi import Depends, Header
from jose import jwt, ExpiredSignatureError
from errors.business_exception import InvalidJWT, ExpiredJWT
from common_sdk.config import settings

# 로그 메시지
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# 인증을 위한 환경변수 세팅
ALGORITHM = "HS256"

# BASE64로 인코딩된 JWT_SECRET 디코딩
JWT_SECRET = base64.urlsafe_b64decode(settings.JWT_SECRET)


# Bearer token 추출 및 디코딩
def get_token_from_header(authorization: str = Header(...)):
    if not authorization:
        logger.error("토큰의 형식이 잘못 되었습니다.")
        # 잘못된 인증 토큰 예외처리 
        raise InvalidJWT()
    token = authorization.split("Bearer ")[1]
    return token


# Token에서 member id 가져오기
async def get_current_member(token: str = Depends(get_token_from_header)):
    
    if isinstance(token, dict):
        return token

    try:
        decoded_payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        member_id: int = decoded_payload.get("sub")

        if member_id is None:
            # 잘못된 인증 토큰 예외처리
            logger.error("토큰에 member_id가 존재하지 않습니다.")
            raise InvalidJWT()
        
        logger.debug(f"토큰 디코딩 member_id: {member_id}")
        return member_id

    except ExpiredSignatureError:
        # 만료된 인증 토큰 예외처리
        logger.error("토큰이 만료 되었습니다.")
        raise ExpiredJWT()