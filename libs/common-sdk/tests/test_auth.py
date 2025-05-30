from typing import Optional
from fastapi import Header
from fastapi import FastAPI

from common_sdk.auth import get_token_from_header, get_current_member

app = FastAPI()


@app.get("/test/token")
def test_token(
    authorization: Optional[str] = Header(None)
):
    try:
        token = get_token_from_header(authorization)
        print("Successfully extracted token from header")

        user_id = get_current_member(token)
        print(f"Successfully decoded token. User ID: {user_id}")
        
        response = {
            "success": True,
            "response": {
                "user_id": user_id
            },
            "error": None
        }
        return response
    
    except Exception as e:
        response = {
            "success": False,
            "response": None,
            "error": str(e)
        }
        return response