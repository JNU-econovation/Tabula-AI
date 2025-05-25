from fastapi import status

# API 응답 구성 예시
get_detail_responses = {
    401: {
        "description": "인증 오류: 잘못된 인증 토큰 또는 만료된 인증 토큰",
        "content": {
            "application/json": {
                "examples": {
                    "InvalidJWT": {
                        "summary": "잘못된 인증 토큰",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SECURITY_401_1",
                                "reason": "잘못된 인증 토큰 형식입니다.",
                                "http_status": status.HTTP_401_UNAUTHORIZED
                            }
                        }
                    },
                    "ExpiredJWT": {
                        "summary": "만료된 인증 토큰",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SECURITY_401_2",
                                "reason": "인증 토큰이 만료되었습니다.",
                                "http_status": status.HTTP_401_UNAUTHORIZED
                            }
                        }
                    }
                }
            }
        }
    },
    404: {
        "description": "데이터 오류: 분석에 필요한 데이터 미존재 또는 분석 기록 없음",
        "content": {
            "application/json": {
                "examples": {
                    "NoAnalysisRecord": {
                        "summary": "분석 기록 없음(해당 유저는 분석이 성공한 경우가 존재하지 않음)",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "DIET_404_3",
                                "reason": "해당 유저에 대한 분석 기록이 존재하지 않습니다.",
                                "http_status": status.HTTP_404_NOT_FOUND
                            }
                        }
                    }
                }
            }
        }
    }
}