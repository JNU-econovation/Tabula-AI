from fastapi import status

# 공통 응답 구성
security_responses = {
    401: {
        "description": "인증 오류",
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
                    },
                    "EmptyJWT": {
                        "summary": "빈 인증 토큰",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SECURITY_401_3",
                                "reason": "인증 토큰이 존재하지 않습니다.",
                                "http_status": status.HTTP_401_UNAUTHORIZED
                            }
                        }
                    }
                }
            }
        }
    }
}

# Note Service 응답 구성
note_service_response = {
    **security_responses,
    400: {
        "description": "요청 데이터 오류",
        "content": {
            "application/json": {
                "examples": {
                    "MissingFieldData": {
                        "summary": "필드 데이터 누락",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SPACE_400_1",
                                "reason": "필드 데이터가 누락되었습니다.",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    },
                    "MissingNoteFileData": {
                        "summary": "파일 데이터 누락",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_400_1",
                                "reason": "파일 데이터(PDF) 누락입니다.",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    },
                    "UnsupportedNoteFileFormat": {
                        "summary": "파일 형식 미지원",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_400_2",
                                "reason": "파일 형식이 유효하지 않습니다.(지원되는 파일 형식: PDF).",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    }
                }
            }
        }
    },
    413: {
        "description": "파일 크기 초과",
        "content": {
            "application/json": {
                "examples": {
                    "NoteFileSizeExceeded": {
                        "summary": "파일 용량 초과",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_413_1",
                                "reason": "파일(PDF) 크기 허용 범위 초과입니다.(허용 범위: 5MB)",
                                "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                            }
                        }
                    }
                }
            }
        }
    }
}

# Note Service Space 응답 구성
note_service_space_response = {
    **security_responses,
    400: {
        "description": "요청 데이터 오류",
        "content": {
            "application/json": {
                "examples": {
                    "MissingSpaceId": {
                        "summary": "Space ID 누락",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SPACE_400_2",
                                "reason": "Request-Header에 spaceId가 누락되었습니다.",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    }
                }
            }
        }
    },
    404: {
        "description": "리소스 미존재",
        "content": {
            "application/json": {
                "examples": {
                    "SpaceIdNotFound": {
                        "summary": "Space ID 미존재",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SPACE_404_1",
                                "reason": "요청 데이터(spaceId)에 해당하는 리소스가 존재하지 않습니다.",
                                "http_status": status.HTTP_404_NOT_FOUND
                            }
                        }
                    }
                }
            }
        }
    }
}

# Result Service 응답 구성
result_service_response = {
    **security_responses,
    400: {
        "description": "요청 데이터 오류",
        "content": {
            "application/json": {
                "examples": {
                    "MissingResultFileData": {
                        "summary": "결과물 파일 데이터 누락",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_400_3",
                                "reason": "파일 데이터(PDF/Image) 누락입니다.",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    },
                    "UnsupportedResultFileFormat": {
                        "summary": "결과물 파일 형식 미지원",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_400_4",
                                "reason": "파일 형식이 유효하지 않습니다.(지원되는 파일 형식: PDF / Image)",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    }
                }
            }
        }
    },
    413: {
        "description": "파일 크기 초과",
        "content": {
            "application/json": {
                "examples": {
                    "ResultFileSizeExceeded": {
                        "summary": "결과물 파일 용량 초과",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_413_2",
                                "reason": "파일(PDF / Image) 크기가 허용 범위 초과입니다.(허용 범위: 5MB)",
                                "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                            }
                        }
                    },
                    "ResultFileUploadPageExceeded": {
                        "summary": "결과물 업로드 페이지 초과",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "FILE_413_3",
                                "reason": "파일(PDF / Image)입력 페이지가 허용 범위 초과입니다.(허용 범위: 6페이지)",
                                "http_status": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                            }
                        }
                    }
                }
            }
        }
    }
}

# Result Service Space 응답 구성
result_service_space_response = {
    **security_responses,
    400: {
        "description": "요청 데이터 오류",
        "content": {
            "application/json": {
                "examples": {
                    "MissingTaskId": {
                        "summary": "Space ID 누락",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SPACE_400_2",
                                "reason": "Request-Header에 spaceId가 누락되었습니다.",
                                "http_status": status.HTTP_400_BAD_REQUEST
                            }
                        }
                    }
                }
            }
        }
    },
    404: {
        "description": "리소스 미존재",
        "content": {
            "application/json": {
                "examples": {
                    "TaskIdNotFound": {
                        "summary": "Space ID 미존재",
                        "value": {
                            "success": False,
                            "response": None,
                            "error": {
                                "code": "SPACE_404_1",
                                "reason": "요청 데이터(spaceId)에 해당하는 리소스가 존재하지 않습니다.",
                                "http_status": status.HTTP_404_NOT_FOUND
                            }
                        }
                    }
                }
            }
        }
    }
}