class ProgressPhase:
    PDF_PARSING = "PDF 파싱"
    MARKDOWN_PROCESSING = "마크다운 처리"
    KEYWORD_GENERATION = "키워드 생성"
    DB_STORAGE = "데이터베이스 적재"

class ProgressRange:
    PDF_PARSING = (0, 30)
    MARKDOWN_PROCESSING = (30, 60)
    KEYWORD_GENERATION = (60, 90)
    DB_STORAGE = (90, 100)

class StatusMessage:
    INITIAL = "처리 준비 중"
    PDF_PARSING = "PDF 파싱 중"
    MARKDOWN_PROCESSING = "마크다운 처리 중"
    KEYWORD_GENERATION = "키워드 생성 중"
    DB_STORAGE = "데이터베이스 저장 중"
    COMPLETE = "처리 완료"
    ERROR = "에러 발생" 