import pytest
from common_sdk.utils import get_embedding

@pytest.fixture
def test_texts():
    """테스트용 텍스트 fixture"""
    return {
        "korean": "안녕하세요. 이것은 한국어 테스트입니다.",
        "english": "Hello. This is an English test.",
        "korean_with_newline": "안녕하세요.\n이것은 줄바꿈이 있는\n한국어 테스트입니다.",
        "english_with_newline": "Hello.\nThis is a test\nwith newlines."
    }

def test_korean_embedding(test_texts):
    """한국어 임베딩 테스트"""
    # 기본 한국어 텍스트 테스트
    embedding = get_embedding(test_texts["korean"], "ko")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # 줄바꿈이 있는 한국어 텍스트 테스트
    embedding_with_newline = get_embedding(test_texts["korean_with_newline"], "ko")
    assert isinstance(embedding_with_newline, list)
    assert len(embedding_with_newline) > 0
    assert all(isinstance(x, float) for x in embedding_with_newline)

def test_english_embedding(test_texts):
    """영어 임베딩 테스트"""
    # 기본 영어 텍스트 테스트
    embedding = get_embedding(test_texts["english"], "en")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
    
    # 줄바꿈이 있는 영어 텍스트 테스트
    embedding_with_newline = get_embedding(test_texts["english_with_newline"], "en")
    assert isinstance(embedding_with_newline, list)
    assert len(embedding_with_newline) > 0
    assert all(isinstance(x, float) for x in embedding_with_newline)

def test_embedding_dimensions(test_texts):
    """임베딩 차원 테스트"""
    # 한국어와 영어 임베딩의 차원이 동일한지 확인
    ko_embedding = get_embedding(test_texts["korean"], "ko")
    en_embedding = get_embedding(test_texts["english"], "en")
    
    assert len(ko_embedding) == len(en_embedding)
    print(f"임베딩 차원: {len(ko_embedding)}")

def test_invalid_language():
    """잘못된 언어 설정 테스트"""
    with pytest.raises(Exception):
        get_embedding("테스트", "invalid_language")

def test_empty_text():
    """빈 텍스트 테스트"""
    with pytest.raises(Exception):
        get_embedding("", "ko")
    with pytest.raises(Exception):
        get_embedding("", "en")