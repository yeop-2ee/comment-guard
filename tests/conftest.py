"""API 테스트용 작은 가짜 모델. 학습된 모델(models/) 없이도 make test가 돌게 합니다."""

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from commentguard.api import create_app
from commentguard.model_store import CommentModel

# 직접 만든 가짜 예시 (공개 데이터 원문 아님)
FAKE = [
    ("오늘 무대 정말 좋았어요", "none", "none"),
    ("응원합니다 힘내세요", "none", "none"),
    ("노래 잘 들었습니다", "none", "none"),
    ("연기 늘었네 보기 좋다", "none", "none"),
    ("진짜 바보 같네", "offensive", "none"),
    ("멍청한 소리 그만해", "offensive", "others"),
    ("생각 좀 하고 말해라 바보야", "offensive", "none"),
    ("여자가 무슨 정치냐 바보", "hate", "gender"),
    ("저런 애들은 다 쫓아내야 함 멍청이", "hate", "others"),
    ("여자들은 다 똑같아 멍청해", "hate", "gender"),
]
TEST_VERSION = "test-20261109-0000000"


@pytest.fixture(scope="session")
def tiny_model():
    texts = [t for t, _, _ in FAKE]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
    X = vec.fit_transform(texts)
    hate = LogisticRegression(max_iter=1000).fit(X, [h for _, h, _ in FAKE])
    bias = LogisticRegression(max_iter=1000).fit(X, [b for _, _, b in FAKE])
    card = {"model_version": TEST_VERSION, "git_commit": "0" * 40, "git_dirty": False,
            "severity": {"thresholds": {"caution": 30, "severe": 60},
                         "sensitivity": {"low": {"normal": "show", "caution": "show", "severe": "hide"},
                                         "medium": {"normal": "show", "caution": "blur", "severe": "hide"},
                                         "high": {"normal": "show", "caution": "hide", "severe": "hide"}}}}
    return CommentModel(vec, hate, bias, card)


@pytest.fixture()
def client(tiny_model):
    with TestClient(create_app(model=tiny_model)) as c:
        yield c
