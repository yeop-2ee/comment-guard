"""M2 입력 형식 검사. torch가 없으면(서빙 환경) 건너뜀."""
import pytest

torch = pytest.importorskip("torch")
from commentguard.m2 import encode  # noqa: E402


class FakeTok:
    """[CLS]=1, [SEP]=2, [PAD]=3, 나머지 글자는 10부터. 특수 토큰을 스스로 붙이지 않는 토크나이저 흉내."""
    special = {"[CLS]": 1, "[SEP]": 2, "[PAD]": 3}

    def convert_tokens_to_ids(self, toks):
        return [self.special[t] for t in toks]

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [10 + (ord(c) % 50) for c in text]}


def test_encode_adds_cls_sep_and_pads():
    b = encode(FakeTok(), ["가나다", "  가  "], max_length=80)
    ids, mask = b["input_ids"].tolist(), b["attention_mask"].tolist()
    assert ids[0][0] == 1 and ids[0][4] == 2
    assert ids[1][:3] == [1, ids[0][1], 2] and ids[1][3:] == [3, 3]  # normalize()로 공백 제거
    assert mask[1] == [1, 1, 1, 0, 0]


def test_encode_truncates_to_max_length():
    b = encode(FakeTok(), ["가" * 200], max_length=16)
    assert b["input_ids"].shape[1] == 16 and b["input_ids"][0, -1].item() == 2
