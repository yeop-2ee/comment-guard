# 댓글 가드 파이프라인 진입점. 각 단계는 실패하면 0이 아닌 코드로 끝나고, Make는 그 자리에서 멈춥니다.
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

PYTHON_BIN ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup data validate train evaluate test all-local serve smoke-local upload-model deploy smoke cleanup

setup:
	$(PYTHON_BIN) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt -r requirements-m2.txt

data:
	$(PY) -m commentguard.runlog wrap --step data -- bash scripts/download_data.sh

validate:
	$(PY) -m commentguard.data

train:
	$(PY) -m commentguard.train

evaluate:
	$(PY) -m commentguard.evaluate

test:
	$(PY) -m pytest -q

all-local: data validate train evaluate test

PORT ?= 8080
# 로컬 서버 (MODEL_URI가 없으면 configs/serving.yaml이 가리키는 모델)
serve:
	$(VENV)/bin/uvicorn commentguard.api:app --host 127.0.0.1 --port $(PORT)

# 다른 터미널에서 make serve 를 띄운 뒤 실행
smoke-local:
	$(PY) scripts/smoke_cloud.py --url http://127.0.0.1:$(PORT) --target local

upload-model:
	@echo "M4에서 구현 예정" >&2; exit 1

deploy: test
	@echo "M4에서 구현 예정" >&2; exit 1

smoke:
	@echo "M4에서 구현 예정" >&2; exit 1

cleanup:
	@echo "M8에서 구현 예정" >&2; exit 1
