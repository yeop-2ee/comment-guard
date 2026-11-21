# 댓글 가드 파이프라인 진입점. 각 단계는 실패하면 0이 아닌 코드로 끝나고, Make는 그 자리에서 멈춥니다.
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c

PYTHON_BIN ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup data validate train evaluate test all-local serve serve-onnx smoke-local export-onnx vercel-setup upload-model deploy smoke cleanup

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

# M2를 ONNX로 변환하고 valid로 PyTorch 결과와 같은지 확인 (Vercel 서빙용, 명세서 6.2장)
export-onnx:
	$(PY) -m commentguard.onnx_export

# 클라우드와 같은 ONNX 경로로 로컬 서버 실행 (가중치 없이 model.onnx만 있는 폴더를 만들어 MODEL_URI로 지정)
ONNX_DIR := .state/onnx-serving
serve-onnx:
	@d="$$($(PY) -c 'from commentguard.model_store import serving_dir; print(serving_dir())')"; \
	rm -rf $(ONNX_DIR) && mkdir -p $(ONNX_DIR) && \
	for f in model_card.json model.onnx tokenizer.json tokenizer_config.json; do ln -s "$$d/$$f" $(ONNX_DIR)/$$f; done
	MODEL_URI=$(CURDIR)/$(ONNX_DIR) $(VENV)/bin/uvicorn commentguard.api:app --host 127.0.0.1 --port $(PORT)

# 실행 기록(run.json, output.txt)에서 Blob 저장소 ID를 가림 (configs/deploy.env의 BLOB_STORE_ID)
MASK = export COMMENTGUARD_MASK="$$(sed -n 's/^BLOB_STORE_ID=//p' configs/deploy.env 2>/dev/null | tr -d '"' | sed 's/^store_//' | awk 'NF{print $$0","tolower($$0)}')";
WRAP = $(PY) -m commentguard.runlog wrap

vercel-setup:
	$(MASK) $(WRAP) --step vercel-setup -- bash scripts/vercel_setup.sh

upload-model:
	$(MASK) $(WRAP) --step upload-model --model-version "$$(basename "$$($(PY) -c 'from commentguard.model_store import serving_dir; print(serving_dir())')")" -- bash scripts/upload_model.sh

# 테스트가 실패하면 배포하지 않음
deploy: test
	$(MASK) $(WRAP) --step deploy --suffix $${RUN_SUFFIX:-main} -- bash scripts/deploy_vercel.sh

smoke:
	$(MASK) $(PY) scripts/smoke_cloud.py --target cloud --logs --suffix $${RUN_SUFFIX:-main} \
		$$($(PY) -c "import json;d=json.load(open('.state/last_deploy.json'));print(f\"--url {d['url']} --region {d['region']} --deployment-id {d['deployment_id']} --deployment-url {d['deployment_url']} --model-uri {d['model_uri']}\")")

cleanup:
	@echo "M8에서 구현 예정" >&2; exit 1
