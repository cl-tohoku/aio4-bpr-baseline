FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy data and model files
COPY work/model/aio_02/biencoder/lightning_logs/version_0/onnx/question_encoder.onnx /work/question_encoder.onnx
COPY work/model/aio_02/reader/lightning_logs/version_0/onnx/reader.onnx /work/reader.onnx
COPY work/model/aio_02/embedder/lightning_logs/version_0/embedings.faiss /work/passages.faiss
COPY work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz /work/passages.json.gz

# Copy codes
COPY aio4_bpr_baseline /code/aio4_bpr_baseline
COPY prediction_loop.py /code/prediction_loop.py
COPY pyproject.toml /code/pyproject.toml
COPY setup.cfg /code/setup.cfg

# Install dependencies
WORKDIR /code
RUN pip install -U pip setuptools wheel && \
    pip install -e '.[onnx]' && \
    pip cache purge

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-v3"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

RUN python -c 'from datasets import Dataset; from aio4_bpr_baseline.utils.data import PASSAGE_DATASET_FEATURES; Dataset.from_json("/work/passages.json.gz", features=PASSAGE_DATASET_FEATURES)'

# Start the prediction loop
WORKDIR /code
ENTRYPOINT ["python", "-m", "prediction_loop"]
