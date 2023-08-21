FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir \
        datasets \
        faiss-cpu \
        lightning \
        torch \
        transformers[ja]

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-v3"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

# Copy data and model files
COPY work/model/aio_02/retriever/lightning_logs/version_0/checkpoints/last.ckpt /work/retriever.ckpt
COPY work/model/aio_02/reader/lightning_logs/version_0/checkpoints/best.ckpt /work/reader.ckpt
COPY work/data/aio_02/passage_index/jawiki-20220404-c400-large.faiss /work/passages.faiss
COPY work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz /work/passages.json.gz

# Copy codes
COPY models /code/models
COPY utils /code/utils
COPY prediction_loop.py /code

# Start the web API
WORKDIR /code
ENTRYPOINT ["python", "prediction_loop.py"]
