FROM nvcr.io/nvidia/pytorch:23.05-py3

# Copy data and model files
COPY work/biencoder.ckpt /work/biencoder.ckpt
COPY work/reader.ckpt /work/reader.ckpt
COPY work/passage_embeddings.npy /work/passage_embeddings.npy
COPY work/passages.json.gz /work/passages.json.gz

# Copy codes
COPY aio4_bpr_baseline /code/aio4_bpr_baseline
# COPY prediction_loop.py /code/prediction_loop.py
COPY prediction_api.py /code/prediction_api.py
COPY pyproject.toml /code/pyproject.toml
COPY setup.cfg /code/setup.cfg

# Install dependencies
WORKDIR /code
RUN pip install -U pip setuptools wheel --no-cache-dir && \
    pip install . --no-cache-dir

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-v3"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

# Start the prediction loop
WORKDIR /code
# ENTRYPOINT ["python", "-m", "prediction_loop"]
CMD ["uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
