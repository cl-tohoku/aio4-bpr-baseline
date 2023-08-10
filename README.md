# AIO4 BPR Baseline

## Example Usage

**Note:** The following example is run on a server with 4 GPUs, each with at least 16 GB of memory.

### Download datasets

```sh
mkdir data
wget https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/datasets.jawiki-20220404-c400-large.aio_02_train.jsonl.gz -P data
wget https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/datasets.jawiki-20220404-c400-large.aio_02_dev.jsonl.gz -P data
wget https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.1/passages.jawiki-20220404-c400-large.jsonl.gz -P data
```

### Preprocess the datasets

```sh
mkdir -p work/data/aio_02/retriever
python convert_retriever_dataset.py \
  --input_dataset_file data/datasets.jawiki-20220404-c400-large.aio_02_train.jsonl.gz \
  --output_dataset_file work/data/aio_02/retriever/aio_02_train.jsonl.gz
python convert_retriever_dataset.py \
  --input_dataset_file data/datasets.jawiki-20220404-c400-large.aio_02_dev.jsonl.gz \
  --output_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz

mkdir -p work/data/aio_02/passages
python convert_passage_dataset.py \
  --passage_file data/passages.jawiki-20220404-c400-large.jsonl.gz \
  --output_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz
```

### Retriever Training

```sh
python cli.py fit \
  --config configs/retriever.yaml \
  --model.train_dataset_file work/data/aio_02/retriever/aio_02_train.jsonl.gz \
  --model.val_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/retriever
```

### Embedder Prediction

```sh
mkdir -p work/data/aio_02/passage_index
python cli.py predict \
  --config configs/embedder.yaml \
  --model.retriever_ckpt_file work/model/aio_02/retriever/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.predict_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/embeddings/jawiki-20220404-c400-large
python gather_numpy_predictions.py \
  --predictions_dir work/model/aio_02/embeddings/jawiki-20220404-c400-large/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/embeddings/jawiki-20220404-c400-large/lightning_logs/version_0/prediction.npy
python build_passage_faiss_index.py \
  --embedder_prediction_file work/model/aio_02/embeddings/jawiki-20220404-c400-large/lightning_logs/version_0/prediction.npy \
  --output_file work/data/aio_02/passage_index/jawiki-20220404-c400-large.faiss
```


### Retriever Prediction

```sh
python cli.py predict \
  --config configs/retriever.yaml \
  --ckpt_path work/model/aio_02/retriever/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.predict_dataset_file work/data/aio_02/retriever/aio_02_train.jsonl.gz \
  --model.passage_faiss_index_file work/data/aio_02/passage_index/jawiki-20220404-c400-large.faiss \
  --trainer.default_root_dir work/model/aio_02/retrieved_passages/aio_02_train
python gather_json_predictions.py \
  --predictions_dir work/model/aio_02/retrieved_passages/aio_02_train/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/retrieved_passages/aio_02_train/lightning_logs/version_0/prediction.jsonl.gz

python cli.py predict \
  --config configs/retriever.yaml \
  --ckpt_path work/model/aio_02/retriever/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.predict_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --model.passage_faiss_index_file work/data/aio_02/passage_index/jawiki-20220404-c400-large.faiss \
  --trainer.default_root_dir work/model/aio_02/retrieved_passages/aio_02_dev
python gather_json_predictions.py \
  --predictions_dir work/model/aio_02/retrieved_passages/aio_02_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/retrieved_passages/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz
```

### Retriever Evaluation

```sh
mkdir -p work/data/aio_02/reader
python evaluate_retriever.py \
  --retriever_input_file  work/data/aio_02/retriever/aio_02_train.jsonl.gz \
  --retriever_prediction_file work/model/aio_02/retrieved_passages/aio_02_train/lightning_logs/version_0/prediction.jsonl.gz \
  --passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --match_type nfkc \
  --output_file work/data/aio_02/reader/aio_02_train.jsonl.gz
# Recall@1: 0.7638 (17059/22335)
# Recall@2: 0.8446 (18864/22335)
# Recall@5: 0.8878 (19828/22335)
# Recall@10: 0.9024 (20154/22335)
# Recall@20: 0.9112 (20351/22335)
# Recall@50: 0.9227 (20608/22335)
# Recall@100: 0.9311 (20797/22335)
# MRR@10: 0.8199
python evaluate_retriever.py \
  --retriever_input_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --retriever_prediction_file work/model/aio_02/retrieved_passages/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz \
  --passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --match_type nfkc \
  --output_file work/data/aio_02/reader/aio_02_dev.jsonl.gz
# Recall@1: 0.5740 (574/1000)
# Recall@2: 0.6800 (680/1000)
# Recall@5: 0.7570 (757/1000)
# Recall@10: 0.8030 (803/1000)
# Recall@20: 0.8410 (841/1000)
# Recall@50: 0.8710 (871/1000)
# Recall@100: 0.8890 (889/1000)
# MRR@10: 0.6587
```

### Reader Training

```sh
python cli.py fit \
  --config configs/reader.yaml \
  --model.train_dataset_file work/data/aio_02/reader/aio_02_train.jsonl.gz \
  --model.val_dataset_file work/data/aio_02/reader/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/reader
```

### Reader Prediction

```sh
python cli.py predict \
  --config configs/reader.yaml \
  --ckpt_path work/model/aio_02/reader/lightning_logs/version_0/checkpoints/best.ckpt \
  --model.predict_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/reader_prediction/aio_02_dev
python gather_json_predictions.py \
  --predictions_dir work/model/aio_02/reader_prediction/aio_02_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/reader_prediction/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz
```

### Reader Evaluation

```sh
python evaluate_reader.py \
  --reader_input_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --reader_prediction_file work/model/aio_02/reader_prediction/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz \
  --normalization_mode nfkc
# Exact Match: 0.5600 (560/1000)
```

### Pipeline Prediction

```sh
python cli.py predict \
  --config configs/pipeline.yaml \
  --model.retriever_ckpt_file work/model/aio_02/retriever/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.reader_ckpt_file work/model/aio_02/reader/lightning_logs/version_0/checkpoints/best.ckpt \
  --model.predict_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --model.passage_faiss_index_file work/data/aio_02/passage_index/jawiki-20220404-c400-large.faiss \
  --model.passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/pipeline/aio_02_dev
python gather_json_predictions.py \
  --predictions_dir work/model/aio_02/pipeline/aio_02_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/pipeline/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz
```

### Building and Running the Web API

```sh
docker build -t aio4-bpr-baseline .
docker run --rm -p 8000:8000 aio4-bpr-baseline
```

```sh
curl -G "http://localhost:8000/answer" --data-urlencode "q=滋賀県の面積のおよそ6分の1を占める、日本で一番広い湖はどこ？"
# {"answer":"琵琶 湖"}
```
