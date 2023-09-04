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
# Recall@1: 0.7597 (16968/22335)
# Recall@2: 0.8409 (18782/22335)
# Recall@5: 0.8869 (19809/22335)
# Recall@10: 0.9019 (20143/22335)
# Recall@20: 0.9123 (20377/22335)
# Recall@50: 0.9233 (20623/22335)
# Recall@100: 0.9311 (20797/22335)
# MRR@10: 0.8170
python evaluate_retriever.py \
  --retriever_input_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --retriever_prediction_file work/model/aio_02/retrieved_passages/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz \
  --passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --match_type nfkc \
  --output_file work/data/aio_02/reader/aio_02_dev.jsonl.gz
# Recall@1: 0.5500 (550/1000)
# Recall@2: 0.6720 (672/1000)
# Recall@5: 0.7750 (775/1000)
# Recall@10: 0.8220 (822/1000)
# Recall@20: 0.8500 (850/1000)
# Recall@50: 0.8740 (874/1000)
# Recall@100: 0.9010 (901/1000)
# MRR@10: 0.6495
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
  --model.predict_dataset_file data/aio_04_dev_unlabeled_v1.0.jsonl \
  --model.passage_faiss_index_file work/data/aio_02/passage_index/jawiki-20220404-c400-large.faiss \
  --model.passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/pipeline/aio_04_dev
python gather_json_predictions.py \
  --predictions_dir work/model/aio_02/pipeline/aio_04_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/pipeline/aio_04_dev/lightning_logs/version_0/prediction.jsonl.gz
```

### Building and Running the Docker Container for Inference

```sh
mkdir input output

docker build -t aio4-bpr-baseline .
docker run --rm -v $(realpath input):/input -v $(realpath output):/output aio4-bpr-baseline

echo '{"qid": "AIO04-0005", "position": 36, "question": "味がまずい魚のことを、猫でさえ見向きもしないということから俗に何という?"}' > input/example.json
cat output/example.json
# {"qid": "AIO04-0005", "position": 36, "prediction": "猫またぎ"}
```
