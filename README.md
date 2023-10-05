# AIO4 BPR Baseline

This is a baseline system for AIO4 competition utilizing [Binary Passage Retriever (BPR)](https://github.com/studio-ousia/bpr).

BPR is an efficient passage retrieval model for a large collection of documents.
BPR integrates a learning-to-hash technique into [Dense Passage Retriever (DPR)](https://github.com/facebookresearch/DPR) to represent the passage embeddings using compact binary codes rather than continuous vectors.
It substantially reduces the memory size without a loss of accuracy when tested on several QA datasets (see [the BPR repository](https://github.com/studio-ousia/bpr) for more detail).

## Installation

```sh
pip install -U pip setuptools
pip install -e '.[lightning,onnx]'
```

## Example Usage

**Note:** The following example is run on a server with 4 GPUs, each with 16 GB of memory.

### Preparing Datasets

**1. Download datasets**

```sh
mkdir data
wget https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/datasets.jawiki-20220404-c400-large.aio_02_train.jsonl.gz -P data
wget https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.0/datasets.jawiki-20220404-c400-large.aio_02_dev.jsonl.gz -P data
wget https://github.com/cl-tohoku/quiz-datasets/releases/download/v1.0.1/passages.jawiki-20220404-c400-large.jsonl.gz -P data
```

**2. Preprocess the datasets**

```sh
mkdir -p work/aio_02/data

python -m aio4_bpr_baseline.utils.convert_passages \
  --passages_file data/passages.jawiki-20220404-c400-large.jsonl.gz \
  --output_passages_file work/aio_02/data/passages.jsonl.gz \
  --output_pid_idx_map_file work/aio_02/data/pid_idx_map.json.gz

python -m aio4_bpr_baseline.utils.convert_dataset \
  --dataset_file data/datasets.jawiki-20220404-c400-large.aio_02_train.jsonl.gz \
  --pid_idx_map_file work/aio_02/data/pid_idx_map.json.gz \
  --output_dataset_file work/aio_02/data/retriever_train.jsonl.gz
python -m aio4_bpr_baseline.utils.convert_dataset \
  --dataset_file data/datasets.jawiki-20220404-c400-large.aio_02_dev.jsonl.gz \
  --pid_idx_map_file work/aio_02/data/pid_idx_map.json.gz \
  --output_dataset_file work/aio_02/data/retriever_dev.jsonl.gz
```

### Training and Evaluating Retriever

**1. Train a biencoder**

```sh
python -m aio4_bpr_baseline.lightning_cli fit \
  --config aio4_bpr_baseline/configs/retriever/bpr/biencoder.yaml \
  --model.train_dataset_file work/aio_02/data/retriever_train.jsonl.gz \
  --model.val_dataset_file work/aio_02/data/retriever_dev.jsonl.gz \
  --model.passages_file work/aio_02/data/passages.jsonl.gz \
  --trainer.default_root_dir work/aio_02/biencoder
```

**2. Build passage embeddings**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/retriever/bpr/embedder.yaml \
  --model.biencoder_ckpt_file work/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passages_file work/aio_02/data/passages.jsonl.gz \
  --trainer.default_root_dir work/aio_02/embedder
python -m aio4_bpr_baseline.utils.gather_numpy_predictions \
  --predictions_dir work/aio_02/embedder/lightning_logs/version_0/predictions \
  --output_file work/aio_02/embedder/lightning_logs/version_0/prediction.npy
```

**3. Retrieve passages for questions**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/retriever/bpr/retriever.yaml \
  --model.biencoder_ckpt_file work/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passage_embeddings_file work/aio_02/embedder/lightning_logs/version_0/prediction.npy \
  --model.predict_dataset_file work/aio_02/data/retriever_train.jsonl.gz \
  --trainer.default_root_dir work/aio_02/retriever/train
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/aio_02/retriever/train/lightning_logs/version_0/predictions \
  --output_file work/aio_02/retriever/train/lightning_logs/version_0/prediction.json.gz

python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/retriever/bpr/retriever.yaml \
  --model.biencoder_ckpt_file work/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passage_embeddings_file work/aio_02/embedder/lightning_logs/version_0/prediction.npy \
  --model.predict_dataset_file work/aio_02/data/retriever_dev.jsonl.gz \
  --trainer.default_root_dir work/aio_02/retriever/dev
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/aio_02/retriever/dev/lightning_logs/version_0/predictions \
  --output_file work/aio_02/retriever/dev/lightning_logs/version_0/prediction.json.gz
```

**4. Evaluate the retriever performance**

```sh
python -m aio4_bpr_baseline.utils.evaluate_retriever \
  --dataset_file work/aio_02/data/retriever_train.jsonl.gz \
  --passages_file work/aio_02/data/passages.jsonl.gz \
  --prediction_file work/aio_02/retriever/train/lightning_logs/version_0/prediction.json.gz \
  --answer_match_type nfkc \
  --output_file work/aio_02/data/reader_train.jsonl.gz
# Recall@1: 0.7597
# Recall@2: 0.8409
# Recall@5: 0.8869
# Recall@10: 0.9019
# Recall@20: 0.9123
# Recall@50: 0.9233
# Recall@100: 0.9311
# MRR@10: 0.8158
python -m aio4_bpr_baseline.utils.evaluate_retriever \
  --dataset_file work/aio_02/data/retriever_dev.jsonl.gz \
  --passages_file work/aio_02/data/passages.jsonl.gz \
  --prediction_file work/aio_02/retriever/dev/lightning_logs/version_0/prediction.json.gz \
  --answer_match_type nfkc \
  --output_file work/aio_02/data/reader_dev.jsonl.gz
# Recall@1: 0.5500
# Recall@2: 0.6720
# Recall@5: 0.7750
# Recall@10: 0.8220
# Recall@20: 0.8500
# Recall@50: 0.8740
# Recall@100: 0.9010
# MRR@10: 0.6463
```

### Training and Evaluating Reader

**1. Train a reader**

```sh
python -m aio4_bpr_baseline.lightning_cli fit \
  --config aio4_bpr_baseline/configs/reader/extractive_reader/reader.yaml \
  --model.train_dataset_file work/aio_02/data/reader_train.jsonl.gz \
  --model.val_dataset_file work/aio_02/data/reader_dev.jsonl.gz \
  --model.passages_file work/aio_02/data/passages.jsonl.gz \
  --trainer.default_root_dir work/aio_02/reader
```

**2. Predict answers for questions**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/reader/extractive_reader/reader_predict.yaml \
  --model.reader_ckpt_file work/aio_02/reader/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.predict_dataset_file work/aio_02/data/reader_dev.jsonl.gz \
  --model.passages_file work/aio_02/data/passages.jsonl.gz \
  --trainer.default_root_dir work/aio_02/reader_predict/aio_02_dev
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/aio_02/reader_predict/aio_02_dev/lightning_logs/version_0/predictions \
  --output_file work/aio_02/reader_predict/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz
```

**3. Evaluate the reader performance**

```sh
python -m aio4_bpr_baseline.utils.evaluate_reader \
  --dataset_file work/aio_02/data/reader_dev.jsonl.gz \
  --passages_file work/aio_02/data/passages.jsonl.gz \
  --prediction_file work/aio_02/reader_predict/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz \
  --answer_normalization_mode nfkc
# Exact Match: 0.5680
```

### Running Pipeline of Retriever and Reader

**1. Predict answers for the questions in AIO4 development data**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/pipeline_aio4/bpr_extractive_reader/pipeline.yaml \
  --model.biencoder_ckpt_file work/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.reader_ckpt_file work/aio_02/reader/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passage_embeddings_file work/aio_02/embedder/lightning_logs/version_0/prediction.npy \
  --model.passages_file work/aio_02/data/passages.jsonl.gz \
  --model.predict_dataset_file data/aio_04_dev_unlabeled_v1.0.jsonl \
  --model.predict_num_passages 10 \
  --model.predict_answer_score_threshold 0.5 \
  --trainer.default_root_dir work/aio_02/pipeline_aio4/aio_04_dev
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/aio_02/pipeline_aio4/aio_04_dev/lightning_logs/version_0/predictions \
  --output_file work/aio_02/pipeline_aio4/aio_04_dev/lightning_logs/version_0/prediction.jsonl
```

**2. Compute the scores**

```sh
python -m compute_score \
  --prediction_file work/aio_02/pipeline_aio4/aio_04_dev/lightning_logs/version_0/prediction.jsonl \
  --gold_file data/aio_04_dev_v1.0.jsonl \
  --limit_num_wrong_answers 3
# num_questions: 500
# num_correct: 288
# num_missed: 196
# num_failed: 16
# accuracy: 57.6%
# accuracy_score: 288.000
# position_score: 76.380
# total_score: 364.380
```

**3. Predict answers for the questions in AIO4 leaderboard test data**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/pipeline_aio4/bpr_extractive_reader/pipeline.yaml \
  --model.biencoder_ckpt_file work/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.reader_ckpt_file work/aio_02/reader/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passage_embeddings_file work/aio_02/embedder/lightning_logs/version_0/prediction.npy \
  --model.passages_file work/aio_02/data/passages.jsonl.gz \
  --model.predict_dataset_file data/aio_04_test_lb_unlabeled_v1.0.jsonl \
  --model.predict_num_passages 10 \
  --model.predict_answer_score_threshold 0.5 \
  --trainer.default_root_dir work/aio_02/pipeline_aio4/aio_04_test_lb
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/aio_02/pipeline_aio4/aio_04_test_lb/lightning_logs/version_0/predictions \
  --output_file work/aio_02/pipeline_aio4/aio_04_test_lb/lightning_logs/version_0/prediction.jsonl
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This
work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative
Commons Attribution-NonCommercial 4.0 International License</a>.

## Citation

If you find this work useful, please cite the following paper:

[Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2106.00882)

```
@inproceedings{yamada2021bpr,
  title={Efficient Passage Retrieval with Hashing for Open-domain Question Answering},
  author={Ikuya Yamada and Akari Asai and Hannaneh Hajishirzi},
  booktitle={ACL},
  year={2021}
}
```
