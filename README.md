# AIO4 BPR Baseline

This is a baseline system for AIO4 competition utilizing [Binary Passage Retriever (BPR)](https://github.com/studio-ousia/bpr).

BPR is an efficient passage retrieval model for a large collection of documents.
BPR integrates a learning-to-hash technique into [Dense Passage Retriever (DPR)](https://github.com/facebookresearch/DPR) to represent the passage embeddings using compact binary codes rather than continuous vectors.
It substantially reduces the memory size without a loss of accuracy when tested on several QA datasets (see [the BPR repository](https://github.com/studio-ousia/bpr) for more detail).

## Installation

```sh
pip install -U pip setuptools wheel
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
mkdir -p work/data/aio_02/retriever
python -m convert_retriever_dataset \
  --input_file data/datasets.jawiki-20220404-c400-large.aio_02_train.jsonl.gz \
  --output_file work/data/aio_02/retriever/aio_02_train.jsonl.gz
python -m convert_retriever_dataset \
  --input_file data/datasets.jawiki-20220404-c400-large.aio_02_dev.jsonl.gz \
  --output_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz

mkdir -p work/data/aio_02/passages
python -m convert_passage_dataset \
  --input_file data/passages.jawiki-20220404-c400-large.jsonl.gz \
  --output_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz
```

### Training and Evaluating Retriever

**1. Train a biencoder**

```sh
python -m aio4_bpr_baseline.lightning_cli fit \
  --config aio4_bpr_baseline/configs/retriever/bpr/biencoder.yaml \
  --model.train_dataset_file work/data/aio_02/retriever/aio_02_train.jsonl.gz \
  --model.val_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/biencoder
```

**2. Build passage embeddings**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/retriever/bpr/embedder.yaml \
  --model.biencoder_ckpt_file work/model/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.predict_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/embedder
python -m aio4_bpr_baseline.utils.gather_numpy_predictions \
  --predictions_dir work/model/aio_02/embedder/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/embedder/lightning_logs/version_0/prediction.npy
python -m aio4_bpr_baseline.models.retriever.bpr.build_faiss_index \
  --embeddings_file work/model/aio_02/embedder/lightning_logs/version_0/prediction.npy \
  --output_file work/model/aio_02/embedder/lightning_logs/version_0/embedings.faiss
```

**3. Retrieve passages for questions**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/retriever/bpr/retriever.yaml \
  --model.biencoder_ckpt_file work/model/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passage_faiss_index_file work/model/aio_02/embedder/lightning_logs/version_0/embedings.faiss \
  --model.predict_dataset_file work/data/aio_02/retriever/aio_02_train.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/retriever/aio_02_train
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/model/aio_02/retriever/aio_02_train/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/retriever/aio_02_train/lightning_logs/version_0/prediction.json.gz

python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/retriever/bpr/retriever.yaml \
  --model.biencoder_ckpt_file work/model/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.passage_faiss_index_file work/model/aio_02/embedder/lightning_logs/version_0/embedings.faiss \
  --model.predict_dataset_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/retriever/aio_02_dev
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/model/aio_02/retriever/aio_02_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/retriever/aio_02_dev/lightning_logs/version_0/prediction.json.gz
```

**4. Evaluate the retriever performance**

```sh
mkdir -p work/data/aio_02/reader
python -m aio4_bpr_baseline.utils.evaluate_retriever \
  --input_file work/data/aio_02/retriever/aio_02_train.jsonl.gz \
  --prediction_file work/model/aio_02/retriever/aio_02_train/lightning_logs/version_0/prediction.json.gz \
  --passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --answer_match_type nfkc \
  --output_file work/data/aio_02/reader/aio_02_train.jsonl.gz
# Recall@1: 0.7597
# Recall@2: 0.8409
# Recall@5: 0.8869
# Recall@10: 0.9019
# Recall@20: 0.9123
# Recall@50: 0.9233
# Recall@100: 0.9311
# MRR@10: 0.8158
python -m aio4_bpr_baseline.utils.evaluate_retriever \
  --input_file work/data/aio_02/retriever/aio_02_dev.jsonl.gz \
  --prediction_file work/model/aio_02/retriever/aio_02_dev/lightning_logs/version_0/prediction.json.gz \
  --passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --answer_match_type nfkc \
  --output_file work/data/aio_02/reader/aio_02_dev.jsonl.gz
# Recall@1: 0.5500
# Recall@2: 0.6720
# Recall@5: 0.7750
# Recall@10: 0.8220
# Recall@20: 0.8500
# Recall@50: 0.8740
# Recall@100: 0.9010
# MRR@10: 0.6463
```

**5. Export the biencoder to ONNX**

```sh
mkdir work/model/aio_02/biencoder/lightning_logs/version_0/onnx
python -m aio4_bpr_baseline.models.retriever.bpr.export_to_onnx \
  --biencoder_ckpt_file work/model/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --output_dir work/model/aio_02/biencoder/lightning_logs/version_0/onnx
```

### Training and Evaluating Reader

**1. Train a reader**

```sh
python -m aio4_bpr_baseline.lightning_cli fit \
  --config aio4_bpr_baseline/configs/reader/extractive_reader/reader.yaml \
  --model.train_dataset_file work/data/aio_02/reader/aio_02_train.jsonl.gz \
  --model.val_dataset_file work/data/aio_02/reader/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/reader
```

**2. Predict answers for questions**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/reader/extractive_reader/reader_prediction.yaml \
  --model.reader_ckpt_file work/model/aio_02/reader/lightning_logs/version_0/checkpoints/best.ckpt \
  --model.predict_dataset_file work/data/aio_02/reader/aio_02_dev.jsonl.gz \
  --trainer.default_root_dir work/model/aio_02/reader_prediction/aio_02_dev
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/model/aio_02/reader_prediction/aio_02_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/reader_prediction/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz
```

**3. Evaluate the reader performance**

```sh
python -m aio4_bpr_baseline.utils.evaluate_reader \
  --input_file work/data/aio_02/reader/aio_02_dev.jsonl.gz \
  --prediction_file work/model/aio_02/reader_prediction/aio_02_dev/lightning_logs/version_0/prediction.jsonl.gz \
  --answer_normalization_mode nfkc
# Exact Match: 0.5770 (577/1000)
```

**4. Export the reader to ONNX**

```sh
mkdir work/model/aio_02/reader/lightning_logs/version_0/onnx
python -m aio4_bpr_baseline.models.reader.extractive_reader.export_to_onnx \
  --reader_ckpt_file work/model/aio_02/reader/lightning_logs/version_0/checkpoints/best.ckpt \
  --output_dir work/model/aio_02/reader/lightning_logs/version_0/onnx
```

### Running Pipeline of Retriever and Reader

**1. Predict answers for the questions in AIO4 development data**

```sh
python -m aio4_bpr_baseline.lightning_cli predict \
  --config aio4_bpr_baseline/configs/pipeline_aio4/bpr/pipeline.yaml \
  --model.biencoder_ckpt_file work/model/aio_02/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
  --model.reader_ckpt_file work/model/aio_02/reader/lightning_logs/version_0/checkpoints/best.ckpt \
  --model.passage_faiss_index_file work/model/aio_02/embedder/lightning_logs/version_0/embedings.faiss \
  --model.passage_dataset_file work/data/aio_02/passages/jawiki-20220404-c400-large.jsonl.gz \
  --model.predict_dataset_file data/aio_04_dev_unlabeled_v1.0.jsonl \
  --model.predict_retriever_k 10 \
  --model.predict_answer_score_threshold 0.3 \
  --trainer.default_root_dir work/model/aio_02/pipeline_aio4/aio_04_dev
python -m aio4_bpr_baseline.utils.gather_json_predictions \
  --predictions_dir work/model/aio_02/pipeline_aio4/aio_04_dev/lightning_logs/version_0/predictions \
  --output_file work/model/aio_02/pipeline_aio4/aio_04_dev/lightning_logs/version_0/prediction.jsonl
```

**2. Compute the scores**

```sh
python -m compute_score \
  --prediction_file work/model/aio_02/pipeline_aio4/aio_04_dev/lightning_logs/version_0/prediction.jsonl \
  --gold_file data/aio_04_dev_v1.0.jsonl \
  --limit_num_wrong_answers 3
# num_questions: 500
# num_correct: 273
# num_missed: 166
# num_failed: 61
# accuracy: 54.6%
# accuracy_score: 273.000
# position_score: 91.272
# total_score: 364.272
```

### Prepare a Docker image for the Final Submission

```sh
mkdir input output

docker build -t aio4-bpr-baseline .
docker run --rm -v $(realpath input):/input -v $(realpath output):/output aio4-bpr-baseline
```

```sh
echo '{"qid": "AIO04-0005", "position": 36, "question": "味がまずい魚のことを、猫でさえ見向きもしないということから俗に何という?"}' > input/example.json
cat output/example.json
# {"qid": "AIO04-0005", "position": 36, "prediction": "猫またぎ"}
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
