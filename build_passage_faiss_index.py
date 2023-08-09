import argparse

import faiss
import numpy as np
from tqdm import trange


def main(args: argparse.Namespace):
    embeddings = np.load(args.embedder_prediction_file)
    num_passages, binary_embedding_size = embeddings.shape

    faiss_index = faiss.IndexBinaryFlat(binary_embedding_size * 8)
    for start in trange(0, num_passages, args.batch_size):
        faiss_index.add(embeddings[start : start + args.batch_size])

    faiss.write_index_binary(faiss_index, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedder_prediction_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)

    args = parser.parse_args()
    main(args)
