import argparse

import faiss
import numpy as np
from tqdm import trange


def build_index_from_file(embeddings_file: str, output_file: str, batch_size: int = 1000):
    embeddings = np.load(embeddings_file)
    num_passages, binary_embedding_size = embeddings.shape

    faiss_index = faiss.IndexBinaryFlat(binary_embedding_size * 8)
    for start in trange(0, num_passages, batch_size):
        faiss_index.add(embeddings[start : start + batch_size])

    faiss.write_index_binary(faiss_index, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    build_index_from_file(args.embeddings_file, args.output_file, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
