import argparse
from pathlib import Path

import torch

from aio4_bpr_baseline.models.retriever.bpr.lightning_modules import BiEncoderLightningModule


def export_question_encoder_to_onnx(biencoder_ckpt_file: str, output_file: str, verbose: bool = False):
    biencoder_module = BiEncoderLightningModule.load_from_checkpoint(
        biencoder_ckpt_file, map_location="cpu", strict=False
    )
    biencoder_module.freeze()

    question_tokenizer = biencoder_module.question_tokenizer
    question_encoder = biencoder_module.question_encoder

    tokenized_questions = question_tokenizer(["これは質問です。"])
    input_names = list(tokenized_questions)
    output_names = ["encoded_questions"]

    dynamic_axes = {name: {0: "question", 1: "sequence"} for name in tokenized_questions}
    dynamic_axes["encoded_questions"] = {0: "question"}

    torch.onnx.export(
        model=question_encoder,
        args=(dict(tokenized_questions), {}),
        f=output_file,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )


def export_passage_encoder_to_onnx(biencoder_ckpt_file: str, output_file: str, verbose: bool = False):
    biencoder_module = BiEncoderLightningModule.load_from_checkpoint(
        biencoder_ckpt_file, map_location="cpu", strict=False
    )
    biencoder_module.freeze()

    passage_tokenizer = biencoder_module.passage_tokenizer
    passage_encoder = biencoder_module.passage_encoder

    tokenized_passages = passage_tokenizer(["これはタイトルです。"], ["これは本文です。"])
    input_names = list(tokenized_passages)
    output_names = ["encoded_passages"]

    dynamic_axes = {name: {0: "passage", 1: "sequence"} for name in tokenized_passages}
    dynamic_axes["encoded_passages"] = {0: "passage"}

    torch.onnx.export(
        model=passage_encoder,
        args=(dict(tokenized_passages), {}),
        f=output_file,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biencoder_ckpt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    question_encoder_output_file = Path(args.output_dir) / "question_encoder.onnx"
    export_question_encoder_to_onnx(args.biencoder_ckpt_file, question_encoder_output_file, verbose=args.verbose)

    passage_encoder_output_file = Path(args.output_dir) / "passage_encoder.onnx"
    export_passage_encoder_to_onnx(args.biencoder_ckpt_file, passage_encoder_output_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
