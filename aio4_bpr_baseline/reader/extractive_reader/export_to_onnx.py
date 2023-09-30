import argparse
from pathlib import Path

import torch

from aio4_bpr_baseline.models.reader.extractive_reader.lightning_modules import ReaderLightningModule


def export_reader_to_onnx(reader_ckpt_file: str, output_file: str, verbose: bool = False):
    reader_module = ReaderLightningModule.load_from_checkpoint(reader_ckpt_file, map_location="cpu", strict=False)
    reader_module.freeze()

    tokenizer = reader_module.tokenizer
    model = reader_module.reader

    tokenized_inputs, _, _ = tokenizer(["これは質問です。"], [["これはタイトルです。"]], [["これは本文です。"]])
    input_names = list(tokenized_inputs)
    output_names = ["classifier_logits", "start_logits", "end_logits"]

    dynamic_axes = {name: {0: "question", 1: "passage", 2: "sequence"} for name in tokenized_inputs}
    dynamic_axes["classifier_logits"] = {0: "question", 1: "passage"}
    dynamic_axes["start_logits"] = {0: "question", 1: "passage", 2: "sequence"}
    dynamic_axes["end_logits"] = {0: "question", 1: "passage", 2: "sequence"}

    torch.onnx.export(
        model=model,
        args=(dict(tokenized_inputs), {}),
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
    parser.add_argument("--reader_ckpt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    reader_output_file = Path(args.output_dir) / "reader.onnx"
    export_reader_to_onnx(args.reader_ckpt_file, reader_output_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
