import torch
from fastapi import FastAPI

from models.pipeline import BPRPipeline


ANSWER_SCORE_THRESHOLD = 0.1


app = FastAPI()

print("Loading BPRPipeline")

pipeline = BPRPipeline(
    retriever_ckpt_file="/work/retriever.ckpt",
    reader_ckpt_file="/work/reader.ckpt",
    passage_faiss_index_file="/work/passages.faiss",
    passage_dataset_file="/work/passages.json.gz",
)

if torch.cuda.is_available():
    pipeline.to("cuda")
    print("Loaded BPRPipeline to GPU")
else:
    print("Loaded BPRPipeline to CPU")


@app.get("/answer")
def answer(q: str):
    with torch.inference_mode():
        pipeline.eval()
        prediction = pipeline.predict_answers([q])[0]

    answer = prediction["answers"][0]
    score = prediction["scores"][0]

    if score > ANSWER_SCORE_THRESHOLD:
        return {"answer": answer}
    else:
        return {"answer": None}
