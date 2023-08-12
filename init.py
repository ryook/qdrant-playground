import time
from typing import TypedDict

import numpy as np
import requests
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PointStruct,
    VectorParams,
)


class Point(TypedDict):
    id_: int
    sentence: str
    embedding: list[float]
    label: int


class Qdrant:
    client = QdrantClient(host="localhost", port=6333)

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=2,
                lowercase=True,
            ),
        )

    def upsert(self, points: list[Point]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point["id_"],
                    vector=point["embedding"],
                    payload={"text": point["sentence"], "label": point["label"]},
                )
                for point in points
            ],
        )


def download_dataset(split="test"):
    dataset = load_dataset("mteb/amazon_reviews_multi", "ja", split=split)
    return dataset


def encode(sentences: list[str]) -> list[list[float]]:
    res = requests.post("http://localhost:8080/encode", json={"sentences": sentences})

    if res.status_code != 200:
        res.raise_for_status()

    res_body = res.json()
    embeddings = res_body["embeddings"]

    if len(sentences) != len(embeddings):
        raise Exception("sentencesとEmbeddingAPIのレスポンスの数が一致しません。")

    return embeddings


def convert(dataset) -> list[Point]:
    """
    datasetをPointに必要な情報に変換
    """
    df = dataset.to_pandas()
    sentences = df["text"]
    labels = df["label"]
    label_sentences = [(label, sentence) for label, sentence in zip(labels, sentences)]
    splited_label_sentences = np.array_split(
        label_sentences, len(label_sentences) // 100
    )

    rtn = []
    id_ = 0
    for label_sentence in splited_label_sentences:
        labels = [l[0] for l in label_sentence]
        sentences = [l[1] for l in label_sentence]
        embeddings = encode(sentences)

        for label, sentence, embedding in zip(labels, sentences, embeddings):
            id_ += 1
            rtn.append(
                Point(
                    id_=id_,
                    label=label,
                    sentence=sentence,
                    embedding=embedding,
                )
            )
    return rtn


if __name__ == "__main__":
    print("downloading dataset...")
    dataset = download_dataset()

    print("encoding dataset...")
    converted_dataset = convert(dataset)

    client = Qdrant("amazon_review")

    print("start upserting to Qdrant.")
    tm = time.time()
    for split_points in np.array_split(
        converted_dataset, len(converted_dataset) // 100
    ):
        client.upsert(split_points)
    exec_tm = time.time() - tm
    print(f"finish upserting to Qdrant. exec_tm: {exec_tm}")
