import dspy
# import milvus
import os
import numpy as np
import ollama
from tqdm import tqdm

from pymilvus import model, MilvusClient, IndexType, DataType
from dspy.retrieve.milvus_rm import MilvusRM

from globals import MILVUS_URI, MILVUS_TOKEN, EMB_MODEL, API_BASE, API_KEY


class RAG(dspy.Module):
    def __init__(self, retriever):
        self.respond = dspy.ChainOfThought('context, question -> response')
        self.retriever = retriever

    def forward(self, question):
        # breakpoint()
        context = self.retriever(question)
        return self.respond(context=context, question=question)
    


def create_collection(collection_name:str, dimension:int, max_length:int) -> MilvusClient:
    # Initialize Milvus client
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    if collection_name not in milvus_client.list_collections():
        milvus_client.create_collection(
            collection_name=collection_name,
            overwrite=True,
            dimension=dimension,
            primary_field_name="id",
            vector_field_name="embedding",
            id_type="int",
            metric_type="IP",
            max_length=max_length,
            enable_dynamic=True,
        )
        return milvus_client
    else:
        return None

def my_embedder(texts:list[str]) -> list[float]:
    embeddings = []
    if type(texts) is not list:
        raise Exception(f"texts is of type {type(texts)} instead of list")
    
    for text in texts:
        response = ollama.embeddings(model=EMB_MODEL, prompt=text)
        embedding = response["embedding"]
        embeddings.append(embedding)
    return embeddings

def upload_embeddings(client, chunks:list[str], collection_name:str) -> None:
    for idx,chunk in enumerate(tqdm(chunks, desc="Loading embeddings in DB")):
        if len(chunk) == 0:
            continue
        client.insert(
            collection_name=collection_name,
            data=[
                {
                    "id": idx,
                    "embedding": my_embedder([chunk])[0],
                    "text": chunk,
                }
            ],
        )
        # breakpoint()

def get_retriever(collection_name:str, k:int) -> MilvusRM:
    # Initialize the MilvusRM retriever
    milvus_retriever = MilvusRM(
        collection_name=collection_name,
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
        embedding_function=my_embedder,
        k=k
    )

    return milvus_retriever


def create_embedder(dimensions:int) -> dspy.Embedder:
    embedder = dspy.Embedder(my_embedder, dimensions=dimensions)
    return embedder

def conn_LLM(model, max_tokens):

    # # Connect to Llama3 hosted with Ollama
    # lm = dspy.OllamaLocal(
    #     model=model,
    #     max_tokens=max_tokens,
    #     timeout_s=480
    # )
    lm = dspy.LM(model, api_base=API_BASE, api_key=API_KEY)
    dspy.configure(lm=lm)
    
    # # Test connection
    # test_query = "What is the latest in AI?"
    # test_response = lm(test_query)
    # print(f"Test {model} response:", test_response)

    return lm

