import dspy
# import milvus
import os
import numpy as np
import ollama
from tqdm import tqdm

from pymilvus import model, MilvusClient, IndexType, DataType
from dspy.retrieve.milvus_rm import MilvusRM

MILVUS_URI = './milvus.db'
MILVUS_TOKEN = ""
COLL_NAME = 'FeMail'

# MODEL = 'llama3:8b-instruct-q5_1'
EMB_MODEL = 'mxbai-embed-large'
EMB_DIM = 1024
GEN_MODEL = 'llama3.2:latest'

class RAG(dspy.Module):
    def __init__(self, retriever):
        self.respond = dspy.ChainOfThought('context, question -> response')
        self.retriever = retriever

    def forward(self, question):
        breakpoint()
        context = self.retriever(question)
        return self.respond(context=context, question=question)
    

def create_client(chunks):
    # Initialize Milvus client
    milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    if COLL_NAME not in milvus_client.list_collections():
        milvus_client.create_collection(
            collection_name=COLL_NAME,
            overwrite=True,
            dimension=EMB_DIM,
            primary_field_name="id",
            vector_field_name="embedding",
            id_type="int",
            metric_type="IP",
            max_length=65535,
            enable_dynamic=True,
        )
        
        for idx,chunk in enumerate(tqdm(chunks, desc="Loading embeddings in DB")):
            if len(chunk) == 0:
                continue
            milvus_client.insert(
                collection_name=COLL_NAME,
                data=[
                    {
                        "id": idx,
                        "embedding": my_embedder([chunk])[0],
                        "text": chunk,
                    }
                ],
            )
            # breakpoint()

    # Initialize the MilvusRM retriever
    milvus_retriever = MilvusRM(
        collection_name=COLL_NAME,
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
        embedding_function=my_embedder,
        k=5
    )

    return milvus_retriever

def my_embedder(texts):
    embeddings = []
    if type(texts) is not list:
        raise Exception(f"texts is of type {type(texts)} instead of list")
    
    for text in texts:
        response = ollama.embeddings(model=EMB_MODEL, prompt=text)
        embedding = response["embedding"]
        embeddings.append(embedding)
    return embeddings

def create_embedder(dimensions=EMB_DIM):
    embedder = dspy.Embedder(my_embedder, dimensions=dimensions)
    return embedder

def conn_LLM(model=GEN_MODEL, max_tokens = 4000):

    # Connect to Llama3 hosted with Ollama
    llm = dspy.OllamaLocal(
        model=model,
        max_tokens=max_tokens,
        timeout_s=480
    )

    # Test connection
    test_query = "What is the latest in AI?"
    test_response = llm(test_query)
    print("Test Llama3 response:", test_response)

    dspy.configure(lm=llm)
    return llm



# TODO
# Create a DSPy embedder with ollama, here:
# https://ollama.com/blog/embedding-models, https://milvus.io/docs/integrate_with_dspy.md