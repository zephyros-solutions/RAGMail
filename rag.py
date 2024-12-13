import dspy

class RAG(dspy.Module):
    def __init__(self, retriever):
        self.respond = dspy.ChainOfThought('context, question -> response')
        self.retriever = retriever

    def forward(self, question):
        # breakpoint()
        context = self.retriever(question)
        return self.respond(context=context, question=question)
    


# def create_collection(collection_name:str, dimension:int, metric_type:str, max_length:int, enable_dynamic:bool) -> MilvusClient:
#     # Initialize Milvus client
#     milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
#     if collection_name not in milvus_client.list_collections():
#         milvus_client.create_collection(
#             collection_name=collection_name,
#             overwrite=True,
#             dimension=dimension,
#             primary_field_name="id",
#             vector_field_name="embedding",
#             id_type="int",
#             metric_type=metric_type,
#             max_length=max_length,
#             enable_dynamic=enable_dynamic,
#         )
#         return milvus_client
#     else:
#         return None


# def upload_embeddings(client, chunks:list[str], collection_name:str) -> None:
#     for idx,chunk in enumerate(tqdm(chunks, desc="Loading embeddings in DB")):
#         if len(chunk) == 0:
#             continue
#         client.insert(
#             collection_name=collection_name,
#             data=[
#                 {
#                     "id": idx,
#                     "embedding": my_embedder([chunk])[0],
#                     "text": chunk,
#                 }
#             ],
#         )
#         # breakpoint()

# def get_retriever(collection_name:str, k:int) -> MilvusRM:
#     # Initialize the MilvusRM retriever
#     milvus_retriever = MilvusRM(
#         collection_name=collection_name,
#         uri=MILVUS_URI,
#         token=MILVUS_TOKEN,
#         embedding_function=my_embedder,
#         k=k
#     )

#     return milvus_retriever


# def create_embedder(dimensions:int) -> dspy.Embedder:
#     # https://github.com/stanfordnlp/dspy/blob/6178c28ce96b2ecb8a21c722ff06cac58b0bb83c/dspy/clients/embedding.py#L5
#     embedder = dspy.Embedder(my_embedder, dimensions=dimensions)
#     return embedder


