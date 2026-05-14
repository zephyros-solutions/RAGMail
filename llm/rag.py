import dspy

# ============================================================================
# DSPy Module for LLM Generation with Retrieval-Augmentation
# ============================================================================
class GenericAnswer(dspy.Signature):
    """Answer questions with context."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Explain the main facts, the main actors, and their motivations and points of view. Be concise but informative.")

class RAG(dspy.Module):
    """
    Retrieval-Augmented Generation pipeline for email analysis.
    
    Supports multiple analysis modes:
    - Perspective: Analyze one person's view
    - Conflict: Analyze conflict between two people
    - Comparison: Compare perspectives
    - Generic: Use generic ChainOfThought for other queries
    """
    
    def __init__(self, retriever, context=None):
        """
        Initialize RAG system.
        
        Args:
            retriever: Function to retrieve relevant context (or None if context provided)
            context: Pre-built context string (if provided, retriever is ignored)
        """
        super().__init__()
        
        # Specialized analyzers using DSPy signatures
        self.perspective_analyzer = dspy.ChainOfThought('person, topic, context -> analysis')
        self.conflict_analyzer = dspy.ChainOfThought('person_a, person_b, context -> root_cause, perspective_a, perspective_b, common_ground, possible_resolution')
        self.comparison_module = dspy.ChainOfThought('person_a, person_b, topic, context -> position_a, position_b, convergences, divergences')
        
        # Generic fallback
        self.generic_respond = dspy.ChainOfThought(GenericAnswer)
        
        self.context = context
        self.retriever = retriever

    def forward(self, question, mode="generic", **kwargs):
        """
        Generate response based on query and selected mode.
        
        Args:
            question: User query
            mode: One of ["generic", "perspective", "conflict", "comparison"]
            **kwargs: Additional arguments for specific modes
                - For "perspective": person, topic
                - For "conflict": person_a, person_b
                - For "comparison": person_a, person_b, topic
        
        Returns:
            Response from the appropriate analyzer
        """
        
        # Retrieve context if not provided
        if self.context is None:
            context = self.retriever(question)
        else:
            context = self.context
        
        breakpoint()
        # Route to appropriate analyzer
        if mode == "perspective":
            return self.perspective_analyzer(
                person=kwargs.get("person", ""),
                topic=kwargs.get("topic", ""),
                context=context
            )
        
        elif mode == "conflict":
            return self.conflict_analyzer(
                person_a=kwargs.get("person_a", ""),
                person_b=kwargs.get("person_b", ""),
                context=context
            )
        
        elif mode == "comparison":
            return self.comparison_module(
                person_a=kwargs.get("person_a", ""),
                person_b=kwargs.get("person_b", ""),
                topic=kwargs.get("topic", ""),
                context=context
            )
        
        else:  # Generic mode
            return self.generic_respond(context=context, question=question)
    


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


