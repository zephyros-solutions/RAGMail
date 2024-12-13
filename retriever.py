from tqdm import tqdm
import dspy
import ollama

from pymilvus import MilvusClient, DataType, Function, FunctionType, WeightedRanker, RRFRanker, AnnSearchRequest

from globals import MILVUS_URI, MILVUS_TOKEN
from globals import TEXT_FIELD_NAME
from globals import SPARSE_FIELD_NAME, SPARSE_INDEX_NAME, SPARSE_INDEX_TYPE, SPARSE_METRIC_TYPE, SPARSE_INDEX_PARAMS
from globals import DENSE_FIELD_NAME, DENSE_INDEX_NAME, DENSE_INDEX_TYPE, DENSE_METRIC_TYPE, DENSE_INDEX_PARAMS
from globals import DENSE_EMB_MODEL

from typing import List, Union, Optional

def my_embedder(texts:list[str]) -> list[float]:
    embeddings = []
    if type(texts) is not list:
        raise Exception(f"texts is of type {type(texts)} instead of list")
    
    for text in texts:
        response = ollama.embeddings(model=DENSE_EMB_MODEL, prompt=text)
        embedding = response["embedding"]
        embeddings.append(embedding)
    return embeddings

def get_emb_size():
    sz = len(my_embedder(['Ciao'])[0])
    print(f"Embeddings are of size {sz}")
    return sz


class RMClient(dspy.Retrieve):

    def __init__(self, collection_name: str, k:int, dim_dense_emb:int, max_length: int, dense_embedding_function, 
                 sparse_embedding_function, rerank_function, use_contextualize_embedding ):
        super().__init__(k=k)

        self.client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        self.collection_name = collection_name

        self.dense_embedding_function = dense_embedding_function
        self.dim_dense_emb = dim_dense_emb
        self.sparse_embedding_function = sparse_embedding_function
        self.rerank_function = rerank_function
        self.use_contextualize_embedding = use_contextualize_embedding
        self.max_length = max_length

    def build_collection(self, enable_dynamic_field):

        if self.collection_name not in self.client.list_collections():
            # Create schema
            schema = MilvusClient.create_schema(
                enable_dynamic_field=enable_dynamic_field,
            )

            # Add fields to schema
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

            schema.add_field(field_name=DENSE_FIELD_NAME, datatype=DataType.FLOAT_VECTOR, dim=self.dim_dense_emb)
            
            if self.sparse_embedding_function:
                schema.add_field(field_name=SPARSE_FIELD_NAME, datatype=DataType.SPARSE_FLOAT_VECTOR)

            index_params = self.client.prepare_index_params()

            index_params.add_index(
                field_name=DENSE_FIELD_NAME,
                index_name=DENSE_INDEX_NAME,
                index_type=DENSE_INDEX_TYPE,
                metric_type=DENSE_METRIC_TYPE,
                params=DENSE_INDEX_PARAMS,
            )

            if self.sparse_embedding_function:
                index_params.add_index(
                    field_name=SPARSE_FIELD_NAME,
                    index_name=SPARSE_INDEX_NAME,
                    index_type=SPARSE_INDEX_TYPE,
                    metric_type=SPARSE_METRIC_TYPE,
                    params=SPARSE_INDEX_PARAMS,
                )

            # Add VARCHAR field
            schema.add_field(
                field_name=TEXT_FIELD_NAME, 
                datatype=DataType.VARCHAR, 
                max_length=self.max_length, 
            )

            self.client.create_collection(
                collection_name=self.collection_name, 
                schema=schema, 
                index_params=index_params
            )
            return True
        
        return False



    def auto_sparse(self, schema, index_params):
            '''
                For the case Milvus does the sparse embedding internally
                https://milvus.io/docs/full-text-search.md
            '''
            # Specifies the language for the stemming process.
            # Supported languages include: "arabic", "danish", "dutch", "english", "finnish", "french", 
            # "german", "greek", "hungarian", "italian", "norwegian", "portuguese", "romanian", "russian", 
            # "spanish", "swedish", "tamil", "turkish"
            analyzer_params = {
                "tokenizer": "standard",
                "filter":[{
                    "type": "stemmer", # Specifies the filter type as stemmer
                    "language": "italian", # Sets the language for stemming
                }],
            }

            # Add VARCHAR field
            schema.add_field(
                field_name=TEXT_FIELD_NAME, 
                datatype=DataType.VARCHAR, 
                max_length=self.max_length, 
                enable_analyzer=True,
                analyzer_params=analyzer_params,
                enable_match=True,
            )

            index_params.add_index(
                field_name=SPARSE_FIELD_NAME,
                index_name=SPARSE_INDEX_NAME,
                index_type="AUTOINDEX", 
                metric_type="BM25"
            )


            bm25_function = Function(
                name="text_bm25_emb", # Function name
                input_field_names=[TEXT_FIELD_NAME], # Name of the VARCHAR field containing raw text data
                output_field_names=[SPARSE_FIELD_NAME], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)



    def upload_embeddings(self, chunks:list[str], metadata) -> None:
        for idx,chunk in enumerate(tqdm(chunks, desc="Loading embeddings in DB")):
            if len(chunk) == 0:
                continue
            
            # breakpoint()
            if self.sparse_embedding_function:
                sparse_result = self.sparse_embedding_function.encode_documents([chunk])
                if type(sparse_result) == dict:
                    sparse_vec = sparse_result["sparse"][[0]]
                else:
                    sparse_vec = sparse_result[[0]]
                
                self.client.insert(
                    collection_name=self.collection_name,
                    data=[
                        {
                            "id": idx,
                            DENSE_FIELD_NAME: self.dense_embedding_function([chunk])[0],
                            SPARSE_FIELD_NAME: sparse_vec,
                            TEXT_FIELD_NAME: chunk,
                            **metadata,
                        }
                    ],
                )
            else:
                self.client.insert(
                    collection_name=self.collection_name,
                    data=[
                        {
                            "id": idx,
                            DENSE_FIELD_NAME: self.dense_embedding_function([chunk])[0],
                            TEXT_FIELD_NAME: chunk,
                            **metadata,
                        }
                    ],
                )


            # breakpoint()

    def forward(self, question:str, k:Optional[int]=None) -> dspy.Prediction:

        k = k if k else self.k
        if self.sparse_embedding_function:
            sparse_search_params = {
                'data' : self.sparse_embedding_function.encode_documents([question])['sparse'],
                'anns_field' : SPARSE_FIELD_NAME,
                'limit' : k,
                'param': {'drop_ratio_search': 0.2},
            }

            sparse_res = self.client.search(
                collection_name=self.collection_name,
                data=self.sparse_embedding_function.encode_documents([question])['sparse'],
                anns_field=SPARSE_FIELD_NAME,
                limit=k,
                search_params=sparse_search_params['param'],
                output_fields=[
                        TEXT_FIELD_NAME
                    ],
            )
            for hits in sparse_res:
                print("***** TopK results sparse search:")
                for hit in hits:
                    print('** ' + hit['entity'][f'{TEXT_FIELD_NAME}'])
            
            sparse_req = AnnSearchRequest(**sparse_search_params)
        
        # breakpoint()
        dense_search_params = {
            'data' : self.dense_embedding_function([question]),
            'anns_field' : DENSE_FIELD_NAME,
            'limit' : k,
            'param': {},
        }

        dense_res = self.client.search(
            collection_name=self.collection_name,
            data=self.dense_embedding_function([question]),
            anns_field=DENSE_FIELD_NAME,
            limit=k,
            search_params=dense_search_params['param'],
            output_fields=[
                    TEXT_FIELD_NAME
                ],
        )
        for hits in dense_res:
            print("***** TopK results dense search:")
            for hit in hits:
                print('** ' + hit['entity'][f'{TEXT_FIELD_NAME}'])
        
        dense_req = AnnSearchRequest(**dense_search_params)

        if self.sparse_embedding_function and self.rerank_function:
            
            res = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req,sparse_req],
                ranker=self.rerank_function,
                limit=k,
                output_fields=[
                        TEXT_FIELD_NAME
                    ],
            )
            for hits in res:
                print("***** TopK results in hybrid search:")
                for hit in hits:
                    print('** ' + hit['entity'][f'{TEXT_FIELD_NAME}'])

            # breakpoint()
            return dspy.Prediction(
                passages=res
            )
        else:
            return dspy.Prediction(
                passages=dense_res
            )





