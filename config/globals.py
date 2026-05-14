from pymilvus import WeightedRanker, RRFRanker

DUMP_DIR = "./dumps"

######  CHUNKING
MAX_CHUNK_LEN = 1000
MAX_CHUNK_EXCESS = 2
TOK2CHAR = 2.2 # Approximate conversion from tokens to characters

######  MILVUS
MILVUS_URI = './milvus.db'
MILVUS_TOKEN = ""

MILVUS_MAX_LENGTH = 65535
# MILVUS_MAX_LENGTH = MAX_CHUNK_LEN*MAX_CHUNK_EXCESS
MILVUS_DYN = True
MILVUS_LEN_CTX = 3
TEXT_FIELD_NAME = 'text'

DENSE_FIELD_NAME = 'dense'
DENSE_INDEX_NAME = 'dense_index'
DENSE_INDEX_TYPE = 'FLAT' # IVF_FLAT AUTOINDEX
DENSE_INDEX_PARAMS = {} # {"nlist": 128}
DENSE_METRIC_TYPE = 'IP' # COSINE

# https://milvus.io/api-reference/pymilvus/v2.4.x/EmbeddingModels/BGEM3EmbeddingFunction/BGEM3EmbeddingFunction.md
# NOTE: BGEM3 import from pymilvus.model.hybrid is unavailable in 2.6.10
# Instead, use None for sparse_embedding_function and rely on Milvus's built-in BM25 indexing

SPARSE_EMB_FUNS = None  # Use Milvus's auto_sparse() with built-in BM25 instead
    

SPARSE_FIELD_NAME = 'sparse'
SPARSE_INDEX_NAME = 'sparse_index'
SPARSE_INDEX_TYPE = 'SPARSE_INVERTED_INDEX' #SPARSE_INVERTED_INDEX  WAND_INVERTED_INDEX AUTOINDEX
SPARSE_METRIC_TYPE =  'IP'
SPARSE_INDEX_PARAMS = {}

# BM25 possible parameters
# SPARSE_INDEX_PARAMS = {
#                         "bm25_k1": 1.5,
#                         "bm25_b": 0.75,
#                         "drop_ratio_build": 0.2
#                         }
 # {"nlist": 128}
# SPARSE_METRIC_TYPE =  'BM25'


# RANKER = RRFRanker(100)
RANKER = WeightedRanker(0.3, 0.8) 

###### OLLAMA
OLLAMA_API_BASE = 'http://localhost:11434'
OLLAMA_API_KEY = ''

# ollama list | cut -f1 -d' ' | grep -v NAME | while read model; do echo "$model"; ollama show "$model"; done
GEN_MODELS = { 
    'deepseek-r1_14b' : {
                  'name': 'ollama_chat/deepseek-r1:14b',
                  'parameters': 14.8 * 10**9,
                  'ctx_len': 131072,
                  'emb_len': 5120
              },
    'deepseek-r1' : {
                  'name': 'ollama_chat/deepseek-r1:latest',
                  'parameters': 7.6 * 10**9,
                  'ctx_len': 131072,
                  'emb_len': 3584
              },
    'llama3.3' : {
                  'name': 'ollama_chat/llama3.3:latest',
                  'parameters': 70.6 * 10**9,
                  'ctx_len': 131072,
                  'emb_len': 8192
              },
    'llama3_inst' : {
                  'name': 'ollama_chat/llama3:8b-instruct-q5_1',
                  'parameters': 8.0 * 10**9,
                  'ctx_len': 8192,
                  'emb_len': 4096
              },
    'llama3.2' : {
                  'name': 'ollama_chat/llama3.2:latest',
                  'parameters': 3.2 * 10**9,
                  'ctx_len': 131072,
                  'emb_len': 3072
              },
    'llama3' : {
                  'name': 'ollama_chat/llama3:latest',
                  'parameters': 8.0 * 10**9,
                  'ctx_len': 8192,
                  'emb_len': 4096
              },
    'vicuna' : {
                  'name': 'ollama_chat/wizard-vicuna-uncensored:latest',
                  'parameters': 6.7 * 10**9,
                  'ctx_len': 2048,
                  'emb_len': 4096
              },
    'smollm2' : {
                  'name': 'ollama_chat/smollm2:135m',
                  'parameters': 134.52 * 10**6,
                  'ctx_len': 8192,
                  'emb_len': 576
              },
    'gemma3' : {
        'name': 'ollama_chat/gemma3:4b',
        'parameters': 4.3 * 10**9,
        'ctx_len': 131072,
        'emb_len': 2560
    },

}

###### OLLAMA EMBEDDING MODELS
# https://ollama.com/blog/embedding-models

DENSE_EMB_MODELS = { 
    'nomic' : {
                  'name': 'nomic-embed-text:latest',  # https://ollama.com/library/nomic-embed-text
                  'parameters': 274 * 10**6,
                  'ctx_len': 8192,
                  'emb_len': 768
              },
    'mxbai' : {
                  'name': 'mxbai-embed-large:latest', # https://ollama.com/library/mxbai-embed-large
                  'parameters': 334.09 * 10**6,
                  'ctx_len': 512,
                  'emb_len': 1024
              },
    'multilingual-e5' : {
                  'name': 'multilingual-e5-small:latest',  # https://ollama.com/library/multilingual-e5-small
                  'parameters': 109 * 10**6,
                  'ctx_len': 512,
                  'emb_len': 384
              },
}





