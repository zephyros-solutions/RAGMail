from pymilvus import WeightedRanker, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


# if there is a folder of eml mails already available.
ORIG_MAILS_DIR = "./orig_mails"
USERNAME = 'SB'


######  CHUNKING
MAX_CHUNK_LEN = 1000
MAX_CHUNK_EXCESS = 2
TOK2CHAR = 2.5

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
# SPARSE_EMB_FUN = BGEM3EmbeddingFunction
# SPARSE_EMB_FUN_NAME = SPARSE_EMB_FUN.__module__
SPARSE_EMB_FUN = BGEM3EmbeddingFunction
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
    'llama3.2' : {
                  'name': 'ollama_chat/llama3.2:latest',
                  'parameters': 3.2 * 10**9,
                  'ctx_len': 131072,
                  'emb_len': 3072
              },
}

###### OLLAMA EMBEDDING MODELS
# https://ollama.com/blog/embedding-models

DENSE_EMB_MODELS = { 
    'cerbero' : {
                  'name': 'galatolo/cerbero-7b-openchat:latest', # https://github.com/galatolofederico/cerbero-7b
                  'parameters': 7.2 * 10**9,
                  'ctx_len': 8192,
                  'emb_len': 4096
              },
    'mxbai' : {
                  'name': 'mxbai-embed-large:latest', # https://ollama.com/library/mxbai-embed-large
                  'parameters': 334.09 * 10**6,
                  'ctx_len': 512,
                  'emb_len': 1024
              },
}





