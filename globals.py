from pymilvus import WeightedRanker, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


# if there is a folder of eml mails already available.
ORIG_MAILS_DIR = "./orig_mails"
USERNAME = 'SB'

######  CHUNKING
MAX_CHUNK_LEN = 1000
MAX_CHUNK_EXCESS = 2

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
SPARSE_EMB_FUN = BGEM3EmbeddingFunction() 
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
# GEN_MODEL = 'ollama_chat/llama3.2:latest'
# GEN_MODEL = 'ollama_chat/llama3:8b-instruct-q5_1'
GEN_MODEL = 'ollama_chat/llama3.3:latest'
# GEN_MODEL = 'ollama_chat/wizard-vicuna-uncensored'
# GEN_MODEL = 'ollama_chat/llama3.2'

###### OLLAMA EMBEDDING MODELS
# https://ollama.com/blog/embedding-models
# DENSE_EMB_MODEL = 'mxbai-embed-large' # https://ollama.com/library/mxbai-embed-large
DENSE_EMB_MODEL = 'galatolo/cerbero-7b-openchat' # https://github.com/galatolofederico/cerbero-7b



