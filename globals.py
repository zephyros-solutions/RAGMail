# if there is a folder of eml mails already available.
ORIG_MAILS_DIR = "./orig_mails"
USERNAME = 'SB'


MILVUS_URI = './milvus.db'
MILVUS_TOKEN = ""

# EMB_MODEL = 'mxbai-embed-large' # https://ollama.com/library/mxbai-embed-large
EMB_MODEL = 'galatolo/cerbero-7b-openchat' # https://github.com/galatolofederico/cerbero-7b


MILVUS_METRIC_TYPE = "IP"
MILVUS_MAX_LENGTH = 65535
MILVUS_DYN = True
MILVUS_LEN_CTX = 20



# GEN_MODEL = 'ollama_chat/llama3.2:latest'
# GEN_MODEL = 'ollama_chat/llama3:8b-instruct-q5_1'
GEN_MODEL = 'ollama_chat/llama3.3:latest'
# GEN_MODEL = 'ollama_chat/wizard-vicuna-uncensored'
# 'ollama_chat/llama3.2'
API_BASE = 'http://localhost:11434'
API_KEY = ''



# Chunking
MAX_CHUNK_LEN = 1000
MAX_CHUNK_EXCESS = 2