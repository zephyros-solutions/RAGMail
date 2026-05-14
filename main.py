# TODO
# Try contextual retrieval:
#   https://milvus.io/docs/contextual_retrieval_with_milvus.md
# Try a complete context with all emails.
# Try other methods than chain of thoughts
# Try different generation models from https://ollama.com/library
# What model for embeddings (ColBERTv2) and Milvus can also do embeddings
#   https://milvus.io/docs/embeddings.md
# Integrate Milvus with DSPy
#   https://milvus.io/docs/integrate_with_dspy.md
# DSPy optimisation
# How to configure the generation LLM?
#   https://github.com/stanfordnlp/dspy/blob/main/dsp/modules/lm.py
# Consider wrapping embedders in dspy.Embedder, maybe it gives batching?
#   https://github.com/stanfordnlp/dspy/blob/6178c28ce96b2ecb8a21c722ff06cac58b0bb83c/dspy/clients/embedding.py#L5
# Old LLM list
# NAME                                   ID              SIZE      MODIFIED      
# phi4:latest                            ac896e5b8b34    9.1 GB    12 months ago    
# deepseek-r1:14b                        ea35dfe18182    9.0 GB    12 months ago    
# deepseek-r1:latest                     0a8c26691023    4.7 GB    13 months ago    
# galatolo/cerbero-7b-openchat:latest    7a59cede270b    14 GB     15 months ago    
# llama3.3:latest                        a6eb4748fd29    42 GB     15 months ago    
# mxbai-embed-large:latest               468836162de7    669 MB    15 months ago    
# wizard-vicuna-uncensored:latest        72fc3c2b99dc    3.8 GB    15 months ago    
# llama3:8b-instruct-q5_1                662158bc9277    6.1 GB    15 months ago    
# llama3:latest                          365c0bd3c000    4.7 GB    15 months ago    
# llama3.2:latest                        a80c4f17acd5    2.0 GB    15 months ago    

import math
import re
import dspy
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from llm.rag import RAG
from mail_processing.mailconverter import MailConverter, EmlxConverter
from config.globals import DUMP_DIR, MAX_CHUNK_LEN, MAX_CHUNK_EXCESS, TOK2CHAR
from config.globals import OLLAMA_API_BASE, OLLAMA_API_KEY
from config.globals import MILVUS_DYN, MILVUS_MAX_LENGTH, MILVUS_LEN_CTX
from config.globals import DENSE_EMB_MODELS, DENSE_METRIC_TYPE
from config.globals import SPARSE_EMB_FUNS
from config.globals import GEN_MODELS
from config.globals import RANKER

from indexing.retriever import RMClient, my_embedder
from llm.prompts import Prompts
from indexing.es import ElSearch


def conn_LLM(model):

    # # Connect to Llama3 hosted with Ollama
    # lm = dspy.OllamaLocal(
    #     model=model,
    #     max_tokens=max_tokens,
    #     timeout_s=480
    # )
    lm = dspy.LM(model['name'], api_base=OLLAMA_API_BASE, api_key=OLLAMA_API_KEY, num_ctx=model['ctx_len'])
    dspy.configure(lm=lm)
    
    # # Test connection
    # test_query = "What is the latest in AI?"
    # test_response = lm(test_query)
    # print(f"Test {model} response:", test_response)

    return lm

def reduce_context(context, ctx_len, llm):
    """
        Reduce context length to fit model limits by splitting into segments and summarizing with the LLM.
    """
    max_len = int(TOK2CHAR*ctx_len)
    prt_len = Prompts.get_blog_prt_len()
    print(f"Context length: {len(context)} chars, prompt len: {prt_len}, limit for model {ctx_len} tokens is {max_len} chars")
    if len(context) + prt_len > max_len:
        seg_max_len = max_len - prt_len
        nr_segs = math.ceil(len(context)/seg_max_len)
        new_ctx = ""
        print(f"Context exceeds model limit, splitting into {nr_segs} segments and summarizing with LLM...")
        for i in tqdm(range(nr_segs), desc="Processing segments"):
            inf = i * seg_max_len if i > 0 else 0
            sup = min((i+1) * seg_max_len,len(context))
            prompt = Prompts.blog_prompt(max_chars=int(max_len/nr_segs), content=context[inf:sup])
            # breakpoint()
            new_ctx = f'{new_ctx}{llm(prompt)} '
    else:
        new_ctx = context

    return new_ctx

def do_blob(mail_source, ctx_len:int, llm):
    print(f"Creating blob from {len(mail_source.proc_mails)} processed mails...")
    context = mail_source.make_blob()
    # breakpoint()
    
    new_ctx = reduce_context(context, ctx_len, llm)

    blog_dir = f'{DUMP_DIR}/methods/{mail_source.mailsId}/blogs'
    if not (p:=Path(blog_dir)).is_dir():
               p.mkdir(parents=True, exist_ok=True) 
    with open(f'{blog_dir}/blob.txt', 'w') as f:
        f.write(new_ctx)

    rag_system = RAG(retriever=None, context=new_ctx)

    return rag_system

def do_grep(mail_source, ctx_len:int, llm):
    mails_arr = mail_source.msgs_array()
    print(f"Creating grep functionality for {len(mails_arr)} processed mails...")
    grep_dir = f'{DUMP_DIR}/methods/{mail_source.mailsId}/grep'
    if not (p:=Path(grep_dir)).is_dir():
               p.mkdir(parents=True, exist_ok=True) 
    
    def retriever(prompt):
        
        grep_prompt = Prompts.grep_prompt(prompt=prompt)
        entities = llm(grep_prompt)[0].split(',')
        context_arr = []
        for mail in mails_arr:
            for entity in entities:
                # breakpoint()
                if entity.lower().strip() in mail.get_content().lower():
                    context_arr.append(mail.get_content())
        context = "\n".join(context_arr)
        new_ctx = reduce_context(context, ctx_len, llm)

        with open(f'{grep_dir}/grep.txt', 'a') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Extracted entities: {entities}\n\n")
            f.write("Retrieved context:\n")
            f.write(new_ctx)
    
        return context
    
    rag_system = RAG(retriever=retriever, context=None)

    return rag_system

def do_es(mail_source, llm):
    es = ElSearch(mail_source.mailsId)
    es.index_mails(mail_source.msgs_array())

    def retriever(prompt):
        context = []
        mail_ids = es.search(prompt)
        # breakpoint()
        
        for mail_id in mail_ids:
            context.append(mail_source.proc_folder[mail_id].get_content())
        # breakpoint()
        return context
    
    rag_system = RAG(retriever=retriever, context=None)

    return rag_system

def do_rag(mail_out_dir, dense_emb, sparse_emb, force):
    
    dim_dense_emb = dense_emb['emb_len']
    
    repo = f"{mail_out_dir}_{dense_emb['name']}_{sparse_emb.__name__ if sparse_emb else 'NS'}"
    # breakpoint()
    # collection name can only contain numbers, letters and underscores
    collection_name = re.sub(r'[^\w\d]', '', repo)

    print(f"Working with collection: {collection_name}")

    rm_client = RMClient(collection_name, k = MILVUS_LEN_CTX, dim_dense_emb=dim_dense_emb, max_length=MILVUS_MAX_LENGTH, 
                         dense_embedding_function=my_embedder(dense_emb['name']), sparse_embedding_function=sparse_emb, rerank_function=RANKER,
                         use_contextualize_embedding=False)

    
    
    if force or rm_client.build_collection(enable_dynamic_field=MILVUS_DYN):
        # breakpoint()
        chunks = MailConverter.make_chunks(mail_out_dir, max_chunk_len=MAX_CHUNK_LEN, max_chunk_excess=MAX_CHUNK_EXCESS)
        rm_client.upload_embeddings(chunks, metadata={})
         
   
    # embedder = create_embedder()
    
    rag_system = RAG(retriever=rm_client, context=None)

    return rag_system



def main(mailbox:str, doThreads:bool, method:str, dense:str, sparse:str, gen:str, username:str, start:str, end:str):
    
    in_fmt = r'%d/%m/%Y'
    if start != None:
        start_date = datetime.strptime(start,in_fmt).replace(tzinfo=timezone.utc)
    else:
        start_date = datetime.strptime('01/01/1970',in_fmt).replace(tzinfo=timezone.utc)

    if end != None:
        end_date = datetime.strptime(end,in_fmt).replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.strptime('01/01/2970',in_fmt).replace(tzinfo=timezone.utc)

    
    # breakpoint()

    
    mail_converter = EmlxConverter(username=username, mailbox=mailbox, doThreads=doThreads, start_date=start_date, end_date=end_date)
    mail_converter.read_mails()
    mail_converter.save_msgs()
    
    llm = conn_LLM(model=GEN_MODELS[f'{gen}'])

    if method == 'blob':
        rag_system = do_blob(mail_source=mail_converter, ctx_len=GEN_MODELS[f'{gen}']['ctx_len'], llm=llm)
    elif method == 'grep':
        rag_system = do_grep(mail_source=mail_converter, ctx_len=GEN_MODELS[f'{gen}']['ctx_len'], llm=llm)
    elif method == 'es':
        rag_system = do_es(mail_source=mail_converter, llm=llm)
    elif method == 'milvus':
        # Prepare mail output directory for chunking
        mail_out_dir = mail_converter.mail_out_dir
        rag_system = do_rag(mail_out_dir=mail_out_dir, 
                           dense_emb=DENSE_EMB_MODELS[f'{dense}'], 
                           sparse_emb=SPARSE_EMB_FUNS.get(f'{sparse}') if sparse else None,
                           force=False)
    else:
        raise Exception(f'Method {method} not known')
    
    while True:
        try:
            prompt = str(input("Please enter your prompt: "))            
        except ValueError:
            print("Sorry, I didn't understand that.")
            #better try again... Return to the start of the loop
            continue
        if prompt == "q":
            break
        # breakpoint()
        # print(rm_client(prompt))
        print(rag_system(prompt))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-d', '--dense',
        dest='dense',
        action='store',
        required=False,
        help='specifies the name of the embedding model',
    )

    parser.add_argument(
        '-e', '--end',
        dest='end',
        action='store',
        required=False,
        help='specifies the latest email to examine',
    )

    parser.add_argument(
        '-g', '--gen',
        dest='gen',
        action='store',
        required=True,
        help='specifies the name of the generation model',
    )

    parser.add_argument(
        '-m', '--mailbox',
        dest='mailbox',
        action='store',
        required=True,
        help='specifies the name of the mailbox to process',
    )

    parser.add_argument(
        '--method',
        dest='method',
        action='store',
        required=True,
        choices=['blob', 'grep', 'es', 'milvus'],
        help='specifies the retrieval method: blob (load all), grep (keyword search), es (Elasticsearch BM25), or milvus (semantic + sparse hybrid)',
    )

    parser.add_argument(
        '-s', '--start',
        dest='start',
        action='store',
        required=False,
        help='specifies the earliest email to examine',
    )

    parser.add_argument(
        '--sparse',
        dest='sparse',
        action='store',
        required=False,
        default = 'BGEM3',
        help='specifies the sparse embedder to use',
    )

    parser.add_argument(
        '-t', '--threaded',
        dest='doThreads',
        action='store_true',
        default=False,
        help='specifies whether to group the emails in threads',
    )
    
    parser.add_argument(
        '-u', '--username',
        dest='username',
        action='store',
        required=True,
        help='specifies the username for the mail directory',
    )


    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(f'Unknown options {unknown}')
        parser.print_help()
        exit(-1)

    if args.method == 'milvus' and args.dense is None:
        parser.error("--method milvus requires -d (--dense) <dense embedder>")

    main(mailbox=args.mailbox, doThreads=args.doThreads, method=args.method, dense=args.dense,
          sparse=args.sparse, gen=args.gen, username=args.username, start=args.start, end=args.end)
