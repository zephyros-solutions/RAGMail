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

from pathlib import Path
import re
import dspy
from datetime import datetime, timezone

from rag import RAG
from mailconverter import MailConverter, EmlxConverter
from globals import MAX_CHUNK_LEN, MAX_CHUNK_EXCESS, TOK2CHAR
from globals import OLLAMA_API_BASE, OLLAMA_API_KEY
from globals import MILVUS_DYN, MILVUS_MAX_LENGTH, MILVUS_LEN_CTX
from globals import DENSE_EMB_MODELS, DENSE_METRIC_TYPE
from globals import SPARSE_EMB_FUNS
from globals import GEN_MODELS
from globals import RANKER

from retriever import RMClient, my_embedder
from vocab import summary_prt, entities_prt
from es import ElSearch


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

def do_blob(mail_source,ctx_len:int, llm):

    context = mail_source.make_blob()
    # breakpoint()
    if len(context) > TOK2CHAR*ctx_len:
        seg_len = int(TOK2CHAR*ctx_len)
        nr_segs = int(len(context)/seg_len)
        new_ctx = ""
        for i in range(nr_segs):
            inf = i * seg_len
            sup = min((i+1) * seg_len,len(context))
            prompt = summary_prt(max_char=int(TOK2CHAR*ctx_len/nr_segs), content=context[inf:sup])
            # breakpoint()
            new_ctx = f'{new_ctx}{llm(prompt)} '
    else:
        new_ctx = context

    rag_system = RAG(retriever=None, context=new_ctx)

    return rag_system

def do_grep(mail_source, llm):

    def retriever(prompt):
        
        entities_prompt = entities_prt(prompt=prompt)
        entities = llm(entities_prompt)[0].split(',')
        print(f"Extracted entities: {entities}")
        context = []
        for mail in mail_source.msgs_array():
            for entity in entities:
                # breakpoint()
                if entity.lower().strip() in mail.get_content().lower():
                    context.append(mail.get_content())
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



def main(mailbox, doThreads, do_elmx, method, dense, sparse, gen, start, end):
    
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

    
    mail_converter = EmlxConverter(mailbox=mailbox,doThreads=doThreads,start_date=start_date, end_date=end_date)
    mail_converter.read_mails()
    mail_converter.save_msgs()
    
    llm = conn_LLM(model=GEN_MODELS[f'{gen}'])

    if method == 'blob':
        rag_system = do_blob(mail_source=mail_converter, ctx_len=GEN_MODELS[f'{gen}']['ctx_len'], llm=llm)
    elif method == 'grep':
        rag_system = do_grep(mail_source=mail_converter, llm=llm)
    elif method == 'es':
        rag_system = do_es(mail_source=mail_converter, llm=llm)
    # elif method == 'rag':
    #     rag_system = do_rag(mail_out_dir=tr_mail_out_dir, force=mail_processed, dense_emb=DENSE_EMB_MODELS[f'{dense}'], sparse_emb=SPARSE_EMB_FUNS[f'{sparse}'])
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
        help='specifies the method to be used',
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
        '-x', '--emlx',
        dest='elmx',
        action='store_true',
        default=True,
        help='specifies whether to process elmx files',
    )

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(f'Unknown options {unknown}')
        parser.print_help()
        exit(-1)

    if args.method == 'rag' and args.dense is None:
        parser.error("--method rag requires -d (--dense) <dense embedder>")

    main(mailbox=args.mailbox, doThreads=args.doThreads, do_elmx=args.elmx, 
         method=args.method, dense=args.dense, sparse=args.sparse, gen=args.gen, start=args.start, end=args.end)
