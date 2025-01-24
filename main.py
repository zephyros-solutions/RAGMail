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

from rag import RAG
from mailconverter import MailConverter, EmlConverter, EmlxConverter
from globals import ORIG_MAILS_DIR
from globals import MAX_CHUNK_LEN, MAX_CHUNK_EXCESS
from globals import OLLAMA_API_BASE, OLLAMA_API_KEY
from globals import MILVUS_DYN, MILVUS_MAX_LENGTH, MILVUS_LEN_CTX
from globals import DENSE_EMB_MODELS, DENSE_METRIC_TYPE
from globals import SPARSE_EMB_FUN
from globals import GEN_MODELS
from globals import RANKER

from retriever import RMClient, get_emb_size, my_embedder


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

def do_blob(mail_out_dir):

    context = MailConverter.make_blob(mail_out_dir=mail_out_dir)

    rag_system = RAG(retriever=None, context=context)

    return rag_system


def do_rag(mail_out_dir, mailbox, doThreads, model, force):
    
    dim_dense_emb = get_emb_size()
    
    repo = f"{mailbox}_{'T' if doThreads else 'NT'}_{model['name']}_{SPARSE_EMB_FUN.__name__ if SPARSE_EMB_FUN else 'NS'}"
    # breakpoint()
    # collection name can only contain numbers, letters and underscores
    collection_name = re.sub(r'[^\w\d]', '', repo)

    print(f"Working with collection: {collection_name}")

    rm_client = RMClient(collection_name, k = MILVUS_LEN_CTX, dim_dense_emb=dim_dense_emb, max_length=MILVUS_MAX_LENGTH, 
                         dense_embedding_function=my_embedder, sparse_embedding_function=SPARSE_EMB_FUN, rerank_function=RANKER,
                         use_contextualize_embedding=False)

    
    
    if force or rm_client.build_collection(enable_dynamic_field=MILVUS_DYN):
        
        chunks = MailConverter.make_chunks(mail_out_dir, max_chunk_len=MAX_CHUNK_LEN, max_chunk_excess=MAX_CHUNK_EXCESS)
        rm_client.upload_embeddings(chunks, metadata={})
         
   
    # embedder = create_embedder()
    
    rag_system = RAG(retriever=rm_client, context=None)

    return rag_system


def do_mail(mail_out_dir, mailbox, doThreads, do_elmx):
    mail_processed = False

    if not (p:=Path(mail_out_dir)).is_dir():
        if mailbox == None:
            raise Exception(f"mailbox needs to be specified")
        p.mkdir(parents=True, exist_ok=True) 

        if do_elmx:
            mail_converter = EmlxConverter(mailbox, doThreads)
            mail_converter.read_mails(mail_out_dir)
        else:
            mail_converter = EmlConverter(ORIG_MAILS_DIR)
            mail_converter.read_mails(mail_out_dir)
        
        mail_processed = True

    return mail_processed

def main(mail_out_dir, mailbox, doThreads, do_elmx, method):
    
    tr_mail_out_dir = f"{mail_out_dir}_{'T' if doThreads else 'NT'}"

    force = do_mail(tr_mail_out_dir, mailbox, doThreads, do_elmx)
    
    llm = conn_LLM(model=GEN_MODELS['llama3.3'])

    if method == 'blob':
        rag_system = do_blob(mail_out_dir)
    elif method == 'rag':
        rag_system = do_rag(mail_out_dir=tr_mail_out_dir, mailbox=mailbox, doThreads=doThreads, force=force, model=DENSE_EMB_MODELS['cerbero'])
    
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
        '-d', '--out_dir',
        dest='out_dir',
        action='store',
        required=False,
        default = './proc_mails',
        help='specifies the name of the directory to store mails',
    )

    parser.add_argument(
        '-m', '--mailbox',
        dest='mailbox',
        action='store',
        required=False,
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
        '-t', '--threaded',
        dest='doThreads',
        action='store_true',
        default=True,
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

    main(args.out_dir, args.mailbox, args.doThreads, args.elmx, args.method)
