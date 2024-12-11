# TODO
# Create a DSPy embedder with ollama, here:
# https://ollama.com/blog/embedding-models, https://milvus.io/docs/integrate_with_dspy.md
# How to configure the generation LLM?
#   https://github.com/stanfordnlp/dspy/blob/main/dsp/modules/lm.py
#   https://github.com/stanfordnlp/dspy/blob/6178c28ce96b2ecb8a21c722ff06cac58b0bb83c/dspy/clients/embedding.py#L5
# Try different generation models from https://ollama.com/library
# parameters for milvus_client.create_collection: dimension, metric_type, max_length, enable_dynamic
# The parameter k for context retrieved
# What model for embeddings and Milvus can also do embeddings: https://milvus.io/docs/embeddings.md
# Try other methods than chain of thoughts

from pathlib import Path
import re

from vector import RAG, create_embedder, create_collection, upload_embeddings, conn_LLM, get_retriever, get_emb_size
from mailconverter import MailConverter, EmlConverter, EmlxConverter
from globals import ORIG_MAILS_DIR
from globals import EMB_MODEL, GEN_MODEL
from globals import MILVUS_METRIC_TYPE, MILVUS_MAX_LENGTH, MILVUS_DYN, MILVUS_LEN_CTX
from globals import MAX_CHUNK_LEN, MAX_CHUNK_EXCESS

def main(mail_out_dir, mailbox, doThreads, do_elmx):
    emb_size = get_emb_size()
    
    repo = f"{mailbox}_{'T' if doThreads else 'NT'}_{EMB_MODEL}_{emb_size}"
    
    # collection name can only contain numbers, letters and underscores
    collection_name = re.sub(r'[^\w\d]', '', repo)

    db_client = create_collection(collection_name=collection_name, dimension=emb_size, metric_type=MILVUS_METRIC_TYPE, max_length=MILVUS_MAX_LENGTH, enable_dynamic=MILVUS_DYN)
    
    mail_out_dir = f"{mail_out_dir}_{'T' if doThreads else 'NT'}"
    
    if db_client != None:
    
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
    
        chunks = MailConverter.make_chunks(mail_out_dir, max_chunk_len=MAX_CHUNK_LEN, max_chunk_excess=MAX_CHUNK_EXCESS)
        upload_embeddings(client=db_client, chunks=chunks, collection_name=collection_name)
         
   
    # embedder = create_embedder()
    milvus_retriever = get_retriever(collection_name=collection_name, k=MILVUS_LEN_CTX)
    
    llm = conn_LLM(model=GEN_MODEL)

    # Instantiate the RAG system
    rag_system = RAG(retriever=milvus_retriever)

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

    main(args.out_dir, args.mailbox, args.doThreads, args.elmx)
