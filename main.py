from pathlib import Path

from vector import RAG, create_embedder, create_client, conn_LLM
from mailconverter import EmlConverter, EmlxConverter

def main(mail_out_dir, mailbox, username, do_elmx):
    # if there is a folder of eml mails already available.
    ORIG_MAILS_DIR = "./orig_mails"

    if do_elmx:
        mail_converter = EmlxConverter()
        
    else:
        mail_converter = EmlConverter()

    if not (p:=Path(mail_out_dir)).is_dir():
        p.mkdir(parents=True, exist_ok=True) 

        if do_elmx:
            mail_converter.read_mails(username, mailbox, mail_out_dir)
        else:
            mail_converter.read_mails(ORIG_MAILS_DIR, mail_out_dir)

        # if not (p:=Path(chunks_dir)).is_dir():
        # p.mkdir(parents=True, exist_ok=True) 
    chunks = mail_converter.make_chunks(mail_out_dir)

 
    embedder = create_embedder()

    # print(embeddings)
   
    milvus_retriever = create_client(chunks)
    llm = conn_LLM()

    # Instantiate the RAG system
    rag_system = RAG(retriever=milvus_retriever, generator=llm)

    print(rag_system("Chi e' Stefano?"))

        # with open(Path(chunks_dir, BaseMailConv.CHUNKS_FILE), "r") as ctx_file:
        #    context = ctx_file.read()


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
        required=True,
        help='specifies the name of the mailbox to process',
    )
    parser.add_argument(
        '-u', '--username',
        dest='username',
        action='store',
        required=True,
        help='specifies the name of the mailbox to process',
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

    main(args.out_dir, args.mailbox, args.username, args.elmx)
