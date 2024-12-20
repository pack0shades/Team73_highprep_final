import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--use_reranker", type=bool,
                    default=False, help="Use reranker or not")
    arg.add_argument("--retrieved_docs", type=int, default=6,
                    help="Number of documents to retrieve")
    arg.add_argument("--collection_name", type=str,
                    default="generate", help="Name of the collection")
    arg.add_argument("--pdf_path", type=str,
                    default="./pdfs/nvidia.pdf", help="Path to the PDF document")
    arg.add_argument("--dfrom", type=int, default=400,
                    help="from which pdf to start")
    arg.add_argument("--dto", type=int, default=509,
                    help="to which pdf to end")
    arg.add_argument("--pipeline", default="naive",
                    help="Specify the pipeline to use. Options are: 'nov4', 'nov9'")
    arg.add_argument("--topk", type=int, default=8,
                    help="Number of documents to retrieve")
    arg.add_argument("--method", type=str, default=None,
                    help="Specify the method to use. Options are: 'cr', 'hs'")
    arg.add_argument("--use_reflection", type=bool,
                    default=False, help="Use reflection or not")
    arg.add_argument("--n_reflection", type=int, default=2,
                    help="Number of reflections to use")
    arg.add_argument("--agent_type", type=str,
                    default="dynamic", help="which agent to use", choices=["dynamic", "multi"])
    arg.add_argument("--use_router", type=bool, default=True,
                    help="whether to use router or not")

    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
