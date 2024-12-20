from .vectorRetriever import VectorStoreRetriever
from collections import defaultdict
from loguru import logger
from typing import List


# For hybrid retrieval, we can use a simple rank fusion approach to combine the results from two different retrievers.
class VectorStoreRetrieverHybrid(object):
    def __init__(
        self,
        vector_retriever_1: VectorStoreRetriever,
        vector_retriever_2: VectorStoreRetriever,
    ) -> None:
        self.vector_retriever_1 = vector_retriever_1  # dense retriever
        self.vector_retriever_2 = vector_retriever_2  # sparse retriever

    def rank_fusion(
        self,
        query: str,
        top_k: int = 3
    ) -> List[str]:
        
        chunks1 = self.vector_retriever_1.get_all_chunks(query) # from dense retriever
        chunks2 = self.vector_retriever_2.get_all_chunks(query) # from sparse retriever

        fusion_scores = defaultdict(float)

        for idx, chunk in enumerate(chunks1):
            fusion_scores[chunk["text"]] += (1.0 / (idx + 10)) # rank fusion score
        
        for idx, chunk in enumerate(chunks2):
            fusion_scores[chunk["text"]] += (1.0 / (idx + 10)) # rank fusion score

        sorted_fusion_scores = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_docs = sorted_fusion_scores[:top_k]
        logger.success(f"Retrieved docs successfully")
        
        return [doc[0] for doc in retrieved_docs] # return top_k docs


if __name__ == "__main__":
    retriever1 = VectorStoreRetriever("127.0.0.1", port=8765)
    retriever2 = VectorStoreRetriever("127.0.0.1", port=8766)
    retriever_hybrid = VectorStoreRetrieverHybrid(retriever1, retriever2)
    print(retriever_hybrid.rank_fusion("who is Whitfield Diffie and Martin E. Hellman?", top_k=10))
