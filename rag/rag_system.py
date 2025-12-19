import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from data.loader import load_bioasq_data
from faiss import IndexFlatL2
from evaluation import (
    evaluate_rag_system, 
    print_evaluation_results,
    save_results,
    generate_all_plots,
    plot_latency_distribution
)
from enum import Enum


class ModelNames(Enum):
    """Supported sentence transformer model names."""
    ALL_MINI_LM_L6_V2 = "all-MiniLM-L6-v2"

class RAGSystem:
    def __init__(self, corpus: Dataset, model_name: str, top_k: int = 10):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.index = None
        self.id_map = None
        self.top_k = top_k

    def embed_corpus(self):
        self.embeddings = self.model.encode(
            self.corpus['passage'],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.id_map = list(self.corpus['id'])


    def build_index(self):
        """
        Build a FAISS index from the embeddings.
        Must be called after embed_corpus().
        """
        if self.embeddings is None:
            raise ValueError("Must call embed_corpus() before build_index()")
        
        self.index = IndexFlatL2(self.embeddings.shape[1])
        
        embeddings_to_add = self.embeddings.astype('float32')
        self.index.add(embeddings_to_add)

    def query(self, query: str):
        """
        Query the index for the top k passages most similar to the query.
         
        Args:
            query: The query string to search for
            
        Returns:
            dict with:
                - 'passage_ids': List of passage IDs (top k most similar)
                - 'distances': List of L2 distances (lower = more similar)
                - 'passages': List of passage texts (optional, for convenience)
        """
        if self.index is None:
            raise ValueError("Must call build_index() before query()")
        
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, k=self.top_k)
        
        passage_ids = [self.id_map[idx] for idx in indices[0]]
        passages = [self.corpus[int(idx)]['passage'] for idx in indices[0]]
        
        return {
            'passage_ids': passage_ids,
            'distances': distances[0].tolist(),
            'passages': passages
        }

    def evaluate(self, data, query_ids=None, k=None):
        """
        Evaluate the RAG system on queries.
        
        Args:
            data: BioASQData instance with queries and ground truth
            query_ids: List of query IDs to evaluate (None = all queries)
            k: Top K to evaluate (None = use self.top_k)
            
        Returns:
            Dict with evaluation metrics
        """
        return evaluate_rag_system(self, data, query_ids, k)

    def run(self):
        pass


if __name__ == "__main__":
    data = load_bioasq_data()
    print(data.corpus[:1])
    rag_system = RAGSystem(data.corpus, ModelNames.ALL_MINI_LM_L6_V2.value)
    rag_system.embed_corpus()
    rag_system.build_index()

    rag_system.query("What is the purpose of the human body?")
    results = rag_system.evaluate(data)
    print_evaluation_results(results)
