import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from data.loader import load_bioasq_data



class RAGSystem:
    def __init__(self, corpus: Dataset, model_name: str):
        self.corpus = corpus
        self.model_name = model_name
        self.embeddings = None
        self.index = None
        self.id_map = None

    def embed_corpus(self):
        model = SentenceTransformer(self.model_name)
        self.embeddings = model.encode(
            self.corpus['passage'],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.id_map = self.corpus['id']


    def build_index(self):
        pass

    def query(self):
        pass

    def evaluate(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    data = load_bioasq_data()
    print(data.corpus[:1])
    rag_system = RAGSystem(data.corpus, "all-MiniLM-L6-v2")
    rag_system.embed_corpus()
    print(rag_system.embeddings[:1])
