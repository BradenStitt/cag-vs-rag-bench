"""Data loader for BioASQ RAG dataset."""

from typing import Dict, List, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
import random


class BioASQData:
    """Container for BioASQ dataset with convenient access methods."""

    def __init__(self, queries: Dataset, corpus: Dataset):
        """
        Initialize with loaded datasets.

        Args:
            queries: Dataset with 'question', 'answer', 'relevant_passage_ids', 'id'
            corpus: Dataset with 'passage', 'id'
        """
        self.queries = queries
        self.corpus = corpus

        # Create lookup dictionaries for fast access
        self._corpus_dict = {row['id']: row['passage'] for row in corpus}
        self._query_dict = {
            row['id']: {
                'question': row['question'],
                'answer': row['answer'],
                'relevant_passage_ids': row['relevant_passage_ids']
            }
            for row in queries
        }

    def get_query(self, query_id: str) -> Dict:
        """Get a query by ID."""
        return self._query_dict[query_id]

    def get_passage(self, passage_id: str) -> str:
        """Get a passage by ID."""
        return self._corpus_dict[passage_id]

    def get_relevant_passages(self, query_id: str) -> List[str]:
        """Get all relevant passages for a query as text."""
        relevant_ids = self._query_dict[query_id]['relevant_passage_ids']
        return [self._corpus_dict[pid] for pid in relevant_ids if pid in self._corpus_dict]

    def get_corpus_subset(self, size: int, seed: Optional[int] = None) -> Dataset:
        """
        Get a random subset of the corpus.

        Args:
            size: Number of passages to include
            seed: Random seed for reproducibility

        Returns:
            Subset of corpus dataset
        """
        if seed is not None:
            random.seed(seed)

        if size >= len(self.corpus):
            return self.corpus

        indices = random.sample(range(len(self.corpus)), size)
        return self.corpus.select(indices)

    def get_corpus_dict_subset(self, size: int, seed: Optional[int] = None) -> Dict[str, str]:
        """
        Get a random subset of the corpus as a dictionary.

        Args:
            size: Number of passages to include
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping passage_id -> passage_text
        """
        subset = self.get_corpus_subset(size, seed)
        return {row['id']: row['passage'] for row in subset}

    def get_query_subset(self, size: int, seed: Optional[int] = None) -> List[str]:
        """
        Get a random subset of query IDs.

        Args:
            size: Number of queries to include
            seed: Random seed for reproducibility

        Returns:
            List of query IDs
        """
        if seed is not None:
            random.seed(seed)

        all_query_ids = list(self._query_dict.keys())
        if size >= len(all_query_ids):
            return all_query_ids

        return random.sample(all_query_ids, size)

    def __len__(self):
        """Return number of queries."""
        return len(self.queries)

    def __repr__(self):
        return (
            f"BioASQData(queries={len(self.queries)}, "
            f"corpus={len(self.corpus)} passages)"
        )


def load_bioasq_data(
    cache_dir: Optional[str] = None,
    verbose: bool = True
) -> BioASQData:
    """
    Load the BioASQ RAG dataset.

    Args:
        cache_dir: Directory to cache datasets (default: HuggingFace cache)
        verbose: Whether to print loading progress

    Returns:
        BioASQData object with queries and corpus
    """
    if verbose:
        print("Loading BioASQ queries...")

    queries_ds = load_dataset(
        "rag-datasets/rag-mini-bioasq",
        "question-answer-passages",
        cache_dir=cache_dir
    )
    queries = queries_ds['test']  # Get the test split

    if verbose:
        print(f"Loaded {len(queries)} queries")
        print("Loading BioASQ corpus...")

    corpus_ds = load_dataset(
        "rag-datasets/rag-mini-bioasq",
        "text-corpus",
        cache_dir=cache_dir
    )
    corpus = corpus_ds['passages']  # Get the passages split

    if verbose:
        print(f"Loaded {len(corpus)} passages")

    return BioASQData(queries, corpus)


def get_corpus_subset(
    data: BioASQData,
    size: int,
    seed: Optional[int] = None
) -> Dict[str, str]:
    """
    Convenience function to get a corpus subset.

    Args:
        data: BioASQData object
        size: Number of passages to include
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping passage_id -> passage_text
    """
    return data.get_corpus_dict_subset(size, seed)


if __name__ == "__main__":
    # Test the data loader
    print("Testing BioASQ data loader...")
    data = load_bioasq_data()

    print(f"\nDataset info: {data}")

    # Test getting a query
    query_ids = list(data._query_dict.keys())
    test_query_id = query_ids[0]
    query = data.get_query(test_query_id)

    print(f"\nSample query (ID: {test_query_id}):")
    print(f"  Question: {query['question'][:100]}...")
    print(f"  Relevant passages: {len(query['relevant_passage_ids'])}")

    # Test getting relevant passages
    relevant = data.get_relevant_passages(test_query_id)
    print(
        f"  First relevant passage: {relevant[0][:100] if relevant else 'None'}...")

    # Test corpus subset
    print(f"\nTesting corpus subsets:")
    for size in [1000, 5000, 10000]:
        subset = data.get_corpus_subset(size, seed=42)
        print(f"  {size} passages: {len(subset)} actual passages")
