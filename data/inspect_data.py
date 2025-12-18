"""Quick script to inspect the data structure."""

from datasets import load_dataset

# Load queries
queries_ds = load_dataset(
    "rag-datasets/rag-mini-bioasq", "question-answer-passages")
queries = queries_ds['test']
print(queries)

# Load corpus
corpus_ds = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")
corpus = corpus_ds['passages']

print("=== First Query ===")
first_query = queries[0]
print(f"Query ID: {first_query['id']}")
print(f"Question: {first_query['question'][:200]}")
print(
    f"Relevant passage IDs type: {type(first_query['relevant_passage_ids'])}")
# First 10
print(f"Relevant passage IDs: {first_query['relevant_passage_ids'][:10]}")
print(f"Total relevant: {len(first_query['relevant_passage_ids'])}")

print("\n=== Corpus Sample ===")
print(f"Total passages: {len(corpus)}")
print(f"First passage ID: {corpus[0]['id']}")
print(f"First passage ID type: {type(corpus[0]['id'])}")
print(f"First 10 passage IDs: {[corpus[i]['id'] for i in range(10)]}")

print("\n=== Checking ID Match ===")
query_relevant_ids = set(first_query['relevant_passage_ids'])
corpus_ids = set(corpus['id'])
print(f"Query relevant IDs (first 5): {list(query_relevant_ids)[:5]}")
print(f"Corpus IDs (first 5): {list(corpus_ids)[:5]}")
print(
    f"Overlap: {len(query_relevant_ids & corpus_ids)} / {len(query_relevant_ids)}")
