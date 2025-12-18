What RAG means in this project:
    RAG means retrieval augmented generation. We are going to take a dataset, ingest it, convert and split it into chunks. Then I will take those chunks and convert them to embeddings. We store those embeddings in a vector database. The user will ask a question and we will embed that questions as well. We will do some similarity search between the user query embeddings and the prep data embeddings in the vector db. We will grab the top k results and pass that into the LLM for retrieval and test for the metrics defined below. 

    Concerns:
    1. Complex to setup
    2. Latency
    3. Retrieval accuracy

    image.png

What CAG means in this prokect:
    CAG means Cache augmented generation. This is much simpler. We are utilizing the size of the context window of the model and storing the prep data inside directly. This simply wasnt possible 2 years ago as often the largest of these context windows would be 4096 tokens. We dump the prep data aand the user question into the model. No need for vectorization, retrieval etc.

What we are measuring:
✅ Recall under memory pressure
% of queries whose relevant passages survive in CAG memory
    Compare to RAG Recall@K

✅ Degradation curves
As corpus grows:
    RAG recall stays relatively stable
    CAG recall drops sharply once memory fills
    This is a real result, not a fake one.

✅ Latency tradeoffs
RAG: per-query cost
CAG: upfront cost, zero per-query retrieval

Dataset Contract

Corpus size: 40,221 passages

Query count: 4,719

Ground truth: relevant_passage_ids

Evaluation unit: passage-level recall