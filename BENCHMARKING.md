# RAG vs CAG Benchmarking Framework

## Core Metrics

### 1. Retrieval Quality Metrics

#### Passage-Level Recall
- **RAG**: `Recall@K = |retrieved_passages ∩ relevant_passages| / |relevant_passages|`
- **CAG**: `Recall = |passages_in_context ∩ relevant_passages| / |relevant_passages|`
- **Query-Level Recall**: Percentage of queries where at least one relevant passage is found

#### Precision
- **Precision@K**: `|retrieved_passages ∩ relevant_passages| / K` (for RAG)
- Useful for understanding retrieval quality vs quantity tradeoff

### 2. Latency Metrics

#### RAG Latency Breakdown
- Embedding time: Query embedding generation
- Retrieval time: Vector search + passage fetching
- Generation time: LLM inference
- **Total per-query latency**: Sum of above

#### CAG Latency
- Generation time: LLM inference (no retrieval overhead)
- **Upfront cost**: Corpus embedding/preparation (one-time, amortized)

### 3. Cost Metrics

#### Token Usage
- **RAG per-query**: `query_tokens + retrieved_passages_tokens + generation_tokens`
- **CAG per-query**: `corpus_tokens + query_tokens + generation_tokens`
- Track token counts for cost estimation

#### Cost Efficiency
- Cost per correct answer
- Cost per query at different recall thresholds

## Experimental Design

### Experiment 1: Degradation Curves
**Purpose**: Show how systems perform as corpus grows

**Method**:
1. Start with 1,000 passages
2. Incrementally add passages: 1K → 5K → 10K → 20K → 40K
3. For each corpus size:
   - RAG: Test with K=1, 3, 5, 10, 20
   - CAG: Fit as many passages as possible in context window
4. Measure recall for all 4,719 queries

**Expected Results**:
- RAG: Stable recall (slight degradation with larger corpus)
- CAG: Sharp drop when context window fills

### Experiment 2: Latency Comparison
**Purpose**: Compare per-query performance

**Method**:
1. Fixed corpus size (e.g., 10K passages)
2. Run 100 queries (sampled from test set)
3. Measure:
   - RAG: End-to-end latency per query
   - CAG: Generation latency per query
4. Calculate percentiles (p50, p95, p99)

### Experiment 3: Cost Analysis
**Purpose**: Compare total cost at different scales

**Method**:
1. Calculate token usage for both systems
2. Estimate costs at different query volumes (100, 1K, 10K queries)
3. Include upfront costs (RAG: embedding, CAG: context prep)

### Experiment 4: Memory Pressure Test
**Purpose**: Find the breaking point for CAG

**Method**:
1. Start with corpus that fits comfortably in context
2. Gradually increase corpus size
3. Measure recall drop-off point
4. Compare to RAG performance at same corpus size

## Implementation Structure

### Recommended Code Organization

```
benchmark/
├── metrics.py          # Metric calculation functions
├── rag_system.py       # RAG implementation
├── cag_system.py       # CAG implementation
├── evaluator.py        # Main evaluation loop
└── experiments.py      # Experiment configurations

results/
├── degradation_curves/ # Results for different corpus sizes
├── latency/           # Latency measurements
└── cost/              # Cost analysis
```

### Key Functions Needed

1. **Recall Calculation**
   ```python
   def calculate_passage_recall(retrieved_ids, relevant_ids):
       """Calculate passage-level recall"""
       intersection = set(retrieved_ids) & set(relevant_ids)
       return len(intersection) / len(relevant_ids) if relevant_ids else 0
   ```

2. **Query-Level Recall**
   ```python
   def calculate_query_recall(results, ground_truth):
       """Calculate % of queries with at least one relevant passage"""
       successful_queries = sum(
           1 for q_id, retrieved in results.items()
           if any(pid in ground_truth[q_id] for pid in retrieved)
       )
       return successful_queries / len(results)
   ```

3. **Latency Measurement**
   ```python
   def measure_latency(system, query, corpus):
       """Measure end-to-end latency"""
       start = time.time()
       result = system.query(query, corpus)
       latency = time.time() - start
       return result, latency
   ```

## Evaluation Protocol

### Step 1: Data Preparation
1. Load corpus (40,221 passages)
2. Load test queries (4,719 questions with ground truth)
3. Create corpus subsets for degradation experiments

### Step 2: System Setup
1. **RAG**:
   - Embed all passages
   - Build vector index
   - Set up retrieval system
2. **CAG**:
   - Determine context window size
   - Prepare corpus for context injection

### Step 3: Run Experiments
1. For each corpus size:
   - Run all queries through both systems
   - Collect: retrieved passages, latency, token counts
2. Calculate metrics for each query
3. Aggregate results

### Step 4: Analysis
1. Plot degradation curves
2. Compare latency distributions
3. Calculate cost breakdowns
4. Generate summary statistics

## Reporting

### Key Visualizations
1. **Degradation Curve**: Recall vs Corpus Size (RAG vs CAG)
2. **Latency Distribution**: Box plots or histograms
3. **Cost Comparison**: Bar charts for different query volumes
4. **Recall@K Curves**: RAG performance at different K values

### Summary Statistics
- Mean/median recall across all queries
- Latency percentiles (p50, p95, p99)
- Cost per query at different scales
- Break-even points (where RAG becomes better than CAG)

## Best Practices

1. **Reproducibility**
   - Set random seeds
   - Save all configurations
   - Version control all code and data

2. **Fair Comparison**
   - Use same LLM for both systems
   - Use same embedding model for RAG
   - Same prompt templates where applicable

3. **Statistical Rigor**
   - Run multiple trials if possible
   - Report confidence intervals
   - Use appropriate statistical tests

4. **Realistic Settings**
   - Use production-like configurations
   - Test with realistic query patterns
   - Consider actual deployment constraints

