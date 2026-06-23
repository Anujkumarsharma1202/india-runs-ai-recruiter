IntelliRank AI: Knowledge-Graph Augmented Candidate Ranking
🌟 Overview
IntelliRank AI is a high-performance candidate ranking system designed to move beyond simple keyword matching. By combining Semantic Vector Search, Bipartite Knowledge Graph Skill Expansion, and LLM-powered Reasoning, the system identifies the best fit for a role based on actual capability and professional trajectory, not just resume buzzwords.

🏗️ The 3-Stage Pipeline (Architecture)
To ensure both speed and precision, we implemented a Retrieve-and-Rerank architecture:

Stage 1: Semantic Retrieval (The Wide Net)
- Technology: FAISS IndexFlatIP + `all-MiniLM-L6-v2` embeddings.
- Process: Converts 100,000+ candidate profiles into high-dimensional vectors, persisting the FAISS index and metadata in `outputs/`.
- Goal: Millisecond retrieval of the Top 50 candidates matching the job description semantic vector.

Stage 2: Knowledge Graph Expansion (The Intuition)
- Technology: NetworkX Bipartite Skill Taxonomy.
- Process: Uses a directed bipartite graph inside `src/skill_graph.py` to map specific tools to broader categories and synonyms across 6 major domains (Vector DBs, Deep Learning, ML/Data Science, LLM Ops, GenAI/LLMs, Cloud Infrastructure).
- Programmatic Logic Match: Employs Python regex word-boundary validation to compare profile skills with equivalent nodes, ensuring `graph_match` accurately triggers "Yes" if the candidate lists an equivalent tool to a must-have JD requirement.

Stage 3: Batch LLM Reasoning & Reranking (The Final Judge)
- Technology: Groq LPU → **Llama 3.3 (70B Versatile)**.
- Token Compression: Extracts only key data (Headline, YoE, Skills, and top 3 jobs with truncated 60-character descriptions) to fit within Groq TPM limits.
- Group Batching: Evaluates candidates in batches of 5, reducing API requests 5x (from 50 to 10), speeding up execution while preserving comparison context.
- Paced Dispatch: Paces out requests by 1.5 seconds to bypass burst rate limits.
- Goal: deterministic suitability scoring (0-10) with human-readable recruiter reasons.

🛠️ Tech Stack
- Language: Python 3.10+
- Vector DB: FAISS
- LLM Engine: Groq (`llama-3.3-70b-versatile`)
- Graph Engine: NetworkX
- NLP: Sentence-Transformers, Pandas, Python-Docx

📂 Project Structure
- `data/`: Contains `candidates.jsonl` and `job_description.docx`.
- `outputs/`: Stores `candidates_index.faiss`, `metadata.json`, and the final `final_ranked.csv`.
- `src/`: Core Python modules (`skill_graph.py`, `candidate_ranker.py`, `reranker.py`).

📈 Execution & Performance Results
- Reranking Throughput: Processed 50 FAISS-retrieved candidates in **178.1 seconds** (~2m 58s).
- Rate-Limit Resilience: Automatic exponential backoff + paced request dispatching prevents Groq free-tier RPM and TPM rate limit blocks.
- Accuracy: 70B parameter model ensures highly accurate reasoning, calibration, and zero terminology penalty.

🎨 Phase 3: The Presentation Deck (The "Pitch")
Slide 1: Title Slide
- Title: IntelliRank AI: Solving the Keyword Filter Problem.
- Subtitle: A 3-Stage Semantic & Knowledge-Graph Pipeline for Intelligent Hiring.
- Your Name & Team.

Slide 2: The Problem
- Bullet 1: Keyword filters miss great talent (The "Terminology Gap").
- Bullet 2: Basic embeddings are too "broad" and don't provide reasons.
- Bullet 3: Recruiters need justification, not just a list.

Slide 3: The Solution (The Architecture)
- Process flow: FAISS -> Bipartite Knowledge Graph -> Paced Batch LLM Reranker.
- Key Point: We move from "Similarity" to "Suitability."

Slide 4: Technical Deep Dive
- Stage 1: FAISS for Millisecond Retrieval.
- Stage 2: NetworkX Bipartite Graph for Skill Taxonomy (Expanding "Vector DB" to include "Pinecone/Milvus").
- Stage 3: Llama 3.3 (70B) via Groq for deterministic, batch-pushed scoring.

Slide 5: Results & Impact
- Screenshot: Show the final table with the LLM Score, Graph Match (Yes/No), and Reason columns.
- Impact: Reduced recruiter screening time by providing pre-analyzed justifications.