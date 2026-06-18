IntelliRank AI: Knowledge-Graph Augmented Candidate Ranking
🌟 Overview
IntelliRank AI is a high-performance candidate ranking system designed to move beyond simple keyword matching. By combining Semantic Vector Search, Knowledge Graph Skill Expansion, and LLM-powered Reasoning, the system identifies the best fit for a role based on actual capability and professional trajectory, not just resume buzzwords.

🏗️ The 3-Stage Pipeline (Architecture)
To ensure both speed and precision, we implemented a Retrieve-and-Rerank architecture:

Stage 1: Semantic Retrieval (The Wide Net)
Technology: FAISS + all-MiniLM-L6-v2 embeddings.
Process: Converts 100,000+ candidate profiles into high-dimensional vectors.
Goal: Quickly prune the dataset from 100k to the Top 50 most semantically similar candidates.
Stage 2: Knowledge Graph Expansion (The Intuition)
Technology: NetworkX Skill Taxonomy.
Process: Uses a directed graph to map specific tools to broader categories (e.g., Pinecone $\rightarrow$ Vector DB $\rightarrow$ Embeddings).
Goal: Identify "Hidden Gems"—candidates who possess the required capabilities but use different terminology than the JD.
Stage 3: LLM Reasoning & Reranking (The Final Judge)
Technology: Groq LPU $\rightarrow$ Llama 3.1 (8B).
Process:
Decomposes the JD into a structured Requirement Rubric (Must-haves vs. Nice-to-haves).
Performs a deep-dive analysis of the Top 50 candidates against this rubric.
Generates a final score (0-10) and a human-readable justification.
Goal: Provide a shortlist that a recruiter can trust, complete with reasoning.
🛠️ Tech Stack
Language: Python 3.10+
Vector DB: FAISS
LLM Engine: Groq (Llama 3.1)
Graph Engine: NetworkX
NLP: Sentence-Transformers, Pandas, Python-Docx
📈 Results
Throughput: Processed 100k candidates to Top 50 in < 1 minute.
Reliability: Implemented exponential backoff for API rate-limiting.
Accuracy: Shifted from a flat similarity distribution to a highly differentiated ranking score.
🎨 Phase 3: The Presentation Deck (The "Pitch")
You need to convert a PPT to PDF. Since I can't make the PPT for you, I will give you the exact content for each slide.

Slide 1: Title Slide

Title: IntelliRank AI: Solving the Keyword Filter Problem.
Subtitle: A 3-Stage Semantic & Knowledge-Graph Pipeline for Intelligent Hiring.
Your Name & Team.
Slide 2: The Problem

Bullet 1: Keyword filters miss great talent (The "Terminology Gap").
Bullet 2: Basic embeddings are too "broad" and don't provide reasons.
Bullet 3: Recruiters need justification, not just a list.
Slide 3: The Solution (The Architecture)

(Insert a diagram here: FAISS $\rightarrow$ Knowledge Graph $\rightarrow$ LLM Reranker).
Key Point: We move from "Similarity" to "Suitability."
Slide 4: Technical Deep Dive

Stage 1: FAISS for Millisecond Retrieval.
Stage 2: NetworkX for Skill Taxonomy (Expanding "Vector DB" to include "Pinecone/Milvus").
Stage 3: Llama 3 via Groq for deterministic, rubric-based scoring.
Slide 5: Results & Impact

Screenshot: Show the final table with the LLM Score and Reason columns.
Impact: Reduced recruiter screening time by providing pre-analyzed justifications.