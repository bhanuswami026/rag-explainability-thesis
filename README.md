# RAG Explainability & Interpretability Workbench

🎓 **M.Tech Thesis Demo Project**: *“Explainability of RAG Systems using post-hoc interpretability methods.”*

This repository contains a complete, local, modular Python project demonstrating **post-hoc interpretability and explainability methods** applied to Retrieval-Augmented Generation (RAG) pipelines. It integrates local dense retrieval (BGE-small embeddings + FAISS) with generative answer synthesis (Google Gemini API) and exposes three powerful, academic-grade post-hoc explanations on a high-fidelity Streamlit dashboard.

---

## 🔬 Core Explainability Methodologies

To address the "black-box" nature of both dense vector search and large language models (LLMs), this project implements three complementary post-hoc interpretability techniques:

### 1. Semantic Similarity & Retrieval Attention Mapping
* **What it does**: Computes the embedding space cosine similarity between the user's query and the retrieved passages.
* **Academic Value**: We map raw similarity scores into a **Retrieval Attention Probability Distribution** using a Softmax normalization function:
  $$A_i = \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}$$
  Where $s_i$ is the cosine similarity score of chunk $i$, and $\tau$ is the temperature hyperparameter (default $\tau = 0.05$). This highlights the prioritized attention weight of each chunk before prompt construction.

### 2. Causal Chunk Occlusion (Counterfactual Perturbation)
* **What it does**: A causal perturbation interpretability method. We systematically withhold one retrieved chunk at a time from the prompt, re-run Gemini API generation, and measure the semantic and lexical shift of the new output compared to the original.
* **Academic Value**: Quantifies the causal necessity of each context chunk. The semantic shift is measured using the cosine distance ($1 - \text{CosineSimilarity}$) between the original and perturbed responses:
  $$CausalNecessity(C_i) = 1 - \text{CosineSimilarity}(\text{Embed}(R_{all}), \text{Embed}(R_{-C_i}))$$
  A high shift indicates the omitted chunk was highly necessary for synthesizing the generated answer.

### 3. Response Saliency Mapping
* **What it does**: Tokenizes the generated answer and retrieved source chunks, filters out English stop words, and highlights identical keywords/phrases in the source text.
* **Academic Value**: Provides a visual lexical alignment, showing exactly which parts of the retrieved documents were directly copied or heavily paraphrased by the LLM generator.

---

## 📂 Project Directory Structure

```text
rag-explainability-thesis/
├── .env                  # API Key configurations (user-supplied)
├── .env.example          # Template for environment variables
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation and running guide
├── run.sh                # Executable startup bash script
├── documents/            # Target directory for uploaded PDF documents
├── app/
│   ├── __init__.py
│   ├── main.py            # Streamlit dashboard entrypoint
│   └── components.py      # Custom glassmorphism UI components & premium styling
├── rag/
│   ├── __init__.py
│   ├── parser.py          # PyMuPDF PDF page parser and text extractor
│   ├── chunker.py         # Text chunker with metadata tracking
│   ├── embedder.py        # HuggingFace BGE-small embedding model wrapper
│   ├── vector_store.py    # Local FAISS index database manager
│   └── pipeline.py        # Combined RAG pipeline orchestrator
├── explainability/
│   ├── __init__.py
│   ├── similarity.py      # Cosine similarity and Softmax attention metrics
│   ├── occlusion.py       # Counterfactual perturbation occlusion engine
│   └── saliency.py        # Keyword/token-level alignment saliency mapper
└── evaluation/
    ├── __init__.py
    └── metrics.py         # RAG Triad (Faithfulness, Relevance, Latency profile)
```

---

## 🛠️ Getting Started (Local Setup)

### Prerequisites
* Python 3.9 or higher (Tested on macOS with Python 3.9.6)
* Google Gemini API Key (Obtain one for free from [Google AI Studio](https://aistudio.google.com/))

### Installation Steps

1. **Clone or navigate** to the project directory:
   ```bash
   cd /Users/bhanuipad/Documents/rag-explainability-thesis
   ```

2. **Configure Environment Variables**:
   Open the `.env` file and paste your Gemini API Key:
   ```env
   GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
   ```
   *(Note: You can also input the key directly inside the dashboard sidebar at runtime if you prefer.)*

3. **Make the Startup Script Executable**:
   ```bash
   chmod +x run.sh
   ```

4. **Launch the Application**:
   Simply run the bash script. It will automatically activate the Python virtual environment and launch Streamlit:
   ```bash
   ./run.sh
   ```

5. **Access the Dashboard**:
   Once launched, open your web browser and navigate to:
   ```text
   http://localhost:8501
   ```

---

## 🎓 Academic Evaluation Verifiers

The **Academic Evaluation** tab on the dashboard computes quantitative metrics representing the "RAG Triad":
1. **Context Relevance**: Evaluates how well vector search mapped chunks to the user request (average cosine similarity).
2. **Groundedness (Faithfulness)**: Measures if the synthesized response sentences are semantically backed by the retrieved context chunks (sentence embedding alignments). Helps verify against **hallucinations**.
3. **Answer Relevance**: Measures the semantic similarity between the user's original query and the final response.
4. **Latency Profiling**: Profiles the execution times of vector retrieval on CPU vs remote Gemini generation.
