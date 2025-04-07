# 🔍 Potential Talents

**Potential Talents** is an intelligent candidate ranking system designed to streamline the process of identifying top-matching candidates for technical and HR roles. By combining traditional text similarity techniques with modern embeddings and large language models (LLMs), the system helps recruiters surface high-quality candidates quickly and accurately.

> 🌐 Live Gradio Demo (LLM-based): [alizoljodi/Potential_Talents](https://huggingface.co/spaces/alizoljodi/Potential_Talents)

---

## 🚀 Project Objective

Sourcing great candidates is hard — understanding role requirements, interpreting candidate backgrounds, and comparing them fairly requires both domain expertise and time. This project automates and enhances that process by:

- Extracting meaningful features from candidate data
- Comparing candidate job titles with job queries
- Ranking candidates based on contextual fit
- Re-ranking based on human-in-the-loop feedback

---

## 🔧 Features

✅ **Multi-model similarity ranking**  
✅ **Tokenization, vectorization, and semantic embedding**  
✅ **Support for TF-IDF, Word2Vec, FastText, GloVe, Sentence-BERT**  
✅ **LLM-based contextual similarity using LLaMA, Gemma, and Qwen**  
✅ **Re-rankable system with "starred" candidate supervision**  
✅ **Gradio-powered demo for real-time testing**

---

## 🧠 Models Used

### 🔹 Traditional Methods
- **TF-IDF + Cosine Similarity**: Fast but literal
- **Word2Vec / GloVe / FastText**: Dense, general-purpose embeddings

### 🔹 Deep Learning-Based
- **Sentence-BERT (SBERT)**: Captures deep semantic meaning of job titles
- **Large Language Models (LLMs)**:
  - **LLaMA 3.2B & 70B** – lightweight to large contextual understanding
  - **Gemma 3.1B-IT** – instruction-tuned relevance scoring
  - **Qwen-QWQ 32B** – high multilingual and ranking accuracy

---

## 📊 Pipeline Overview

1. **Exploratory Data Analysis**: Job title length, word counts, and token inspection
2. **Embedding Generation**: Using various models to vectorize job titles
3. **Similarity Computation**: Cosine similarity between query and candidate vectors
4. **Ranking + Re-Ranking**: Based on similarity scores and human feedback
5. **LLM Integration**: High-precision semantic scoring and re-ranking

---

## 📦 Requirements

- `pandas`, `numpy`, `scikit-learn`
- `gensim`
- `sentence-transformers`
- `gradio` (for the interactive demo)
- Pretrained LLM access (local or via Groq, HF, etc.)

---

## 🚀 Run the Gradio App

Try the interactive LLM-based candidate ranking demo:

```bash
git clone https://huggingface.co/spaces/alizoljodi/Potential_Talents
cd Potential_Talents
gradio app.py
