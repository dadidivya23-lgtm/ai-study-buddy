# 📚 AI Study Buddy
### A Personalized RAG Learning Agent Powered by Google Gemini

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.5-orange)

## 📖 Overview
**AI Study Buddy** is an intelligent educational assistant designed to help students learn faster and more efficiently. By leveraging **Retrieval-Augmented Generation (RAG)**, this agent transforms static PDF textbooks, lecture notes, and research papers into interactive, conversational study partners.

Unlike generic chatbots, AI Study Buddy eliminates hallucinations by grounding its answers strictly in the user's uploaded documents. It uses **Local Embeddings** to ensure data privacy and zero API costs for processing, while utilizing **Google Gemini 2.5** for high-level reasoning and summarization.

---

## 🚀 Key Features

* **📄 Instant PDF Analysis:** Upload extensive textbooks or research papers and start asking questions in seconds.
* **💬 Conversational Memory:** Features intelligent chat history, allowing for back-and-forth dialogue and follow-up questions without losing context.
* **🧠 Local Embeddings (Privacy-First):** Uses the `HuggingFace` model (`all-MiniLM-L6-v2`) running locally on the CPU. This ensures user data privacy and prevents hitting API rate limits/quotas.
* **🤖 Advanced Reasoning:** Powered by **Google Gemini 2.5 Flash & Pro**, providing state-of-the-art accuracy for complex academic topics.
* **⚡ Smart Context Retrieval:** Utilizes **FAISS** (Facebook AI Similarity Search) to retrieve only the most relevant text chunks for precise answering.
* **🎯 Zero-Cost Architecture:** Designed to run entirely using free-tier resources.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Frontend:** [Streamlit](https://streamlit.io/) (for the web interface)
* **LLM Orchestration:** [LangChain](https://www.langchain.com/)
* **LLM:** Google Gemini 2.5 (via `google-generativeai`)
* **Vector Database:** FAISS (CPU)
* **Embeddings:** HuggingFace (`sentence-transformers`)
* **PDF Processing:** PyPDF2

---

## ⚙️ How It Works (The RAG Pipeline)

1.  **Ingestion:** The user uploads a PDF. The app uses `PyPDF2` to extract raw text.
2.  **Chunking:** The text is split into smaller, manageable chunks (1000 characters) using a `RecursiveCharacterTextSplitter` to preserve semantic meaning.
3.  **Embedding:** Each chunk is converted into a vector (a list of numbers) using the local HuggingFace model.
4.  **Storage:** These vectors are stored in a transient `FAISS` vector store.
5.  **Retrieval & Generation:**
    * The user asks a question.
    * The system searches the Vector Store for the top 3 most relevant text chunks.
    * These chunks + the user's question + chat history are sent to **Google Gemini**.
    * Gemini generates a coherent answer based *only* on the provided context.

---

## 💻 Installation & Setup

Follow these steps to run the project locally on your machine.

### Prerequisites
* Python 3.10 or 3.11 installed.
* A free Google API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
* Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ai-study-buddy.git](https://github.com/YOUR_USERNAME/ai-study-buddy.git)
