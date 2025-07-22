# OllamaRAG: A Full-Stack Retrieval-Augmented Generation System

OllamaRAG is a full-stack Retrieval-Augmented Generation (RAG) system that integrates a **Streamlit frontend**, a **FastAPI backend**, and **Ollama's LLM** for generating responses. The system also includes evaluation capabilities using **DeepEval** to assess the quality of the RAG pipeline.

# Demo
[streamlit-main-2025-04-30-22-04-34.webm](https://github.com/user-attachments/assets/cae75225-61fe-4164-9a57-53038a2398ac)

## 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 2. Running the System

```bash
# starting ollama server
ollama serve

# pulling llm model
ollama pull llama3

# Basic usage
python run.py

# With a custom model
python run.py --model-name llama3
```

## 3. Accessing the System

- **Frontend**: [http://localhost:8501](http://localhost:8501)
- **API**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)


## 4. Evaluating the RAG Pipeline

### Using Ollama Models

To evaluate the RAG pipeline with Ollama models, use the following commands:

```bash
# Set the judge LLM
deepeval set-ollama llama3:8b

# Unset the judge LLM
deepeval unset-ollama
```

### Using Gemini Models

To evaluate with Gemini models, run:

```bash
deepeval set-gemini \
    --model-name=<model_name> \ # e.g., "gemini-2.0-flash-001"
    --google-api-key=<api_key>
```


## System Components

### 1. **Streamlit Frontend**
   - Provides an interactive interface for users to upload documents, ask questions, and view responses.
   - Displays document metadata and retrieval sources.

### 2. **FastAPI Backend**
   - Handles document processing, retrieval, and LLM integration.
   - Provides endpoints for uploading documents, querying, and managing the system.

### 3. **Retriever**
   - Uses ChromaDB for semantic search and retrieval.
   - Supports document chunking and hybrid search with reranking.

### 4. **LLM Integration**
   - Leverages Ollama's LLM for generating responses based on retrieved documents.
   - Supports configurable parameters like `temperature` and `top_p`.

### 5. **Evaluation with DeepEval**
   - Benchmarks the RAG pipeline using llm-based metrics like:
     - **Answer Relevancy**
     - **Faithfulness**
     - **Contextual Precision**
     - **Contextual Recall**



## Example Workflow

1. **Upload Documents**: Use the Streamlit interface to upload PDF, TXT, or DOCX files.
2. **Ask Questions**: Enter a query in the chat interface.
3. **View Responses**: See the generated answer along with the retrieved document sources.
4. **Evaluate**: Use DeepEval to benchmark the system's performance.

## Future Improvements

- Add support for more document formats.
- Optimize retrieval and generation for larger datasets.
- Extend evaluation metrics for more comprehensive benchmarking.


## License

This project is licensed under the MIT License.

