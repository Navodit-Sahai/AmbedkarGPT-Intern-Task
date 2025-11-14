# AmbedkarGPT 🚀

**AI Intern Assignment - Kalpit Pvt Ltd, UK**

A Retrieval Augmented Generation (RAG) system that answers questions based on Dr. B.R. Ambedkar's "Annihilation of Caste" speech using local LLMs with **zero API costs**.

Repository: [https://github.com/Navodit-Sahai/AmbedkarGPT-Intern-Task](https://github.com/Navodit-Sahai/AmbedkarGPT-Intern-Task)

## 📋 Overview

AmbedkarGPT is a command-line Q&A system that implements a complete RAG pipeline:
- ✅ Loads and processes text from Dr. Ambedkar's speech
- ✅ Splits text into manageable chunks
- ✅ Creates semantic embeddings using **sentence-transformers/all-MiniLM-L6-v2**
- ✅ Stores embeddings in **ChromaDB** vector database (local, persistent)
- ✅ Retrieves relevant context using similarity search
- ✅ Generates accurate answers using **Mistral 7B via Ollama**
- ✅ **Strictly answers only from provided context** - no hallucinations!

## 🎯 Assignment Requirements Met

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Python 3.8+ | ✅ | Compatible with Python 3.8+ |
| LangChain Framework | ✅ | Core orchestration using LangChain |
| ChromaDB Vector Store | ✅ | Local persistent storage |
| HuggingFace Embeddings | ✅ | sentence-transformers/all-MiniLM-L6-v2 |
| Ollama + Mistral 7B | ✅ | 100% free, no API keys |
| Well-commented Code | ✅ | Detailed docstrings and comments |
| requirements.txt | ✅ | All dependencies listed |
| README.md | ✅ | Complete setup guide |
| speech.txt | ✅ | Provided text included |

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

### 1. Python 3.8+
```bash
python --version  # Should be 3.8 or higher
```

### 2. Ollama
Ollama is required to run the Mistral model locally.

#### Installation on Linux/macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Installation on Windows:
Download and install from [https://ollama.com/download](https://ollama.com/download)

#### Verify Ollama Installation:
```bash
ollama --version
```

### 3. Pull the Mistral Model
After installing Ollama, pull the Mistral model:
```bash
ollama pull mistral
```

This will download the Mistral LLM (approximately 4GB). Wait for the download to complete.

#### Verify Model Installation:
```bash
ollama list
```
You should see `mistral` in the list of available models.

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Navodit-Sahai/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `langchain-core` - Core LangChain framework
- `langchain-community` - Community integrations
- `chromadb` - Vector database for embeddings
- `langchain-groq` - Groq integration
- `sentence-transformers` - Embedding models
- `langchain-ollama` - Ollama integration
- `ollama` - Ollama Python client

## 📁 Project Structure

```
.
├── src/
│   └── RAG/
│       ├── __init__.py
│       └── RAG_class.py      # Main RAG implementation
├── speech.txt                 # Input text file
├── requirements.txt           # Python dependencies
├── chroma_db/                # Vector database (auto-created)
└── README.md                 # This file
```

## 🚀 Usage

### Running the Application

1. **Ensure Ollama is Running**
   ```bash
   # Start Ollama service (if not already running)
   ollama serve
   ```
   
   Leave this terminal open and open a new terminal for the next steps.

2. **Run AmbedkarGPT**
   
   From the project root directory, run:
   ```bash
   python src/RAG/RAG_class.py
   ```
   
   **Note**: The command must be run from the project root, not from inside the `src/RAG` folder, so that the relative path to `speech.txt` works correctly.

### Example Interaction

```
🚀 AmbedkarGPT is ready!
Ask any question related to the speech or type 'exit' to quit.

You: What is the real remedy according to the speech?
AmbedkarGPT: The real remedy is to destroy the belief in the sanctity of the shastras...

------------------------------------------------------------
You: What is the problem with social reform?
AmbedkarGPT: The problem with social reform is that it's like a gardener constantly pruning leaves and branches without attacking the roots...

------------------------------------------------------------
You: exit
👋 Goodbye!
```

## 🔧 Configuration

### RAG Pipeline Components

The system uses the following technologies (all **100% free, no API keys required**):

- **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Lightweight and fast (~90MB)
  - Produces 384-dimensional embeddings
  - Excellent for semantic similarity search
  - Downloaded automatically on first run from HuggingFace

- **Vector Database**: ChromaDB
  - Stores embeddings locally in `chroma_db/` folder
  - Persistent storage (data survives restarts)
  - Fast similarity search using cosine distance

- **LLM**: Mistral 7B via Ollama
  - Runs completely locally
  - No internet required after initial download
  - Excellent instruction following
  - ~4GB model size

### Adjusting RAG Parameters

You can customize the RAG system by modifying parameters in `RAG_class.py`:

```python
# In __init__ method
rag = AmbedkarRAG(
    speech_path="speech.txt",           # Path to your text file
    persist_directory="./chroma_db"     # Vector DB storage location
)

# Chunk size for text splitting
def split_documents(self, chunk_size=200, chunk_overlap=50):
    # Adjust chunk_size and chunk_overlap as needed
    pass

# Number of relevant chunks to retrieve
def create_retriever(self, k=3):
    # Adjust k to retrieve more/fewer chunks
    pass
```

### Using a Different Embedding Model

```python
# In create_embeddings method, change the model_name:
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # More accurate but slower
)
```

### Using a Different LLM

```python
# In __init__ method, change the model:
self.llm = OllamaLLM(model="llama2")  # or "codellama", "phi", etc.
```

Make sure to pull the model first:
```bash
ollama pull llama2
```

## 🔍 How It Works

### The RAG Pipeline (5 Steps):

1. **Document Loading**: 
   - Uses `TextLoader` from LangChain to load `speech.txt`
   - Reads the entire text into memory

2. **Text Splitting**: 
   - `RecursiveCharacterTextSplitter` breaks text into chunks
   - Default: 200 characters per chunk with 50 character overlap
   - Overlap preserves context across chunk boundaries

3. **Embedding Creation**: 
   - Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2`
   - Converts each text chunk into a 384-dimensional vector
   - These vectors capture semantic meaning

4. **Vector Storage**: 
   - Stores all embeddings in ChromaDB (local SQLite database)
   - Creates an index for fast similarity search
   - Persists to disk in `chroma_db/` folder

5. **Query Processing**:
   - When you ask a question:
     - Your question is converted to an embedding (same model)
     - ChromaDB finds the top K most similar chunks (default K=3)
     - Retrieved chunks are combined into context
     - Context + question + strict prompt → sent to Mistral LLM
     - LLM generates answer **only from the provided context**

### Why This Approach Works:

- **No Hallucinations**: Strict prompting ensures answers come only from retrieved context
- **Semantic Search**: Vector embeddings find conceptually similar text, not just keyword matches
- **Scalable**: Works with documents of any size (just add more chunks)
- **Local & Private**: Everything runs on your machine, no data sent to cloud services

## 🐛 Troubleshooting

### Ollama Connection Error
```
Error: Failed to connect to Ollama
```
**Solution**: Ensure Ollama service is running:
```bash
ollama serve
```

### Model Not Found
```
Error: model 'mistral' not found
```
**Solution**: Pull the Mistral model:
```bash
ollama pull mistral
```

### ChromaDB Permission Error
```
Error: Permission denied when creating chroma_db
```
**Solution**: Ensure you have write permissions in the directory or change `persist_directory` path.

### Import Errors
```
ModuleNotFoundError: No module named 'langchain_core'
```
**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### Slow First Run
The first run will be slower as it:
- Downloads the embedding model `sentence-transformers/all-MiniLM-L6-v2` (~90MB) from HuggingFace
- Creates embeddings for all text chunks
- Builds and persists the ChromaDB vector database
- **Subsequent runs will be much faster** as the database is already built!

## 📝 Adding Your Own Documents

1. Replace or modify `speech.txt` with your own text content
2. Update the path in the code if needed:
   ```python
   rag = AmbedkarRAG(speech_path="your_document.txt")
   ```
3. Delete the `chroma_db/` folder to rebuild the vector database
4. Run the script again

## 🎓 Assignment Context

This project was developed as part of the **AI Intern Hiring Assignment for Kalpit Pvt Ltd, UK**. The assignment required building a functional RAG prototype demonstrating understanding of:

- LangChain framework orchestration
- Vector embeddings and semantic search
- Local LLM integration
- RAG pipeline architecture
- Clean, documented code

**Key Achievement**: 100% free, locally-running system with no API keys, no accounts, and no recurring costs.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dr. B.R. Ambedkar for his profound speeches and writings
- LangChain for the RAG framework
- Ollama for making local LLMs accessible
- ChromaDB for efficient vector storage

## 📧 Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Built with ❤️ for preserving and making accessible the wisdom of Dr. B.R. Ambedkar**
