"""
AmbedkarGPT – A simple RAG (Retrieval Augmented Generation) system.

This script:
1. Loads the speech.txt file
2. Splits it into small chunks
3. Converts chunks into embeddings
4. Stores them in a Chroma vector database
5. Retrieves the most relevant chunks when a user asks a question
6. Uses an LLM (Ollama + Mistral) to answer ONLY from the retrieved context
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM


class AmbedkarRAG:
    def __init__(self, speech_path: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the entire RAG pipeline.

        - speech_path: Path to speech.txt file.
        - persist_directory: Folder where Chroma DB will store embeddings.

        All major components (loader, splitter, embeddings, vectorstore, retriever)
        are created immediately when an object is created.
        """

        self.speech_path = speech_path
        self.persist_directory = persist_directory

        # Will store loaded + processed data
        self.documents = None
        self.docs = None
        self.embeddings = None
        self.vectorstore = None

        # LLM from Ollama (make sure "mistral" is pulled locally)
        self.llm = OllamaLLM(model="mistral")

        # Retriever object
        self.retriever = None

        # Setup complete pipeline automatically
        self._initialize_pipeline()

    # 1. Load the text file
    def load_documents(self):
        """Load speech.txt using LangChain's TextLoader."""
        loader = TextLoader(self.speech_path)
        self.documents = loader.load()

    # 2. Split the text into chunks
    def split_documents(self, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Break large text into smaller chunks so the model can handle them.
        Example:
        - chunk_size=500 → max 500 characters per chunk
        - chunk_overlap=50 → keeps some overlap for better meaning retention
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.docs = splitter.split_documents(self.documents)

    # 3. Convert chunks into embeddings
    def create_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Create sentence embeddings using a HuggingFace model.
        These embeddings allow semantic search.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 4. Store embeddings in Chroma DB
    def create_vectorstore(self):
        """
        Save all chunk embeddings into Chroma, a persistent vector database.
        This allows fast similarity search for any query.
        """
        self.vectorstore = Chroma.from_documents(
            documents=self.docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    # 5. Create Retriever
    def create_retriever(self, k: int = 3):
        """
        Convert vector store into a retriever.
        - k = number of top relevant chunks to fetch for every question.
        - search_type="mmr" helps maximize relevance + diversity.
        """
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k}
        )

    # Pipeline Setup (automatically runs steps 1–5)
    def _initialize_pipeline(self):
        """Run all pipeline steps in order."""
        self.load_documents()
        self.split_documents()
        self.create_embeddings()
        self.create_vectorstore()
        self.create_retriever()

    # 6. RAG Query Function
    def ask(self, query: str):
        """
        Takes user's question, retrieves relevant chunks,
        builds a strict prompt, and generates an answer.
        """

        # Fetch top relevant chunks from vector store
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # A strict prompt to avoid hallucination
        prompt = f"""
You are an AI assistant that must answer ONLY using the information provided
below from speech.txt.

STRICT RULES:
- If the answer IS in the context, answer clearly.
- If the answer is NOT in the context, reply exactly with:
  "The provided context does not contain this information."
- Do NOT use outside knowledge.
- Do NOT assume anything.
- Do NOT add extra details.

Context:
{context}

Question:
{query}

Answer:
"""
        # Ask the LLM to generate answer
        answer = self.llm.invoke(prompt)
        return answer


#                 COMMAND-LINE INTERFACE
if __name__ == "__main__":
    # Initialize RAG system
    rag = AmbedkarRAG(speech_path="speech.txt")

    print("\n🚀 AmbedkarGPT is ready!")
    print("Ask any question related to the speech or type 'exit' to quit.\n")

    # Chat loop
    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        answer = rag.ask(query)
        print("AmbedkarGPT:", answer)
        print("-" * 60)
