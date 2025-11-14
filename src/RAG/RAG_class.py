from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM 


class AmbedkarRAG:
    def __init__(self, speech_path: str, persist_directory: str = "./chroma_db"):
        self.speech_path = speech_path
        self.persist_directory = persist_directory
        self.documents = None
        self.docs = None
        self.embeddings = None
        self.vectorstore = None
        self.llm = OllamaLLM(model="mistral")
        self.retriever = None
        self._initialize_pipeline()

    def load_documents(self):
        loader = TextLoader(self.speech_path)
        self.documents = loader.load()

    def split_documents(self, chunk_size: int = 500, chunk_overlap: int = 50):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.docs = splitter.split_documents(self.documents)

    def create_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def create_vectorstore(self):
        self.vectorstore = Chroma.from_documents(
            documents=self.docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def create_retriever(self, k: int = 3):
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": k}
        )

    def _initialize_pipeline(self):
        self.load_documents()
        self.split_documents()
        self.create_embeddings()
        self.create_vectorstore()
        self.create_retriever()

    def ask(self, query: str):
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are an AI assistant that must answer ONLY using the information provided in the following context extracted from speech.txt.

STRICT RULES:
- If the answer is present in the context, provide it clearly.
- If the answer is NOT present in the context, reply strictly with:
  "The provided context does not contain this information."
- Do NOT use prior knowledge.
- Do NOT make assumptions.
- Do NOT add extra details not found in the context.

Context (from speech.txt):
{context}

Question:
{query}

Answer:
"""

        
        answer = self.llm.invoke(prompt)
        return answer


if __name__ == "__main__":
    rag = AmbedkarRAG(speech_path="speech.txt")
    print("\n AmbedkarGPT is ready. Type your question or 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = rag.ask(query)
        print("AmbedkarGPT:", answer)
        print("-" * 60)


