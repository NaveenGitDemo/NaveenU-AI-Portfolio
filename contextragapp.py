from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
print("✅ .env file loaded:", True)

# Initialize the HuggingFace LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation"  # ✅ correct task (not test-generation)
)

chat_model = ChatHuggingFace(llm=llm)

# Initialize embeddings model with explicit model name
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store (persistent)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Add documents (run only once to avoid duplicates)
if not os.path.exists("./chroma_db/index.sqlite3"):
    print("📄 Adding documents to vectorstore...")
    docs = [
        "India has many famous cities including Mumbai, Delhi, Bangalore.",
        "Mumbai is the financial capital of India.",
        "Delhi is the capital city of India.",
        "Bangalore is the tech hub of India, also called the Silicon Valley of India.",
    ]
    doc_objs = [Document(page_content=text) for text in docs]
    vectorstore.add_documents(doc_objs)
    vectorstore.persist()
    print("✅ Documents added and persisted.")

# User query
query = "Tell me about famous Indian cities."

# Search relevant documents
results = vectorstore.similarity_search(query, k=3)

# Create context from retrieved documents
context = "\n".join([doc.page_content for doc in results])

# Construct RAG prompt
prompt = f"""Use the following information to answer the question:

{context}

Question: {query}
Answer:"""

# Setup memory per user session
def get_memory(session_id: str) -> BaseChatMessageHistory:
    os.makedirs("history", exist_ok=True)
    return FileChatMessageHistory(file_path=f"history/{session_id}.json")

# Create runnable with memory
runnable = RunnableWithMessageHistory(
    chat_model,
    get_memory,
)

# Unique session for chat history
session_id = "user1"

# Invoke LLM with RAG prompt
response = runnable.invoke(
    prompt,
    config={"configurable": {"session_id": session_id}}
)

# Print final response
print("\n🤖 Chatbot Response:\n", response.content)
