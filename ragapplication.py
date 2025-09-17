import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation"
)
chat_model = ChatHuggingFace(llm=llm)

# Embeddings and Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Document indexing (optional: only first time)
if not os.path.exists("./chroma_db/index.sqlite3"):
    docs = [
        "India has many famous cities including Mumbai, Delhi, Bangalore.",
        "Mumbai is the financial capital of India.",
        "Delhi is the capital city of India.",
        "Bangalore is the tech hub of India, also called the Silicon Valley of India.",
    ]
    doc_objs = [Document(page_content=text) for text in docs]
    vectorstore.add_documents(doc_objs)
    vectorstore.persist()

# Memory function
def get_memory(session_id: str) -> BaseChatMessageHistory:
    os.makedirs("history", exist_ok=True)
    return FileChatMessageHistory(file_path=f"history/{session_id}.json")

# RAG pipeline
runnable = RunnableWithMessageHistory(
    chat_model,
    get_memory,
)

# Streamlit UI
st.set_page_config(page_title="🇮🇳 Indian Cities Chatbot", page_icon="🧠")
st.title("🧠 Chat with Indian Cities RAG Bot")

# Unique session id (for now: user1)
session_id = "user1"

# User input
user_input = st.text_input("Ask a question about Indian cities:")

if user_input:
    # Retrieve docs
    results = vectorstore.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in results])

    # RAG prompt
    prompt = f"""Use the following information to answer the question:

{context}

Question: {user_input}
Answer:"""

    with st.spinner("Thinking..."):
        response = runnable.invoke(
            prompt,
            config={"configurable": {"session_id": session_id}}
        )

    st.markdown(" Answer:")
    st.write(response.content)
