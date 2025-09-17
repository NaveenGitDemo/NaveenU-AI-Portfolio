from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()
print("Loaded .env file?", True)

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="test-generation"
)

chat_model = ChatHuggingFace(llm=llm)

def get_memory(session_id: str) -> BaseChatMessageHistory:
    os.makedirs("history", exist_ok=True)
    return FileChatMessageHistory(file_path=f"history/{session_id}.json")

# Notice: no input_messages_key or history_messages_key here
runnable = RunnableWithMessageHistory(
    chat_model,
    get_memory,
)

session_id = "user1"

response1 = runnable.invoke(
    "Tell me about famous Indian cities.",
    config={"configurable": {"session_id": session_id}}
)
print(response1.content)

response2 = runnable.invoke(
    "Make it more summarized city-wise.",
    config={"configurable": {"session_id": session_id}}
)
print(response2.content)
