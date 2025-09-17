import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
import os

# Setup session
session_id = "user1"
os.makedirs("history", exist_ok=True)

# Setup model
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-alpha", task="text-generation")
chat_model = ChatHuggingFace(llm=llm)

#from langchain_core.chat_history import ChatMessageHistory

#def get_memory(session_id: str):
#    return ChatMessageHistory()


# History function
def get_memory(session_id):
    return FileChatMessageHistory(file_path=f"history/{session_id}.json")

# Wrap model
runnable = RunnableWithMessageHistory(
    chat_model,
    get_memory,
    input_messages_key="input",
    history_messages_key="history"
)

# Streamlit UI
st.title("🧠 Contextual Chatbot")

user_input = st.text_input("You:", placeholder="Ask me about anything...")

if user_input:
    with st.spinner("Thinking..."):
        response = runnable.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.markdown(f"**Bot:** {response.content}")
