from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

success = load_dotenv()
print("Loaded .env file?", success)


llm = HuggingFaceEndpoint(
 repo_id="HuggingFaceH4/zephyr-7b-alpha",
 task="test-generation"
)
model = ChatHuggingFace(llm=llm)


result = model.invoke("who won 2011 cricket world cup ?")
print(result.content)