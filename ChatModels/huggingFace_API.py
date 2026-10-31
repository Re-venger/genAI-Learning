from langchain_ollama import ChatOllama



model = ChatOllama(
        model="tinyllama:latest",
        temperature=0.97
        )

response = model.invoke("What is the capital of India?")
print(response.content)
