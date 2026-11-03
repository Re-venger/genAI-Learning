from langchain_ollama import ChatOllama

# here I am using Ollama running locally on my system
# for LLM - Tinyllama:lastest
# for embeddings - nomic-embed-text


# temprature here how stale or how creative and random the output should be
model = ChatOllama(model="tinyllama:latest", temperature=0)

response = model.invoke("Writ a poem on cricket only 5 lines")


print(response.content)
