from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, content

from langchain_ollama import ChatOllama


model = ChatOllama(model='llama3.2:latest')

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain')
]

response = model.invoke(messages)
messages.append(AIMessage(content=response.content))

print(messages)
