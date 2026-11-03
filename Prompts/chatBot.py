from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama

# Getting the LLMs 
model = ChatOllama(model="qwen3:4b")

# creating chat context
chat_history: list[BaseMessage] = [SystemMessage(content="You are a helpful AI Assistant")]


try:
    while True:
        u_input = str(input("You: ").strip())
        
        if u_input in ['exit' , 'bye']:
            break

        chat_history.append(HumanMessage(content=u_input))

        result = model.invoke(chat_history)
        
        chat_history.append(AIMessage(content=result.content))

        print(f"Agent: {result.content}")
    
    print(chat_history)
except KeyboardInterrupt:
    print("\nExiting.... Bye !!!")

