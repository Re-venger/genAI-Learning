from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
    
# simulating multiline/multidocs embeddings
documents = [
        "The Sun rises in the East",
        "There are moons and stars in the night sky",
        "Chain is the larget EV cars manufacturer in the world"
        ]

# Generate an embedding for multiple documents
vector = embeddings.embed_documents(documents)

print("Embedding dimensions:", len(vector))
print("First 10 values:", vector)
