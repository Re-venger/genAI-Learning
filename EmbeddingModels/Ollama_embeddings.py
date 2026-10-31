from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    dimensions=32
)

# Generate an embedding for one text
vector = embeddings.embed_query(
    "What is the capital of India?"
)

print("Embedding dimensions:", len(vector))
print("First 10 values:", vector)
