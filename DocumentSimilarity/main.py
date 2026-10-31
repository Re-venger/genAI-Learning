from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 



# create embedding model 
embedding = OllamaEmbeddings(model="nomic-embed-text", dimensions=300)

# my psuedo document
cricketers = [
    "Virat Kohli is one of India's most famous cricketers, known for his aggressive batting, excellent fitness, and remarkable performances in international cricket.",
    "Rohit Sharma is a talented Indian cricketer famous for his elegant batting, powerful sixes, and record-breaking performances as an opening batsman.",
    "MS Dhoni is a legendary Indian cricketer known for his calm leadership, outstanding wicketkeeping, and guiding India to major international cricket victories.",
    "Sachin Tendulkar is one of the greatest cricketers in history, admired for his extraordinary batting skills and his incredible contribution to Indian cricket.",
    "Jasprit Bumrah is a world-famous Indian fast bowler known for his unique bowling action, accurate yorkers, and ability to take important wickets."
]

User_query = str(input("Query: "))


doc_embeds = embedding.embed_documents(cricketers)
uqry_embeds = embedding.embed_query(User_query)

sim_scores = cosine_similarity([uqry_embeds],doc_embeds)[0]

index, score = sorted(list(enumerate(sim_scores)), key=lambda x:x[1])[-1]

print(f"User: {User_query}")
print(f"Response: {cricketers[index]}")
print(f"Similarity Score : {score}")
