from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="faculty_handbook",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Perform a raw search
query = "who is the dean of academics"
results = vector_store.similarity_search(query, k=3)

# 3. Print the raw results
print(f"Found {len(results)} chunks of text.\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print("-------------------\n")