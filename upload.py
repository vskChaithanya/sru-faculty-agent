import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("1. Loading PDF...")
loader = PyPDFLoader("handbook.pdf")
docs = loader.load()

print(f"2. Splitting {len(docs)} pages...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(splits)}")

print("3. Initializing embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

print("4. Uploading to Pinecone in batches to respect API limits...")
vectorstore = PineconeVectorStore(index_name="sru-handbook", embedding=embeddings)

batch_size = 80
for i in range(0, len(splits), batch_size):
    batch = splits[i:i + batch_size]
    print(f"Uploading chunks {i + 1} to {min(i + batch_size, len(splits))}...")
    vectorstore.add_documents(batch)
    
    # Pause if there are still more chunks left to upload
    if i + batch_size < len(splits):
        print("Sleeping for 60 seconds to clear Google rate limit...")
        time.sleep(60)

print("--> SUCCESS: All chunks uploaded to Pinecone!")