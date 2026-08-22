import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "faculty_handbook.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "faculty_handbook"

def ingest_handbook():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Could not find '{PDF_PATH}'. Make sure the PDF is in the project root folder.")

    print(f"📄 Loading document: {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    print("✂️ Splitting document into semantic chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Generated {len(chunks)} total text chunks.")

    print("🔢 Initializing local Hugging Face model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"💾 Storing vectors into local ChromaDB at '{CHROMA_DIR}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print("✅ Ingestion complete! ChromaDB is populated and ready for queries.")

if __name__ == "__main__":
    ingest_handbook()