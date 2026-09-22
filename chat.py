import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the API key from .env
load_dotenv()

def ask_faculty_bot(query):
    # 1. Load our local embeddings and Chroma database
    print("🔍 Searching the handbook...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="faculty_handbook",
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # Retrieve the top 3 most relevant chunks of text
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0 # Keep it at 0 so it doesn't hallucinate
    )

    # 3. Create the prompt structure
    system_prompt = (
        "You are an official administrative assistant for faculty at SR University. "
        "Use the following pieces of retrieved handbook context to answer the question. "
        "If the answer is not in the context, strictly say 'I do not have that information in the Faculty Handbook.' "
        "Do not make up answers.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Helper function to format the retrieved text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 4. The Modern LCEL RAG Chain (Replaces langchain.chains)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Generate the answer
    print("🤖 Thinking...")
    response = rag_chain.invoke(query)
    
    print("\n--- ANSWER ---")
    print(response)
    print("--------------\n")

if __name__ == "__main__":
    # Test your chatbot here! Replace this with a real question from your handbook.
    test_question = "who is the dean of the faculty?"
    ask_faculty_bot(test_question)