import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # Added the text filter

load_dotenv()

st.set_page_config(page_title="SRU Faculty Bot", page_icon="🎓", layout="centered")
st.title("SRU Faculty Assistant 🎓")
st.caption("Grounded strictly on the Official Faculty Handbook with source citations.")

@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="faculty_handbook",
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an official administrative assistant for faculty at SR University. "
            "Answer the question strictly using the provided context. "
            "If the answer is not present, respond: 'I do not have that information in the Faculty Handbook.' "
            "Do not fabricate details.\n\n"
            "Context:\n{context}"
        )),
        ("human", "{input}"),
    ])
    return retriever, llm, prompt

retriever, llm, prompt = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 View Source Excerpts"):
                for src in msg["sources"]:
                    st.caption(f"**Page {src['page']}:** {src['content']}")

if user_input := st.chat_input("Ask a question about faculty policies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching handbook context..."):
            docs = retriever.invoke(user_input)
            context_str = "\n\n".join(d.page_content for d in docs)
            
            # THE FIX: Added the StrOutputParser() to the chain to strip out the junk
            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"context": context_str, "input": user_input})
            
            st.markdown(response)
            
            sources = []
            if "I do not have that information" not in response:
                with st.expander("📌 View Source Excerpts"):
                    for d in docs:
                        page_num = d.metadata.get("page", "Unknown")
                        clean_text = d.page_content.strip()[:200] + "..."
                        st.caption(f"**Page {page_num}:** {clean_text}")
                        sources.append({"page": page_num, "content": clean_text})

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })