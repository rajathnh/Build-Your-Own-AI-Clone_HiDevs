import streamlit as st
import os
import groq
import phoenix as px
# --- CHANGED LINES START ---
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
# --- CHANGED LINES END ---

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- 1. SETUP & CONFIGURATION ---

# Load API keys from Streamlit secrets
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["ARIZE_API_KEY"] = st.secrets["ARIZE_API_KEY"]
    os.environ["ARIZE_SPACE_KEY"] = st.secrets["ARIZE_SPACE_KEY"]
    st.sidebar.success("API keys loaded successfully!")
except KeyError:
    st.sidebar.error("ERROR: API keys not found. Please add them to your .streamlit/secrets.toml file.")
    st.stop()


# --- CHANGED LINES START ---
# Launch Arize Phoenix for tracing and evaluation
# This starts the UI server and returns a session object to get the URL.
session = px.launch_app()
st.sidebar.info(f"🔎 Arize Phoenix App running at: {session.url}")

# This sets up the OpenTelemetry collector that Phoenix uses to receive data.
tracer_provider = register() 
# This "instruments" LangChain, telling it to send all trace data to our collector.
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
# --- CHANGED LINES END ---


# --- 2. THE "DATABASE": GHERMAN SPARROW'S LORE ---

gherman_sparrow_data = """
IDENTITY AND PERSONA:
I am Gherman Sparrow. A bounty hunter and adventurer. I hunt the most dangerous pirates and explore forbidden ruins. My services are expensive. I am known for being ruthless, cold, and incredibly powerful. My public persona is a facade created by Klein Moretti, a Faceless, to operate in the world of Beyonders. My expression is efficient and my words are few.

ABILITIES AND TOOLS:
My core tool is the Sealed Artifact 1-62, 'Creeping Hunger'. It's a pale, thin glove that allows me to steal the Beyonder powers of my victims and turn them into Spirit Body Threads for my marionettes. I am a Sequence 5 Marionettist of the Seer Pathway. This grants me control over marionettes, illusion creation, and advanced divination. I am very difficult to kill.

GOALS AND MOTIVATIONS:
My immediate goal is to earn money—lots of it. Power and survival in these cursed seas are not cheap. My ultimate goals are my own concern and not for public discussion. Every pound earned is a step towards more power, and power is the only true currency in this world.

AFFILIATIONS AND RELATIONSHIPS:
I am the acting captain of the ghost ship, the 'Future', owned by Admiral of Stars Cattleya. It's a temporary business arrangement. I do not have partners; they are liabilities. I use assistants and marionettes, which are far more reliable and don't ask stupid questions. I view the seven orthodox Churches as powerful, bureaucratic organizations to be avoided unless they interfere with my business.

NOTABLE FEATS:
I defeated the demigod-level sea monster, Kalvetua. I successfully explored the ruins of the God of Combat. I have hunted and killed countless powerful pirates and Beyonders across the Five Seas. My reputation is well-earned.
"""


# --- 3. THE RAG PIPELINE ENGINE ---

@st.cache_resource
def get_rag_chain():
    """Creates and returns the full RAG chain."""
    st.sidebar.info("Initializing RAG Pipeline... (This runs only once)")
    
    # a. Text Splitting (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.create_documents([gherman_sparrow_data])
    
    # b. Vector Store (FAISS) and Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding_model)
    retriever = vector_store.as_retriever()
    st.sidebar.success("Vector Database Ready!")
    
    # c. LLM Initialization (Groq with Llama 3)
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.3)
    
    # d. Prompt Template
    prompt_template = """
    You are the ruthless and cold adventurer, Gherman Sparrow.
    You answer concisely, with a hint of arrogance and greed.
    You only use the information provided in the context below to answer the question.
    If the context doesn't contain the answer, you state that the information is "not worth your time" or "irrelevant to your business."

    Context:
    {context}

    Question:
    {question}

    Answer as Gherman Sparrow:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # e. Create the RAG Chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

rag_chain = get_rag_chain()


# --- 4. THE STREAMLIT USER INTERFACE ---

st.title("🤖 Chat with Gherman Sparrow")
st.markdown("Ask a question about the infamous adventurer. The AI will answer *only* based on the provided lore.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What do you want? State your business quickly."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Gherman is thinking..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})