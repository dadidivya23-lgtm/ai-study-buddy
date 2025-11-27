import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# 1. SETUP THE PAGE
st.set_page_config(page_title="My Personal Study Assistant")
st.header("📚 AI Study Buddy")

# 2. SIDEBAR: SETTINGS
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Google API Key:", type="password")
    st.markdown("[Get your key here](https://aistudio.google.com/app/apikey)")
    
    # --- MANUAL MODEL SELECTOR (UPDATED FOR NOV 2025) ---
    # Gemini 1.5 is retired. We now use Gemini 2.0 and 2.5.
    chat_models = [
        "gemini-2.5-flash",       # Fastest, Newest
        "gemini-2.5-pro",         # Smartest
        "gemini-2.0-flash",       # Stable backup
        "gemini-2.0-flash-lite"   # Extremely fast
    ]
    selected_model = st.selectbox("Choose AI Model:", chat_models, index=0)
    # ----------------------------------------------------

    uploaded_file = st.file_uploader("Upload your Study Material (PDF)", type="pdf")

# 3. MAIN LOGIC
if api_key and uploaded_file:
    try:
        # Read the PDF file
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into small chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Create a "Search Engine" using LOCAL CPU
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # 4. CHAT INTERFACE
        user_question = st.text_input("Ask a question about your notes:")

        if user_question:
            # Search your PDF for relevant text
            docs = vector_store.similarity_search(user_question)
            
            # Initialize the Gemini Model
            llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)
            
            # Create the answer chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Get the answer
            with st.spinner(f"Thinking using {selected_model}..."):
                response = chain.run(input_documents=docs, question=user_question)
                
            st.write(response)
            st.success("Answer generated!")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

elif not api_key:
    st.warning("Please enter your Google API Key in the sidebar to start.")