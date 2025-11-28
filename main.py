import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# 1. SETUP THE PAGE
st.set_page_config(page_title="My Personal Study Assistant", layout="wide")
st.header("📚 AI Study Buddy")

# 2. SIDEBAR: SETTINGS
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter Google API Key:", type="password")
    st.markdown("[Get your key here](https://aistudio.google.com/app/apikey)")
    
    # Model Selector
    chat_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash"
    ]
    selected_model = st.selectbox("Choose AI Model:", chat_models, index=0)

    uploaded_file = st.file_uploader("Upload your Study Material (PDF)", type="pdf")

# 3. INITIALIZE SESSION STATE (MEMORY)
# This keeps the chat history even when you click buttons
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. MAIN LOGIC
if api_key and uploaded_file:
    try:
        # Process PDF only once (using session state to avoid reloading)
        if "vector_store" not in st.session_state:
            with st.spinner("Analyzing PDF... (This happens only once)"):
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                st.success("PDF Loaded Successfully!")

        # 5. DISPLAY CHAT HISTORY
        # Loop through memory and show old messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 6. CHAT INPUT (THE NEW WAY)
        # This shows a proper chat box at the bottom
        if prompt := st.chat_input("Ask a question about your notes..."):
            
            # Show User Message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add to memory
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate Answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs = st.session_state.vector_store.similarity_search(prompt)
                    llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    response = chain.run(input_documents=docs, question=prompt)
                    st.markdown(response)
            
            # Add AI Answer to memory
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif not api_key:
    st.warning("Please enter your Google API Key in the sidebar to start.")