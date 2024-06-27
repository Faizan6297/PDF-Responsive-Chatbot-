import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from design import css, bot_template, user_template
from openai.error import RateLimitError
from tenacity import retry, stop_after_attempt, wait_fixed

# Define a decorator to retry on RateLimitError
@retry(stop=stop_after_attempt(5), wait=wait_fixed(4))
def embed_with_retry(embeddings, **kwargs):
    return embeddings._embed_with_retry(**kwargs)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def read_pdf_chunks(uploaded_file, chunk_size=8192):
    with BytesIO(uploaded_file.getvalue()) as file:
        pdf_reader = PdfReader(file)
        return [page.extract_text() for page in pdf_reader.pages]

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_pdf_text(pdf_docs):
    text_chunks = [chunk for pdf in pdf_docs for chunk in read_pdf_chunks(pdf)]
    raw_text = ''.join(text_chunks)

    # Move st.alert() or st.write() outside the cached function
    if len(raw_text) > 10000:  # Adjust the threshold as needed
        st.warning("The text in the PDF is too long. Consider uploading a shorter PDF.")
    
    return raw_text

@st.cache(allow_output_mutation=True)
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

@st.cache(allow_output_mutation=True)
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    try:
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except RateLimitError:
        st.error("Rate limit exceeded. Please try again later or check your API plan and billing details.")
        st.stop()
    return vector_store

@st.cache(allow_output_mutation=True)
def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

def response_user(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Get your answer from the book:", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    pdf_docs = st.sidebar.file_uploader(
        "Upload the pdf", accept_multiple_files=True)

    if st.sidebar.button("Submit"):
        with st.spinner("Processing"):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conver = get_conversation_chain(vector_store)
            else:
                st.warning("Please upload a PDF file before submitting.")

    st.header("Get your answer :books:")

    if "conver" in st.session_state and st.session_state.conver:
        user_input = st.text_input("Type your question")

        if user_input:
            with st.spinner("Getting the response"):
                response_user(user_input, st.session_state.conver)
    else:
        st.warning("Submit a PDF file first to start the conversation.")

if __name__ == '__main__':
    main()
