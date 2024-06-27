import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from design import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
         text += page.extract_text()
    return text
         
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
       separator="\n",
       chunk_size=1000,
       chunk_overlap=200,
       length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_con_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain
def response_user(user_question):
    response = st.session_state.conver({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(
    page_title="Get your answer from the book:", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conver" not in st.session_state:
       st.session_state.conver = None
    if "chat_histroy" not in st.session_state:
       st.session_state.chat_histry = None
       
    
    with st.sidebar:
        st.subheader("Pdf files")
        pdf_docs = st.file_uploader(
            "Upload the pdf", accept_multiple_files=True)



        if st.button("Submit"):
          with st.spinner("processing"):
            #get the text
            raw_text= get_pdf_text(pdf_docs)
            
            #get chunks
            text_chunks = get_text_chunks(raw_text)
           
            #create vector
            vector_store = get_vector_store(text_chunks)

            #conversation chain
            st.session_state.conver = get_con_chain(vector_store)
        st.header("Get your answer :books:")
        user_input = st.text_input("Type your question")
    if user_input:
         with st.spinner("Getting the response"):
          response_user(user_input)


if __name__ == '__main__':
    main()
