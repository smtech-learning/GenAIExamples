from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import streamlit as st 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import AzureOpenAIEmbeddings

#load environment variables
load_dotenv()

#create an llm
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")

)

#load the pdf document via pyPDFLoader
fle_name= "data/A_Brief_Introduction_To_AI.pdf"
pdf_loader = PyPDFLoader(fle_name)
docs = pdf_loader.load()

# Now chunk the document
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

#Create a prompt
prompt = ChatPromptTemplate.from_template(
    """
    When user ask question, respond only from the provided context
    <context>
    {context}
    </context>
    user : {input}
    """
)

AzureEmbedings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDINGS_BASE"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDINGS_VERSION"),
    chunk_size=512
)

if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index",AzureEmbedings,allow_dangerous_deserialization=True)
else:
    #Create a vector store
    db = FAISS.from_documents(chunks,AzureEmbedings )
    #Save the vector store
    db.save_local("faiss_index")
    print("stored locally")

#Create a retriever
retriever = db.as_retriever()
#Create a document chain
document_chain = create_stuff_documents_chain(llm,prompt)
#Create a retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

st.title("Document Reader")

input= st.text_input("Enter your question here")
if input:
    with st.spinner("Searching for answer"):

        result = retrieval_chain.invoke({"input" : input})
        st.write(result['answer'])

#result = retrieval_chain.invoke({"input" : "What is the document about can you write in 1 line"})

#print(result['answer'])



