import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Azure OpenAI configuration
import os



llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")

)

# Load and preprocess data
azure_data = pd.read_csv("azure_costs.csv")
aws_data = pd.read_csv("aws_costs.csv")
oci_data = pd.read_csv("oci_costs.csv")

combined_data = pd.concat([azure_data, aws_data, oci_data])



embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDINGS_BASE"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDINGS_VERSION"),
    chunk_size=512
)

vectorstore = FAISS.from_texts(combined_data.to_string(), embeddings)

# Set up Azure OpenAI model




# Custom prompt template for graph generation
graph_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    Based on the following question, provide Python code to generate the requested graph using matplotlib or seaborn:
    Question: {question}
    
    The data is available in a pandas DataFrame called 'combined_data' with columns: [list your actual columns here]
    
    Respond with only the Python code to generate the graph. Do not include any explanations.
    """
)

qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Streamlit interface
st.title("FinOps Cloud Cost Analysis Tool")

user_input = st.text_input("Ask a question about cloud costs:")

if user_input:
    response = qa_chain({"question": user_input, "chat_history": []})
    
    st.write("Answer:", response['answer'])
    
    if any(keyword in user_input.lower() for keyword in ["graph", "chart", "plot", "visualize"]):
        graph_code = llm(graph_prompt.format(question=user_input))
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            exec(graph_code)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating graph: {str(e)}")

# Function to display raw data
def show_raw_data():
    st.subheader("Raw Data")
    st.dataframe(combined_data)

# Add a button to show raw data
if st.button("Show Raw Data"):
    show_raw_data()
