from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


#create an llm
#llm = ChatOpenAI(model_name="gpt-3.5-turbo")

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")

)

prompt = ChatPromptTemplate([
    ("system", "Hello, I am a language model. I can help you with your questions. What would you like to know?"),
    ("user" , "{question}")
])

output_parser = StrOutputParser()
#Call LLM chain

chain = prompt|llm|output_parser
result = chain.invoke({"question" : "What is the capital of France?"})
print(result)

