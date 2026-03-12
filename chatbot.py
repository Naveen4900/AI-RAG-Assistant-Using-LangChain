# AI-RAG-Assistant-Using-LangChain
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')

# Set up credentials and project_id
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="your_api_key_here"  # Replace this with your actual API key
)
project_id = "skills-network"

## LLM
def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url=credentials.url,
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

## Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.INPUT: None,
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: "none"
    }

    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-embeddings",
        url=credentials.url,
        project_id=project_id,
        params=embed_params,
    )
    return watsonx_embedding

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=True
    )
    response = qa.invoke({"query": query})
    return response['result']

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Document-based QA Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7860)
