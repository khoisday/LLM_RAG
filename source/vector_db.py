from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chainlit as cl
from chainlit.types import AskFileResponse
from data_processing import process_file

def get_embeddings():
    return HuggingFaceEmbeddings()

def get_vector_db(file: AskFileResponse):
    docs = process_file(file=file)
    cl.user_session.set("docs", docs)
    vector_db = Chroma.from_documents(
        documents = docs,
        embegging = get_embeddings()
    )
    return vector_db
