from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from chainlit.types import AskFileResponse


def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        loader = TextLoader
    elif file.type == "application/pdf":
        loader = PyPDFLoader

    loaderr = loader(file.path)
    text_splitter = get_text_splitter()
    documents = loaderr.load()


    docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata['source'] = f"source_{i}"
    return docs