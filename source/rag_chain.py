from langchain.chains import ConversationalRetrievalChain


def create_rag_chain(llm, retriever, memory):
    
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = 'stuff',
        retriever = retriever,
        memory = memory,
        return_source_documents = True
    )
    return chain