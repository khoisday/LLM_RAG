from vector_db import get_vector_db
from llm import get_huggingface_llm
from rag_chain import create_rag_chain

import chainlit as cl
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory


LLM = get_huggingface_llm()
welcome_message = """
1. Upload a PDF or text file
2. Ask a question about the file
"""

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:  # Wait until a file is uploaded
        files = await cl.AskFileMessage(
            content=welcome_message,  # Message prompting for file
            accept=["text/plain", "application/pdf"],  # Allowed file types
            max_size_mb=20,
            timeout=180,  # Wait 3 minutes for file
        ).send()
    file = files[0]  # Get the uploaded file

    # Processing message
    msg = cl.Message(content=f"Processing '{file.name}'...", disable_feedback=True)
    await msg.send()

    # Create vector database from the file (asynchronously)
    vector_db = await cl.make_async(get_vector_db)(file)

    # Initialize conversation history and memory
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a retriever for searching the vector database
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    # Create a conversational retrieval chain using the LLM
    chain = create_rag_chain(
        LLM,
        retriever,
        memory
    )

    msg.content = f"‘{file.name}’ processed. You can now ask questions!"  # Update the processing message
    await msg.update()

    cl.user_session.set("chain", chain)  # Store the chain in the user session

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


if __name__ == "__main__":
    cl.run()


