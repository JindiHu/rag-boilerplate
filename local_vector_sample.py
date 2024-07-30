import os

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

from callback import CallbackHandler
from util import format_docs

if __name__ == "__main__":
    print("hi")
    load_dotenv()
    pdf_path = "docs/G2C0006-CCP_Applicant_Guide_V3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    proxy_url = "http://proxy.ci.mcf.sh:3128"

    # Define the client with proxy settings
    http_client = httpx.Client(proxies={
        "http://": proxy_url,
        "https://": proxy_url,
    })

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        http_client=http_client,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_local")

    new_vectorstore = FAISS.load_local(
        "faiss_index_local", embeddings, allow_dangerous_deserialization=True
    )

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        http_client=http_client,
        callbacks=[CallbackHandler()]
    )

    retrieval_qa_chat_prompt = """
    Answer any use questions based on the context below:

    <context>
    {context}
    </context>

    {input}
    """
    prompt = PromptTemplate.from_template(retrieval_qa_chat_prompt)


    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    llm.invoke("how to apply CCP?")

    res = rag_chain.invoke("how to apply CCP?")
