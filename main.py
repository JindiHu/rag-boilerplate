import os
import httpx
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from callback import CallbackHandler
from util import format_docs

load_dotenv()


if __name__ == "__main__":
    # Define the proxy URL with the correct scheme
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

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        http_client=http_client,
        callbacks=[CallbackHandler()]
    )

    query = "what is Pinecone in machine learning?"

    # prompt = PromptTemplate.from_template(query)
    #
    # chain = prompt | llm
    # result = chain.invoke(input={})

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = """
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>
    
    {input}
    """
    prompt = PromptTemplate.from_template(retrieval_qa_chat_prompt)

    # to use stuff documents chain
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # retrieval_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    #
    # result = retrieval_chain.invoke(input={"input": query})

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    res = rag_chain.invoke(query)
