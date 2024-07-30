import os

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader, PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


def ingest_docs():
    # https://www.singstat.gov.sg/standards/standards-and-classifications/ssoc taking Complete report (SSOC 2024)
    pdf_path = "docs/ssoc2024a-detailed-definitions.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    print(f"Going to add {len(documents)} documents to Pinecone")

    proxy_url = "http://proxy.ci.mcf.sh:3128"

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

    PineconeVectorStore.from_documents(docs, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])


if __name__ == '__main__':
    load_dotenv()
    ingest_docs()
