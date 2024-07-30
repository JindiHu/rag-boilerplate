import os

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


if __name__ == '__main__':
    load_dotenv()

    loader = TextLoader("./mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

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

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])