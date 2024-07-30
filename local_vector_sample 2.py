import os

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from callback import CallbackHandler

if __name__ == "__main__":
    print("hi")
    load_dotenv()

    # https://www.singstat.gov.sg/standards/standards-and-classifications/ssoc taking Complete report (SSOC 2024)
    pdf_path = "docs/ssoc2024a-detailed-definitions.pdf"
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
        temperature=0.2,
        callbacks=[CallbackHandler()]
    )

    retrieval_qa_chat_prompt = """You are a career coach guiding me through my next job search or career transition. I 
    need your expertise to help me strategize my next steps and break them down into monthly milestones. Here’s the 
    context for my situation:

    <context>
    {context}
    </context>
    
    <situation>
    {situation}
    </situation>

    {input}
    """
    prompt = PromptTemplate.from_template(retrieval_qa_chat_prompt)

    situation = """I am currently a product owner. These are my skills gap: sales process, b2b, sales management, 
    selling, account management, business development, key account management, customer relationships, new business 
    development, negotiation, direct sales, customer success. These are courses recommended to me: Digital Marketing 
    by Ngee Ann Poly, Logistics Solutions Marketing by Singapore Chinese Chamber Institute and Sales Performance 
    Strategy by Pioneer Training & Consultancy."""

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    query = "How can i transition into a Sales Manager within 3 months?"

    res = retrieval_chain.invoke(input={"input": query, "situation": situation})

    res_without_context = combine_docs_chain.invoke(input={"input": query, "situation": situation, "context": ""})

