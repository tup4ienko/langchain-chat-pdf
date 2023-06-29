from pathlib import Path

import openai
from environs import Env
from langchain import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env file
env = Env()
env.read_env(".env")
openai.api_key = env.str("OPENAI_API_KEY")


def run_llm(query: str):
    # Load text from a PDF document
    dir_path = Path.cwd()
    path = str(Path(dir_path, "data", "bio.pdf"))
    loader = PyPDFLoader(path)
    pages = loader.load()

    # Break the document into smaller chunks of text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents=pages)

    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()

    # Specify the name of the index
    index_name = "bio-index-react"

    try:
        # Load the pre-existing index if it exists
        vectorstore = FAISS.load_local(index_name, embeddings)
    except Exception as e:
        print("Creating index...")
        # Create a new index using the text chunks and embeddings
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_name)
        print("The index was created successfully")

    # Initialize the retrieval-based question answering model
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature="0", model="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Retrieve the answer for the given query
    result = qa({"query": query})
    return result["result"]


if __name__ == "__main__":
    while True:
        # Prompt the user to enter a question and get the answer
        print(run_llm(input("Enter question: ")))
