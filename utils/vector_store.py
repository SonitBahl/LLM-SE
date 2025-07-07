from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

def create_faiss_index(docs):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    return FAISS.from_texts(docs, embedding=embeddings)