# !pip install langchain langchain-community wikipedia-api
# !pip install wikipedia
# !pip install chromadb

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os

PERSIST_DIR = "./chroma"

def doc_loader(topics,max_doc=2,lang="en"):
  if isinstance(topics,str):
    topic=[topics]
  all_doc=[]
  for topic in topics:
    loader = WikipediaLoader(query=topic,load_max_docs=max_doc,lang=lang)
    docs =loader.load()
    all_doc.extend(docs)
  return all_doc
  
def split_docs(docs, chunk_size=100,chunk_overlap=100):
  for d in docs:
    assert isinstance(d.page_content,str), f"Page_content is not str: {type(d.page_content)}"
  splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  return splitter.split_documents(docs)

def upload_chroma(docs, persist_dir = PERSIST_DIR, collection_name="wiki"):
  texts = [d.page_content for d in docs]
  metas = [d.metadata for d in docs]
  embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectordb = Chroma(collection_name=collection_name,embedding_function=embed,persist_directory=persist_dir)
  vectordb.add_texts(texts=texts,metadatas=metas)
  vectordb.persist()
  return vectordb

topics = ["Mollywood","Kollywood"]
raw_docs = doc_loader(topics,max_doc=2)
chunks = split_docs(raw_docs,chunk_size=100, chunk_overlap=100)
db = upload_chroma(chunks)
print(f"Successfully ingested {len(chunks)} chunks into {PERSIST_DIR}")