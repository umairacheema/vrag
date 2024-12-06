"""
vectorstore.py
-------------

Description:
    This module creates a vector store from the PDF documents stored in the documents folder.

Author:
    Umair Cheema <cheemzgpt@gmail.com>

Version:
    1.0.0

License:
    Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Date Created:
    2024-11-30

Last Modified:
    2024-11-30

Python Version:
    3.8+

Usage:
    Import this module and call the available functions for data processing tasks.
    Example:
        from vectorstore import create_vectordb
        result = create_vectordb

Dependencies:

"""

from vragconfig import VRAGConfig
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectordb():
    config = VRAGConfig(file_path='./vrag.yaml').read()
    embedding_model = config['vs_embeddings_model_path']
    embedding_model_name = config['vs_embeddings_model_name']
    pdf_document_folder = config['vs_pdf_documents_folder']
    vector_database_path = config['vs_output_folder']
    document_chunk_size = config['vs_chunk_size']
    document_chunk_overlap = config['vs_chunk_overlap']

    #Load the documents from the folder
    loader = PyPDFDirectoryLoader(pdf_document_folder)
    documents = []
    
    print(f'Processing files in: {pdf_document_folder}')
    for document in loader.load():
        documents.append(document)
    
     #Initialize embedding model
    print(f'Initializing embedding model: {embedding_model_name}')
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    #Split documents into chunks
    print(f'Creating chunks with settings: size={document_chunk_size}, overlap={document_chunk_overlap}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=document_chunk_size, chunk_overlap=document_chunk_overlap)
    splits = text_splitter.split_documents(documents)
    
    #Create and persist vector store
    vector_store = Chroma.from_documents(splits, embeddings, persist_directory=vector_database_path)

if __name__ == "__main__":
    create_vectordb()

