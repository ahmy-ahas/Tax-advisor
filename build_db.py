#!/usr/bin/env python3

"""
Build a local Chroma vector database from .txt files
inside the knowledge_base folder.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def main():
    # Project folders
    knowledge_base = Path("./knowledge_base")
    chroma_db = Path("./chroma_db")

    # Check knowledge base folder
    if not knowledge_base.exists() or not knowledge_base.is_dir():
        raise SystemExit(
            f"Knowledge base folder not found: {knowledge_base.resolve()}"
        )

    # Load all .txt files
    print("Step 1/4: Loading text files...")

    loader = DirectoryLoader(
        str(knowledge_base),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    if not documents:
        raise SystemExit("No .txt files found in the knowledge_base folder.")

    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    print("Step 2/4: Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Create local embeddings
    print("Step 3/4: Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma vector database
    print("Step 4/4: Creating Chroma database...")

    chroma_db.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_db),
    )

    print(f"Chroma database created at: {chroma_db.resolve()}")
    print("Build complete.")


if __name__ == "__main__":
    main()