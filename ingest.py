"""Load Atlas documents, split them into chunks, and index them in Chroma."""

from __future__ import annotations

import argparse
import hashlib
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import COLLECTION_NAME, DATA_DIR, VECTOR_DIR
from providers import get_embeddings, load_environment

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

METADATA_FIELDS = {
    "Document ID": "document_id",
    "Department": "department",
    "Document type": "document_type",
    "Version": "version",
    "Effective date": "effective_date",
    "Owner": "owner",
}


def extract_metadata(text: str, source: str) -> dict[str, str]:
    """Extract the document-control fields located at the top of each file."""
    metadata = {
        "source": source,
        "tenant_id": "atlas-logistics",
        "access_scope": "employees",
    }

    for line in text.splitlines()[:15]:
        label, separator, value = line.partition(":")
        if separator and label in METADATA_FIELDS:
            metadata[METADATA_FIELDS[label]] = value.strip()

    return metadata


def load_documents() -> list[Document]:
    """Load all Markdown knowledge-base files and attach governance metadata."""
    paths = sorted(DATA_DIR.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown documents found in {DATA_DIR}")

    documents: list[Document] = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata=extract_metadata(content, path.name),
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents and attach stable chunk metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    source_positions: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index = source_positions.get(source, 0)
        source_positions[source] = chunk_index + 1
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = chunk_id(chunk)

    return chunks


def chunk_id(chunk: Document) -> str:
    """Create a stable identifier from the source, position, and content."""
    identity = (
        f"{chunk.metadata['source']}:{chunk.metadata['chunk_index']}:"
        f"{chunk.page_content}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def index_documents(chunks: list[Document]) -> None:
    """Replace the generated Chroma collection with the current chunks."""
    load_environment()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to .env before running ingestion."
        )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=get_embeddings(),
    )

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    embeddings = get_embeddings().embed_documents(texts)

    existing = vectorstore.get(include=[])["ids"]
    if existing:
        vectorstore.delete(ids=existing)
    vectorstore._collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and split documents without creating embeddings.",
    )
    args = parser.parse_args()

    load_environment()
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")

    if args.dry_run:
        for chunk in chunks:
            print(
                f"- {chunk.metadata['source']} "
                f"chunk {chunk.metadata['chunk_index']} "
                f"({len(chunk.page_content)} characters)"
            )
        print("Dry run complete. No embeddings were created.")
        return

    index_documents(chunks)
    print(f"Indexed {len(chunks)} chunks in {VECTOR_DIR}.")


if __name__ == "__main__":
    main()
