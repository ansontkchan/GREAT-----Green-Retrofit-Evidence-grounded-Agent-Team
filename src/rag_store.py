from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from .config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, CHROMA_DIR


class RAGStore:
    """
    Minimal Chroma-based vector store for GREAT.

    Collections:
      - breeam: BREEAM corpus
      - wlca: Whole Life Carbon Assessment corpus
      - energy: energy/retrofit corpus
      - cost: cost/business-case corpus
      - ALL: all chunks for the single-agent RAG baseline
    """

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
        )
        self._collections: Dict[str, chromadb.Collection] = {}

    def _get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.ef,
            )
        return self._collections[name]

    def reset_collection(self, name: str) -> None:
        """Delete and recreate a collection. Useful before clean re-ingestion."""
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

        if name in self._collections:
            del self._collections[name]

        self._get_collection(name)

    @staticmethod
    def extract_pages_from_pdf(path: Path) -> List[Tuple[int, str]]:
        """Return [(page_number, text), ...]. Page numbers are 1-indexed."""
        reader = PdfReader(str(path))
        pages: List[Tuple[int, str]] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                pages.append((i + 1, text))

        return pages

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> List[str]:
        """
        Split text into overlapping character chunks.

        Rationale:
        - PDFs are too long to retrieve or embed as one document.
        - Chunking creates retrievable evidence units.
        - Overlap helps preserve continuity across section boundaries.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if overlap < 0:
            raise ValueError("overlap must be non-negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        chunks: List[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = end - overlap

        return chunks

    @staticmethod
    def _batched_upsert(
        collection,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
        batch_size: int = 100,
    ) -> None:
        """
        Upsert documents into Chroma in smaller batches.

        This avoids OpenAI embedding API token-per-request limits.
        """
        total = len(documents)

        if total == 0:
            return

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            collection.upsert(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

            print(f"  Upserted batch {start}-{end} of {total}")

    def ingest_pdfs(
        self,
        agent_name: str,
        pdf_paths: List[Path],
        reset_agent_collection: bool = False,
    ) -> None:
        """
        Ingest PDFs into an agent-specific collection and the global ALL collection.

        Uses upsert instead of add, so re-running ingestion updates existing IDs
        rather than crashing on duplicates.
        """
        if reset_agent_collection:
            print(f"[{agent_name}] Resetting collection '{agent_name}' ...")
            self.reset_collection(agent_name)
            # Do not reset ALL here because other agents may already be ingested.

        col_agent = self._get_collection(agent_name)
        col_all = self._get_collection("ALL")

        docs_agent: List[str] = []
        metas_agent: List[dict] = []
        ids_agent: List[str] = []

        docs_all: List[str] = []
        metas_all: List[dict] = []
        ids_all: List[str] = []

        for pdf_path in pdf_paths:
            print(f"[{agent_name}] Ingesting {pdf_path} ...")
            pages = self.extract_pages_from_pdf(pdf_path)

            for page_number, page_text in pages:
                chunks = self.chunk_text(page_text)

                for chunk_index, chunk in enumerate(chunks):
                    safe_stem = pdf_path.stem.replace(" ", "_")
                    uid = f"{agent_name}-{safe_stem}-p{page_number}-c{chunk_index}"

                    meta = {
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "agent": agent_name,
                    }

                    docs_agent.append(chunk)
                    metas_agent.append(meta)
                    ids_agent.append(uid)

                    docs_all.append(chunk)
                    metas_all.append(meta)
                    ids_all.append(uid)

        if docs_agent:
            self._batched_upsert(
                col_agent,
                docs_agent,
                metas_agent,
                ids_agent,
                batch_size=100,
            )

            self._batched_upsert(
                col_all,
                docs_all,
                metas_all,
                ids_all,
                batch_size=100,
            )

            print(
                f"[{agent_name}] Upserted {len(docs_agent)} chunks "
                f"into '{agent_name}' and 'ALL'."
            )
        else:
            print(f"[{agent_name}] No text chunks found.")

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 6,
    ) -> str:
        col = self._get_collection(collection_name)

        result = col.query(
            query_texts=[query_text],
            n_results=n_results,
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]

        if not docs:
            return "No retrieved context found."

        lines: List[str] = []

        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            src = meta.get("source", "unknown")
            page = meta.get("page", "?")
            agent = meta.get("agent", collection_name)

            lines.append(
                f"[{i}] source={src}, page={page}, corpus={agent}\n{doc}"
            )

        return "\n\n".join(lines)
