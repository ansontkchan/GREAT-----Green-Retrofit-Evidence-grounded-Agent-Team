from __future__ import annotations

import argparse
from pathlib import Path

from .rag_store import RAGStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into GREAT RAG store.")
    parser.add_argument("--agent", required=True, help="Agent/corpus name: breeam, wlca, energy, cost")
    parser.add_argument("--reset", action="store_true", help="Reset this agent collection before ingestion")
    parser.add_argument("paths", nargs="+", help="PDF path patterns, e.g. 'data/breeam/*.pdf'")
    args = parser.parse_args()

    pdf_files = []
    for pattern in args.paths:
        for p in Path(".").glob(pattern):
            if p.is_file() and p.suffix.lower() == ".pdf":
                pdf_files.append(p)

    if not pdf_files:
        raise SystemExit("No PDF files found for the given patterns.")

    store = RAGStore()
    store.ingest_pdfs(args.agent, pdf_files, reset_agent_collection=args.reset)


if __name__ == "__main__":
    main()
