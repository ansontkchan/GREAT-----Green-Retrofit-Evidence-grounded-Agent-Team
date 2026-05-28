# GREAT RAG Corpus Summary

Corpus version: v0.1-pilot

Vector store: ChromaDB  
Vector store folder: `.chroma_great`  
Embedding model: `text-embedding-3-large`  
Chunk size: 1200 characters  
Chunk overlap: 200 characters  
Specialist agent retrieval top-k: 6  
Single-agent RAG retrieval top-k: 8  

## Corpus sizes

| Agent / corpus | Number of PDFs | Number of chunks | Status |
|---|---:|---:|---|
| BREEAM Agent | 4 | 2652 | Ingested |
| Whole Life Carbon Assessment Agent | 4 | 765 | Ingested |
| Energy Efficiency Agent | 6 | 633 | Ingested |
| Cost and Feasibility Agent | 4 | 529 | Ingested |

## Method note

The UKGBC case studies are not included in the RAG corpus. They are reserved as held-out evaluation cases for strategy-alignment testing.

The PDF documents themselves are not committed to GitHub because some standards and guidance documents may be copyrighted or licence-restricted. The reproducibility record is therefore provided through the corpus inventory, chunk counts, chunking settings, embedding model, and retrieval settings.