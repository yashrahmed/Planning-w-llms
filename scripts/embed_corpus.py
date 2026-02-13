from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openai import OpenAI
import yaml


def load_documents(corpus_dir: Path) -> list[Path]:
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return files


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    chunks: list[str] = []
    step = chunk_chars - overlap_chars
    for start in range(0, len(clean), step):
        part = clean[start : start + chunk_chars].strip()
        if part:
            chunks.append(part)
    return chunks


def load_openai_api_key(keys_config_path: Path) -> str:
    if not keys_config_path.exists():
        raise FileNotFoundError(f"Keys config file not found: {keys_config_path}")

    payload = yaml.safe_load(keys_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {keys_config_path}")

    api_key = payload.get("OPENAI_API_KEY") or payload.get("openai_api_key")
    if not api_key:
        raise KeyError(
            f"Missing OPENAI_API_KEY in {keys_config_path}. "
            "Add OPENAI_API_KEY: <your-key>."
        )
    return str(api_key).strip()


def embed_texts(texts: list[str], model: str, batch_size: int, api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch_texts)
        vectors.extend(item.embedding for item in response.data)

    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed .txt files in a corpus directory and persist embeddings as a NumPy array."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("resources/corpus"),
        help="Directory containing .txt documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/corpus/embeddings.npy"),
        help="Path to output .npy file.",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("resources/corpus/embeddings_index.json"),
        help="Path to JSON index mapping rows to source files.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--keys-config",
        type=Path,
        default=Path("keys-config.yml"),
        help="Path to YAML file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of documents to embed per request.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=3000,
        help="Approximate max characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=300,
        help="Character overlap between adjacent chunks.",
    )
    args = parser.parse_args()

    api_key = load_openai_api_key(args.keys_config)
    files = load_documents(args.corpus_dir)
    chunk_texts: list[str] = []
    chunk_sources: list[dict[str, int | str]] = []
    for file_path in files:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(
            raw, chunk_chars=args.chunk_chars, overlap_chars=args.chunk_overlap_chars
        )
        for idx, chunk in enumerate(chunks):
            chunk_texts.append(chunk)
            chunk_sources.append({"file": str(file_path), "chunk_index": idx})

    if not chunk_texts:
        raise ValueError(f"No chunkable text found in {args.corpus_dir}")

    embeddings = embed_texts(
        chunk_texts, model=args.model, batch_size=args.batch_size, api_key=api_key
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)

    index_payload = {
        "model": args.model,
        "shape": list(embeddings.shape),
        "chunking": {
            "chunk_chars": args.chunk_chars,
            "chunk_overlap_chars": args.chunk_overlap_chars,
        },
        "chunks": chunk_sources,
    }
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.index_output.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    print(f"Embedded {len(files)} documents into {len(chunk_texts)} chunk(s)")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved array to: {args.output}")
    print(f"Saved index to: {args.index_output}")


if __name__ == "__main__":
    main()
