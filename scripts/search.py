from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openai import OpenAI
import yaml


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


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {index_path}. Run scripts/embed_corpus.py first."
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / max(np.linalg.norm(query), 1e-12)
    m = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return m @ q


def make_snippet(path: Path, max_chars: int = 240) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    step = chunk_chars - overlap_chars
    return [clean[i : i + chunk_chars] for i in range(0, len(clean), step) if clean[i : i + chunk_chars]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search corpus docs using precomputed embeddings."
    )
    parser.add_argument("query", help="Natural language search query.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("resources/corpus/embeddings.npy"),
        help="Path to saved embeddings array.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("resources/corpus/embeddings_index.json"),
        help="Path to embeddings index JSON.",
    )
    parser.add_argument(
        "--keys-config",
        type=Path,
        default=Path("keys-config.yml"),
        help="Path to YAML file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    index = load_index(args.index)
    chunks = index.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"No chunks listed in index: {args.index}")

    if not args.embeddings.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {args.embeddings}. Run scripts/embed_corpus.py first."
        )
    matrix = np.load(args.embeddings)
    if matrix.shape[0] != len(chunks):
        raise ValueError(
            f"Row/chunk mismatch: {matrix.shape[0]} embeddings vs {len(chunks)} chunks in index."
        )

    model = index.get("model", "text-embedding-3-large")
    api_key = load_openai_api_key(args.keys_config)
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=[args.query])
    query_vector = np.asarray(response.data[0].embedding, dtype=np.float32)

    scores = cosine_similarity(query_vector, matrix)
    top_k = min(args.top_k, len(chunks))
    top_idx = np.argsort(-scores)[:top_k]

    print(f'Query: "{args.query}"')
    print(f"Model: {model}")
    print(f"Top {top_k} result(s):")
    for rank, idx in enumerate(top_idx, start=1):
        chunk_meta = chunks[idx]
        path = Path(str(chunk_meta["file"]))
        chunk_index = int(chunk_meta["chunk_index"])
        chunking = index.get("chunking", {})
        chunk_chars = int(chunking.get("chunk_chars", 3000))
        overlap_chars = int(chunking.get("chunk_overlap_chars", 300))
        text = path.read_text(encoding="utf-8", errors="replace")
        chunk_list = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        snippet = chunk_list[chunk_index][:240].rstrip() + ("..." if len(chunk_list[chunk_index]) > 240 else "")
        print(f"\n{rank}. {path} (chunk {chunk_index})")
        print(f"   score: {scores[idx]:.4f}")
        print(f"   snippet: {snippet}")


if __name__ == "__main__":
    main()
