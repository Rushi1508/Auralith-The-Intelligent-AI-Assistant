"""Download the evaluation corpus and verify it byte-for-byte.

The PDFs are fetched rather than committed. Two reasons: the repo stays small,
and arXiv keeps distribution rights unambiguous. The SHA-256 pins matter more
than they look - every recall@k number in results/ is only meaningful against
these exact bytes. A silently updated v2 of a paper would shift page numbers
and quietly invalidate the ground-truth labels.

Usage:  python eval/corpus/fetch_corpus.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent

CORPUS = [
    {
        "doc_id": "attention",
        "filename": "attention.pdf",
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "sha256": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
        "pages": 15,
    },
    {
        "doc_id": "bert",
        "filename": "bert.pdf",
        "arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for "
        "Language Understanding",
        "sha256": "5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e",
        "pages": 16,
    },
    {
        "doc_id": "rag",
        "filename": "rag.pdf",
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "sha256": "23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233",
        "pages": 19,
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(entry: dict, force: bool = False) -> bool:
    dest = CORPUS_DIR / entry["filename"]
    if dest.exists() and not force:
        actual = sha256_of(dest)
        if actual == entry["sha256"]:
            print(f"  ok       {entry['filename']} (checksum verified)")
            return True
        print(f"  MISMATCH {entry['filename']} - refetching")

    url = f"https://arxiv.org/pdf/{entry['arxiv_id']}"
    print(f"  fetching {entry['filename']} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "auralith-eval/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())

    actual = sha256_of(dest)
    if actual != entry["sha256"]:
        print(
            f"  FAIL     {entry['filename']} checksum mismatch\n"
            f"           expected {entry['sha256']}\n"
            f"           got      {actual}\n"
            f"           arXiv may have published a new version. Ground-truth "
            f"page labels in questions.json are pinned to the old bytes.",
            file=sys.stderr,
        )
        return False
    print(f"  ok       {entry['filename']} (checksum verified)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="Redownload even if present")
    args = ap.parse_args()

    print(f"Corpus directory: {CORPUS_DIR}")
    results = [fetch(e, force=args.force) for e in CORPUS]
    ok = sum(results)
    print(f"\n{ok}/{len(CORPUS)} documents verified.")
    return 0 if ok == len(CORPUS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
