"""torch vs ONNX (fastembed) must agree: same models, so same retrieval.

Run: py -m pytest tests/test_onnx_equivalence.py -m "slow and corpus"
"""

import json

import numpy as np
import pytest

from auralith.config import CORPUS_DIR, PRODUCTION_CONFIG, REPO_ROOT, Settings
from auralith.providers import get_cross_encoder, get_embeddings

pytestmark = [pytest.mark.slow, pytest.mark.corpus]


def test_onnx_matches_torch():
    pytest.importorskip("fastembed")
    pytest.importorskip("sentence_transformers")
    if not any(CORPUS_DIR.glob("*.pdf")):
        pytest.skip("Corpus missing. Run: python eval/corpus/fetch_corpus.py")
    from auralith.ingest import chunk_documents, load_corpus

    chunks = chunk_documents(load_corpus(CORPUS_DIR), PRODUCTION_CONFIG)
    texts = [c.page_content for c in chunks][:300]
    t, o = Settings(ml_backend="torch"), Settings(ml_backend="onnx")

    a = np.array(get_embeddings(t).embed_documents(texts))
    b = np.array(get_embeddings(o).embed_documents(texts))
    assert a.shape == b.shape and b.shape[1] == 384
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-3)
    cos = (a * b).sum(axis=1)
    assert cos.min() >= 0.999, cos.min()

    qs = json.loads((REPO_ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    questions = [q["question"] for q in qs["questions"][:10]]
    ct, co = get_cross_encoder(t), get_cross_encoder(o)
    for i, q in enumerate(questions):
        pairs = [(q, x) for x in texts[i * 10 : i * 10 + 20]]
        st, so = ct.score(pairs), co.score(pairs)
        assert np.argsort(st).tolist() == np.argsort(so).tolist(), q
