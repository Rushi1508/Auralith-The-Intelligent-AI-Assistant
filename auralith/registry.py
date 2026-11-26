"""API-key auth and the per-owner document registry (Postgres)."""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import os
import re
import secrets
import sys

from sqlalchemy import text

from auralith.index import _pg_engine

_USER = re.compile(r"^[a-z0-9_-]{1,64}$")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_keys() -> dict[str, str]:
    """AURALITH_API_KEYS="alice:<sha256hex>,bob:<sha256hex>" -> {hash: user}."""
    out: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in os.getenv("AURALITH_API_KEYS", "").split(","))):
        user, sep, digest = entry.partition(":")
        if not sep or not _USER.match(user) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("AURALITH_API_KEYS entries must look like user:<sha256hex>")
        out[digest] = user
    return out


def user_for_key(key: str) -> str | None:
    found = None
    for digest, user in load_keys().items():  # no early exit: constant-ish time
        if hmac.compare_digest(digest, hash_key(key)):
            found = user
    return found


@functools.lru_cache
def ensure_schema(url: str) -> None:
    with _pg_engine(url).begin() as c:
        c.execute(text("SELECT pg_advisory_xact_lock(4242)"))
        c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        c.execute(
            text(
                "CREATE TABLE IF NOT EXISTS auralith_docs ("
                "owner text NOT NULL, doc_id text NOT NULL, filename text NOT NULL, "
                "info jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(), "
                "PRIMARY KEY (owner, doc_id))"
            )
        )


def save_doc(url: str, owner: str, info: dict) -> None:
    ensure_schema(url)
    with _pg_engine(url).begin() as c:
        c.execute(
            text(
                "INSERT INTO auralith_docs (owner, doc_id, filename, info) "
                "VALUES (:o, :d, :f, CAST(:i AS jsonb)) ON CONFLICT DO NOTHING"
            ),
            {"o": owner, "d": info["doc_id"], "f": info["filename"], "i": json.dumps(info)},
        )


def get_doc(url: str, owner: str, doc_id: str) -> dict | None:
    with _pg_engine(url).connect() as c:
        row = c.execute(
            text("SELECT info FROM auralith_docs WHERE owner = :o AND doc_id = :d"),
            {"o": owner, "d": doc_id},
        ).first()
    return row[0] if row else None


# shortcut: env key list, adding/revoking needs an api recreate; move to a table if keys must change at runtime
def _main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "create-key" or not _USER.match(sys.argv[2]):
        sys.exit("usage: python -m auralith.registry create-key <user>  (user: [a-z0-9_-]{1,64})")
    key = secrets.token_urlsafe(32)
    print(f"API key (shown once): {key}")
    print(f"AURALITH_API_KEYS entry: {sys.argv[2]}:{hash_key(key)}")


if __name__ == "__main__":
    _main()
