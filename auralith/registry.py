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
import time

from sqlalchemy import text

from auralith.index import _pg_engine

_USER = re.compile(r"^[a-z0-9_-]{1,64}$")
GUEST_PREFIX = "guest-"
SAMPLE_OWNER = "sample"  # public read-only demo doc, owned by this reserved user
GUEST_TTL_S = int(os.getenv("AURALITH_GUEST_TTL_HOURS", "24")) * 3600


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
        c.execute(
            text(
                "CREATE TABLE IF NOT EXISTS auralith_usage ("
                "owner text NOT NULL, bucket text NOT NULL, win bigint NOT NULL, "
                "n integer NOT NULL, PRIMARY KEY (owner, bucket, win))"
            )
        )
        c.execute(
            text(
                "CREATE TABLE IF NOT EXISTS auralith_keys ("
                "id bigserial PRIMARY KEY, key_hash text UNIQUE NOT NULL, owner text NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now())"
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


def list_docs(url: str, owner: str) -> list[dict]:
    ensure_schema(url)
    with _pg_engine(url).connect() as c:
        rows = c.execute(
            text("SELECT info FROM auralith_docs WHERE owner = :o ORDER BY created_at DESC"),
            {"o": owner},
        ).all()
    return [r[0] for r in rows]


def admins() -> set[str]:
    """AURALITH_ADMINS="alice,bob": users allowed to manage API keys."""
    return {u.strip() for u in os.getenv("AURALITH_ADMINS", "").split(",") if u.strip()}


def db_user_for_key(url: str, key: str) -> str | None:
    ensure_schema(url)
    with _pg_engine(url).connect() as c:
        return c.execute(
            text(
                "SELECT owner FROM auralith_keys WHERE key_hash = :h AND "
                "(owner NOT LIKE :g OR created_at > now() - make_interval(secs => :ttl))"
            ),
            {"h": hash_key(key), "g": GUEST_PREFIX + "%", "ttl": GUEST_TTL_S},
        ).scalar()


def create_key(url: str, owner: str) -> dict:
    """New key for owner; the raw key is returned once and only its hash is stored."""
    ensure_schema(url)
    key = secrets.token_urlsafe(32)
    with _pg_engine(url).begin() as c:
        row = c.execute(
            text(
                "INSERT INTO auralith_keys (key_hash, owner) VALUES (:h, :o) "
                "RETURNING id, created_at"
            ),
            {"h": hash_key(key), "o": owner},
        ).one()
    return {"id": row[0], "user": owner, "key": key, "created_at": row[1].isoformat()}


def list_keys(url: str) -> list[dict]:
    ensure_schema(url)
    with _pg_engine(url).connect() as c:
        rows = c.execute(text("SELECT id, owner, created_at FROM auralith_keys ORDER BY id")).all()
    return [{"id": i, "user": o, "created_at": t.isoformat()} for i, o, t in rows]


def revoke_key(url: str, key_id: int) -> bool:
    ensure_schema(url)
    with _pg_engine(url).begin() as c:
        return c.execute(
            text("DELETE FROM auralith_keys WHERE id = :i RETURNING id"), {"i": key_id}
        ).first() is not None


def count_usage(url: str, owner: str, windows: list[tuple[str, int]], now: float) -> list[int]:
    """Count one event in each (bucket, window_seconds) fixed window; returns counts incl. this one."""
    ensure_schema(url)
    out = []
    with _pg_engine(url).begin() as c:
        for bucket, seconds in windows:
            p = {"o": owner, "b": bucket, "w": int(now // seconds)}
            c.execute(
                text("DELETE FROM auralith_usage WHERE owner = :o AND bucket = :b AND win < :w"), p
            )
            out.append(
                c.execute(
                    text(
                        "INSERT INTO auralith_usage (owner, bucket, win, n) VALUES (:o, :b, :w, 1) "
                        "ON CONFLICT (owner, bucket, win) DO UPDATE SET n = auralith_usage.n + 1 "
                        "RETURNING n"
                    ),
                    p,
                ).scalar_one()
            )
    return out


def delete_doc(url: str, owner: str, doc_id: str) -> bool:
    """Drop the registry row and every `<doc_id>__*` pgvector collection (embeddings cascade).

    One per embedding provider/chunking scheme the doc was ever indexed under.
    """
    ensure_schema(url)
    with _pg_engine(url).begin() as c:
        gone = c.execute(
            text("DELETE FROM auralith_docs WHERE owner = :o AND doc_id = :d RETURNING doc_id"),
            {"o": owner, "d": doc_id},
        ).first()
        if gone:
            c.execute(
                text("DELETE FROM langchain_pg_collection WHERE name LIKE :p ESCAPE '\\'"),
                {"p": doc_id + "\\_\\_%"},  # doc_id is 12 hex (row existed): no wildcards
            )
    return gone is not None


def sweep_guests(url: str, limit: int = 20) -> int:
    """Bounded cleanup of expired guest data; only guest-% / ip:% owners are ever touched."""
    ensure_schema(url)
    p = {"g": GUEST_PREFIX + "%", "ttl": GUEST_TTL_S, "n": limit}
    with _pg_engine(url).connect() as c:
        stale = c.execute(
            text(
                "SELECT owner, doc_id FROM auralith_docs WHERE owner LIKE :g AND "
                "created_at < now() - make_interval(secs => :ttl) LIMIT :n"
            ),
            p,
        ).all()
    n = sum(delete_doc(url, owner, doc_id) for owner, doc_id in stale)
    with _pg_engine(url).begin() as c:
        n += c.execute(
            text(
                "DELETE FROM auralith_keys WHERE id IN (SELECT id FROM auralith_keys WHERE "
                "owner LIKE :g AND created_at < now() - make_interval(secs => :ttl) LIMIT :n)"
            ),
            p,
        ).rowcount
        c.execute(
            text(
                "DELETE FROM auralith_usage WHERE owner LIKE :g AND "
                "owner NOT IN (SELECT owner FROM auralith_keys)"
            ),
            p,
        )
        c.execute(  # per-IP issuance counters use 1h windows (bucket guest:3600)
            text("DELETE FROM auralith_usage WHERE owner LIKE :ip AND win < :w"),
            {"ip": "ip:%", "w": int(time.time() // 3600)},
        )
    return n


# Bootstrap keys come from AURALITH_API_KEYS; more are added at runtime via /admin/keys.
def _main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "create-key" or not _USER.match(sys.argv[2]):
        sys.exit("usage: python -m auralith.registry create-key <user>  (user: [a-z0-9_-]{1,64})")
    key = secrets.token_urlsafe(32)
    print(f"API key (shown once): {key}")
    print(f"AURALITH_API_KEYS entry: {sys.argv[2]}:{hash_key(key)}")


if __name__ == "__main__":
    _main()
