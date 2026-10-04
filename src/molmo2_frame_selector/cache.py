"""Small, safe SQLite cache for frozen image embeddings."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from math import isfinite
from pathlib import Path


def video_file_identity(path: str | Path) -> str:
    """Create a stable-enough cache identity without hashing a large video."""

    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    payload = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_namespace(
    *,
    encoder_fingerprint: str,
    video_identity: str,
) -> str:
    payload = f"{encoder_fingerprint}|{video_identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SQLiteEmbeddingCache:
    """Cache embeddings by encoder/video namespace and original frame index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL,
                item_key TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (namespace, item_key)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteEmbeddingCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _encode(vector: Sequence[float]) -> tuple[int, bytes]:
        values = [float(value) for value in vector]
        if not values:
            raise ValueError("cannot cache an empty embedding")
        if any(not isfinite(value) for value in values):
            raise ValueError("cannot cache non-finite embedding values")
        return len(values), struct.pack(f"<{len(values)}f", *values)

    @staticmethod
    def _decode(dimension: int, payload: bytes) -> list[float]:
        expected_bytes = dimension * 4
        if len(payload) != expected_bytes:
            raise ValueError("cached embedding payload has an invalid size")
        return list(struct.unpack(f"<{dimension}f", payload))

    def get(self, namespace: str, item_key: str) -> list[float] | None:
        row = self._connection.execute(
            "SELECT dimension, vector FROM embeddings WHERE namespace=? AND item_key=?",
            (namespace, item_key),
        ).fetchone()
        if row is None:
            return None
        return self._decode(int(row[0]), row[1])

    def get_many(
        self, namespace: str, item_keys: Iterable[str]
    ) -> dict[str, list[float]]:
        return {
            item_key: vector
            for item_key in item_keys
            if (vector := self.get(namespace, item_key)) is not None
        }

    def put(self, namespace: str, item_key: str, vector: Sequence[float]) -> None:
        dimension, payload = self._encode(vector)
        self._connection.execute(
            """
            INSERT INTO embeddings(namespace, item_key, dimension, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, item_key)
            DO UPDATE SET dimension=excluded.dimension, vector=excluded.vector
            """,
            (namespace, item_key, dimension, payload),
        )
        self._connection.commit()

    def put_many(
        self,
        namespace: str,
        items: Iterable[tuple[str, Sequence[float]]],
    ) -> None:
        encoded = []
        for item_key, vector in items:
            dimension, payload = self._encode(vector)
            encoded.append((namespace, item_key, dimension, payload))
        self._connection.executemany(
            """
            INSERT INTO embeddings(namespace, item_key, dimension, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, item_key)
            DO UPDATE SET dimension=excluded.dimension, vector=excluded.vector
            """,
            encoded,
        )
        self._connection.commit()

