"""
query_cache.py - Cache for raw search results (no summarization).

Stores the full list of search results (title/snippet/link) for a query,
keyed by (query, engine). On cache hit, returns top_k results from the
stored list without hitting the search API again.
"""

import hashlib
import json
import os
import sqlite3
import re
import asyncio
from typing import Optional


class QueryCache:
    """SQLite cache for raw search results (title/snippet/link lists).

    Concurrency safety:
    - _write_lock (asyncio.Lock) serializes all writes so concurrent coroutines
      (e.g. --max-concurrent 64) never corrupt the DB by writing simultaneously.

    Degradation policy:
    - If the DB file is corrupted (DatabaseError) or unreadable, reads return None
      (cache miss) and writes are silently skipped. Evaluation continues normally.
    - _db_degraded flag is set on first error to avoid hammering a broken file.
    """

    def __init__(self, cache_dir: str):
        # Use absolute path to avoid issues in multi-process/worker contexts
        self.cache_dir = os.path.abspath(cache_dir)
        self.db_path = os.path.join(self.cache_dir, "query_cache.db")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0
        # Serialize concurrent writes; critical when --max-concurrent is large.
        self._write_lock = asyncio.Lock()
        # Flag: True when DB is known-broken; all writes skipped, reads degrade to miss
        self._db_degraded = False
        self._init_db()
        count = self._get_entry_count()
        print(f"Query cache: {self.db_path} ({count} entries)")

    def _connect(self, timeout: float = 10.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        # WAL mode is unsafe on network file systems (NFS/DolphinFS/HDFS) due to
        # unreliable shared-memory and advisory locking semantics. Use DELETE mode
        # (the SQLite default) which is safe on all file systems.
        # On local-disk runs WAL would be faster, but DELETE is safe everywhere.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        if self._db_degraded:
            return
        try:
            conn = self._connect()
            try:
                # # Integrity check: mark degraded immediately if DB is already broken
                # result = conn.execute("PRAGMA integrity_check").fetchone()
                # if result and result[0] != "ok":
                #     print(f"[QUERY CACHE] DB integrity check failed ({result[0]}): {self.db_path}. "
                #           f"Cache degraded (reads -> miss, writes skipped).")
                #     self._db_degraded = True
                #     return
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            finally:
                conn.close()
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
            print(f"[QUERY CACHE] DB init failed ({e}): {self.db_path}. "
                  f"Cache degraded (reads → miss, writes skipped).")
            self._db_degraded = True

    def _get_entry_count(self) -> int:
        if self._db_degraded:
            return 0
        try:
            conn = self._connect()
            try:
                return conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            finally:
                conn.close()
        except Exception:
            return 0

    def _make_key(self, query: str, engine: str) -> str:
        """Cache key depends only on query text and search engine."""
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        parts = [f"q={normalized}", f"engine={engine}"]
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

    def get(self, query: str, engine: str, top_k: int) -> Optional[list]:
        """Return top_k results from cache, or None on miss.

        Returns None instead of raising when DB is missing or corrupted.
        """
        if self._db_degraded:
            self.misses += 1
            return None
        if not os.path.exists(self.db_path):
            self.misses += 1
            return None
        key = self._make_key(query, engine)
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM query_cache WHERE key = ?", (key,)
                ).fetchone()
            finally:
                conn.close()
        except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError) as e:
            print(f"[QUERY CACHE] Read error ({e}): degrading to miss.")
            self._db_degraded = True
            self.misses += 1
            return None

        if row:
            self.hits += 1
            print(f"[QUERY CACHE HIT] {query[:50]}...")
            try:
                results = json.loads(row[0])
                return results[:top_k]
            except (json.JSONDecodeError, TypeError):
                return None
        self.misses += 1
        return None

    async def set(self, query: str, engine: str, results: list):
        """Store full result list (store all, slice on get).

        Uses asyncio.Lock to prevent concurrent writes from corrupting the DB.
        Silently skips write if DB is degraded. Never raises.
        """
        if self._db_degraded:
            return
        key = self._make_key(query, engine)
        data = json.dumps(results, ensure_ascii=False)
        # Serialize all writes to prevent high-concurrency corruption.
        async with self._write_lock:
            for attempt in range(5):
                try:
                    conn = self._connect(timeout=10.0)
                    try:
                        conn.execute(
                            "INSERT OR REPLACE INTO query_cache (key, value) VALUES (?, ?)",
                            (key, data)
                        )
                        conn.commit()
                        print(f"[QUERY CACHE STORE] {query[:50]}...")
                        return
                    finally:
                        conn.close()
                except sqlite3.DatabaseError as e:
                    # Corruption: mark degraded and stop retrying.
                    print(f"[QUERY CACHE] Write failed, DB corrupted ({e}): {self.db_path}. "
                          f"Disabling writes for this session.")
                    self._db_degraded = True
                    return
                except (sqlite3.OperationalError, OSError) as e:
                    err_str = str(e).lower()
                    if ("locked" in err_str or "unable to open" in err_str) and attempt < 4:
                        await asyncio.sleep(0.1 * (2 ** attempt))
                    else:
                        print(f"[QUERY CACHE] Write failed after {attempt + 1} attempts ({e}): "
                              f"disabling writes for this session.")
                        self._db_degraded = True
                        return

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": (self.hits / total * 100) if total > 0 else 0.0,
            "degraded": self._db_degraded,
        }

    def close(self):
        # Nothing to do for DELETE journal mode (no WAL files to checkpoint).
        pass
