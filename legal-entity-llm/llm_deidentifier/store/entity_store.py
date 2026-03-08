from __future__ import annotations

import sqlite3
from pathlib import Path

from llm_deidentifier.config import DB_PATH


class EntityStore:
    """SQLite-backed cross-document entity → token persistence.

    Ensures the same real-world entity always maps to the same replacement
    token, even across different documents processed at different times.
    """

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                token_prefix TEXT NOT NULL,
                token TEXT NOT NULL UNIQUE,
                UNIQUE(canonical_text, entity_type)
            );

            CREATE TABLE IF NOT EXISTS aliases (
                alias_text TEXT NOT NULL,
                entity_id INTEGER NOT NULL REFERENCES entities(id),
                PRIMARY KEY(alias_text)
            );
        """)
        self._conn.commit()

    def get_or_create_token(
        self,
        canonical_text: str,
        entity_type: str,
        token_prefix: str,
    ) -> str:
        """Return an existing token or create a new one for the given entity."""
        row = self._conn.execute(
            "SELECT token FROM entities WHERE canonical_text = ? AND entity_type = ?",
            (canonical_text, entity_type),
        ).fetchone()

        if row:
            return row[0]

        # Determine the next numeric suffix for this prefix
        like_pattern = f"{token_prefix}_%"
        max_row = self._conn.execute(
            "SELECT token FROM entities WHERE token LIKE ? ORDER BY id DESC LIMIT 1",
            (like_pattern,),
        ).fetchone()

        if max_row:
            # Extract the trailing number
            existing_token: str = max_row[0]
            parts = existing_token.rsplit("_", 1)
            try:
                next_num = int(parts[-1]) + 1
            except ValueError:
                next_num = 1
        else:
            next_num = 1

        token = f"{token_prefix}_{next_num}"

        self._conn.execute(
            "INSERT INTO entities (canonical_text, entity_type, token_prefix, token) VALUES (?, ?, ?, ?)",
            (canonical_text, entity_type, token_prefix, token),
        )
        self._conn.commit()
        return token

    def register_alias(self, alias_text: str, canonical_text: str, entity_type: str) -> None:
        """Register an alias surface form for an existing canonical entity."""
        row = self._conn.execute(
            "SELECT id FROM entities WHERE canonical_text = ? AND entity_type = ?",
            (canonical_text, entity_type),
        ).fetchone()
        if not row:
            return

        self._conn.execute(
            "INSERT OR IGNORE INTO aliases (alias_text, entity_id) VALUES (?, ?)",
            (alias_text, row[0]),
        )
        self._conn.commit()

    def lookup_alias(self, alias_text: str) -> str | None:
        """Look up a token by alias text. Returns the token or None."""
        row = self._conn.execute(
            """SELECT e.token FROM aliases a
               JOIN entities e ON a.entity_id = e.id
               WHERE a.alias_text = ?""",
            (alias_text,),
        ).fetchone()
        return row[0] if row else None

    def lookup_token(self, canonical_text: str, entity_type: str) -> str | None:
        """Look up a token by canonical text and type."""
        row = self._conn.execute(
            "SELECT token FROM entities WHERE canonical_text = ? AND entity_type = ?",
            (canonical_text, entity_type),
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()
