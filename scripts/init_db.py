"""Create an empty DuckDB database from sql/schema.sql.

Usage:  python scripts/init_db.py            -> data/acoustics.duckdb
        python scripts/init_db.py --fresh    -> delete the old file first
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "acoustics.duckdb"
SCHEMA = ROOT / "sql" / "schema.sql"


def init_db(db_path=DB_PATH, fresh=False):
    """Run schema.sql on db_path and return an open connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA.read_text())
    return con


if __name__ == "__main__":
    con = init_db(fresh="--fresh" in sys.argv)
    print(con.sql("SELECT table_name, estimated_size AS n_rows, column_count "
                  "FROM duckdb_tables() ORDER BY table_name"))
