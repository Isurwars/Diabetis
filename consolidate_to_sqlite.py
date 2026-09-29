#!/usr/bin/env python3
"""
ENSANUT 2018 SQLite Database Consolidation Pipeline
Extracts raw CSV files from INEGI ENSANUT 2018 and consolidates them into an indexed SQLite database.
"""

import os
import glob
import sqlite3
import pandas as pd
from typing import List, Dict

DB_PATH = "/home/isurwars/Projects/Diabetis/ensanut_2018.db"
BASE_DIR = "/home/isurwars/Projects/Diabetis/conjunto_de_datos_cs_ensanut_2018_csv"

# Mapping from folder prefix to target table name
TABLE_MAPPING: Dict[str, str] = {
    "cs_adultos": "adultos",
    "cs_residentes": "residentes",
    "cs_hogares": "hogares",
    "cs_viviendas": "viviendas",
    "cs_serv_salud": "serv_salud",
    "cs_act_fis_ado": "act_fis_ado",
    "cs_ayuda_alimentaria": "ayuda_alimentaria",
    "cs_seguridad_alimentaria": "seguridad_alimentaria",
    "cs_etiquetado_frontal": "etiquetado_frontal"
}


def clean_column_name(col: str) -> str:
    """Strip BOM, quotation marks, and standardize column names to lowercase."""
    return col.replace('\ufeff', '').replace('ï»¿', '').replace('"', '').replace("'", "").strip().lower()


def consolidate_table(conn: sqlite3.Connection, folder_pattern: str, table_name: str) -> None:
    csv_candidates = glob.glob(os.path.join(BASE_DIR, f"*{folder_pattern}*", "conjunto_de_datos", "*.csv"))
    if not csv_candidates:
        print(f"[-] No CSV found for {table_name} matching {folder_pattern}")
        return

    csv_path = csv_candidates[0]
    print(f"\n[+] Processing '{table_name}' from: {os.path.basename(csv_path)}")

    # Read and clean headers first
    header_df = pd.read_csv(csv_path, encoding="latin1", nrows=0)
    cleaned_columns = [clean_column_name(c) for c in header_df.columns]

    print(f"    - Columns: {len(cleaned_columns)}")

    # Stream chunks into SQLite
    chunksize = 20000
    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(csv_path, encoding="latin1", chunksize=chunksize, low_memory=False):
        chunk.columns = cleaned_columns
        # Write to SQLite
        chunk.to_sql(
            table_name,
            conn,
            if_exists="replace" if first_chunk else "append",
            index=False
        )
        total_rows += len(chunk)
        first_chunk = False

    print(f"    [✔] Successfully written {total_rows:,} rows to table '{table_name}'")


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create composite indexes on keys (upm, viv_sel, hogar, numren) for fast joins."""
    cursor = conn.cursor()
    print("\n[+] Creating indexes for fast relational joins...")

    index_definitions = [
        ("idx_adultos_pk", "adultos", ["upm", "viv_sel", "hogar", "numren"]),
        ("idx_residentes_pk", "residentes", ["upm", "viv_sel", "hogar", "numren"]),
        ("idx_hogares_pk", "hogares", ["upm", "viv_sel", "hogar"]),
        ("idx_viviendas_pk", "viviendas", ["upm", "viv_sel"]),
        ("idx_adultos_p3_1", "adultos", ["p3_1"])  # Target variable index
    ]

    for idx_name, table, cols in index_definitions:
        col_list = ", ".join(cols)
        try:
            sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col_list});"
            cursor.execute(sql)
            print(f"    [✔] Created index {idx_name} on {table}({col_list})")
        except sqlite3.OperationalError as e:
            print(f"    [-] Skipped {idx_name}: {e}")

    conn.commit()


def main():
    print(f"=== Starting ENSANUT 2018 SQLite Consolidation ===")
    print(f"Target Database: {DB_PATH}")

    # Remove existing db if desired, or overwrite tables
    conn = sqlite3.connect(DB_PATH)

    try:
        for pattern, table_name in TABLE_MAPPING.items():
            consolidate_table(conn, pattern, table_name)

        create_indexes(conn)

        # Quick summary count
        cursor = conn.cursor()
        print("\n=== Database Summary ===")
        for _, table_name in TABLE_MAPPING.items():
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  • {table_name:25s}: {count:>8,d} rows")
            except sqlite3.OperationalError:
                pass

        print(f"\n[✔] Consolidation completed successfully: {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
