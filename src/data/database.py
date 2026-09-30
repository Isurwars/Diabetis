import os
import sqlite3
import pandas as pd
from typing import Optional

def get_db_connection(db_path: str = "ensanut_2018.db") -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at '{db_path}'. Please run consolidate_to_sqlite.py first.")
    return sqlite3.connect(db_path)

def load_raw_cohort(
    db_path: str = "ensanut_2018.db",
    exclude_gestational: bool = True
) -> pd.DataFrame:
    """
    Extracts the adult cohort joining adultos, residentes, and hogares.
    Excludes gestational cases (p3_1 == 2) if exclude_gestational is True.
    Strictly excludes post-diagnosis conditional questions (p3_2 through p3_18)
    to eliminate data leakage.
    """
    conn = get_db_connection(db_path)
    
    where_clause = "WHERE a.p3_1 IN (1, 3)" if exclude_gestational else "WHERE a.p3_1 IN (1, 2, 3)"
    
    query = f"""
    WITH res_agg AS (
        SELECT upm, viv_sel, hogar, 
               COUNT(*) as tam_hogar,
               SUM(CASE WHEN edad < 18 THEN 1 ELSE 0 END) as n_menores,
               SUM(CASE WHEN edad >= 60 THEN 1 ELSE 0 END) as n_adultos_mayores
        FROM residentes
        GROUP BY upm, viv_sel, hogar
    ),
    seg_agg AS (
        SELECT upm, viv_sel, hogar, p1 as inseguridad_alim_p1
        FROM seguridad_alimentaria
        GROUP BY upm, viv_sel, hogar
    ),
    ayu_agg AS (
        SELECT upm, viv_sel, hogar, p1 as recibe_ayuda_alim
        FROM ayuda_alimentaria
        GROUP BY upm, viv_sel, hogar
    )
    SELECT 
        a.upm, 
        a.viv_sel, 
        a.hogar, 
        a.numren,
        a.p3_1,
        a.f_20mas as factor_expansion,
        -- Demographics from residentes
        r.edad,
        r.sexo,
        r.nivel as nivel_educativo,
        r.estrato as estrato_socioeconomico,
        a.dominio,
        a.region,
        -- Anthropometrics & clinical history from adultos
        a.p1_1 as dx_obesidad,
        a.p1_4 as silueta_corporal,
        a.p1_5 as peso_habitual,
        a.p1_7 as cambio_peso,
        a.p1_8 as kg_cambio,
        a.p4_1 as dx_hipertension,
        a.p6_3 as dx_colesterol_trigliceridos,
        a.p7_1_1 as ant_padre_diab,
        a.p7_1_2 as ant_madre_diab,
        a.p7_1_3 as ant_hermano_diab,
        a.p7_2_1 as ant_padre_hta,
        a.p7_2_2 as ant_madre_hta,
        a.p7_2_3 as ant_hermano_hta,
        a.p13_1 as fuma_100_cigarros,
        a.p13_2 as fuma_actualmente,
        a.p14_1 as consume_alcohol,
        -- Household assets & structure from hogares
        h.p2_9_1 as tiene_refri,
        h.p2_9_2 as tiene_lavadora,
        h.p2_9_3 as tiene_auto,
        -- Household epidemiological indicators
        COALESCE(res_agg.tam_hogar, 1) as tam_hogar,
        COALESCE(res_agg.n_menores, 0) as n_menores,
        COALESCE(res_agg.n_adultos_mayores, 0) as n_adultos_mayores,
        COALESCE(seg_agg.inseguridad_alim_p1, 2) as inseguridad_alim_p1,
        COALESCE(ayu_agg.recibe_ayuda_alim, 2) as recibe_ayuda_alim
    FROM adultos a
    LEFT JOIN residentes r 
        ON a.upm = r.upm AND a.viv_sel = r.viv_sel AND a.hogar = r.hogar AND a.numren = r.numren
    LEFT JOIN hogares h 
        ON a.upm = h.upm AND a.viv_sel = h.viv_sel AND a.hogar = h.hogar
    LEFT JOIN res_agg 
        ON a.upm = res_agg.upm AND a.viv_sel = res_agg.viv_sel AND a.hogar = res_agg.hogar
    LEFT JOIN seg_agg 
        ON a.upm = seg_agg.upm AND a.viv_sel = seg_agg.viv_sel AND a.hogar = seg_agg.hogar
    LEFT JOIN ayu_agg 
        ON a.upm = ayu_agg.upm AND a.viv_sel = ayu_agg.viv_sel AND a.hogar = ayu_agg.hogar
    {where_clause};
    """
    
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
        
    return df
