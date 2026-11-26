from cohort_database import *

LAB_CODES = {
    "hb": [51222,50811,51640,51645],
    "platelets":[51265], "inr":[51237], "pt":[51274], "ptt":[51275,227466],
    "creatinine":[50912], "bun":[51006], "wbc":[51301],
    "bicarbonate":[50882], "lactate":[50813,52442,53154], "albumin":[50862],
    "bilirubin":[50885], "tnt":[51003],
}
E_LABS = {
    "hgb":"hb","hemoglobin":"hb","platelets x 1000":"platelets","pt - inr":"inr",
    "pt":"pt","ptt":"ptt","creatinine":"creatinine","bun":"bun","wbc x 1000":"wbc",
    "bicarbonate":"bicarbonate","lactate":"lactate","albumin":"albumin","total bilirubin":"bilirubin",
}
from config import LAB_CACHE

def verify_release():
    for name,key in (("patients","subject_id"),("admissions","hadm_id")):
        p, r = pq("MIMIC_hosp",name),raw("hosp",name)
        check=one(f"""SELECT (SELECT count(*) FROM {p}), (SELECT count(*) FROM {r}),
            (SELECT count(*) FROM (SELECT {key} FROM {p} EXCEPT ALL SELECT {key} FROM {r})),
            (SELECT count(*) FROM (SELECT {key} FROM {r} EXCEPT ALL SELECT {key} FROM {p}))""")
        require(check[0]==check[1] and check[2:]==(0,0), f"v3.1 {name} key mismatch")
    bad=one(f"""SELECT count(*) FILTER(WHERE i IS NULL OR (outtime IS NOT NULL AND o IS NULL)),
       count(*) FILTER(WHERE abs(date_diff('second',i,o)/86400.0-los)>1.0/1440 OR o<=i), count(*) FILTER(WHERE o IS NULL OR los IS NULL)
       FROM (SELECT {ts('intime')} i,{ts('outtime')} o,los,outtime FROM {raw('icu','icustays')})""")
    require(bad[:2]==(0,0),f"ICU timestamp/LOS errors: {tuple(map(mask,bad))}")
    build("m_transfers",f"""SELECT subject_id,hadm_id,careunit,eventtype,{ts('intime')} t_in,{ts('outtime')} t_out
         FROM {raw('hosp','transfers',all_varchar=True)}
         WHERE regexp_matches(lower(coalesce(careunit,'')), 'intensive care unit|(^|[^a-z])icu([^a-z]|$)')
            OR lower(eventtype)='ed'""")
    build("m_dx_dict",f"""SELECT {icd('icd_code')} code,icd_version ver,long_title FROM {raw('hosp','d_icd_diagnoses')}""")
    build("m_lab_dict",f"""SELECT * FROM {raw('hosp','d_labitems')}""")
    checks=one("""SELECT count(*) FILTER(WHERE itemid IN (51222,50811) AND lower(label)='hemoglobin'),
            count(*) FILTER(WHERE itemid IN (50813,52442,53154) AND lower(label)='lactate'),
            count(*) FILTER(WHERE itemid=51003 AND lower(label)='troponin t')
        FROM m_lab_dict""")
    require(checks==(2,3,1),"v3.1 Hb, lactate or TnT dictionary mismatch")

def prepare_m_labs():
    codes=",".join(lit(str(x)) for items in LAB_CODES.values() for x in items if x!=227466)
    # Candidate stays and hospital windows come from the high-risk GIB phenotype.
    build("m_lab_cache",f"""
      WITH selected AS (
         SELECT try_cast(x.subject_id AS BIGINT) subject_id,try_cast(x.hadm_id AS BIGINT) hadm_id,
             try_cast(x.labevent_id AS BIGINT) lid,x.itemid,
             {ts('x.charttime')} chart_at,{ts('x.storetime')} available_at,
             try_cast(x.valuenum AS DOUBLE) val,lower(trim(x.valueuom)) uom,
             try_cast(x.ref_range_upper AS DOUBLE) ul
         FROM {raw('hosp','labevents',all_varchar=True)} x
         SEMI JOIN (SELECT DISTINCT person FROM m_base) b ON try_cast(x.subject_id AS BIGINT)=b.person
         WHERE x.itemid IN ({codes})
      )
      SELECT x.*,d.label official_label FROM selected x LEFT JOIN m_lab_dict d ON try_cast(x.itemid AS BIGINT)=d.itemid
      WHERE EXISTS (
         SELECT 1 FROM m_base b WHERE x.subject_id=b.person
           AND (x.hadm_id=b.hosp OR x.hadm_id IS NULL)
           AND x.chart_at BETWEEN least(b.adm_at,b.ed_at,b.entry_at-INTERVAL '1 day')-INTERVAL '2 hours'
                              AND coalesce(b.discharge_at,b.exit_at)
      )
    """)
    require(one("SELECT count(*)-count(DISTINCT lid), count(*) FILTER(WHERE official_label IS NULL OR chart_at IS NULL) FROM m_lab_cache")== (0,0),
        "Official lab cache uniqueness / dictionary validation failed")
    save("m_lab_cache",LAB_CACHE)
    build("m_labs",f"""
       SELECT b.id,date_diff('second',b.entry_at,x.chart_at)/60.0 t,
             x.itemid,x.val,x.uom,x.ul,x.lid,x.chart_at at_time
       FROM read_parquet({lit(LAB_CACHE.as_posix())}) x JOIN m_base b
           ON x.subject_id=b.person AND (x.hadm_id=b.hosp OR x.hadm_id IS NULL)
       WHERE x.itemid IN ('51222','50811','51640','51645','50813','51003')
         AND (x.hadm_id IS NOT NULL OR x.chart_at BETWEEN
                    least(b.entry_at-INTERVAL '2 hours',b.ed_at,b.adm_at) AND least(b.discharge_at,b.exit_at))
    """)
