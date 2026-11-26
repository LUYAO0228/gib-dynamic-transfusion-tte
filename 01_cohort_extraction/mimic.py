from cohort_database import *
from clinical_sources import prepare_m_labs

H = "MIMIC_hosp"
I = "MIMIC_ICU"
D = "mimiciv_derived"
ppi = "pantoprazol|protonix|omeprazol|prilosec|esomeprazol|nexium|lansoprazol|prevacid|rabeprazol|aciphex|dexlansoprazol|dexilant"
gib9_exact = "53021 5307 53082 53783 4560 45620 56202 56203 56212 56213 5693 56985 5780 5781 5789".split()
gib9_prefix = [f"53{organ}{cat}" for organ in range(1,5) for cat in (0,2,4,6)]
gib10 = """K2211 K226 I8501 I8511 K250 K252 K254 K256 K260 K262 K264 K266 K270 K272 K274 K276 K280 K282 K284 K286 K2901 K31811 K5521 K5701 K5711 K5713 K5721 K5731 K5733 K5741 K5751 K5753 K5781 K5791 K5793 K625 K920 K921 K922""".split()

def inlist(items):
    return "(" + ",".join(map(lit,items)) + ")"

def gib_code(code="code", ver="ver"):
    return f"(({ver}=9 AND ({code} IN {inlist(gib9_exact)} OR left({code},4) IN {inlist(gib9_prefix)})) OR ({ver}=10 AND {code} IN {inlist(gib10)}))"

def elapsed(expr, anchor="b.entry_at"):
    return f"date_diff('second',{anchor},{expr})/60.0"

def start_mimic():
    build("m0",f"""
        SELECT i.stay_id id, i.subject_id person, i.hadm_id hosp,
               {ts('i.intime')} entry_at, {ts('i.outtime')} exit_at,
               {ts('a.admittime')} adm_at, {ts('a.dischtime')} discharge_at,
               {ts('a.edregtime')} ed_at,{ts('a.edouttime')} ed_out,
               {ts('a.deathtime')} death_at,{ts('p.dod')}::DATE dod,
               a.hospital_expire_flag=1 hospitalized_death,
               p.anchor_age + year({ts('a.admittime')}) - p.anchor_year age
        FROM {raw('icu','icustays')} i LEFT JOIN {pq(H,'admissions')} a USING(subject_id,hadm_id)
        LEFT JOIN {pq(H,'patients')} p USING(subject_id)
    """)
    assert q("SELECT count(*)-count(DISTINCT id),count(*) FILTER(WHERE entry_at IS NULL OR exit_at <= entry_at) FROM m0")[0]==(0,0)
    build("m_dx",f"""
       WITH dx AS (
           SELECT hadm_id hosp, {icd("icd_code")} code,icd_version ver
           FROM {raw('hosp','diagnoses_icd')} SEMI JOIN m0 ON hadm_id=hosp
       )
       SELECT hosp, bool_or({gib_code()}) gib,
           bool_or((ver=9 AND (left(code,3) BETWEEN '410' AND '414' OR code IN ('V4581','V4582')))
               OR (ver=10 AND (left(code,3) BETWEEN 'I20' AND 'I25' OR code IN ('Z951','Z955','Z9861')))) ihd,
           bool_or((ver=9 AND left(code,3)='410') OR (ver=10 AND left(code,3) IN ('I21','I22'))) ami,
           bool_or((ver=9 AND code IN ('4560','45620','4561','45621','5712','5715','5716','5723'))
               OR (ver=10 AND (left(code,3)='I85' OR code IN ('K703','K7030','K7031','K717','K743','K744','K745','K746','K7460','K7469','K766')))) varlc,
           bool_or((ver=9 AND (code='9584' OR left(code,3) IN ('851','852','853','854','901','902')))
               OR (ver=10 AND regexp_matches(code,'^T0[0-7]|^T794|^S06[3456]|^S25|^S35'))) trauma
       FROM dx GROUP BY hosp HAVING bool_or({gib_code()})
    """)
    build("m_base", """
        SELECT b.*,d.* EXCLUDE(hosp,gib),
            date_diff('second',entry_at,exit_at)/60.0 exit_min,
            date_diff('second',entry_at,death_at)/60.0 death_min,
            date_diff('second',entry_at,discharge_at)/60.0 discharge_min,
            least(0,date_diff('second',entry_at,adm_at)/60.0,
                   date_diff('second',entry_at,ed_at)/60.0,-1440) history_start,
            date_diff('second',entry_at,ed_at)/60.0 ed_min,
            ed_at IS NOT NULL AND ed_out IS NOT NULL AND
                ed_out BETWEEN entry_at-INTERVAL '2 hours' AND entry_at+INTERVAL '2 hours' ed_link
        FROM m0 b JOIN m_dx d USING(hosp)
    """)
    build("m_rbc_raw",f"""
        SELECT DISTINCT b.id, {elapsed(ts('x.starttime'))} t, {elapsed(ts('x.endtime'))} end_t,
             x.itemid, x.linkorderid, x.orderid, x.stay_id source_stay, x.amount,
             CASE lower(trim(x.amountuom)) WHEN 'ml' THEN amount WHEN 'µl' THEN amount/1000
                 WHEN 'ul' THEN amount/1000 WHEN 'l' THEN amount*1000 END ml,
             CASE WHEN lower(trim(x.amountuom)) IN ('unit','units','u') THEN amount ELSE 0 END units,
             coalesce(CAST({ts('x.starttime')} AS TIME) <> TIME '00:00:00',false) nonmidnight,
             date_trunc('minute',{ts('x.starttime')})={ts('x.starttime')} exact_minute
        FROM {pq(I,'inputevents')} x JOIN m_base b ON x.hadm_id=b.hosp
        WHERE itemid IN (225168,226368,227070) AND amount>0
          AND lower(coalesce(statusdescription,''))<>'rewritten'
    """)
    build("m_rbc", """
        WITH previous AS (
          SELECT *,max(greatest(t,end_t)) OVER
              (PARTITION BY id,source_stay,itemid,coalesce(linkorderid,orderid)
               ORDER BY t,end_t,orderid ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end
          FROM m_rbc_raw WHERE t IS NOT NULL
        ), runs AS (
          SELECT *,sum(CASE WHEN prev_end IS NULL OR t>prev_end+1 THEN 1 ELSE 0 END) OVER
             (PARTITION BY id,source_stay,itemid,coalesce(linkorderid,orderid)
              ORDER BY t,end_t,orderid ROWS UNBOUNDED PRECEDING) segment
          FROM previous
        )
        SELECT id,min(t) t,sum(ml) ml, sum(units) units
        FROM runs GROUP BY id,source_stay,itemid,coalesce(linkorderid,orderid),segment
    """)
    prepare_m_labs()
    build("m_hb_raw",f"""
        SELECT id,t,val hb, CASE itemid WHEN '51222' THEN 1 WHEN '51640' THEN 1
                                      WHEN '50811' THEN 2 ELSE 4 END priority,
             'lab:'||itemid hb_source,try_cast(lid AS BIGINT) tie
        FROM m_labs WHERE itemid IN ('51222','50811','51640','51645') AND val BETWEEN 2 AND 22
          AND regexp_replace(uom,'[^a-z]','','g') IN ('gdl','gmdl')
        UNION ALL
        SELECT b.id,{elapsed(ts('x.charttime'))},x.valuenum,3,'chart:220228',0
        FROM {pq(I,'chartevents')} x JOIN m_base b ON x.hadm_id=b.hosp
        WHERE x.itemid=220228 AND x.valuenum BETWEEN 2 AND 22
          AND (x.valueuom IS NULL OR regexp_replace(lower(x.valueuom),'[^a-z]','','g') IN ('gdl','gmdl'))
    """)
    return q("SELECT itemid,uom,count(DISTINCT id) FROM m_labs GROUP BY ALL ORDER BY 1,3 DESC")

def finish_mimic():
    build("m_procs",f"""
       SELECT b.id, {ts('x.chartdate')}::DATE procedure_day,
           (x.icd_version=9 AND trim(x.icd_code)='3961') OR
           (x.icd_version=10 AND trim(x.icd_code)='5A1221Z') cpb,
           (x.icd_version=9 AND trim(x.icd_code) IN
             ('4222','4223','4224','4233','4412','4413','4414','4443','4513','4514','4516','4523','4524','4525','4542','4543','4822','4823','4824'))
           OR (x.icd_version=10 AND regexp_matches(lower(d.long_title),
                 'via natural or artificial opening endoscopic|control bleeding in gastrointestinal tract, via natural')
               AND regexp_matches(lower(d.long_title),'esophag|stomach|pylor|duoden|intestin|ileum|jejun|colon|rectum|sigmoid|cecum|gastric|gastrointestinal'))
           AS endo
       FROM {pq(H,'procedures_icd')} x JOIN m_base b ON x.hadm_id=b.hosp
       LEFT JOIN {pq(H,'d_icd_procedures')} d
           ON trim(x.icd_code)=trim(d.icd_code) AND x.icd_version=d.icd_version
    """)
    build("m_activity",f"""
       SELECT DISTINCT b.id FROM m_base b JOIN m_hb_raw h USING(id)
        WHERE h.t BETWEEN -1440 AND 4320 AND h.hb < 12
       UNION
       SELECT id FROM m_rbc WHERE t BETWEEN -1440 AND 4320
       UNION
       SELECT b.id FROM {pq(H,'emar')} x JOIN m_base b ON x.hadm_id=b.hosp
        WHERE regexp_matches(lower(x.medication),{lit(ppi)})
          AND regexp_matches(lower(event_txt),'administered|^started')
          AND {ts('x.charttime')} BETWEEN b.entry_at-INTERVAL '1 day' AND b.entry_at+INTERVAL '3 days'
       UNION
       SELECT b.id FROM {pq(I,'inputevents')} x JOIN m_base b ON x.hadm_id=b.hosp
        WHERE x.itemid=225910 AND x.amount>0 AND lower(coalesce(x.statusdescription,''))<>'rewritten'
          AND {ts('x.starttime')} BETWEEN b.entry_at-INTERVAL '1 day' AND b.entry_at+INTERVAL '3 days'
       UNION
       SELECT b.id FROM {pq(I,'procedureevents')} x JOIN m_base b ON x.hadm_id=b.hosp
        WHERE x.itemid IN (225434,225439) AND lower(coalesce(x.statusdescription,''))<>'rewritten'
          AND {ts('x.starttime')} BETWEEN b.entry_at-INTERVAL '1 day' AND b.entry_at+INTERVAL '3 days'
       UNION
       SELECT b.id FROM m_procs x JOIN m_base b USING(id)
        WHERE endo AND procedure_day BETWEEN (b.entry_at-INTERVAL '1 day')::DATE
                                          AND (b.entry_at+INTERVAL '3 days')::DATE
       UNION
       SELECT b.id FROM {pq(H,'hcpcsevents')} x JOIN m_base b ON x.hadm_id=b.hosp
        WHERE regexp_matches(trim(hcpcs_cd),'^(432[2345][0-9]|443[6789][0-9]|453[378][0-9]|G0105|G0121)$')
          AND regexp_matches(lower(short_description),'endo|colono|sigmoid|eg[dso]|esoph|entero|exam')
          AND {ts('x.chartdate')}::DATE BETWEEN (b.entry_at-INTERVAL '1 day')::DATE
                                          AND (b.entry_at+INTERVAL '3 days')::DATE
    """)
    build("m1","SELECT b.* FROM m_base b SEMI JOIN m_activity a USING(id)")
    build("m2","""
       SELECT * FROM m1 WHERE age>=18
       QUALIFY row_number() OVER (PARTITION BY person ORDER BY adm_at,entry_at,id)=1
    """)
    build("m3","""
       SELECT * FROM m2 b WHERE exit_min>=360 AND NOT coalesce(trauma,false)
           AND NOT EXISTS (SELECT 1 FROM m_procs p WHERE p.id=b.id AND cpb)
    """)
    build("m_shock",f"""
       WITH vitals AS (
           SELECT b.id,{elapsed(ts('v.charttime'))} t, heart_rate,
                CASE WHEN sbp>0 THEN sbp END art_sbp, CASE WHEN sbp_ni>0 THEN sbp_ni END cuff_sbp,
                CASE WHEN mbp>0 THEN mbp END art_map, CASE WHEN mbp_ni>0 THEN mbp_ni END cuff_map
           FROM {pq(D,'vitalsign')} v JOIN m3 b ON v.stay_id=b.id
       ), near_vitals AS (
           SELECT *, last_value(heart_rate IGNORE NULLS) OVER (
             PARTITION BY id ORDER BY t RANGE BETWEEN 5 PRECEDING AND CURRENT ROW) paired_hr
           FROM vitals WHERE t BETWEEN 0 AND 4320
       )
       SELECT id,t FROM near_vitals
       WHERE least(art_sbp,cuff_sbp)<90 OR least(art_map,cuff_map)<65
          OR paired_hr/nullif(least(art_sbp,cuff_sbp),0)>=1
       UNION ALL
       SELECT b.id,greatest(0,{elapsed(ts('v.starttime'))})
          FROM {pq(D,'vasoactive_agent')} v JOIN m3 b ON v.stay_id=b.id
          WHERE {ts('v.starttime')} IS NOT NULL AND {ts('v.endtime')}>=b.entry_at
            AND greatest(norepinephrine,epinephrine,phenylephrine,vasopressin,dopamine)>0
       UNION ALL
       SELECT id,t FROM m_labs WHERE itemid='50813' AND val>=2 AND uom='mmol/l'
    """)
    build("m_adapt","SELECT * REPLACE(least(exit_min,death_min) AS exit_min) FROM m3")
