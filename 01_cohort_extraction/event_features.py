from cohort_database import *
from mimic import I,H,D,elapsed
from clinical_sources import LAB_CODES,E_LABS
from eicu import match

BLOOD = r"fresh frozen plasma|(^|[^a-z])ffp([^a-z]|$)|platelets?|cryoprecip"
OMIT_BLOOD = r"urine|output \(ml\)|loss|saline|\bns for\b|number of|count"
RR_ITEMS=(220210,224690)
# priority 1: arterial pressure and official laboratory result, 2: cuff/bedside.
VITALS={
    220045:("hr",1),220050:("sbp",1),225309:("sbp",1),220179:("sbp",2),
    220052:("map",1),225312:("map",1),220181:("map",2),
    220210:("rr",1),224690:("rr",2),220277:("spo2",1),
}
LAB_RANGE = {
    "hb":(2,22),"platelets":(1,2000),"inr":(0.4,25),"pt":(5,200),"ptt":(5,300),
    "creatinine":(0.1,40),"bun":(1,350),"wbc":(0.01,500),"bicarbonate":(2,65),
    "lactate":(0.1,40),"albumin":(0.5,8),"bilirubin":(0,90),
}
VITAL_RANGE={"hr":(20,300),"sbp":(25,300),"map":(15,220),"rr":(2,100),"spo2":(20,100)}

def ranges(values):
    return " OR ".join(f"(kind={lit(key)} AND val BETWEEN {low} AND {high})"
                       for key,(low,high) in values.items())

def mapped(column, dictionary):
    return f"CASE {column} " + " ".join(f"WHEN {lit(str(key))} THEN {lit(value)}"
                                        for key,value in dictionary.items()) + " END"

def bounds(k):
    e_lower="greatest(a.adm_min-120,a.t0-1440)" if k=="e" else """greatest(
               date_diff('second',a.entry_at,coalesce(a.ed_at,a.adm_at))/60.0-120,a.t0-1440)"""
    v_lower="greatest(a.adm_min,a.t0-360)" if k=="e" else "greatest(coalesce(a.ed_min,a.t0-360),a.t0-360)"
    build(f"{k}_bounds",f"""
        SELECT a.*, {e_lower} lab_start,{v_lower} vital_start
        FROM {k}_trial_stays a
    """)
    build(f"{k}_need",f"""
        SELECT id,any_value(hosp) hosp,min(lab_start) lab_min,min(vital_start) vital_min,
            max(t0+grace) feature_max FROM {k}_bounds GROUP BY id
    """)

def m_events():
    vals=",".join(f"({item},{lit(name)},{priority})" for item,(name,priority) in VITALS.items())
    chart_items=",".join(map(str,(*VITALS,220228,227466,223848,223849,226732,229314)))
    build("m_chart",f"""
      SELECT b.id,x.itemid,x.valuenum val,x.value,lower(x.valueuom) unit,
          {elapsed(ts('x.charttime'))} t,
          greatest({elapsed(ts('x.charttime'))},{elapsed(ts('x.storetime'))}) known
      FROM {pq(I,'chartevents')} x JOIN m_need a ON x.hadm_id=a.hosp JOIN m0 b ON a.id=b.id
      WHERE x.itemid IN ({chart_items})
        AND {ts('x.charttime')} BETWEEN b.entry_at+(least(a.lab_min,a.vital_min)-60)*INTERVAL '1 minute'
                                   AND b.entry_at+a.feature_max*INTERVAL '1 minute'
    """)
    build("m_labe",f"""
      WITH dictionary(itemid,kind) AS (VALUES {','.join(f"({lit(str(i))},{lit(kind)})"
                       for kind,items in LAB_CODES.items() for i in items if kind!='tnt')}),
        mapped_lab AS (
           SELECT b.id,d.kind,x.val,x.chart_at,
               {elapsed('x.chart_at')} t,greatest({elapsed('x.chart_at')},{elapsed('x.available_at')}) known,
               CASE x.itemid WHEN '51222' THEN 1 WHEN '50811' THEN 2 ELSE 3 END pref,
               x.uom unit
           FROM m_lab_cache x JOIN dictionary d USING(itemid) JOIN m_base b ON x.subject_id=b.person
                AND (x.hadm_id=b.hosp OR x.hadm_id IS NULL)
           JOIN m_need a ON a.id=b.id AND {elapsed('x.chart_at')} BETWEEN a.lab_min AND a.feature_max
           UNION ALL
           SELECT x.id,CASE itemid WHEN 227466 THEN 'ptt' ELSE 'hb' END,val,NULL,t,known,4,unit
           FROM m_chart x WHERE itemid IN (220228,227466)
           UNION ALL
           SELECT b.id,'hb',x.hemoglobin,{ts('x.charttime')},{elapsed(ts('x.charttime'))},
               {elapsed(ts('x.charttime'))},5,'g/dl'
           FROM {pq(D,'bg')} x JOIN m_base b ON x.hadm_id=b.hosp JOIN m_need a USING(id)
           WHERE x.hemoglobin IS NOT NULL AND {elapsed(ts('x.charttime'))} BETWEEN a.lab_min AND a.feature_max
             AND NOT EXISTS(SELECT 1 FROM m_lab_cache l WHERE l.subject_id=b.person
                   AND (l.hadm_id=b.hosp OR l.hadm_id IS NULL) AND l.itemid='50811'
                   AND l.chart_at={ts('x.charttime')})
        )
        SELECT id,kind,val,t,known,pref FROM mapped_lab WHERE ({ranges(LAB_RANGE)})
          AND (kind<>'hb' OR unit IN ('g/dl','gm/dl','g/dl.'))
    """)
    build("m_vitale",f"""
      WITH raw_chart AS (SELECT x.id,d.kind,x.val,x.t,x.known,d.pref FROM m_chart x
                         JOIN (VALUES {vals}) d(itemid,kind,pref) USING(itemid)),
        derived AS (
            SELECT a.id,d.kind,d.val,{elapsed(ts('x.charttime'))} t,
                   {elapsed(ts('x.charttime'))} known,d.pref+2 pref
            FROM {pq(D,'vitalsign')} x JOIN m_need a ON x.stay_id=a.id JOIN m0 b ON a.id=b.id
            CROSS JOIN LATERAL (VALUES ('hr',heart_rate,1),('sbp',sbp,1),('sbp',sbp_ni,2),
                ('map',mbp,1),('map',mbp_ni,2),('rr',resp_rate,1),('spo2',spo2,1)) d(kind,val,pref)
            WHERE d.val IS NOT NULL AND {elapsed(ts('x.charttime'))} BETWEEN a.vital_min AND a.feature_max
        )
        SELECT * FROM raw_chart WHERE {ranges(VITAL_RANGE)}
        UNION ALL SELECT * FROM derived z WHERE ({ranges(VITAL_RANGE)})
          AND NOT EXISTS (SELECT 1 FROM raw_chart r WHERE r.id=z.id AND r.t=z.t AND r.kind=z.kind
                          AND mod(r.pref,2)=mod(z.pref,2) AND r.val IS NOT NULL)
    """)
    build("m_blood",f"""
       SELECT a.id,{elapsed(ts('x.starttime'))} t,
           greatest({elapsed(ts('x.starttime'))},{elapsed(ts('x.storetime'))}) known
       FROM {pq(I,'inputevents')} x JOIN m_need a ON a.hosp=x.hadm_id JOIN m0 b ON a.id=b.id
       JOIN {pq(I,'d_items')} d USING(itemid)
       WHERE {match('d.label',BLOOD)} AND NOT {match('d.label',r'pheresis|autolog')}
         AND x.amount>0 AND lower(coalesce(x.statusdescription,''))<>'rewritten'
    """)

def e_events():
    lab_expr=mapped("lower(trim(x.labname))",E_LABS)
    build("e_labe",f"""
       WITH selected AS (
         SELECT a.id,{lab_expr} kind,x.labresult val,x.labresultoffset::DOUBLE t,
                greatest(x.labresultoffset,x.labresultrevisedoffset)::DOUBLE known,
                1 pref,x.labid,x.labmeasurenamesystem unit
         FROM {ec('lab')} x JOIN e_need a ON x.patientunitstayid=a.id
         WHERE x.labresult IS NOT NULL AND x.labresultoffset BETWEEN a.lab_min AND a.feature_max
       )
       SELECT id,kind,val,t,known,pref FROM selected WHERE ({ranges(LAB_RANGE)})
       AND (kind<>'hb' OR lower(unit) IN ('g/dl','gm/dl'))
    """)
    build("e_vitale",f"""
        WITH periodic AS (
          SELECT a.id,d.kind,d.val,x.observationoffset::DOUBLE t,x.observationoffset::DOUBLE known,1 pref
          FROM {ec('vitalperiodic')} x JOIN e_need a ON x.patientunitstayid=a.id
          CROSS JOIN LATERAL (VALUES ('hr',heartrate),('sbp',systemicsystolic),('map',systemicmean),
                                     ('rr',respiration),('spo2',sao2)) d(kind,val)
          WHERE x.observationoffset BETWEEN a.vital_min AND a.feature_max
        ), cuff AS (
          SELECT a.id,d.kind,d.val,x.observationoffset::DOUBLE t,x.observationoffset::DOUBLE known,2 pref
          FROM {ec('vitalaperiodic')} x JOIN e_need a ON x.patientunitstayid=a.id
          CROSS JOIN LATERAL (VALUES ('sbp',noninvasivesystolic),('map',noninvasivemean)) d(kind,val)
          WHERE x.observationoffset BETWEEN a.vital_min AND a.feature_max
        )
        SELECT * FROM periodic WHERE {ranges(VITAL_RANGE)}
        UNION ALL SELECT * FROM cuff WHERE {ranges(VITAL_RANGE)}
    """)
    build("e_blood",f"""
       SELECT a.id,x.intakeoutputoffset::DOUBLE t,
         greatest(x.intakeoutputoffset,x.intakeoutputentryoffset)::DOUBLE known
       FROM {ec('intakeoutput')} x JOIN e_need a ON x.patientunitstayid=a.id
       WHERE x.cellvaluenumeric>0 AND lower(cellpath) LIKE '%intake (ml)%' AND {match('x.celllabel',BLOOD)}
             AND NOT {match('x.celllabel',OMIT_BLOOD)}
       UNION
       SELECT a.id,x.treatmentoffset::DOUBLE t,x.treatmentoffset::DOUBLE known
       FROM {ec('treatment')} x JOIN e_need a ON x.patientunitstayid=a.id
       WHERE lower(treatmentstring) LIKE '%blood product administration%'
         AND {match("regexp_extract(x.treatmentstring,'[^|]+$')",BLOOD)}
    """)

def prepare_events(k):
    bounds(k)
    (m_events if k=="m" else e_events)()
    for kind in ("lab","vital"):
        require(one(f"SELECT count(*) FILTER(WHERE t IS NULL OR known<t) FROM {k}_{kind}e")[0]==0,
                f"{k} {kind}: measurement/availability timestamps malformed")
