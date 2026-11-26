from cohort_database import *
from mimic import ppi, lit

txrx = r"red[ -]+blood[ -]+cells?|(^|[^a-z])p?rbc[s]?([^a-z]|$)|packed[ -]+cells?"
badrx = r"platelet|ffp|plasma|cryoprecip|autolog|cell.saver|urine|blood.loss|non.prbc"
pressor = r"norepinephrin|levophed|epinephrin|phenylephrin|neosynephrin|neo.sy[nm]ephrin|vasopressin|dopamine|adrenalin"
acuterx = r"acute coronary syndrome|acute myocardial infarct|unstable angina|\bstemi\b|\bnstemi\b"
chronicrx = r"coronary arter(y|ial) disease|atherosclerotic heart|ischemic heart disease|chronic (ischemia|coronary)|stable angina|old myocardial infarct|history of (myocardial infarct|coronary)|s/p (cabg|ptca)|coronary stent"
historyrx = r"coronary|myocardial infarct|angina|cabg|ptca|(^|/)stent( placement)?($|/)"
varlcrx = r"varice|variceal|portal hypertension|cirrho"
traumarx = r"multiple (trauma|injur)|polytrauma|traumatic shock|cardiopulmonary bypass|\|with cpb|intracranial injury\|with (subarachnoid hemorrhage|subdural hematoma|intracerebral hemorrhage)|trauma[^|]*\|([^|]+\|)*(liver laceration|splenic injury|hemothorax|aortic injury)"
gibrx = (r"(^|\|)[^|]*(\bgi bleed|gastrointestinal (bleed|hemorrha)|upper gi (bleed|hemorrha)|lower gi (bleed|hemorrha)|"
         r"hematemes|melen[ae]|hematochez|ulcer[^|]*hemorrha|varice[^|]*bleed|rectal (bleed|hemorrha))"
         r"|^bleeding[, -]+(upper gi|lower gi|other gi|gi\b|variceal)")
notgib = r"without (mention of )?(bleeding|hemorrha)|hemorrhage \(for gastrointestinal bleeding gi-see"

def match(col, pattern):
    return f"regexp_matches(lower({col}),{lit(pattern)})"

def is_gib(col):
    return f"({match(col,gibrx)} AND NOT {match(col,notgib)} AND NOT regexp_matches(lower({col}),'\\|gi bleeding / pud\\|peptic ulcer disease($|\\|)'))"

def start_eicu():
    build("e0",f"""
       SELECT patientunitstayid id,uniquepid person,patienthealthsystemstayid hosp,
           hospitalid hospital, apacheadmissiondx dxadmit,gender,
           CASE WHEN regexp_matches(age,'^>\\s*89$') THEN 90 ELSE try_cast(age AS INTEGER) END age,
           hospitaladmitoffset adm_min,unitdischargeoffset::DOUBLE exit_min,
           CASE WHEN lower(unitdischargestatus) IN ('expired','dead') THEN unitdischargeoffset
                WHEN lower(hospitaldischargestatus) IN ('expired','dead') THEN hospitaldischargeoffset END death_min,
           hospitaldischargeoffset discharge_min,lower(unitdischargestatus) unit_status,
           lower(hospitaldischargestatus) hospital_status,hospitaldischargeyear year_discharge,
           least(hospitaladmitoffset,-120,0) history_start,
           lower(unitadmitsource)='emergency department' ed_link,unitvisitnumber
       FROM {ec('patient')}
    """)
    assert q("SELECT count(*)-count(DISTINCT id) FROM e0")[0][0] == 0
    build("e_diagnoses",f"""
        WITH dx AS (SELECT patientunitstayid id, diagnosisoffset t, diagnosisstring s,
                 regexp_replace(coalesce(cast(icd9code AS VARCHAR),''),'[. ]','','g') codes
             FROM {ec('diagnosis',all_varchar=True)}
             WHERE diagnosisstring IS NOT NULL),
          standard AS (
            SELECT try_cast(id AS BIGINT) id, try_cast(t AS DOUBLE) t,s,codes FROM dx
          )
        SELECT id,
            bool_or(t<=1440 AND {is_gib('s')}) gib,
            bool_or(t<=1440 AND ({match('s',acuterx)} OR regexp_matches(codes,'(^|[,;])(410|I21|I22)[0-9A-Z]*($|[,;])'))
                 AND NOT regexp_matches(lower(s),'s/p ptca|old myocardial|history of|rule out|ruled out')) ami_acs,
            bool_or(({match('s',chronicrx)} OR regexp_matches(codes,
                 '(^|[,;])((412|413|414|I25)[0-9A-Z]*|V4581|V4582|Z951|Z955|Z9861)($|[,;])'))
                 AND NOT regexp_matches(lower(s),'rule out|ruled out|no known|possible')) chronic,
            bool_or(t<=1440 AND regexp_matches(codes,'(^|[,;])((41[0-4]|I2[0-5])[0-9A-Z]*|V4581|V4582|Z951|Z955|Z9861)($|[,;])')) ihd_code,
            bool_or({match('s',varlcrx)} OR regexp_matches(codes,
              '(^|[,;])(4560|45620|4561|45621|5712|5715|5716|5723|I85[0-9]*|K703[0-9]*|K717|K74[3-6][0-9]*|K766)($|[,;])')) varlc,
            bool_or({match('s',traumarx)} OR regexp_matches(codes,
              '(^|[,;])(9584|(85[1-4]|90[12]|T0[0-7]|T794|S06[3456]|S25|S35)[A-Z0-9]*)($|[,;])')) trauma
        FROM standard GROUP BY id
    """)
    build("e_hist",f"""
        SELECT try_cast(patientunitstayid AS BIGINT) id,
           bool_or({match('pasthistorypath', historyrx)} AND
              NOT regexp_matches(lower(pasthistorypath),'biliary|ureter|esophag|carotid|renal|peripheral')) ihd,
           bool_or({match('pasthistorypath',varlcrx)}) varlc
        FROM {ec('pasthistory',all_varchar=True)} GROUP BY id
    """)
    build("e1",f"""
        SELECT b.*,coalesce(d.ami_acs,false) acute,
          coalesce(d.chronic,false) OR coalesce(h.ihd,false) chronic,
          coalesce(d.ami_acs,false) OR coalesce(d.chronic,false) OR coalesce(h.ihd,false)
              OR coalesce(d.ihd_code,false) ihd,
          coalesce(d.varlc,false) OR coalesce(h.varlc,false) OR coalesce({match('dxadmit',varlcrx)},false) varlc,
          coalesce(d.trauma,false) OR coalesce({match('dxadmit',traumarx)},false) trauma
        FROM e0 b LEFT JOIN e_diagnoses d USING(id) LEFT JOIN e_hist h USING(id)
        WHERE {is_gib('dxadmit')} OR d.gib
    """)
    build("e2","""
        SELECT * FROM e1 WHERE age>=18 AND person IS NOT NULL
        QUALIFY row_number() OVER
            (PARTITION BY person ORDER BY year_discharge,hosp,adm_min DESC,unitvisitnumber,id)=1
    """)
    build("e3","SELECT * FROM e2 WHERE exit_min>=360 AND NOT trauma")
    build("e_adapt","SELECT * REPLACE(least(exit_min,death_min) AS exit_min) FROM e3")

    build("e_io",f"""
       SELECT try_cast(patientunitstayid AS BIGINT) id,try_cast(intakeoutputoffset AS DOUBLE) t,
              try_cast(cellvaluenumeric AS DOUBLE) ml,celllabel AS term,cellpath AS term_path
       FROM {ec('intakeoutput',all_varchar=True)}
       WHERE {match("concat_ws(' ',celllabel,cellpath)",txrx)}
         AND NOT {match("concat_ws(' ',celllabel,cellpath)",badrx)}
         AND lower(cellpath) LIKE '%intake (ml)%'
         AND try_cast(cellvaluenumeric AS DOUBLE)>0
    """)
    build("e_treatment_rbc",f"""
       SELECT try_cast(patientunitstayid AS BIGINT) id,try_cast(treatmentoffset AS DOUBLE) t,treatmentstring AS term
       FROM {ec('treatment',all_varchar=True)}
       WHERE regexp_matches(lower(treatmentstring),'\\|packed red blood cells(\\||$)|transfusion[^|]*(p?rbc|red blood cell)')
         AND NOT {match("regexp_extract(treatmentstring,'[^|]+$')",badrx)}
    """)
    build("e_hospital_log","""
       SELECT hospital,count(DISTINCT person) people FROM e0
       SEMI JOIN (SELECT id FROM e_io WHERE t IS NOT NULL
              UNION SELECT id FROM e_treatment_rbc WHERE t IS NOT NULL) r USING(id)
       GROUP BY hospital HAVING count(DISTINCT person)>=10
    """)
    build("e_rbc","""
        WITH io AS (
            SELECT id,t,max(ml) ml FROM e_io WHERE t IS NOT NULL GROUP BY id,t
        ), tx AS (
            SELECT DISTINCT id,t FROM e_treatment_rbc WHERE t IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM io WHERE io.id=e_treatment_rbc.id
                                        AND abs(io.t-e_treatment_rbc.t)<=60)
        ), tx_lead AS (
            SELECT id,t,lag(t) OVER(PARTITION BY id ORDER BY t) prev FROM tx
        )
        SELECT id,t,ml,0 units,'intakeOutput' channel FROM io
        UNION ALL
        SELECT id,t,NULL,0,'treatment' FROM tx_lead WHERE prev IS NULL OR t-prev>60
    """)
    build("e_labs",f"""
       SELECT try_cast(l.patientunitstayid AS BIGINT) id,try_cast(labresultoffset AS DOUBLE) t,
             lower(trim(labname)) lab_kind,try_cast(labresult AS DOUBLE) val,
             lower(regexp_replace(coalesce(labmeasurenamesystem,labmeasurenameinterface),
                                  '[^A-Za-z]','','g')) unit,
             try_cast(labid AS BIGINT) lab_key,
             try_cast(labresultrevisedoffset AS BIGINT) revised
       FROM {ec('lab',all_varchar=True)} l SEMI JOIN e3 b ON try_cast(l.patientunitstayid AS BIGINT)=b.id
       WHERE lower(trim(labname)) IN ('hgb','hemoglobin','platelets x 1000','pt - inr',
                                    'pt','creatinine','bun','lactate')
    """)
    build("e_hb_raw","""
         SELECT id,t,val hb,1 priority,lab_kind hb_source,lab_key tie
         FROM e_labs WHERE lab_kind IN ('hgb','hemoglobin') AND val BETWEEN 2 AND 22
           AND unit IN ('gdl','gmdl')
         QUALIFY row_number() OVER(PARTITION BY id,t ORDER BY revised DESC NULLS LAST,lab_key DESC)=1
    """)
    print("eICU laboratory and transfusion sources scanned.",flush=True)


def finish_eicu():
    build("e_press",f"""
        SELECT b.id,try_cast(infusionoffset AS DOUBLE) t,
         try_cast(replace(drugrate,',','') AS DOUBLE) drug_rate,
         try_cast(replace(infusionrate,',','') AS DOUBLE) infusion_rate
        FROM {ec('infusiondrug',all_varchar=True)} x JOIN e3 b
          ON try_cast(x.patientunitstayid AS BIGINT)=b.id
        WHERE {match('drugname',pressor)}
          AND NOT regexp_matches(lower(drugname),'epinephrine.*(racem|neb)|racemic|intranasal|local anesthe')
    """)
    build("e_vp",f"""
        SELECT b.id,try_cast(observationoffset AS DOUBLE) t,try_cast(heartrate AS DOUBLE) hr,
           try_cast(systemicsystolic AS DOUBLE) sbp,try_cast(systemicmean AS DOUBLE) mean_bp
        FROM {ec('vitalperiodic',all_varchar=True)} x JOIN e3 b
            ON try_cast(x.patientunitstayid AS BIGINT)=b.id
        WHERE try_cast(observationoffset AS DOUBLE) BETWEEN 0 AND 4320
          AND (heartrate IS NOT NULL OR systemicsystolic IS NOT NULL OR systemicmean IS NOT NULL)
    """)
    build("e_va",f"""
        SELECT b.id,try_cast(observationoffset AS DOUBLE) t,
           try_cast(noninvasivesystolic AS DOUBLE) sbp,try_cast(noninvasivemean AS DOUBLE) mean_bp
        FROM {ec('vitalaperiodic',all_varchar=True)} x JOIN e3 b
            ON try_cast(x.patientunitstayid AS BIGINT)=b.id
        WHERE try_cast(observationoffset AS DOUBLE) BETWEEN 0 AND 4320
          AND (noninvasivesystolic IS NOT NULL OR noninvasivemean IS NOT NULL)
    """)
    build("e_shock","""
        WITH bp AS (
          SELECT id,t,sbp,mean_bp FROM e_vp WHERE sbp>0 OR mean_bp>0
          UNION ALL SELECT id,t,sbp,mean_bp FROM e_va WHERE sbp>0 OR mean_bp>0
        ), heart AS (
          SELECT id,t,max(hr) hr FROM e_vp WHERE hr>0 GROUP BY id,t
        )
        SELECT bp.id,bp.t FROM bp ASOF LEFT JOIN heart h ON bp.id=h.id AND bp.t>=h.t
          WHERE bp.sbp BETWEEN 1 AND 89 OR bp.mean_bp BETWEEN 1 AND 64
                OR (bp.t-h.t<=5 AND bp.sbp>0 AND h.hr/bp.sbp>=1)
        UNION ALL
        SELECT id,t FROM e_press WHERE greatest(drug_rate,infusion_rate)>0
        UNION ALL
        SELECT id,t FROM e_labs WHERE lab_kind='lactate' AND val>=2 AND unit='mmoll'
    """)

def e_covariates():
    build("e_apache",f"""
        SELECT DISTINCT b.id FROM {ec('apachepatientresult',all_varchar=True)} a
           JOIN e_final b ON try_cast(a.patientunitstayid AS BIGINT)=b.id AND b.variant='W1'
        WHERE lower(apacheversion)='iv' AND try_cast(apachescore AS INTEGER)>=0
    """)
    build("e_airway",f"""
        SELECT DISTINCT b.id FROM {ec('respiratorycharting',all_varchar=True)} a
            JOIN e_final b ON try_cast(a.patientunitstayid AS BIGINT)=b.id AND b.variant='W1'
        WHERE try_cast(respchartoffset AS DOUBLE) BETWEEN 0 AND b.t0
          AND (regexp_matches(lower(respchartvaluelabel),
                  '^o2 device$|^oxygen delivery method$|^mechanical ventilator mode$|^non-invasive ventilation mode$')
               OR regexp_matches(lower(respchartvaluelabel),'endotracheal tube|ett position|tracheostomy tube'))
          AND nullif(trim(respchartvalue),'') IS NOT NULL
        UNION
        SELECT DISTINCT b.id FROM {ec('treatment',all_varchar=True)} a
            JOIN e_final b ON try_cast(a.patientunitstayid AS BIGINT)=b.id AND b.variant='W1'
        WHERE try_cast(treatmentoffset AS DOUBLE) BETWEEN 0 AND b.t0
          AND regexp_matches(lower(treatmentstring),
              '\\|(mechanical ventilation|non-invasive ventilation|oxygen therapy|endotracheal intubation)(\\|| |$)')
        UNION
        SELECT DISTINCT b.id FROM {ec('ventilation_events',all_varchar=True)} a
            JOIN e_final b ON try_cast(a.patientunitstayid AS BIGINT)=b.id AND b.variant='W1'
        WHERE try_cast(hrs AS DOUBLE)*60 BETWEEN 0 AND b.t0
          AND lower(event) IN ('mechvent start','mechvent end','niv start','niv end','o2 start','roomair end')
    """)
    build("e_ppi",f"""
       SELECT DISTINCT b.id FROM {ec('medication',all_varchar=True)} x
          JOIN e_final b ON try_cast(x.patientunitstayid AS BIGINT)=b.id AND b.variant='W1'
       WHERE try_cast(drugorderoffset AS DOUBLE) BETWEEN 0 AND b.t0
         AND NOT coalesce(try_cast(drugordercancelled AS BOOLEAN),false)
         AND {match('drugname',ppi+'|famotidine|ranitidine|cimetidine|nizatidine|pepcid|zantac|tagamet|axid')}
    """)
    return q("""
       SELECT count(*),
         count(*) FILTER(WHERE age IS NOT NULL),
         count(*) FILTER(WHERE lower(gender) IN ('male','female','other')),
         count(*) FILTER(WHERE t0>=1440 AND a.id IS NOT NULL),
         count(a.id),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_vp x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND x.hr>0)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_vp x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND (x.sbp>0 OR x.mean_bp>0))
                            OR EXISTS(SELECT 1 FROM e_va x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND (x.sbp>0 OR x.mean_bp>0))),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_hb_raw x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_labs x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND x.lab_kind='platelets x 1000' AND x.val IS NOT NULL)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_labs x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND x.lab_kind IN ('pt','pt - inr') AND x.val IS NOT NULL)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_labs x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND x.lab_kind='creatinine' AND x.val IS NOT NULL)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_labs x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND x.lab_kind='bun' AND x.val IS NOT NULL)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_labs x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND x.lab_kind='lactate' AND x.val IS NOT NULL)),
         count(*) FILTER(WHERE EXISTS(SELECT 1 FROM e_press x WHERE x.id=b.id AND x.t BETWEEN 0 AND b.t0 AND greatest(x.drug_rate,x.infusion_rate) IS NOT NULL)),
         count(v.id), count(p.id)
       FROM e_final b LEFT JOIN e_apache a USING(id) LEFT JOIN e_airway v USING(id) LEFT JOIN e_ppi p USING(id)
       WHERE variant='W1'
    """)[0]
