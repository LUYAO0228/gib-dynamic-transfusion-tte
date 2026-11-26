from cohort_database import *
from mimic import H,I,D,elapsed,ppi
from eicu import match,pressor,acuterx,chronicrx,historyrx,varlcrx,is_gib

ACID=ppi+"|famotidine|ranitidine|cimetidine|nizatidine|pepcid|zantac|tagamet|axid"
ANTITHROM=r"aspirin|acetylsalicy|clopidogrel|prasugrel|ticagrelor|ticlopidine|warfarin|coumadin|apixaban|rivaroxaban|dabigatran|edoxaban|eliquis|xarelto|pradaxa|savaysa"
SCOPE=r"endoscop|colonoscop|sigmoidoscop|esophagogastroduodeno|\begd\b"
INFD_PRBC=r"volume-transfuse rbc|volume-transfuse unmatched o-neg rbc|^\s*prbc|^\s*rbc|^packed (red blood )?cells|^blood \(prbcs\)"
INFD_EXCLUDE=r"non.prbc|ns for blood|flush|ffp|plasma|platelet|cryo"
ORAL_VENT=r"invasivevent|invasive ventilation|mechanical ventila|ventilat(or|ed)|endotracheal|\bett\b|tracheost|\btrach"
NONINV_VENT=r"non.?invasive|\bniv\b|bipap|cpap|high.?flow|hfnc|optiflow|vapotherm"
NORMAL_O2=r"nasal cannula|room air|roomair|simple mask|face mask|non.rebreath|humidifier|oxygen therapy|nasal prong|t.?piece|extubat"
CHRONIC={
  "chf":("428","I50","heart failure|cardiac failure"),
  "af":("42731","I48","atrial fibrillation|atrial flutter"),
  "ckd_esrd":("585","N18","chronic kidney|chronic renal|end.stage renal|dialysis.dependent|\besrd\b"),
  "copd":("490|491|492|496","J4[0-4]","\bcopd\b|chronic obstructive|emphysema|chronic bronchitis"),
  "diabetes":("250","E0[89]|E1[013]","diabet"),
  "malignancy":("1[4-9][0-9]|20[0-8]","C[0-9][0-9]","malignan|\bcancer\b|\bcarcinoma\b|\blymphoma\b|\bleukemia\b"),
}
def event_vent(col):
    return f"""CASE WHEN {match(col,NONINV_VENT)} THEN 1
                    WHEN {match(col,ORAL_VENT)} THEN 2
                    WHEN {match(col,NORMAL_O2)} THEN 0 END"""

def m_clinical():
    # Hospital billing diagnoses are explicitly named as encounter-level phenotype proxies.
    build("m_comorb",f"""
      WITH dx AS (
        SELECT distinct o.hosp, {icd('x.icd_code')} code,x.icd_version ver,d.long_title description
        FROM {raw('hosp','diagnoses_icd')} x JOIN (SELECT DISTINCT hosp FROM m_need) o ON x.hadm_id=o.hosp
        LEFT JOIN m_dx_dict d ON {icd('x.icd_code')}=d.code AND x.icd_version=d.ver
      )
      SELECT hosp,{','.join(f"bool_or((ver=9 AND regexp_matches(code,'^({us})')) OR (ver=10 AND regexp_matches(code,'^({who})'))) AS {name}"
                                for name,(us,who,_) in CHRONIC.items())},
        bool_or(regexp_matches(code,'^(53[1-4][0246]|K2[5-8][0246])')) pud_flag,
        bool_or(regexp_matches(code,'^(530(21|7|82)|53[1-4][0246]|53783|4560|45620|5780|K2211|K226|I85[01]1|K2[5-8][0246]|K2901|K31811|K920)')) upper,
        bool_or(regexp_matches(code,'^(5620[23]|5621[23]|5693|56985|K5521|K57(01|11|13|21|31|33|41|51|53|81|91|93)|K625)')) lower_gib
      FROM dx GROUP BY hosp
    """)
    build("m_comorb_pre",f"""
       WITH prior AS (
           SELECT b.id,b.cohort,a.hadm_id
           FROM m_trial_stays b JOIN {pq(H,'admissions')} a ON b.person=a.subject_id
           WHERE b.hosp<>a.hadm_id AND {ts('a.dischtime')}<b.entry_at+b.t0*INTERVAL '1 minute'
       ), dx AS (
           SELECT x.hadm_id,{icd('x.icd_code')} code,x.icd_version ver
           FROM {raw('hosp','diagnoses_icd')} x SEMI JOIN prior p ON x.hadm_id=p.hadm_id
       )
       SELECT p.id,p.cohort,true prior_hospital_history_flag,
          {','.join(f"coalesce(bool_or((ver=9 AND regexp_matches(code,'^({us})')) OR (ver=10 AND regexp_matches(code,'^({who})'))),false) AS {name}"
                 for name,(us,who,_) in CHRONIC.items())}
       FROM prior p LEFT JOIN dx USING(hadm_id) GROUP BY p.id,p.cohort
    """)
    build("m_demo",f"""
        SELECT a.id,x.gender sex,y.race race_eth,z.first_careunit icu_unit_type,z.first_careunit raw_source,
               y.admission_location, a.ed_at IS NOT NULL has_ed
        FROM (SELECT DISTINCT id,person,hosp,ed_at FROM m_trial_stays) a
        JOIN {pq(H,'patients')} x ON a.person=x.subject_id
        JOIN {pq(H,'admissions')} y ON a.hosp=y.hadm_id
        JOIN {raw('icu','icustays')} z ON a.id=z.stay_id
    """)
    build("m_press_intervals",f"""
       SELECT a.id,{elapsed(ts('x.starttime'))} t,{elapsed(ts('x.endtime'))} stop_t,
               greatest(norepinephrine,epinephrine,phenylephrine,vasopressin,dopamine)>0 active,
               n.norepinephrine_equivalent_dose nee
       FROM {pq(D,'vasoactive_agent')} x JOIN m_need a ON x.stay_id=a.id JOIN m0 b ON a.id=b.id
       LEFT JOIN {pq(D,'norepinephrine_equivalent_dose')} n
         ON x.stay_id=n.stay_id AND x.starttime=n.starttime AND x.endtime=n.endtime
       WHERE {ts('x.starttime')} IS NOT NULL AND {ts('x.endtime')} IS NOT NULL
    """)
    build("m_resp",f"""
      WITH airway AS (
        SELECT a.id,{elapsed(ts('x.starttime'))} t,
            greatest({elapsed(ts('x.starttime'))},{elapsed(ts('x.storetime'))}) known,
            CASE WHEN x.itemid=225794 THEN 1 WHEN x.itemid=225477 THEN 0 ELSE 2 END val,
            1 pref
        FROM {pq(I,'procedureevents')} x JOIN m_need a ON x.stay_id=a.id JOIN m0 b ON a.id=b.id
        WHERE x.itemid IN (224385,225448,225792,225794,226237,226260,225477)
           AND coalesce(lower(x.statusdescription),'')<>'rewritten'
      ), all_resp AS (
        SELECT * FROM airway
        UNION ALL
        SELECT id,t,known,{event_vent('value')},2 FROM m_chart
         WHERE itemid IN (223848,223849,226732,229314) AND value IS NOT NULL
        UNION ALL
        SELECT a.id,{elapsed(ts('x.starttime'))},{elapsed(ts('x.starttime'))},
          CASE lower(x.ventilation_status) WHEN 'invasivevent' THEN 2 WHEN 'trach' THEN 2
              WHEN 'noninvasivevent' THEN 1 WHEN 'hfnc' THEN 1 WHEN 'supplementalooxygen' THEN 0
              WHEN 'supplementalo2' THEN 0 WHEN 'none' THEN 0 END,3
        FROM {pq(D,'ventilation')} x JOIN m_need a ON x.stay_id=a.id JOIN m0 b ON a.id=b.id
      )
      SELECT * FROM all_resp WHERE val IS NOT NULL AND t IS NOT NULL
    """)
    build("m_drug_endo",f"""
      WITH meds AS (
        SELECT a.id,{elapsed(ts('x.charttime'))} t,
          greatest({elapsed(ts('x.charttime'))},{elapsed(ts('x.storetime'))}) known,
          {match('x.medication',ACID)} acid,
          {match('x.medication',ANTITHROM)} antithrom,false endoscopy
        FROM {pq(H,'emar')} x JOIN m_need a ON x.hadm_id=a.hosp JOIN m0 b ON a.id=b.id
        WHERE regexp_matches(lower(event_txt),'administered|^started')
          AND ({match('x.medication',ACID)} OR {match('x.medication',ANTITHROM)})
        UNION ALL
        SELECT a.id,{elapsed(ts('x.starttime'))},
            greatest({elapsed(ts('x.starttime'))},{elapsed(ts('x.storetime'))}),
            itemid IN (225907,225910),itemid=225152,false
        FROM {pq(I,'inputevents')} x JOIN m_need a ON x.hadm_id=a.hosp JOIN m0 b ON a.id=b.id
        WHERE itemid IN (225907,225910,225152) AND x.amount>0
           AND lower(coalesce(x.statusdescription,''))<>'rewritten'
        UNION ALL
        SELECT a.id,{elapsed(ts('x.starttime'))},
            greatest({elapsed(ts('x.starttime'))},{elapsed(ts('x.storetime'))}),false,false,true
        FROM {pq(I,'procedureevents')} x JOIN m_need a ON x.hadm_id=a.hosp JOIN m0 b ON a.id=b.id
        WHERE itemid IN (225434,225439) AND lower(coalesce(x.statusdescription,''))<>'rewritten'
      )
      SELECT * FROM meds WHERE t IS NOT NULL
    """)
    build("m_day_endoscopy",f"""
        SELECT p.id,p.procedure_day FROM m_procs p SEMI JOIN m_need a USING(id)
          WHERE p.endo AND p.procedure_day IS NOT NULL
        UNION
        SELECT a.id,{ts('x.chartdate')}::DATE FROM {pq(H,'hcpcsevents')} x JOIN m_need a ON x.hadm_id=a.hosp
           WHERE regexp_matches(trim(hcpcs_cd),'^(432[2345][0-9]|443[6789][0-9]|453[378][0-9]|G0105|G0121)$')
             AND {match('x.short_description',SCOPE+r'|esoph|entero|exam')}
    """)

def e_clinical():
    build("e_demo",f"""
        SELECT a.id,p.gender sex,p.ethnicity race_eth,p.unittype icu_unit_type,p.unitadmitsource raw_source,
               try_cast(p.unitadmittime24 AS TIME) entry_clock,p.admissionweight
        FROM e_need a JOIN {ec('patient')} p ON a.id=p.patientunitstayid
    """)
    build("e_dx_time",f"""
      WITH events AS (
        SELECT a.id,x.diagnosisoffset::DOUBLE t,x.diagnosisoffset::DOUBLE known,
           lower(x.diagnosisstring) term, {icd("coalesce(x.icd9code,'')")} code,false is_history
        FROM {ec('diagnosis')} x JOIN e_need a ON x.patientunitstayid=a.id
        UNION ALL
        SELECT a.id,x.pasthistoryoffset::DOUBLE,
           greatest(x.pasthistoryoffset,x.pasthistoryenteredoffset)::DOUBLE,
           lower(x.pasthistorypath),'',true
        FROM {ec('pasthistory')} x JOIN e_need a ON x.patientunitstayid=a.id
      )
      SELECT * FROM events WHERE t IS NOT NULL AND term IS NOT NULL
    """)
    build("e_prehist",f"""
      SELECT b.id,b.cohort,
          {','.join(f"coalesce(bool_or(regexp_matches(x.term,{lit(regex)}) OR regexp_matches(x.code,{lit('(^|[,;])('+us+'|'+who+')')})),false) AS {name}"
                        for name,(us,who,regex) in CHRONIC.items())},
          coalesce(bool_or(NOT is_history AND {match('x.term',acuterx)}
             AND NOT regexp_matches(x.term,'history|old|s/p ptca|ruled out|rule out')),false) eicu_acute_acs_flag,
          coalesce(bool_or({match('x.term',chronicrx)} OR (is_history AND {match('x.term',historyrx)}
             AND NOT regexp_matches(x.term,'biliary|ureter|esophag|peripheral|carotid|renal'))),false) eicu_chronic_cad_flag,
          coalesce(bool_or({match('x.term',varlcrx)}),false) varlc_pre_t0,
          coalesce(bool_or(regexp_matches(x.term,'peptic ulcer|bleeding\\|ulcer|gastric ulcer|duodenal ulcer')),false) pud_flag,
          coalesce(bool_or({is_gib('x.term')} AND regexp_matches(x.term,'upper gi|hematemes|varice|melena|esophag|duoden')),false) upper,
          coalesce(bool_or({is_gib('x.term')} AND regexp_matches(x.term,'lower gi|hematochez|rectal|colonic|divertic')),false) lower_gib
      FROM e_bounds b LEFT JOIN e_dx_time x ON x.id=b.id AND x.known<=b.t0 GROUP BY b.id,b.cohort
    """)
    build("e_site24",f"""
        WITH encounter_site AS (
            SELECT a.id,lower(a.dxadmit) term FROM e0 a SEMI JOIN e_need USING(id)
            UNION ALL
            SELECT id,term FROM e_dx_time WHERE NOT is_history AND t BETWEEN -1440 AND 1440
        ), sites AS (
            SELECT id,bool_or({is_gib('term')} AND regexp_matches(term,'upper gi|hematemes|esophag|melena|varice')) up,
                 bool_or({is_gib('term')} AND regexp_matches(term,'lower gi|hematochez|rectal|colon')) low
            FROM encounter_site GROUP BY id
        )
        SELECT id,CASE WHEN up AND low THEN 'mixed' WHEN up THEN 'upper' WHEN low THEN 'lower'
                       ELSE 'unspecified' END gib_site_24h FROM sites
    """)
    build("e_apache_isolated",f"""
        SELECT a.id,max(x.apachescore) apache_iv_do_not_adjust_main
        FROM {ec('apachepatientresult')} x JOIN e_need a ON x.patientunitstayid=a.id
        WHERE lower(x.apacheversion)='iv' AND x.apachescore>=0 GROUP BY a.id
    """)
    build("e_infd",f"""
      SELECT x.patientunitstayid id, x.infusionoffset::DOUBLE t,lower(x.drugname) term,
         try_cast(replace(x.drugrate,',','') AS DOUBLE) drug_rate,x.infusionrate infusion_rate,
         coalesce(CASE WHEN x.patientweight BETWEEN 20 AND 350 THEN x.patientweight END,
                  CASE WHEN d.admissionweight BETWEEN 20 AND 350 THEN d.admissionweight END) wt,
         x.infusiondrugid record_id
      FROM {ec('infusiondrug')} x JOIN e_demo d ON x.patientunitstayid=d.id
      WHERE x.infusionoffset IS NOT NULL
    """)
    build("e_rxname","""
      SELECT *,CASE WHEN regexp_matches(term,'norepinephrin|levophed|noradrenalin') THEN 'ne'
             WHEN regexp_matches(term,'epinephrin|adrenalin') THEN 'epi'
             WHEN regexp_matches(term,'phenylephrin|neo.?synephrin') THEN 'phenyl'
             WHEN regexp_matches(term,'vasopressin') THEN 'vasopressin'
             WHEN regexp_matches(term,'dopamine') THEN 'dopamine' END agent
      FROM e_infd WHERE NOT regexp_matches(term,'racemic|intranasal|nebulized|local anesthe')
    """)
    build("e_press_events",f"""
      WITH rx AS (
        SELECT id,t,agent,greatest(drug_rate,infusion_rate)>0 active,
           CASE
             WHEN agent='vasopressin' AND regexp_matches(term,'units?/(min|minute)') THEN drug_rate*2.5
             WHEN agent='vasopressin' AND regexp_matches(term,'units?/(hr|hour)') THEN drug_rate*2.5/60
             WHEN agent<>'vasopressin' AND regexp_matches(term,'(mcg|ug|µg)/kg/min') THEN drug_rate
             WHEN agent<>'vasopressin' AND regexp_matches(term,'(mcg|ug|µg)/min') THEN drug_rate/wt
           END * CASE agent WHEN 'phenyl' THEN 0.1 WHEN 'dopamine' THEN 0.01 ELSE 1 END nee,1 pref,record_id
        FROM e_rxname WHERE agent IS NOT NULL AND greatest(drug_rate,infusion_rate) IS NOT NULL
        UNION ALL
        SELECT a.id,x.treatmentoffset::DOUBLE,
           CASE WHEN {match('x.treatmentstring','norepinephrin|levophed|noradrenalin')} THEN 'ne'
                WHEN {match('x.treatmentstring','epinephrin|adrenalin')} THEN 'epi'
                WHEN {match('x.treatmentstring','phenylephrin|neo.?synephrin')} THEN 'phenyl'
                WHEN {match('x.treatmentstring','dopamine')} THEN 'dopamine'
                WHEN {match('x.treatmentstring','vasopressin($|[ |])')} THEN 'vasopressin'
                ELSE 'nonspecific' END,true,NULL,2,x.treatmentid
        FROM {ec('treatment')} x JOIN e_need a ON x.patientunitstayid=a.id
        WHERE regexp_matches(lower(x.treatmentstring),'\\|vasopressors(\\||$)|\\|inotropic agent\\|.*(dopamine|epineph|phenyleph|vasopress)')
          AND x.treatmentoffset IS NOT NULL
      )
      SELECT * FROM rx
    """)
    build("e_resp",f"""
       WITH all_resp AS (
        SELECT a.id,x.respchartoffset::DOUBLE t,greatest(x.respchartoffset,x.respchartentryoffset)::DOUBLE known,
         CASE WHEN {match('x.respchartvaluelabel','non-invasive ventilation mode')} THEN 1
           WHEN {match('x.respchartvaluelabel','mechanical ventilator mode')}
               AND nullif(trim(x.respchartvalue),'') IS NOT NULL
               AND NOT {match('x.respchartvalue',r'^(off|none|na|n/a)$')} THEN 2
           ELSE {event_vent('x.respchartvalue')} END val, 1 pref
        FROM {ec('respiratorycharting')} x JOIN e_need a ON x.patientunitstayid=a.id
        WHERE {match('x.respchartvaluelabel','ventilator mode|airway attachments|o2 device|oxygen delivery method|ventilator support mode')}
        UNION ALL
        SELECT a.id,round(x.hrs*60,5),round(x.hrs*60,5),
           CASE lower(event) WHEN 'mechvent start' THEN 2 WHEN 'trach' THEN 2 WHEN 'niv start' THEN 1
                WHEN 'mechvent end' THEN 0 WHEN 'niv end' THEN 0 WHEN 'o2 start' THEN 0
                WHEN 'nivend' THEN 0 END,2
        FROM {ec('ventilation_events')} x JOIN e_need a ON x.patientunitstayid=a.id
        UNION ALL
        SELECT a.id,x.treatmentoffset::DOUBLE,x.treatmentoffset::DOUBLE,
            {event_vent('x.treatmentstring')},3
        FROM {ec('treatment')} x JOIN e_need a ON x.patientunitstayid=a.id
        WHERE regexp_matches(lower(treatmentstring),'ventilation and oxygenation|endotracheal intubat|tracheostom|mechanical ventilation')
       )
       SELECT * FROM all_resp WHERE val IS NOT NULL AND t IS NOT NULL
    """)
    build("e_drug_endo",f"""
       WITH meds AS (
        SELECT id,t,t known,{match('term',ACID)} acid,
             {match('term',ANTITHROM)} OR regexp_matches(term,'heparin.*(units/hr|units/kg/hr)')
                   AND NOT regexp_matches(term,'prophy|flush|crrt') antithrom,false endoscopy
        FROM e_infd WHERE greatest(drug_rate,infusion_rate)>0
          AND ({match('term',ACID)} OR {match('term',ANTITHROM+r'|heparin')})
        UNION ALL
        SELECT a.id,x.drugstartoffset::DOUBLE,greatest(x.drugstartoffset,x.drugorderoffset)::DOUBLE,
            {match('x.drugname',ACID)},
            {match('x.drugname',ANTITHROM)} OR
              ({match('x.drugname','heparin')} AND {match("concat_ws(' ',x.routeadmin,x.dosage)",r'infusion|units/hr|units/kg/hr')}
              AND NOT {match("concat_ws(' ',x.drugname,x.routeadmin)",r'subcut|flush|prophy|crrt')}),
            false
        FROM {ec('medication')} x JOIN e_need a ON x.patientunitstayid=a.id
        WHERE NOT coalesce(try_cast(x.drugordercancelled AS BOOLEAN),false)
          AND ({match('x.drugname',ACID)} OR {match('x.drugname',ANTITHROM+r'|heparin')})
        UNION ALL
        SELECT a.id,x.treatmentoffset::DOUBLE,x.treatmentoffset::DOUBLE,
           {match('x.treatmentstring','proton pump inhibitor|h2.blocker|h2.antagonist|'+ACID)},
           {match('x.treatmentstring',ANTITHROM+r'|heparin infusion|therapeutic (anti)?coagulation')},
           {match('x.treatmentstring',SCOPE)} AND NOT {match('x.treatmentstring','plan|consider|scheduled|recommended')}
        FROM {ec('treatment')} x JOIN e_need a ON x.patientunitstayid=a.id
       )
       SELECT * FROM meds WHERE t IS NOT NULL AND (acid OR antithrom OR endoscopy)
    """)
    build("e_rbc_inf",f"""
       SELECT DISTINCT id,t FROM e_infd WHERE greatest(drug_rate,infusion_rate)>0
         AND {match('term',INFD_PRBC)} AND NOT {match('term',INFD_EXCLUDE)}
    """)
    build("e_dominance","""
       WITH observed AS (
          SELECT hospital,person,
             EXISTS(SELECT 1 FROM e_io x WHERE x.id=b.id AND x.t IS NOT NULL) io,
             EXISTS(SELECT 1 FROM e_treatment_rbc x WHERE x.id=b.id AND x.t IS NOT NULL) tx
          FROM e0 b SEMI JOIN e_hospital_log USING(hospital)
       )
       SELECT hospital,count(DISTINCT person) FILTER(WHERE io)*1.0/count(DISTINCT person) FILTER(WHERE io OR tx) rate
       FROM observed GROUP BY hospital
    """)

def prepare_clinical(k):
    (m_clinical if k=="m" else e_clinical)()
