-- One row per patient. The first release keeps only primary tumor samples.
-- Source uniqueness is checked in Python before this join.
SELECT
    p.PATIENT_ID AS patient_id,
    s.SAMPLE_ID AS sample_id,
    p.AGE AS age_raw,
    p.SEX AS sex_raw,
    p.AJCC_PATHOLOGIC_TUMOR_STAGE AS stage_raw,
    p.OS_STATUS AS os_status_raw,
    p.OS_MONTHS AS os_months_raw,
    s.TMB_NONSYNONYMOUS AS tmb_raw
FROM patients AS p
INNER JOIN samples AS s USING (PATIENT_ID)
WHERE p.CANCER_TYPE_ACRONYM = 'LUAD'
  AND s.ONCOTREE_CODE = 'LUAD'
  AND s.SAMPLE_TYPE = 'Primary'
ORDER BY p.PATIENT_ID
