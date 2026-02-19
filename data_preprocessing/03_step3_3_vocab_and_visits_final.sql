CREATE TABLE trajectory_step3_visits_final AS
SELECT
    v.panelist_id,
    v.visit_date,
    v.arrival_ts,
    v.departure_ts,
    v.duration_minutes,
    v.latitude,
    v.longitude,
    v.location_name,
    v.city,
    v.category,
    v.poi_raw_id,
    v.poi_id,
    CASE
        WHEN v.poi_token IN ('POI_HOME','POI_WORK') THEN v.poi_token
        WHEN p.poi_token IS NOT NULL THEN v.poi_token
        ELSE 'POI_OTHER'
    END AS poi_token
FROM trajectory_step2_visits_model v
LEFT JOIN poi_vocab_topK p
  ON v.poi_token = p.poi_token;