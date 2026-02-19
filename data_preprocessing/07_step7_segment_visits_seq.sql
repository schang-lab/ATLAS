CREATE TABLE trajectory_step7_segment_visits_seq AS
SELECT
    s.segment_id,
    s.panelist_id,
    s.segment_start_date,
    s.segment_end_date,
    v.visit_date,
    v.arrival_ts,
    v.departure_ts,
    v.poi_id,
    v.poi_raw_id,
    v.poi_token,
    v.category,
    v.latitude,
    v.longitude,
    v.city,
    v.location_name,
    v.duration_minutes
FROM trajectory_step5_segments_14d_stride7 s
JOIN trajectory_step6_visits_seq v
  ON s.panelist_id = v.panelist_id
 AND v.visit_date BETWEEN s.segment_start_date AND s.segment_end_date