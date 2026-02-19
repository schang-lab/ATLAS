CREATE TABLE trajectory_step6_visits_seq AS
SELECT
    panelist_id,
    visit_date,
    arrival_ts,
    departure_ts,
    duration_minutes,
    latitude,
    longitude,
    location_name,
    category,
    city,
    poi_raw_id,
    poi_id,
    poi_token
FROM (
    SELECT
        v.*,
        LAG(poi_id) OVER (PARTITION BY panelist_id ORDER BY arrival_ts) AS prev_poi
    FROM trajectory_step3_visits_final v
) t
WHERE prev_poi IS NULL OR poi_id <> prev_poi