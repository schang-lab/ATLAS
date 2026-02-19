CREATE TABLE trajectory_step2_visits_model AS
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
    poi_id,

    -- for debugging/inspection only (kept)
    CONCAT(
      COALESCE(location_name, 'UNK_NAME'), '_',
      CAST(ROUND(CAST(latitude  AS DOUBLE), 5) AS VARCHAR), '_',
      CAST(ROUND(CAST(longitude AS DOUBLE), 5) AS VARCHAR)
    ) AS poi_raw_id,

    -- modeling token
    CASE
      WHEN (location_name = 'home') THEN 'POI_HOME'
      WHEN (location_name = 'work') THEN 'POI_WORK'
      WHEN poi_id IS NOT NULL THEN CONCAT('POI::', poi_id)
      ELSE 'POI_MISSING_ID'
    END AS poi_token
FROM trajectory_step1_visits_raw