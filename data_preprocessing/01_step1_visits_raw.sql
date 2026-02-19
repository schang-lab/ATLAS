CREATE TABLE trajectory_step1_visits_raw AS
WITH raw_visits AS (
    SELECT 
        CONCAT('EMBUSA-', third_party_user_id) AS panelist_id,
        latitude, 
        longitude,
        city,
        category,
        poi_id,
        location_name,
        CAST(arrival_local   AS TIMESTAMP) AS arr_ts,
        CAST(departure_local AS TIMESTAMP) AS dep_ts
    FROM "poi"
    WHERE State = 'CA'
      AND latitude IS NOT NULL 
      AND longitude IS NOT NULL
      AND (location_name IS NULL OR location_name <> 'non_poi')
)

-- same-day visits
SELECT 
    panelist_id,
    DATE_TRUNC('day', arr_ts) AS visit_date,
    arr_ts  AS arrival_ts,
    dep_ts  AS departure_ts,
    DATE_DIFF('minute', arr_ts, dep_ts) AS duration_minutes,
    latitude,
    longitude,
    city,
    location_name,
    category,
    poi_id
FROM raw_visits
WHERE DATE_TRUNC('day', arr_ts) = DATE_TRUNC('day', dep_ts)

UNION ALL

-- overnight part A: start → midnight
SELECT 
    panelist_id,
    DATE_TRUNC('day', arr_ts) AS visit_date,
    arr_ts AS arrival_ts,
    DATE_TRUNC('day', arr_ts) + INTERVAL '1' DAY AS departure_ts,
    DATE_DIFF('minute', arr_ts, DATE_TRUNC('day', arr_ts) + INTERVAL '1' DAY) AS duration_minutes,
    latitude,
    longitude,
    city,
    location_name,
    category,
    poi_id
FROM raw_visits
WHERE DATE_TRUNC('day', arr_ts) < DATE_TRUNC('day', dep_ts)

UNION ALL

-- overnight part B: midnight → end
SELECT 
    panelist_id,
    DATE_TRUNC('day', dep_ts) AS visit_date,
    DATE_TRUNC('day', dep_ts) AS arrival_ts,
    dep_ts AS departure_ts,
    DATE_DIFF('minute', DATE_TRUNC('day', dep_ts), dep_ts) AS duration_minutes,
    latitude,
    longitude,
    city,
    location_name,
    category,
    poi_id
FROM raw_visits
WHERE DATE_TRUNC('day', arr_ts) < DATE_TRUNC('day', dep_ts)