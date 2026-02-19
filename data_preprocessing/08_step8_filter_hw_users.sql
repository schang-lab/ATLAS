CREATE TABLE users_with_hw AS
WITH segment_users AS (
    SELECT DISTINCT panelist_id
    FROM trajectory_step5_segments_14d_stride7
),
hw_users AS (
    SELECT 
        CONCAT('EMBUSA-', third_party_user_id) AS panelist_id,
        MAX(CASE WHEN location_name = 'home' THEN latitude  END) AS home_lat,
        MAX(CASE WHEN location_name = 'home' THEN longitude END) AS home_lon,
        MAX(CASE WHEN location_name = 'work' THEN latitude  END) AS work_lat,
        MAX(CASE WHEN location_name = 'work' THEN longitude END) AS work_lon
    FROM "poi"
    WHERE State = 'CA'
      AND location_name IN ('home','work')
    GROUP BY third_party_user_id
)
SELECT
    h.panelist_id, h.home_lat, h.home_lon, h.work_lat, h.work_lon
FROM hw_users h
JOIN segment_users s
  ON h.panelist_id = s.panelist_id
WHERE h.home_lat IS NOT NULL
  AND h.work_lat IS NOT NULL