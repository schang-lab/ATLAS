-- Step 5 — Valid 14-day windows (stride 7 for va and 14 for ca) with spatiotemporal constraints
-- Thresholds:
--   bad spatial day: daily_dist_km > 800
--   good temporal day: coverage_ratio >= 0.2
--   window accept: bad_ratio <= 0.10 and good_ratio >= 0.80
CREATE TABLE trajectory_step5_segments_14d_stride7 AS
WITH flagged_days AS (
    SELECT 
        panelist_id,
        visit_date,
        daily_dist_km,
        coverage_ratio,
        CASE WHEN daily_dist_km > 800 THEN 1 ELSE 0 END AS is_bad_spatial,
        CASE WHEN coverage_ratio >= 0.2 THEN 1 ELSE 0 END AS is_good_temporal
    FROM trajectory_step4_daily_metrics
),
ordered_days AS (
    SELECT
        panelist_id,
        visit_date,
        daily_dist_km,
        coverage_ratio,
        is_bad_spatial,
        is_good_temporal,
        ROW_NUMBER() OVER (PARTITION BY panelist_id ORDER BY visit_date) AS rn
    FROM flagged_days
),
start_days AS (
    SELECT * FROM ordered_days
    WHERE ((rn - 1) % 14) = 0
),
raw_windows AS (
    SELECT
        a.panelist_id,
        a.visit_date AS segment_start_date,
        MAX(b.visit_date) AS segment_end_date,
        COUNT(*) AS days_in_window,
        SUM(b.is_bad_spatial)   AS bad_spatial_days,
        SUM(b.is_good_temporal) AS good_temporal_days,
        MIN(b.daily_dist_km)    AS min_daily_dist_km,
        MAX(b.daily_dist_km)    AS max_daily_dist_km,
        AVG(b.coverage_ratio)   AS avg_coverage_ratio
    FROM start_days a
    JOIN ordered_days b
      ON a.panelist_id = b.panelist_id
     AND b.rn BETWEEN a.rn AND a.rn + 13
    GROUP BY a.panelist_id, a.visit_date
),
valid_windows AS (
    SELECT *
    FROM raw_windows
    WHERE days_in_window = 14
      AND (bad_spatial_days * 1.0 / days_in_window) <= 0.10
      AND (good_temporal_days * 1.0 / days_in_window) >= 0.80
)
SELECT
    panelist_id,
    segment_start_date,
    segment_end_date,
    days_in_window,
    bad_spatial_days,
    good_temporal_days,
    min_daily_dist_km,
    max_daily_dist_km,
    avg_coverage_ratio,
    ROW_NUMBER() OVER (PARTITION BY panelist_id ORDER BY segment_start_date) AS segment_index,
    CONCAT(
        panelist_id, '_seg14d_',
        CAST(ROW_NUMBER() OVER (PARTITION BY panelist_id ORDER BY segment_start_date) AS VARCHAR)
    ) AS segment_id
FROM valid_windows;