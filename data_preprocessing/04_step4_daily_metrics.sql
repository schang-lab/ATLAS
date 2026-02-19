CREATE TABLE trajectory_step4_daily_metrics AS
SELECT 
    panelist_id,
    visit_date,
    CASE WHEN COUNT(*) > 1 THEN 
        SQRT(
            POWER(MAX(TRY_CAST(latitude AS DOUBLE))  - MIN(TRY_CAST(latitude AS DOUBLE)), 2) + 
            POWER(MAX(TRY_CAST(longitude AS DOUBLE)) - MIN(TRY_CAST(longitude AS DOUBLE)), 2)
        ) * 111 
    ELSE 0 END AS daily_dist_km,
    LEAST(SUM(duration_minutes) / 1440.0, 1.0) AS coverage_ratio
FROM trajectory_step3_visits_final
GROUP BY panelist_id, visit_date;
