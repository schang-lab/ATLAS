-- Step 9 — User-level split (80/10/10) on filtered users (Athena-safe hash)
CREATE TABLE user_splits_hw AS
WITH u AS (
    SELECT DISTINCT panelist_id
    FROM segments14_hw
),
h AS (
    SELECT
        panelist_id,
        CAST(from_base(substr(to_hex(md5(to_utf8(panelist_id))), 1, 15), 16) AS BIGINT) AS h_int
    FROM u
)
SELECT
    panelist_id,
    CASE
        WHEN (h_int % 10) BETWEEN 0 AND 7 THEN 'train'
        WHEN (h_int % 10) = 8 THEN 'val'
        ELSE 'test'
    END AS split
FROM h;

CREATE TABLE segments14_hw_with_split AS
SELECT s.*, u.split
FROM segments14_hw s
JOIN user_splits_hw u
  ON s.panelist_id = u.panelist_id;

CREATE TABLE segment_visits14_hw_with_split AS
SELECT v.*, s.split
FROM segment_visits14_hw v
JOIN segments14_hw_with_split s
  ON v.segment_id = s.segment_id;
