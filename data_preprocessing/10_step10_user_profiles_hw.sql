-- Step 10 — Final profile table (only HW users used in modeling)
CREATE TABLE user_profiles_hw AS
SELECT
    u.panelist_id,
    u.home_lat,
    u.home_lon,
    u.work_lat,
    u.work_lon,
    d.gender,
    d.age,
    d.edu_level,
    d.ethnicity,
    d.hh_income,
    d.hh_size,
    d.occupational_status
FROM users_with_hw u
LEFT JOIN "demo" d
  ON u.panelist_id = d.panelist_id;
