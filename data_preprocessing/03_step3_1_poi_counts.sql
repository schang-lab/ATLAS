CREATE TABLE poi_vocab_counts AS
SELECT poi_id_model, COUNT(*) AS cnt
FROM trajectory_step2_visits_model
GROUP BY poi_id_model;