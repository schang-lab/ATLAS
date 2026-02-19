CREATE TABLE poi_vocab_topK AS
SELECT poi_token
FROM poi_vocab_counts
WHERE poi_token NOT IN ('POI_MISSING_ID')
ORDER BY cnt DESC
LIMIT 9500;