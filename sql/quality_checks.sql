-- Passing checks return no rows or zero counts.
SELECT record_id,COUNT(*) duplicates FROM feedback GROUP BY record_id HAVING COUNT(*)>1;
SELECT COUNT(*) invalid_priority FROM feedback WHERE priority NOT IN ('low','medium','high');
SELECT COUNT(*) invalid_satisfaction FROM feedback WHERE satisfaction<1 OR satisfaction>5 OR satisfaction IS NULL;
SELECT COUNT(*) blank_text FROM feedback WHERE TRIM(text)='';
