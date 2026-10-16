SELECT priority,COUNT(*) records FROM feedback GROUP BY priority ORDER BY records DESC;
SELECT category,COUNT(resolution_hours) resolved_records,ROUND(AVG(resolution_hours),2) mean_resolution_hours,ROUND(MAX(resolution_hours),2) maximum_resolution_hours,SUM(is_outlier) flagged_outliers FROM feedback GROUP BY category ORDER BY mean_resolution_hours DESC;
SELECT source_type,source_file,COUNT(*) canonical_records FROM feedback GROUP BY source_type,source_file ORDER BY source_type;
SELECT category,ROUND(AVG(satisfaction),2) satisfaction,ROUND(AVG(affected_users),1) affected_users FROM feedback GROUP BY category;
