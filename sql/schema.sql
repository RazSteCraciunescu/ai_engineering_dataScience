PRAGMA foreign_keys = ON;
CREATE TABLE feedback (
 record_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, created_at TEXT, collected_at TEXT NOT NULL,
 source_type TEXT NOT NULL, source_file TEXT NOT NULL, channel TEXT NOT NULL, product TEXT NOT NULL,
 category TEXT NOT NULL, title TEXT NOT NULL, message TEXT NOT NULL, text TEXT NOT NULL, language TEXT NOT NULL,
 priority TEXT NOT NULL CHECK(priority IN ('low','medium','high')), status TEXT NOT NULL,
 satisfaction_raw REAL, satisfaction REAL NOT NULL CHECK(satisfaction BETWEEN 1 AND 5),
 satisfaction_was_missing INTEGER NOT NULL CHECK(satisfaction_was_missing IN (0,1)),
 resolution_hours_raw REAL, resolution_hours REAL, resolution_hours_model REAL,
 is_outlier INTEGER NOT NULL CHECK(is_outlier IN (0,1)), affected_users INTEGER NOT NULL CHECK(affected_users>=1),
 attachment_count INTEGER NOT NULL CHECK(attachment_count>=0), text_length INTEGER NOT NULL, word_count INTEGER NOT NULL
);
CREATE INDEX idx_feedback_category ON feedback(category);
CREATE INDEX idx_feedback_priority ON feedback(priority);
CREATE INDEX idx_feedback_created ON feedback(created_at);
CREATE TABLE pipeline_runs(id INTEGER PRIMARY KEY AUTOINCREMENT,run_name TEXT NOT NULL,row_count INTEGER NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE VIEW category_summary AS SELECT category,COUNT(*) records,ROUND(AVG(satisfaction),2) average_satisfaction,ROUND(AVG(resolution_hours),2) average_resolution_hours,SUM(CASE WHEN priority='high' THEN 1 ELSE 0 END) high_priority_records FROM feedback GROUP BY category;
