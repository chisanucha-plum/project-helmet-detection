-- SQL สำหรับสร้าง table history_status
-- สำหรับเก็บประวัติการวิเคราะห์ helmet compliance

CREATE TABLE IF NOT EXISTS history_status (
    id VARCHAR PRIMARY KEY,                    -- ID จากชื่อไฟล์ snapshot เช่น capture_20250925_110415_013_helmet_mc_mc_3_7
    helmet_status BOOLEAN,                     -- สถานะการสวมหมวกกันน็อค (true = สวม, false = ไม่สวม)
    passenger_count INTEGER,                   -- จำนวนผู้โดยสาร
    violations TEXT,                           -- รายละเอียดการฝ่าฝืน
    timestamp VARCHAR                          -- เวลาที่ทำการวิเคราะห์ (Thailand timezone)
);

-- เพิ่ม index เพื่อเพิ่มประสิทธิภาพการค้นหา
CREATE INDEX IF NOT EXISTS idx_history_status_id ON history_status (id);
CREATE INDEX IF NOT EXISTS idx_history_status_timestamp ON history_status (timestamp);
CREATE INDEX IF NOT EXISTS idx_history_status_helmet_status ON history_status (helmet_status);

-- ตัวอย่างข้อมูล
-- INSERT INTO history_status (id, helmet_status, passenger_count, violations, timestamp) 
-- VALUES (
--     'capture_20250925_110415_013_helmet_mc_mc_3_7', 
--     false, 
--     3, 
--     'Driver 1 not wearing helmet', 
--     '2025-09-25 13:39:03'
-- );

-- Query ตัวอย่างสำหรับดูข้อมูล
-- SELECT * FROM history_status ORDER BY timestamp DESC LIMIT 10;
-- SELECT COUNT(*) FROM history_status WHERE helmet_status = false;
-- SELECT * FROM history_status WHERE violations IS NOT NULL;