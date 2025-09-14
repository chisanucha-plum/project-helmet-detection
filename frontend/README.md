# Frontend — Helmet Detection UI

สรุปสั้นๆ สำหรับการติดตั้งและรันส่วน Frontend ของโปรเจกต์

Prerequisites
- Node.js (แนะนำ: 18+ หรือเวอร์ชันที่โปรเจกต์ต้องการ)
- npm (มาพร้อมกับ Node.js)

การติดตั้ง
1. เปิด terminal แล้วไปที่โฟลเดอร์ `frontend`:

```powershell
cd C:\project-helmet-detection\frontend
```

2. ติดตั้ง dependency:

```powershell
npm install
```

รันเซิร์ฟเวอร์ (โหมดพัฒนา)
- คำสั่งนี้จะรัน Next.js ในโหมด dev (hot reload):

```powershell
npm run dev
```

- ค่าเริ่มต้น: ใช้พอร์ตที่ Next กำหนด (ปกติ `http://localhost:3000`) — ดู output ของคำสั่งเพื่อทราบพอร์ตที่ถูกใช้งาน

สร้างสำหรับ production และรัน
- สร้างไฟล์ build (จะสร้างไดเรกทอรี `.next`):

```powershell
npm run build
```

- เรียกใช้ production server (ต้องรันหลังจาก build):

```powershell
npm run start
```

Environment / Backend
- ถ้าต้องการให้ frontend เรียก backend ที่รันบนเครื่อง ให้เซ็ตตัวแปรสภาพแวดล้อมในไฟล์ `.env.local` (ถ้ายังไม่มี ให้สร้างไฟล์นี้ในโฟลเดอร์ `frontend`)
- ตัวอย่าง `.env.local`:

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

