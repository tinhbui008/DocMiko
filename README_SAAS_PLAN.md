# Kế hoạch phát triển SaaS — PDF Translator

> Tài liệu định hướng sản phẩm. Cập nhật lần cuối: 2026-06-30.

## 1. Mục tiêu

Phát triển công cụ dịch PDF hiện tại thành một **SaaS** (không còn tune thủ công theo từng file).

**Tiêu chuẩn chất lượng:** giống bản gốc **95–99%**, bao gồm cả **text thường lẫn text nằm trong ảnh**.

Phát triển **song song 2 track** dưới chung một sản phẩm.

---

## 2. Nguyên tắc nền tảng

Một sản phẩm — một cổng upload — một **bộ định tuyến (router)** tự phân loại từng trang và đẩy vào engine phù hợp. Người dùng không cần biết file thuộc loại nào.

```
            ┌─────────── Upload PDF ───────────┐
            │         (1 UI, 1 hàng đợi)         │
            └───────────────┬──────────────────┘
                            ▼
                  ROUTER (phân loại từng trang)
                /                              \
   text-layer rõ ràng                  ảnh / scan / slide phẳng
        ▼                                       ▼
  TRACK A: dịch in-place              TRACK B: Vision-LLM + inpaint
  (giữ font/màu/vị trí)               (OCR+layout+dịch → xóa nền → vẽ lại)
  mục tiêu 95-99%                     mục tiêu 80-90% (+ review tùy chọn)
                \                              /
                 └──────► Ghép lại → PDF kết quả + bản xem trước
```

> **Lưu ý quan trọng:** "song song 2 track" KHÔNG phải code 2 sản phẩm rời. Phải xây **1 hạ tầng SaaS chung trước (Pha 0)**, rồi cắm 2 engine vào cùng router. Nếu không sẽ tốn công gấp đôi và khó ghép.

---

## 3. Phân loại PDF & mức chính xác khả thi

| Loại PDF | Đạt 95–99%? | Cách xử lý đúng |
|----------|-------------|-----------------|
| **Có text-layer thật** (xuất từ Word/InDesign) | ✅ Khả thi cao | Dịch & thay chữ ngay trong text-layer, giữ nguyên font/màu/vị trí |
| **Chữ nung trong ảnh / scan / slide phẳng** (như LONGMAN) | ⚠️ Rất khó — biên giới công nghệ | OCR + **inpaint nền** + vẽ lại khớp style |

**Sự thật cần nhớ:** 95–99% pixel-perfect cho MỌI file upload (kể cả chữ trong ảnh) là bài toán gần như chưa ai giải trọn vẹn ở mức tự động hoàn toàn. Track B cần chấp nhận ~80–90% hoặc kèm review thủ công.

Cách `fill:"sample"` (đè 1 màu phẳng) hiện tại **không bao giờ** đạt mức cao trên nền ảnh/gradient — phải thay bằng **inpainting thật**.

---

## 4. Hạ tầng dùng chung (phần "SaaS thật sự", xây 1 lần)

Độc lập với chất lượng dịch:

- **Hàng đợi job**: mỗi upload = 1 job async, có trạng thái + progress.
- **Lưu trữ** file input/output (hạn dùng, tự xóa).
- **Auth + multi-tenant**: user, tổ chức.
- **Đếm dùng + billing**: tính theo trang (Track B tốn API/compute).
- **UI**: upload → xem tiến trình → preview so sánh gốc/bản dịch → tải về.
- **Quản lý API key/cost tập trung** (không để lộ như `.env` hiện tại).

---

## 5. Track A — PDF text-layer (làm trước, bán được sớm)

- Engine: cải tiến từ `pdf_generator_single_anthropic_merged_v16_full_ocr_rebuild.py` (đã chạy được, có cache dịch).
- Đạt 95–99% vì sửa thẳng text-layer.
- **Đây là MVP** — ra mắt nhanh, có doanh thu, rủi ro thấp.

---

## 6. Track B — PDF ảnh/thiết kế (làm song song, R&D)

Bỏ hẳn tune thủ công per-deck. 3 thành phần mới:

1. **Vision-LLM (Claude)** thay RapidOCR: 1 lần gọi ra JSON `{vị trí, text, màu, role, bản dịch}`. Tự suy ra, không hardcode LONGMAN/CXVIEW.
2. **Inpainting thật** (LaMa / diffusion) thay `fill:"sample"`: xóa chữ gốc và vá nền ảnh.
3. **Re-render khớp style**: màu/font/size lấy từ bước 1.

Kèm **chế độ review (human-in-the-loop)** cho khách cần hoàn hảo.

---

## 7. Lộ trình theo pha

| Pha | Mục tiêu | Nội dung |
|-----|----------|----------|
| **0** | Nền SaaS | Web upload + hàng đợi + auth + billing, bọc quanh `text-translate` hiện có |
| **1** | Track A hoàn chỉnh | Router phát hiện text-layer, dịch in-place, preview, tải về → **ra mắt MVP** |
| **2** | Track B prototype | Vision-LLM OCR+dịch cho trang ảnh, dùng redact tạm → đo chất lượng |
| **3** | Track B chất lượng | Thêm inpainting + re-render khớp màu/font |
| **4** | Review UI | Cho user sửa tay vùng sai (đẩy fidelity gần 99%) |

**Khuyến nghị bắt đầu:** Pha 0 + Pha 1 (nền SaaS + Track A) — tái dùng code đang chạy, ra sản phẩm sớm.

---

## 8. Stack gợi ý

- **Backend:** FastAPI (Python — cùng hệ sinh thái code hiện tại).
- **Hàng đợi:** Celery hoặc RQ + Redis.
- **Storage:** S3-compatible (MinIO tự host, hoặc AWS S3).
- **DB:** PostgreSQL (user, job, usage).
- **Frontend:** React/Next.js (hoặc đơn giản hơn lúc đầu).
- **Triển khai:** Docker Compose trên server Ubuntu hiện tại; sau scale thì tách worker.

---

## 9. Hiện trạng (điểm xuất phát)

- Đã Dockerized, chạy trên Ubuntu (`/root/pdf-translator/DocMiko`), volumes `input/output/cache`.
- 2 luồng CLI sẵn có:
  - `translator` → pipeline v28.2 CXVIEW (local, không tốn API) — chỉ đúng cho deck CXVIEW.
  - `text-translate` → dịch text-layer qua LLM (`claude-sonnet-4-6`), có cache.
- Đã thử nghiệm pipeline per-deck cho LONGMAN (v1 → v29.2): cứu được ảnh nhưng còn mất màu chữ, chưa dịch, **không scale cho SaaS** → sẽ thay bằng hướng tự động ở trên.

---

## 10. Quyết định còn mở

- [ ] Chốt schema DB + danh sách API endpoints cho Pha 0.
- [ ] Chọn cơ chế billing (theo trang / theo gói).
- [ ] Mô hình review cho Track B.
- [ ] Thu hồi & thay API key Anthropic đã lộ trong quá trình dev.
