"""Pre-download OCR models at build time so the runtime container works offline.
Best-effort: failures here must NOT break the image build."""

def warm_rapidocr():
    try:
        try:
            from rapidocr import RapidOCR  # rapidocr 3.x
        except Exception:
            from rapidocr_onnxruntime import RapidOCR  # older name
        RapidOCR()
        print("[warmup] RapidOCR models ready")
    except Exception as e:  # noqa
        print(f"[warmup] RapidOCR skipped: {e}")


def warm_paddle():
    try:
        from paddleocr import PaddleOCR
        # triggers detection/recognition/cls model download into ~/.paddleocr
        PaddleOCR(use_angle_cls=True, lang="en")
        print("[warmup] PaddleOCR models ready")
    except Exception as e:  # noqa
        print(f"[warmup] PaddleOCR skipped: {e}")


if __name__ == "__main__":
    warm_rapidocr()
    warm_paddle()
