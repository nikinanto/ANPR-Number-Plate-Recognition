"""
ANPR Dashboard Backend
- FastAPI server that loads YOLO + EasyOCR ONCE at startup (not per-request)
- POST /api/upload  -> detect plates in an uploaded image
- GET  /api/history -> return last 50 detections from SQLite
- GET  /           -> serve the frontend dashboard
"""

import os, sqlite3, re
from datetime import datetime

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from evaluation import get_performance_stats

# ---------- ensure folders exist before app mounts them ----------
for d in ["static/output", "static/uploads", "static/css", "static/js"]:
    os.makedirs(d, exist_ok=True)

# ---------- initialise app ----------
app = FastAPI(title="ANPR Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ---------- global ML models (loaded once) ----------
print("Loading EasyOCR reader…")
import easyocr
_reader = easyocr.Reader(["en"], gpu=True)

print("Loading YOLOv8 model…")
from ultralytics import YOLO
_model = YOLO("ANPR_best.pt")

print("✅ Models ready.")

# ---------- database ----------
DB_PATH = "anpr_history.db"

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            image_name      TEXT,
            plate_text      TEXT,
            confidence      REAL,
            annotated_image_path TEXT,
            crop_path       TEXT,
            binary_path     TEXT,
            is_correct      INTEGER DEFAULT NULL -- 1: correct, 0: wrong, NULL: pending
        )
    """)
    con.commit()
    con.close()

_init_db()

def _apply_dip(crop):
    """Generates multiple versions of the plate for OCR voting.
    Returns (raw_color, clahe_gray, binary_inv).
    """
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filt)
    _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return up, clahe, cv2.bitwise_not(binary)


def _run_ocr(crop_bgr):
    """Run EasyOCR on the 3x upscaled original color crop.
    Color image preserves all tonal cues EasyOCR needs to distinguish
    D vs O, 4 vs 1, etc. Two passes — with/without allowlist.
    Returns (raw_text, confidence).
    """
    up = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    def _read(img, use_allowlist):
        kw = dict(detail=1, paragraph=False)
        if use_allowlist:
            kw["allowlist"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        res = _reader.readtext(img, **kw)
        if not res:
            return "", 0.0
        res = sorted(res, key=lambda r: min(pt[0] for pt in r[0]))
        txt  = re.sub(r"[^A-Z0-9]", "", "".join(r[1].upper() for r in res))
        conf = sum(r[2] for r in res) / len(res)
        return txt, conf

    raw_a, conf_a = _read(up, use_allowlist=True)
    raw_b, conf_b = _read(up, use_allowlist=False)
    raw, conf = (raw_a, conf_a) if conf_a >= conf_b else (raw_b, conf_b)
    print(f"[OCR DEBUG] allowlist={raw_a!r}({conf_a:.2f})  free={raw_b!r}({conf_b:.2f})  chosen={raw!r}")
    return raw, conf


# Digit↔letter and letter↔digit maps for slot-type coercion
_D2L = {"0":"O","1":"I","2":"Z","3":"J","4":"A","5":"S","6":"G","7":"T","8":"B","9":"P"}
_L2D = {"O":"0","Q":"0","I":"1","L":"1","Z":"2","J":"3","A":"4","S":"5","G":"6","T":"7","B":"8"}

_OCR_FIXES = [("1H","M"),("IH","M"),("TH","M"),("1N","M"),("VV","W")]
_PLATE_RE = re.compile(r"([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})")

def _coerce(text, kind):
    out = ""
    for c in text:
        if kind == "letters":
            out += _D2L.get(c, c) if c.isdigit() else (c if c.isalpha() else "")
        else:
            out += _L2D.get(c, c) if c.isalpha() else (c if c.isdigit() else "")
    return out

def _format_plate(raw):
    """Returns (formatted_text, validity_score).
    Scores: 3 (Full Regex), 2 (Full Coerced), 1 (Partial/Raw)
    """
    t = re.sub(r"[^A-Z0-9]", "", raw.upper())
    t2 = t
    for wrong, right in _OCR_FIXES: t2 = t2.replace(wrong, right)

    # 1. Regex Match (Score 3)
    for cand in (t, t2):
        m = _PLATE_RE.search(cand)
        if m: return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}", 3

    # 2. Full Coercion (Score 2)
    if len(t2) >= 8:
        state, rto, num, series = _coerce(t2[0:2],"letters"), _coerce(t2[2:4],"numbers"), _coerce(t2[-4:],"numbers"), _coerce(t2[4:-4],"letters")
        if len(state)==2 and len(rto)==2 and 1<=len(series)<=3 and len(number:=num)==4:
            return f"{state} {rto} {series} {num}", 2

    return t, 1

def _run_ocr(crop):
    """Multi-Source OCR Voting."""
    up_color, clahe, up_bin = _apply_dip(crop)
    variants = {"color": up_color, "clahe": clahe, "binary": up_bin}
    candidates = []

    for name, img in variants.items():
        res = _reader.readtext(img, detail=1, paragraph=False)
        if not res: continue
        res = sorted(res, key=lambda r: min(pt[0] for pt in r[0]))
        raw = "".join(r[1].upper() for r in res)
        conf = sum(r[2] for r in res) / len(res)
        text, score = _format_plate(raw)
        candidates.append({"text": text, "score": score, "conf": conf, "source": name, "raw": raw})

    if not candidates: return "", 0.0, None

    # Pick best: Highest score first, then highest confidence
    best = sorted(candidates, key=lambda x: (x["score"], x["conf"]), reverse=True)[0]
    print(f"[OCR VOTE] Winner: {best['source']} | Result: {best['text']} | Raw: {best['raw']}")
    return best["text"], best["conf"], best["source"]

# ---------- detection ----------
def _detect(img, conf_thr=0.35):
    results = _model(img, conf=conf_thr)
    plates  = []
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_conf = float(box.conf[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue

            # Step 1: Run Multi-Source OCR Voting
            plate_text, ocr_conf, source = _run_ocr(crop)

            # Step 2: Generate DIP binary for visual display
            # We already have _apply_dip but let's just get the display version
            _, _, dip_display = _apply_dip(crop)

            safe      = re.sub(r"[^A-Za-z0-9]", "_", plate_text) or f"plate_{len(plates)}"
            crop_name = f"{safe}_crop_{ts}_{len(plates)}.jpg"
            bin_name  = f"{safe}_binary_{ts}_{len(plates)}.jpg"
            cv2.imwrite(f"static/output/{crop_name}", crop)
            cv2.imwrite(f"static/output/{bin_name}",  dip_display)

            # Annotate the main frame
            colour = (0, 255, 80)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            label  = plate_text or "UNKNOWN"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), colour, cv2.FILLED)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            plates.append({
                "text":        plate_text,
                "ocr_conf":    round(ocr_conf, 4),
                "box_conf":    round(box_conf, 4),
                "crop_path":   f"/static/output/{crop_name}",
                "binary_path": f"/static/output/{bin_name}"
            })

    ann_name = f"annotated_{ts}.jpg"
    cv2.imwrite(f"static/output/{ann_name}", img)
    return {
        "success":  True,
        "annotated_image": f"/static/output/{ann_name}",
        "plates":   plates
    }

# ---------- API routes ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        raw  = await file.read()
        arr  = np.frombuffer(raw, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "Cannot decode image."}, status_code=400)

        result = _detect(img)

        # persist detections with plate text
        if result["plates"]:
            con = sqlite3.connect(DB_PATH)
            ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for p in result["plates"]:
                if p["text"]:
                    con.execute(
                        "INSERT INTO history VALUES (NULL,?,?,?,?,?,?,?,NULL)",
                        (ts, file.filename, p["text"], p["ocr_conf"],
                         result["annotated_image"], p["crop_path"], p["binary_path"])
                    )
            con.commit()
            con.close()

        return result

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/history")
async def history():
    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, timestamp, image_name, plate_text, confidence, annotated_image_path, crop_path, binary_path, is_correct FROM history ORDER BY id DESC LIMIT 100"
        ).fetchall()
        con.close()
        return {"history": [dict(r) for r in rows]}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/api/history/{record_id}/validate")
async def validate_detection(record_id: int, status: int):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("UPDATE history SET is_correct = ? WHERE id = ?", (status, record_id))
        con.commit()
        con.close()
        return {"success": True}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/api/stats")
async def get_stats():
    try:
        stats = get_performance_stats(DB_PATH)
        return stats
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
