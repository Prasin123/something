"""
Nepal KYC IDP Portal
=====================
An Intelligent Document Processing (IDP) and KYC verification tool for
Nepali citizenship certificates, built with Streamlit, OpenCV, and
Tesseract OCR (Devanagari / Nepali language pack).

Pipeline
--------
1. Ingest      - accept a JPG/PNG photo or a multi-page PDF for each
                 document; PDF pages are rasterised with pypdfium2.
2. Preprocess  - grayscale -> non-local means denoising -> adaptive
                 Gaussian thresholding, to handle glare, shadows and
                 uneven lighting from mobile camera captures.
3. Localize    - crop configurable Regions of Interest (ROI) for the
                 fields that matter for KYC: citizenship number, name,
                 father's name, date of birth, and address. Each field
                 points at a specific page, so a multi-page KYC packet
                 (main form + annex) works from a single upload.
4. Recognize   - run Tesseract with a configurable Devanagari model
                 (Nepali, a generic Devanagari script model, or both
                 combined), using single-line segmentation (--psm 7)
                 for most fields and block mode (--psm 6) for the
                 multi-line address field.
5. Validate    - clean and normalise the Devanagari output (including
                 Devanagari -> Arabic digit conversion), then apply
                 field-specific regular expressions.
6. Cross-check - compare the citizenship certificate against the
                 physical KYC form field-by-field and flag mismatches.

Run:
    streamlit run app.py

System requirements (NOT installable via pip - see SETUP.md):
    - Tesseract OCR binary (tesseract-ocr)
    - Nepali trained data (tesseract-ocr-nep / nep.traineddata) and/or
      a Devanagari script model (Devanagari.traineddata) dropped into
      your Tesseract tessdata folder

NOTE ON ACCURACY: real citizenship certificates and KYC forms vary a lot
in layout, era, print vs. handwriting, and scan quality. The default ROI
boxes below are illustrative starting points, not a guarantee of correct
localisation for every document. Use the "Calibrate ROI regions" panel
in the sidebar to tune them (including which page a field lives on) for
your own scanner/camera setup, and treat this app as a
reviewable-assistant, not an unattended decision-maker.
"""

from __future__ import annotations

import difflib
import io
import json
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --- Optional heavy dependencies are imported defensively so a missing ---
# --- package produces a friendly in-app message instead of a hard crash --
try:
    import cv2
    CV2_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - exercised only if opencv missing
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = str(exc)

try:
    import pytesseract
    from pytesseract import TesseractNotFoundError, TesseractError
    PYTESSERACT_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - exercised only if pytesseract missing
    pytesseract = None  # type: ignore
    PYTESSERACT_IMPORT_ERROR = str(exc)

    class TesseractNotFoundError(Exception):
        pass

    class TesseractError(Exception):
        pass

try:
    import pypdfium2 as pdfium
    PDFIUM_IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - exercised only if pypdfium2 missing
    pdfium = None  # type: ignore
    PDFIUM_IMPORT_ERROR = str(exc)


# =====================================================================
# Page config + visual identity
# =====================================================================
st.set_page_config(
    page_title="Nepal KYC IDP Portal",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Devanagari:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper: #FAF6EE;
    --paper-line: #E4D9BC;
    --ink: #211B1F;
    --ink-soft: #6B6470;
    --crimson: #A91E32;
    --crimson-soft: #F4DEE1;
    --blue: #0B2F73;
    --blue-soft: #DDE6F5;
    --success: #1B7F4C;
    --success-bg: #DEF3E7;
    --warning: #9C6B08;
    --warning-bg: #FBF0D2;
    --danger: #A91E32;
    --danger-bg: #F7DEE1;
}

.stApp { background: var(--paper); }

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }

h1, h2, h3 {
    font-family: 'Noto Serif Devanagari', serif !important;
    color: var(--ink) !important;
    letter-spacing: 0.2px;
}

code, .kyc-mono { font-family: 'IBM Plex Mono', monospace !important; }

.kyc-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.3rem 1.8rem;
    background: linear-gradient(135deg, var(--blue) 0%, #123a86 100%);
    border-radius: 14px;
    margin-bottom: 0.6rem;
}
.kyc-header .kyc-flagbar {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: var(--crimson);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    border: 3px solid #FFFFFF;
}
.kyc-header h1 { color: #FFFFFF !important; font-size: 1.6rem; margin: 0; }
.kyc-header p { color: #DCE6F7; margin: 0.2rem 0 0 0; font-size: 0.92rem; }

.kyc-rule {
    height: 6px;
    margin: 0 0 1.5rem 0;
    border-radius: 3px;
    background: repeating-linear-gradient(90deg,
        var(--crimson), var(--crimson) 10px,
        var(--blue) 10px, var(--blue) 20px);
    opacity: 0.55;
}

.kyc-card {
    background: #FFFFFF;
    border: 1px solid var(--paper-line);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}

.kyc-field-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-soft);
}

.kyc-stamp {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 128px;
    height: 128px;
    border-radius: 50%;
    border: 3px double currentColor;
    box-shadow: inset 0 0 0 5px var(--paper);
    outline: 1px solid currentColor;
    outline-offset: -9px;
    transform: rotate(-6deg);
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    font-size: 0.78rem;
    line-height: 1.25;
    padding: 0.4rem;
    margin: 0.2rem 0.4rem 0.6rem 0;
}
.kyc-stamp--verified { color: var(--success); }
.kyc-stamp--review { color: var(--warning); }
.kyc-stamp--failed { color: var(--danger); }
.kyc-stamp .kyc-stamp-sub {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    opacity: 0.8;
    margin-top: 0.15rem;
}

.stButton>button[kind="primary"] { background: var(--crimson); border: none; }
.stButton>button[kind="primary"]:hover { background: #8A1828; }

[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }

section[data-testid="stSidebar"] { background: #F3ECDD; border-right: 1px solid var(--paper-line); }

@media (prefers-reduced-motion: no-preference) {
    .kyc-card { transition: box-shadow 0.15s ease; }
    .kyc-card:hover { box-shadow: 0 2px 10px rgba(33, 27, 31, 0.08); }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# Domain constants
# =====================================================================
DEV_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")
NUMERIC_WHITELIST = "0123456789०१२३४५६७८९-/. "

PDF_RENDER_DPI = 300
MAX_PDF_PAGES = 20

# Language model choices for Tesseract. "nep" is dictionary/numeral-tuned
# for Nepali; a generic Devanagari script model can help on fonts or
# handwriting styles it wasn't trained on but isn't Nepali-tuned; the
# combined option lets Tesseract score both per character and take the
# better hit - in a quick comparison against these test documents, the
# combined setting matched or beat "nep" alone, so it's the default.
OCR_LANG_OPTIONS = {
    "Nepali (nep)": "nep",
    "Devanagari script - generic": "Devanagari",
    "Nepali + Devanagari - combined": "nep+Devanagari",
}
DEFAULT_OCR_LANG_LABEL = "Nepali + Devanagari - combined"

FIELD_DEFINITIONS = [
    {"key": "citizenship_no", "label": "Citizenship Number", "kind": "number", "psm": 7},
    {"key": "full_name", "label": "Full Name (Devanagari)", "kind": "text", "psm": 7},
    {"key": "father_name", "label": "Father's Name", "kind": "text", "psm": 7},
    {"key": "dob", "label": "Date of Birth (B.S.)", "kind": "date", "psm": 7},
    {"key": "address", "label": "Permanent Address", "kind": "text", "psm": 6},
]
FIELD_KEYS = [f["key"] for f in FIELD_DEFINITIONS]
FIELD_LABELS = {f["key"]: f["label"] for f in FIELD_DEFINITIONS}
CRITICAL_FIELDS = {"citizenship_no", "full_name"}

# Each field's ROI is {"page": <0-indexed page>, "box": (x1, y1, x2, y2)}
# with box coordinates normalised 0-1. Multi-page documents (e.g. a PDF
# with the main KYC form on page 1 and an annex on page 2) are handled
# by pointing a field at the page it actually appears on.
DEFAULT_ROI = {
    # Calibrated against a District Administration Office citizenship
    # certificate layout: photo lower-left; label/value rows to its
    # right running name -> birthplace -> permanent address -> DOB ->
    # father's name -> mother's name -> spouse. Still a starting point -
    # older/newer certificate formats shift things, so recalibrate from
    # the sidebar if your rows land differently.
    "citizenship": {
        "citizenship_no": {"page": 0, "box": (0.02, 0.250, 0.35, 0.300)},
        "full_name": {"page": 0, "box": (0.21, 0.350, 0.62, 0.388)},
        "address": {"page": 0, "box": (0.21, 0.500, 0.62, 0.567)},
        "dob": {"page": 0, "box": (0.21, 0.572, 0.62, 0.603)},
        "father_name": {"page": 0, "box": (0.21, 0.607, 0.62, 0.639)},
    },
    # Calibrated against a Siddhartha Bank "Personal Account Opening
    # Form" (page 1). Only the applicant-name row is present on this
    # page - citizenship_no / father_name / dob / address aren't part
    # of this particular page (they typically live on the accompanying
    # "Individual Customer Information Form" annex). They default to
    # page 1 too and keep illustrative placeholders below until you
    # upload the annex and point them at its actual page.
    "kyc": {
        "citizenship_no": {"page": 0, "box": (0.08, 0.10, 0.55, 0.18)},
        "full_name": {"page": 0, "box": (0.03, 0.635, 0.97, 0.685)},
        "father_name": {"page": 0, "box": (0.08, 0.32, 0.92, 0.40)},
        "dob": {"page": 0, "box": (0.08, 0.42, 0.55, 0.50)},
        "address": {"page": 0, "box": (0.08, 0.52, 0.92, 0.68)},
    },
}

ROI_COLORS = {
    "citizenship_no": (169, 30, 50),
    "full_name": (11, 47, 115),
    "father_name": (27, 127, 76),
    "dob": (156, 107, 8),
    "address": (110, 60, 150),
}

FIELD_LABEL_STRIP = {
    "citizenship_no": [r"नागरिकता(को)?\s*(प्रमाणपत्र)?\s*नं\.?", r"Citizenship\s*No\.?"],
    "full_name": [r"^\s*नाम\s*(थर)?", r"\bName\b"],
    "father_name": [r"बाबुको\s*नाम", r"Father'?s?\s*Name"],
    "dob": [r"जन्म\s*मिति", r"Date\s*of\s*Birth", r"D\.?O\.?B\.?"],
    "address": [r"स्थायी\s*ठेगाना", r"\bठेगाना\b", r"\bAddress\b"],
}

CITIZENSHIP_NO_PATTERN = re.compile(r"(\d{1,3}[-/]\d{1,3}[-/]\d{1,3}(?:[-/]\d{1,6})?|\d{6,15})")
DATE_PATTERN = re.compile(r"(\d{4})[\-/.](\d{1,2})[\-/.](\d{1,2})")

STATUS_MATCH, STATUS_REVIEW, STATUS_MISMATCH = "✅ Match", "⚠️ Review", "❌ Mismatch"
STATUS_SEVERITY = {STATUS_MATCH: 0, STATUS_REVIEW: 1, STATUS_MISMATCH: 2}


# =====================================================================
# Preprocessing (OpenCV)
# =====================================================================
def preprocess_document(image: np.ndarray, denoise_strength: int = 10) -> np.ndarray:
    """Grayscale -> non-local means denoise -> adaptive Gaussian threshold.

    This is the core step that makes mobile-camera captures (uneven
    lighting, glare, shadows) usable by Tesseract.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image passed to preprocess_document().")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    denoised = cv2.fastNlMeansDenoising(gray, None, h=denoise_strength, templateWindowSize=7, searchWindowSize=21)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh


def crop_roi(image: np.ndarray, roi_norm: tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Crop a normalised (0-1) ROI box from an image, clamped to bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi_norm
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    px1, px2 = int(x1 * w), min(w, max(int(x2 * w), int(x1 * w) + 1))
    py1, py2 = int(y1 * h), min(h, max(int(y2 * h), int(y1 * h) + 1))
    crop = image[py1:py2, px1:px2]
    return crop if crop.size > 0 else None


def draw_roi_overlay(image: np.ndarray, roi_dict: dict) -> np.ndarray:
    """Draw labelled ROI rectangles (a flat key -> box map) on a copy of
    the page, for visual QA."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    for key, box in roi_dict.items():
        x1, y1, x2, y2 = box
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        color = ROI_COLORS.get(key, (100, 100, 100))
        cv2.rectangle(overlay, p1, p2, color, max(2, w // 400))
        label_y = max(18, p1[1] - 8)
        cv2.putText(
            overlay, FIELD_LABELS.get(key, key), (p1[0], label_y),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 2200), color, 2, cv2.LINE_AA,
        )
    return overlay


def draw_roi_overlay_for_page(page_image: np.ndarray, roi_dict: dict, page_idx: int) -> np.ndarray:
    """Draw only the ROI boxes whose fields are configured for this page."""
    boxes_here = {k: v["box"] for k, v in roi_dict.items() if int(v.get("page", 0)) == page_idx}
    if not boxes_here:
        return page_image
    return draw_roi_overlay(page_image, boxes_here)


# =====================================================================
# OCR engine (Tesseract)
# =====================================================================
def check_tesseract_ready(lang: str = "nep") -> tuple[bool, str]:
    """Verify the Tesseract binary and the requested language model(s)
    are available. `lang` may combine models with "+", e.g. "nep+Devanagari".

    Returns (ok, message) - never raises, so callers can always show a
    friendly banner instead of letting the app crash.
    """
    if CV2_IMPORT_ERROR:
        return False, f"OpenCV failed to import ({CV2_IMPORT_ERROR}). Run: pip install opencv-python-headless"
    if PYTESSERACT_IMPORT_ERROR:
        return False, f"pytesseract failed to import ({PYTESSERACT_IMPORT_ERROR}). Run: pip install pytesseract"
    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError:
        return False, (
            "Tesseract OCR binary was not found on this system. Install it first, e.g. "
            "`sudo apt-get install tesseract-ocr` (Ubuntu/Debian), `brew install tesseract` (macOS), "
            "or the UB-Mannheim installer on Windows. See SETUP.md for details."
        )
    except Exception as exc:
        return False, f"Could not verify the Tesseract installation ({exc})."

    try:
        installed_langs = pytesseract.get_languages(config="")
    except Exception:
        installed_langs = []
    requested = [part for part in lang.split("+") if part]
    missing = [part for part in requested if part not in installed_langs]
    if missing:
        return False, (
            f"Tesseract is installed, but this language model is missing: {', '.join(missing)}. "
            "Install the Nepali pack with `sudo apt-get install tesseract-ocr-nep`, or drop a "
            "`Devanagari.traineddata` file into your tessdata folder for the generic script model. "
            "See SETUP.md."
        )
    return True, f"Tesseract is ready ({lang})."


def ocr_devanagari(image: np.ndarray, psm: int = 7, numeric_only: bool = False, lang: str = "nep") -> dict:
    """Run Tesseract configured for Devanagari text on a single crop.

    Returns a dict with text, mean word confidence, and an error field
    that is None on success - callers check `error` rather than relying
    on exceptions propagating into the UI.
    """
    config_parts = [f"--psm {psm}", "--oem 3"]
    if numeric_only:
        config_parts.append(f'-c tessedit_char_whitelist={NUMERIC_WHITELIST}')
    config = " ".join(config_parts)
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data.get("conf", []) if str(c) not in ("-1",)]
        mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
        return {"text": text, "confidence": mean_conf, "error": None}
    except TesseractNotFoundError:
        return {"text": "", "confidence": 0.0, "error": "Tesseract binary not found."}
    except TesseractError as exc:
        return {"text": "", "confidence": 0.0, "error": f"Tesseract error: {exc}"}
    except Exception as exc:  # defensive catch-all so one bad crop never crashes the run
        return {"text": "", "confidence": 0.0, "error": str(exc)}


# =====================================================================
# Post-processing & validation
# =====================================================================
def devanagari_digits_to_arabic(text: str) -> str:
    return text.translate(DEV_DIGIT_MAP)


def clean_devanagari_text(text: str) -> str:
    """Keep Devanagari, Latin letters/digits, spaces and light punctuation."""
    if not text:
        return ""
    allowed = re.compile(r"[^\u0900-\u097F0-9A-Za-z\s,./-]")
    cleaned = allowed.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def strip_field_label(field_key: str, text: str) -> str:
    result = text
    for pattern in FIELD_LABEL_STRIP.get(field_key, []):
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result.strip(" :।.,-")


def extract_citizenship_number(text: str) -> Optional[str]:
    normalized = devanagari_digits_to_arabic(text).replace(" ", "")
    match = CITIZENSHIP_NO_PATTERN.search(normalized)
    return match.group(0) if match else None


def validate_citizenship_number(number: Optional[str]) -> bool:
    if not number:
        return False
    digits_only = re.sub(r"[^\d]", "", number)
    return 6 <= len(digits_only) <= 15


def extract_dob(text: str) -> Optional[str]:
    normalized = devanagari_digits_to_arabic(text)
    match = DATE_PATTERN.search(normalized)
    return match.group(0) if match else None


def postprocess_field(field_def: dict, raw_text: str) -> str:
    """Apply the right cleaning/extraction strategy for a field's `kind`."""
    cleaned = clean_devanagari_text(raw_text)
    cleaned = strip_field_label(field_def["key"], cleaned)
    kind = field_def["kind"]
    if kind == "number":
        extracted = extract_citizenship_number(cleaned)
        return extracted if extracted else cleaned
    if kind == "date":
        extracted = extract_dob(cleaned)
        return extracted if extracted else cleaned
    return cleaned


# =====================================================================
# Document ingestion - images and multi-page PDFs
# =====================================================================
def render_pdf_pages(file_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> tuple[list[np.ndarray], list[str]]:
    """Rasterise every page of a PDF to an RGB numpy array at `dpi`."""
    warnings: list[str] = []
    pages: list[np.ndarray] = []
    try:
        pdf_doc = pdfium.PdfDocument(file_bytes)
        n_pages = len(pdf_doc)
        if n_pages == 0:
            return [], ["This PDF has no pages."]
        truncated = n_pages > MAX_PDF_PAGES
        n_to_render = min(n_pages, MAX_PDF_PAGES)
        scale = dpi / 72.0
        for i in range(n_to_render):
            bitmap = pdf_doc[i].render(scale=scale)
            pages.append(np.array(bitmap.to_pil().convert("RGB")))
        pdf_doc.close()
        if truncated:
            warnings.append(f"This PDF has {n_pages} pages; only the first {MAX_PDF_PAGES} were loaded.")
    except Exception as exc:
        warnings.append(f"Could not read this PDF: {exc}")
    return pages, warnings


@st.cache_data(show_spinner="Rendering document pages...")
def _load_pages_from_bytes(file_bytes: bytes, is_pdf_file: bool) -> tuple[list[np.ndarray], list[str]]:
    if is_pdf_file:
        if PDFIUM_IMPORT_ERROR:
            return [], [f"PDF support needs pypdfium2 (pip install pypdfium2): {PDFIUM_IMPORT_ERROR}"]
        return render_pdf_pages(file_bytes)
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return [np.array(image)], []
    except (UnidentifiedImageError, OSError) as exc:
        return [], [f"Could not read this file as an image: {exc}"]


def load_document_pages(uploaded_file) -> tuple[list[np.ndarray], list[str]]:
    """Load an uploaded JPG/PNG/PDF into a list of RGB page images.
    A plain image becomes a single-page list; a PDF becomes one entry
    per page. Cached on file content so re-running the app (e.g. moving
    an unrelated slider) doesn't re-render the same file repeatedly."""
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    is_pdf_file = file_bytes[:5] == b"%PDF-"
    return _load_pages_from_bytes(file_bytes, is_pdf_file)


def extract_fields_from_document(pages: list[np.ndarray], roi_dict: dict, denoise_strength: int, lang: str) -> dict:
    """Run the full per-field pipeline for one (possibly multi-page) document.

    Returns fields (key -> cleaned value), confidences, pages_used
    (key -> 1-indexed page actually read, or None if skipped), warnings,
    and one ROI-annotated overlay image per page for QA.
    """
    fields, confidences, pages_used, warnings = {}, {}, {}, []
    for field_def in FIELD_DEFINITIONS:
        key = field_def["key"]
        field_roi = roi_dict.get(key)
        if not field_roi:
            warnings.append(f"No ROI configured for '{FIELD_LABELS[key]}'.")
            fields[key], confidences[key], pages_used[key] = "", 0.0, None
            continue
        page_idx = int(field_roi.get("page", 0))
        if page_idx < 0 or page_idx >= len(pages):
            warnings.append(
                f"'{FIELD_LABELS[key]}' is set to page {page_idx + 1}, but this document only has "
                f"{len(pages)} page(s) - check the page number in ROI calibration."
            )
            fields[key], confidences[key], pages_used[key] = "", 0.0, None
            continue
        crop = crop_roi(pages[page_idx], field_roi["box"])
        if crop is None:
            warnings.append(f"Could not crop a region for '{FIELD_LABELS[key]}' - check ROI calibration.")
            fields[key], confidences[key], pages_used[key] = "", 0.0, page_idx + 1
            continue
        try:
            processed_crop = preprocess_document(crop, denoise_strength=denoise_strength)
        except ValueError:
            warnings.append(f"Empty crop for '{FIELD_LABELS[key]}'.")
            fields[key], confidences[key], pages_used[key] = "", 0.0, page_idx + 1
            continue
        result = ocr_devanagari(
            processed_crop, psm=field_def["psm"],
            numeric_only=(field_def["kind"] in ("number", "date")), lang=lang,
        )
        if result["error"]:
            warnings.append(f"OCR issue on '{FIELD_LABELS[key]}': {result['error']}")
        value = postprocess_field(field_def, result["text"])
        if key == "citizenship_no" and value and not validate_citizenship_number(value):
            warnings.append(
                f"'{value}' doesn't look like a valid citizenship number (expected 6-15 digits) - please verify manually."
            )
        fields[key] = value
        confidences[key] = result["confidence"]
        pages_used[key] = page_idx + 1

    overlays = [draw_roi_overlay_for_page(pages[i], roi_dict, i) for i in range(len(pages))]
    return {
        "fields": fields,
        "confidences": confidences,
        "pages_used": pages_used,
        "warnings": warnings,
        "overlays": overlays,
        "page_count": len(pages),
    }


# =====================================================================
# Cross-referencing
# =====================================================================
def similarity(a: str, b: str) -> float:
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 100.0
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a, b).ratio() * 100, 1)


def status_for_score(score: float, match_threshold: float) -> str:
    if score >= match_threshold:
        return STATUS_MATCH
    if score >= match_threshold - 30:
        return STATUS_REVIEW
    return STATUS_MISMATCH


def compare_fields(citizenship_fields: dict, kyc_fields: dict, match_threshold: float) -> list[dict]:
    rows = []
    for field_def in FIELD_DEFINITIONS:
        key = field_def["key"]
        val_a, val_b = citizenship_fields.get(key, ""), kyc_fields.get(key, "")
        score = similarity(val_a, val_b)
        rows.append({
            "key": key,
            "Field": field_def["label"],
            "Citizenship Certificate": val_a or "—",
            "KYC Form": val_b or "—",
            "Similarity (%)": score,
            "Status": status_for_score(score, match_threshold),
        })
    return rows


def compute_overall_status(rows: list[dict]) -> tuple[str, str]:
    """Returns (status_class, status_text) where status_class is one of
    'verified' / 'review' / 'failed', used to style the stamp."""
    worst_overall = max((STATUS_SEVERITY.get(r["Status"], 1) for r in rows), default=1)
    worst_critical = max(
        (STATUS_SEVERITY.get(r["Status"], 1) for r in rows if r["key"] in CRITICAL_FIELDS), default=1
    )
    if worst_critical == 2 or worst_overall == 2:
        return "failed", "Verification failed"
    if worst_overall == 1:
        return "review", "Needs manual review"
    return "verified", "Verified"


def apply_status_styles(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Colour the Status column; works across pandas versions where
    Styler.applymap was renamed/removed in favour of Styler.map."""
    def _style(val: str) -> str:
        if "Match" in val:
            return "background-color: #DEF3E7; color: #14532D; font-weight: 600;"
        if "Review" in val:
            return "background-color: #FBF0D2; color: #78430A; font-weight: 600;"
        if "Mismatch" in val:
            return "background-color: #F7DEE1; color: #7A1626; font-weight: 600;"
        return ""
    styler = df.style
    if hasattr(styler, "map"):
        return styler.map(_style, subset=["Status"])
    return styler.applymap(_style, subset=["Status"])  # older pandas fallback


# =====================================================================
# Sidebar
# =====================================================================
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### 🪪 Nepal KYC IDP Portal")
        st.caption("Intelligent Document Processing for citizenship-based KYC.")

        with st.expander("ℹ️ How this works", expanded=False):
            st.markdown(
                "1. **Ingest** - upload a photo or a multi-page PDF.\n"
                "2. **Preprocess** - denoise + adaptive threshold each page.\n"
                "3. **Localize** - crop the ROI for each field, on its configured page.\n"
                "4. **Recognize** - Tesseract OCR with your chosen Devanagari model.\n"
                "5. **Validate** - clean text, extract numbers/dates.\n"
                "6. **Cross-check** - compare the certificate against the KYC form."
            )

        st.markdown("#### Settings")
        ocr_lang_labels = list(OCR_LANG_OPTIONS.keys())
        ocr_lang_label = st.selectbox(
            "OCR language model", options=ocr_lang_labels,
            index=ocr_lang_labels.index(DEFAULT_OCR_LANG_LABEL),
            help=(
                "Nepali (nep) is tuned for Nepali vocabulary and numerals. The generic "
                "Devanagari script model can help on unusual fonts or handwriting but isn't "
                "Nepali-tuned. The combined option lets Tesseract score both per character and "
                "keep the better match - a safe default when both models are installed."
            ),
        )
        ocr_lang = OCR_LANG_OPTIONS[ocr_lang_label]

        st.markdown("#### Engine status")
        ok, msg = check_tesseract_ready(lang=ocr_lang)
        if ok:
            st.success(msg, icon="✅")
        else:
            st.error(msg, icon="🚫")

        match_threshold = st.slider(
            "Match threshold (%)", min_value=50, max_value=100, value=80, step=5,
            help="Similarity score at or above this is a Match. Roughly the next 30 points down is flagged for Review.",
        )
        denoise_strength = st.slider(
            "Denoise strength", min_value=3, max_value=25, value=10, step=1,
            help="Higher values remove more noise/glare but can blur fine strokes on low-resolution photos.",
        )

        if "roi_config" not in st.session_state:
            st.session_state.roi_config = {
                "citizenship": {k: dict(v) for k, v in DEFAULT_ROI["citizenship"].items()},
                "kyc": {k: dict(v) for k, v in DEFAULT_ROI["kyc"].items()},
            }

        with st.expander("🎯 Calibrate ROI regions", expanded=False):
            st.caption(
                "Default boxes are illustrative. Adjust them to match your own scans. "
                "For a multi-page PDF, set which page each field actually appears on "
                "(e.g. citizenship number on page 1, address on an annex page 2)."
            )
            doc_choice = st.selectbox(
                "Document", ["citizenship", "kyc"],
                format_func=lambda d: "Citizenship Certificate" if d == "citizenship" else "KYC Form",
            )
            field_choice = st.selectbox("Field", FIELD_KEYS, format_func=lambda k: FIELD_LABELS[k])
            current = st.session_state.roi_config[doc_choice].get(field_choice, DEFAULT_ROI[doc_choice][field_choice])
            key_prefix = f"roi_{doc_choice}_{field_choice}"
            page_no = st.number_input(
                "Page (1 = first page)", min_value=1, max_value=MAX_PDF_PAGES,
                value=int(current.get("page", 0)) + 1, step=1, key=f"{key_prefix}_page",
            )
            box = current["box"]
            x1 = st.slider("Left (x1)", 0.0, 1.0, float(box[0]), 0.01, key=f"{key_prefix}_x1")
            y1 = st.slider("Top (y1)", 0.0, 1.0, float(box[1]), 0.01, key=f"{key_prefix}_y1")
            x2 = st.slider("Right (x2)", 0.0, 1.0, float(box[2]), 0.01, key=f"{key_prefix}_x2")
            y2 = st.slider("Bottom (y2)", 0.0, 1.0, float(box[3]), 0.01, key=f"{key_prefix}_y2")
            st.session_state.roi_config[doc_choice][field_choice] = {
                "page": int(page_no) - 1, "box": (x1, y1, x2, y2),
            }
            if st.button("Reset this field to default", key=f"{key_prefix}_reset"):
                st.session_state.roi_config[doc_choice][field_choice] = dict(DEFAULT_ROI[doc_choice][field_choice])
                st.rerun()

        st.divider()
        st.caption(
            "Prototype for internal review workflows. Uploaded files are processed "
            "in memory for this session only and are not written to disk or sent "
            "anywhere outside this app. Confirm your own data-retention and KYC/AML "
            "policy requirements before using this in production."
        )

    return {
        "match_threshold": float(match_threshold),
        "denoise_strength": int(denoise_strength),
        "ocr_lang": ocr_lang,
    }


# =====================================================================
# Results rendering
# =====================================================================
def render_stamp(status_class: str, status_text: str, matched_count: int, total_count: int) -> str:
    sub = f"{matched_count}/{total_count} fields"
    return (
        f'<div class="kyc-stamp kyc-stamp--{status_class}">{status_text}'
        f'<span class="kyc-stamp-sub">{sub}</span></div>'
    )


def build_export_payload(results: dict) -> dict:
    rows = results["comparison_rows"]
    status_class, status_text = results["overall_status"]
    return {
        "verification_id": results["verification_id"],
        "generated_at": results["timestamp"],
        "overall_status": status_text,
        "match_threshold_percent": results["match_threshold"],
        "ocr_language": results.get("ocr_lang", "nep"),
        "fields": [
            {
                "field": r["Field"],
                "citizenship_certificate": r["Citizenship Certificate"],
                "kyc_form": r["KYC Form"],
                "similarity_percent": r["Similarity (%)"],
                "status": r["Status"],
            }
            for r in rows
        ],
        "ocr": {
            "citizenship_certificate": {
                "page_count": results["citizenship"]["page_count"],
                "fields": results["citizenship"]["fields"],
                "field_pages": results["citizenship"]["pages_used"],
                "confidence": results["citizenship"]["confidences"],
                "warnings": results["citizenship"]["warnings"],
            },
            "kyc_form": {
                "page_count": results["kyc"]["page_count"],
                "fields": results["kyc"]["fields"],
                "field_pages": results["kyc"]["pages_used"],
                "confidence": results["kyc"]["confidences"],
                "warnings": results["kyc"]["warnings"],
            },
        },
    }


def render_results(results: dict) -> None:
    rows = results["comparison_rows"]
    status_class, status_text = results["overall_status"]
    matched_count = sum(1 for r in rows if r["Status"] == STATUS_MATCH)
    all_confidences = list(results["citizenship"]["confidences"].values()) + list(results["kyc"]["confidences"].values())
    avg_conf = round(sum(all_confidences) / len(all_confidences), 1) if all_confidences else 0.0

    st.markdown("## Verification result")
    col_stamp, col_metrics = st.columns([1, 3])
    with col_stamp:
        st.markdown(render_stamp(status_class, status_text, matched_count, len(rows)), unsafe_allow_html=True)
    with col_metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Fields matched", f"{matched_count}/{len(rows)}")
        m2.metric("Avg. OCR confidence", f"{avg_conf}%")
        m3.metric("Match threshold", f"{results['match_threshold']:.0f}%")
        if results["citizenship"]["warnings"] or results["kyc"]["warnings"]:
            with st.expander("⚠️ Processing warnings", expanded=False):
                for w in results["citizenship"]["warnings"]:
                    st.warning(f"Citizenship certificate: {w}")
                for w in results["kyc"]["warnings"]:
                    st.warning(f"KYC form: {w}")

    tab_table, tab_roi, tab_raw, tab_json = st.tabs(
        ["📊 Comparison table", "🖼️ ROI preview", "📄 Raw OCR text", "🧾 JSON export"]
    )

    with tab_table:
        display_cols = ["Field", "Citizenship Certificate", "KYC Form", "Similarity (%)", "Status"]
        display_df = pd.DataFrame(rows)[display_cols]
        st.dataframe(apply_status_styles(display_df), use_container_width=True, hide_index=True)

    with tab_roi:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Citizenship certificate**")
            for i, ov in enumerate(results["citizenship"]["overlays"]):
                st.image(ov, caption=f"Page {i + 1}", use_container_width=True)
        with c2:
            st.markdown("**KYC form**")
            for i, ov in enumerate(results["kyc"]["overlays"]):
                st.image(ov, caption=f"Page {i + 1}", use_container_width=True)

    with tab_raw:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Citizenship certificate - extracted fields**")
            for k in FIELD_KEYS:
                page = results["citizenship"]["pages_used"].get(k)
                page_note = f" · page {page}" if page else ""
                st.markdown(f'<span class="kyc-field-label">{FIELD_LABELS[k]}{page_note}</span>', unsafe_allow_html=True)
                st.code(results["citizenship"]["fields"].get(k, "") or "(empty)", language=None)
        with c2:
            st.markdown("**KYC form - extracted fields**")
            for k in FIELD_KEYS:
                page = results["kyc"]["pages_used"].get(k)
                page_note = f" · page {page}" if page else ""
                st.markdown(f'<span class="kyc-field-label">{FIELD_LABELS[k]}{page_note}</span>', unsafe_allow_html=True)
                st.code(results["kyc"]["fields"].get(k, "") or "(empty)", language=None)

    with tab_json:
        payload = build_export_payload(results)
        st.json(payload)
        json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON",
            data=json_bytes,
            file_name=f"kyc_verification_{results['verification_id']}.json",
            mime="application/json",
            type="primary",
        )


# =====================================================================
# Main app
# =====================================================================
def render_page_preview(pages: list[np.ndarray], key_prefix: str) -> None:
    """Show a single preview image, or a page-flip slider for multi-page docs."""
    if len(pages) == 1:
        st.image(pages[0], use_container_width=True)
        return
    page_no = st.slider(
        "Preview page", min_value=1, max_value=len(pages), value=1, key=f"{key_prefix}_page_select",
    )
    st.image(pages[page_no - 1], caption=f"Page {page_no} of {len(pages)}", use_container_width=True)


def main() -> None:
    settings = render_sidebar()

    st.markdown(
        '<div class="kyc-header"><div class="kyc-flagbar">🪪</div>'
        "<div><h1>Nepal KYC IDP Portal</h1>"
        "<p>Devanagari OCR for citizenship certificates, cross-checked against your KYC form.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="kyc-rule"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 1. Citizenship certificate")
        citizenship_upload = st.file_uploader(
            "Upload the citizenship certificate - image or PDF, multiple pages supported",
            type=["png", "jpg", "jpeg", "pdf"], key="citizenship_upload",
        )
        citizenship_pages: list[np.ndarray] = []
        if citizenship_upload is not None:
            citizenship_pages, load_warnings = load_document_pages(citizenship_upload)
            for w in load_warnings:
                st.warning(w)
            if citizenship_pages:
                render_page_preview(citizenship_pages, key_prefix="citizenship_preview")
            else:
                st.error("Could not read this file. Please upload a JPG, PNG, or PDF.")

    with col_b:
        st.markdown("#### 2. Physical KYC form")
        kyc_upload = st.file_uploader(
            "Upload the completed KYC form - image or PDF, multiple pages supported",
            type=["png", "jpg", "jpeg", "pdf"], key="kyc_upload",
        )
        kyc_pages: list[np.ndarray] = []
        if kyc_upload is not None:
            kyc_pages, load_warnings = load_document_pages(kyc_upload)
            for w in load_warnings:
                st.warning(w)
            if kyc_pages:
                render_page_preview(kyc_pages, key_prefix="kyc_preview")
            else:
                st.error("Could not read this file. Please upload a JPG, PNG, or PDF.")

    st.write("")
    ready_to_process = len(citizenship_pages) > 0 and len(kyc_pages) > 0
    process_clicked = st.button(
        "🔍 Process KYC & Extract Data", type="primary",
        use_container_width=True, disabled=not ready_to_process,
    )
    if not ready_to_process:
        st.caption("Upload both documents to enable processing.")

    if process_clicked:
        engine_ok, engine_msg = check_tesseract_ready(lang=settings["ocr_lang"])
        if not engine_ok:
            st.error(f"Can't run OCR yet: {engine_msg}")
        else:
            with st.spinner("Processing documents - preprocessing, OCR, validation..."):
                try:
                    citizenship_result = extract_fields_from_document(
                        citizenship_pages, st.session_state.roi_config["citizenship"],
                        settings["denoise_strength"], settings["ocr_lang"],
                    )
                    kyc_result = extract_fields_from_document(
                        kyc_pages, st.session_state.roi_config["kyc"],
                        settings["denoise_strength"], settings["ocr_lang"],
                    )
                    comparison_rows = compare_fields(
                        citizenship_result["fields"], kyc_result["fields"], settings["match_threshold"]
                    )
                    overall_status = compute_overall_status(comparison_rows)
                    st.session_state.kyc_results = {
                        "verification_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "citizenship": citizenship_result,
                        "kyc": kyc_result,
                        "comparison_rows": comparison_rows,
                        "overall_status": overall_status,
                        "match_threshold": settings["match_threshold"],
                        "ocr_lang": settings["ocr_lang"],
                    }
                except Exception as exc:
                    st.error(f"Processing failed unexpectedly: {exc}")
                    with st.expander("Technical details"):
                        st.exception(exc)

    if "kyc_results" in st.session_state:
        st.divider()
        render_results(st.session_state.kyc_results)


main()
