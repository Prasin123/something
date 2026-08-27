import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import json
import difflib
import pypdfium2 as pdfium

# =====================================================================
# Constants & Defaults
# =====================================================================
OCR_LANG_OPTIONS = {
    "Nepali (nep)": "nep",
    "English (eng)": "eng",
    "Devanagari Script (deva)": "script/Devanagari",
    "Nepali + Devanagari": "nep+script/Devanagari",
    "English + Nepali": "eng+nep"
}

DEFAULT_ROIS = {
    "citizenship": {
        "Citizenship Number": (0.05, 0.15, 0.40, 0.08),
        "Full Name": (0.05, 0.25, 0.60, 0.08),
        "Date of Birth": (0.05, 0.35, 0.40, 0.08),
        "Father's Name": (0.05, 0.45, 0.60, 0.08),
        "District of Issue": (0.05, 0.55, 0.40, 0.08),
    },
    "kyc": {
        "Citizenship Number": (0.10, 0.20, 0.40, 0.05),
        "Full Name": (0.10, 0.30, 0.60, 0.05),
        "Date of Birth": (0.10, 0.40, 0.40, 0.05),
        "Father's Name": (0.10, 0.50, 0.60, 0.05),
        "District of Issue": (0.10, 0.60, 0.40, 0.05),
    }
}

# =====================================================================
# Helper Functions
# =====================================================================
def check_tesseract():
    """Verify if Tesseract is installed and accessible."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def load_document_pages(uploaded_file):
    """Load PDFs using pypdfium2 or image files into a list of OpenCV BGR images."""
    pages = []
    warnings = []
    try:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".pdf"):
            pdf_doc = pdfium.PdfDocument(file_bytes)
            for i in range(len(pdf_doc)):
                page = pdf_doc[i]
                image = page.render(scale=2.0).to_pil() # 2x scale for better OCR clarity
                pages.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        else:
            nparr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                pages.append(img)
            else:
                warnings.append("Could not decode image file.")
    except Exception as e:
        warnings.append(f"Error loading document: {str(e)}")
    
    return pages, warnings

def preprocess_image(img, denoise_strength):
    """Convert to grayscale, denoise, and binarize for OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise_strength > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_strength)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_fields_from_document(pages, roi_config, denoise_strength, ocr_lang):
    """Extract text from designated Regions of Interest (ROI)."""
    if not pages:
        return {"fields": {}, "confidences": {}, "overlays": [], "warnings": ["No pages to process."]}

    base_img = pages[0]
    height, width = base_img.shape[:2]
    
    overlay_img = base_img.copy()
    fields = {}
    confidences = {}
    warnings = []

    custom_config = r'--oem 3 --psm 6'
    
    for field_name, (fx, fy, fw, fh) in roi_config.items():
        x, y = int(fx * width), int(fy * height)
        w, h = int(fw * width), int(fh * height)
        
        roi = base_img[y:y+h, x:x+w]
        
        if roi.size == 0:
            warnings.append(f"Invalid crop for field: {field_name}")
            continue
            
        processed_roi = preprocess_image(roi, denoise_strength)
        
        cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(overlay_img, field_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        try:
            text = pytesseract.image_to_string(processed_roi, lang=ocr_lang, config=custom_config).strip()
            data = pytesseract.image_to_data(processed_roi, lang=ocr_lang, config=custom_config, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data['conf'] if int(c) != -1]
            avg_conf = sum(confs) / len(confs) if confs else 0
            
            fields[field_name] = text
            confidences[field_name] = avg_conf
        except Exception as e:
            warnings.append(f"OCR Error on {field_name}: {str(e)}")
            fields[field_name] = ""
            confidences[field_name] = 0

    return {
        "fields": fields,
        "confidences": confidences,
        "overlays": [cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)],
        "warnings": warnings
    }

def string_similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100

def compare_fields(cit_fields, kyc_fields, threshold):
    results = []
    for field in cit_fields.keys():
        val1 = cit_fields.get(field, "").strip()
        val2 = kyc_fields.get(field, "").strip()
        
        if not val1 and not val2:
            sim = 0.0
            status = "Missing"
        else:
            sim = string_similarity(val1, val2)
            if sim >= threshold:
                status = "Match"
            elif sim >= threshold - 15:
                status = "Review"
            else:
                status = "Mismatch"
                
        results.append({
            "Field": field,
            "Citizenship Certificate": val1,
            "KYC Form": val2,
            "Similarity (%)": round(sim, 1),
            "Status": status
        })
    return results

def compute_overall_status(comparison_rows):
    if not comparison_rows:
        return "rejected", "NO DATA"
    statuses = [r["Status"] for r in comparison_rows]
    if "Mismatch" in statuses:
        return "rejected", "MISMATCH DETECTED"
    elif "Review" in statuses or "Missing" in statuses:
        return "review", "MANUAL REVIEW REQUIRED"
    else:
        return "approved", "VERIFIED & APPROVED"

def apply_status_styles(df):
    def color_status(val):
        if val == "Match":
            return "color: green; font-weight: bold;"
        elif val == "Review":
            return "color: orange; font-weight: bold;"
        elif val == "Mismatch":
            return "color: red; font-weight: bold;"
        return ""
    return df.style.map(color_status, subset=["Status"])

# =====================================================================
# UI Elements
# =====================================================================
def render_sidebar():
    st.sidebar.title("⚙️ IDP Pipeline Settings")
    
    tesseract_ok = check_tesseract()
    if tesseract_ok:
        st.sidebar.success("✅ Tesseract Engine Ready")
    else:
        st.sidebar.error("❌ Tesseract Engine Not Found.")

    st.sidebar.markdown("### OCR Configuration")
    lang_choice = st.sidebar.selectbox("OCR Language Mode", list(OCR_LANG_OPTIONS.keys()), index=2)
    ocr_lang = OCR_LANG_OPTIONS[lang_choice]
    
    st.sidebar.markdown("### Image Preprocessing")
    denoise_strength = st.sidebar.slider("Denoising Strength", 0, 20, 10)
    
    st.sidebar.markdown("### Matching Logic")
    match_threshold = st.sidebar.slider("Fuzzy Match Threshold (%)", 50, 100, 85)

    if "roi_config" not in st.session_state:
        st.session_state.roi_config = DEFAULT_ROIS

    return {
        "ocr_lang": ocr_lang,
        "match_threshold": match_threshold,
        "denoise_strength": denoise_strength,
        "tesseract_ok": tesseract_ok,
    }

def main():
    st.set_page_config(page_title="Nepal KYC IDP", layout="wide")
    settings = render_sidebar()

    st.markdown("""
        <style>
        .kyc-header { display: flex; align-items: center; gap: 20px; padding-bottom: 10px; }
        .kyc-flagbar { font-size: 3rem; }
        .kyc-rule { border-bottom: 2px solid #ccc; margin-bottom: 20px; }
        .kyc-stamp { 
            border: 4px solid; padding: 20px; text-align: center; 
            font-size: 1.5rem; font-weight: bold; border-radius: 10px; 
            text-transform: uppercase; margin: 10px 0;
            display: inline-block; transform: rotate(-5deg);
        }
        .kyc-stamp--approved { color: #28a745; border-color: #28a745; }
        .kyc-stamp--rejected { color: #dc3545; border-color: #dc3545; }
        .kyc-stamp--review { color: #ffc107; border-color: #ffc107; }
        .kyc-stamp-sub { display: block; font-size: 0.8rem; margin-top: 5px; opacity: 0.8; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="kyc-header">
            <div class="kyc-flagbar">🪪</div>
            <div>
                <h1 style="margin-bottom:0;">Nepal KYC IDP Portal</h1>
                <p style="margin-top:0; color: gray;">Intelligent Document Processing & Automated Cross-Verification</p>
            </div>
        </div>
        <div class="kyc-rule"></div>
        """,
        unsafe_allow_html=True,
    )

    if not settings["tesseract_ok"]:
        st.error("⚠️ Tesseract OCR is not accessible. Check your packages.txt file.")
        st.stop()

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.markdown("### 1. Citizenship Certificate")
        cit_file = st.file_uploader("Upload Citizenship File", type=["jpg", "jpeg", "png", "pdf"], key="cit_upload")

    with col_up2:
        st.markdown("### 2. KYC Form")
        kyc_file = st.file_uploader("Upload KYC Form File", type=["jpg", "jpeg", "png", "pdf"], key="kyc_upload")

    if not cit_file or not kyc_file:
        st.info("👆 Please upload both documents above to begin cross-verification.")
        return

    cit_pages, cit_warnings = load_document_pages(cit_file)
    kyc_pages, kyc_warnings = load_document_pages(kyc_file)

    for w in cit_warnings:
        st.warning(f"Citizenship warning: {w}")
    for w in kyc_warnings:
        st.warning(f"KYC warning: {w}")

    if not cit_pages or not kyc_pages:
        st.error("Could not process uploaded files.")
        return

    with st.spinner(f"Extracting text (OCR Language: {settings['ocr_lang']})..."):
        cit_result = extract_fields_from_document(
            cit_pages, st.session_state.roi_config["citizenship"], settings["denoise_strength"], settings["ocr_lang"]
        )
        kyc_result = extract_fields_from_document(
            kyc_pages, st.session_state.roi_config["kyc"], settings["denoise_strength"], settings["ocr_lang"]
        )

    comparison_rows = compare_fields(cit_result["fields"], kyc_result["fields"], settings["match_threshold"])
    df_compare = pd.DataFrame(comparison_rows)
    stamp_class, stamp_text = compute_overall_status(comparison_rows)

    st.markdown("---")
    st.markdown("### 🔍 Verification & Cross-Check Results")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.markdown("#### Field-by-Field Reconciliation")
        display_df = df_compare[["Field", "Citizenship Certificate", "KYC Form", "Similarity (%)", "Status"]]
        st.dataframe(apply_status_styles(display_df), use_container_width=True, hide_index=True)

    with res_col2:
        st.markdown("#### Audit Decision Stamp")
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1rem;">
                <div class="kyc-stamp kyc-stamp--{stamp_class}">
                    {stamp_text}
                    <span class="kyc-stamp-sub">Nepal IDP Gateway</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
