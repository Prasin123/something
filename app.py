import re
from io import BytesIO
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Nepali IDP & KYC Extractor", page_icon="🪪", layout="wide"
)


# Initialize EasyOCR Reader for Nepali ('ne') and English ('en') once using caching
@st.cache_resource
def load_ocr_reader():
  return easyocr.Reader(["ne", "en"])


reader = load_ocr_reader()


def scan_effect(img):
  """Enhances image contrast and sharpness for better OCR accuracy on identity documents."""
  gray_img = img.convert("L")
  enhancer = ImageEnhance.Contrast(gray_img)
  contrast_img = enhancer.enhance(2.0)
  sharp_img = contrast_img.filter(ImageFilter.SHARPEN)
  return sharp_img


def extract_kyc_fields(ocr_texts):
  """Applies heuristic regex patterns to extract structured KYC details from raw OCR text lines."""
  kyc_data = {
      "Citizenship Number": None,
      "Date of Birth (DOB)": None,
      "Raw Extracted Text": ocr_texts,
  }

  # Regular expression heuristics for Nepali citizenship patterns
  cit_pattern = re.compile(r"\d{2}-\d{2}-\d{2}-\d{5}|\d{4}-\d{2}-\d{2}-\d{4}")
  dob_pattern = re.compile(
      r"\b(?:20\d{2}|19\d{2})[./-]\d{1,2}[./-]\d{1,2}\b|\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b"
  )

  for text in ocr_texts:
    if not kyc_data["Citizenship Number"]:
      match_cit = cit_pattern.search(text)
      if match_cit:
        kyc_data["Citizenship Number"] = match_cit.group()

    if not kyc_data["Date of Birth (DOB)"]:
      match_dob = dob_pattern.search(text)
      if match_dob:
        kyc_data["Date of Birth (DOB)"] = match_dob.group()

  return kyc_data


# --- Streamlit UI Layout ---
st.title("🪪 Intelligent Document Processing (IDP) - KYC Extractor")
st.markdown(
    "Automated Devanagari OCR pipeline built to extract structured KYC fields"
    " from Nepali identity documents."
)

uploaded_file = st.file_uploader(
    "Upload Nepali Citizenship / ID Document", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Original Document")
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

  # Process image
  scanned_image = scan_effect(image)

  with col2:
    st.subheader("Enhanced Scan Layer")
    st.image(scanned_image, use_column_width=True)

  with st.spinner("Extracting Devanagari text and structuring KYC fields..."):
    # Run EasyOCR
    output = reader.readtext(np.array(scanned_image))
    detected_texts = [text for (_, text, _) in output]
    kyc_results = extract_kyc_fields(detected_texts)

  # --- Results Section ---
  st.markdown("---")
  st.subheader("📊 Structured KYC Output")

  m_col1, m_col2 = st.columns(2)
  with m_col1:
    st.metric(
        label="Extracted Citizenship ID",
        value=(
            kyc_results["Citizenship Number"]
            if kyc_results["Citizenship Number"]
            else "Not Automatically Detected"
        ),
    )
  with m_col2:
    st.metric(
        label="Detected Date of Birth",
        value=(
            kyc_results["Date of Birth (DOB)"]
            if kyc_results["Date of Birth (DOB)"]
            else "Not Automatically Detected"
        ),
    )

  # Display Bounding Boxes & Visual Overlays
  st.subheader("🔍 OCR Bounding Box Overlay")
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.imshow(scanned_image, cmap="gray")

  for bbox, text, conf in output:
    rect_x = [p[0] for p in bbox]
    rect_y = [p[1] for p in bbox]
    ax.plot(rect_x + [rect_x[0]], rect_y + [rect_y[0]], "r-", linewidth=2)
    ax.text(
        rect_x[0],
        rect_y[0] - 5,
        f"{text} ({conf:.2f})",
        color="yellow",
        fontsize=9,
        backgroundcolor="black",
    )

  ax.axis("off")
  st.pyplot(fig)

  # Expandable raw text & JSON view
  with st.expander("View Full Raw OCR Breakdown & JSON"):
    for i, (bbox, text, confidence) in enumerate(output):
      st.write(f"**{i+1}.** {text} *(Confidence: {confidence:.2f})*")

    st.json(kyc_results)

else:
  st.info("Please upload a scan or clear photo of a Nepali identity document.")
