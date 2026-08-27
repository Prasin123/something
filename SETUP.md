# Setup — Nepal KYC IDP Portal

The Python packages in `requirements.txt` are only half the picture.
Tesseract OCR itself, and its Nepali language model, are **system-level**
installs — `pip` cannot install them. If you skip this step the app will
still launch, but the sidebar will show a red "engine status" warning
instead of crashing, and processing will refuse to run until it's fixed.

## 1. Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Tesseract OCR + Nepali language data

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-nep
```

**macOS (Homebrew)**
```bash
brew install tesseract tesseract-lang   # tesseract-lang bundles nep.traineddata
```

**Windows**
1. Install Tesseract via the UB-Mannheim build: https://github.com/UB-Mannheim/tesseract/wiki
2. During setup, tick "Nepali" under additional language data — or download
   `nep.traineddata` from https://github.com/tesseract-ocr/tessdata and
   copy it into `C:\Program Files\Tesseract-OCR\tessdata`.
3. If `tesseract` isn't on your PATH, set it explicitly before running
   Streamlit:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
   (add this near the top of `app.py` if needed for your machine.)

**Docker (any OS)**
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-nep && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## 3. (Optional) Add the generic Devanagari script model

The sidebar's "OCR language model" setting can combine Nepali (`nep`)
with a generic Devanagari script model for potentially better results
on numerals and unusual fonts — the app defaults to this combined mode.
The Nepali pack from step 2 is required either way; the script model is
an optional add-on:

1. Download `Devanagari.traineddata` from the tesseract-ocr `tessdata_best`
   GitHub repository (script-based models, Apache 2.0 licensed).
2. Copy it into the same tessdata folder as `nep.traineddata` — on
   Ubuntu that's typically `/usr/share/tesseract-ocr/5/tessdata/` (check
   your version with `tesseract --version`); on Windows it's
   `C:\Program Files\Tesseract-OCR\tessdata`.
3. If it's missing, the sidebar still tells you exactly which model
   isn't found rather than crashing — you can switch the dropdown back
   to plain "Nepali (nep)" at any time.

## 4. Verify the install

```bash
tesseract --list-langs
```
You should see `nep` in the list (and `Devanagari` too, if you added it
from step 3). If not, re-check step 2.

## 5. Run the app

```bash
streamlit run app.py
```

## Notes on accuracy

- **PDF support** needs no extra system install — `pypdfium2` (in
  `requirements.txt`) renders PDF pages to images entirely via pip, no
  poppler or other system binary required.
- **Multi-page documents**: each field's ROI is bound to a specific
  page, not just a box. If your KYC packet has the applicant's name on
  page 1 but citizenship number / DOB / address on an annex page 2
  (common with Nepali bank forms — the main "Personal Account Opening
  Form" often doesn't carry those fields itself), upload the full
  multi-page PDF and set each field's page number in the sidebar's
  "Calibrate ROI regions" panel.
- The ROI (Region of Interest) boxes that locate each field on the
  document are illustrative defaults, not a universal template — real
  citizenship certificates vary by print era, format, and scan quality.
  Use the **"Calibrate ROI regions"** panel in the sidebar to nudge the
  boxes for your own camera/scanner setup; it updates live as you drag
  the sliders.
- This app is built as a reviewable assistant for a human KYC
  reviewer — it flags discrepancies and confidence scores, but nothing
  here should be wired into an unattended auto-approve/reject decision
  without human sign-off and your own compliance review.
- No uploaded image or extracted field is written to disk or sent
  anywhere by this app; everything lives in the Streamlit session's
  memory and is cleared when the session ends.
