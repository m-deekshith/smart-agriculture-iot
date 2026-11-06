#!/usr/bin/env python3
"""
Comprehensive 20+ Page Academic Project Thesis / Research Report Generator
Course: IoT and Industrial IoT (Problem Statement 10)
Platform: ESP32 Dev Module, RS485 Modbus RTU, Optical Camera, TinyML, Cloud Decision Fusion
Output: Smart_Agriculture_IoT_Research_Report.pdf (>= 20 Pages with Embedded Figures)
"""

import os
import sys
import shutil
from pathlib import Path
from xhtml2pdf import pisa
from pypdf import PdfReader

BASE_DIR = Path("/home/deek/Projects/smart_agriculture")
HTML_TEMPLATE = BASE_DIR / "report_template.html"
PDF_OUTPUT = BASE_DIR / "Smart_Agriculture_IoT_Research_Report.pdf"
DOWNLOADS_OUTPUT = Path("/home/deek/Downloads/Smart_Agriculture_IoT_Research_Report.pdf")
ARTIFACT_DIR = Path("/home/deek/.gemini/antigravity-cli/brain/405c7ef2-6fb8-4274-a828-bedbf5a5b0e6")
ARTIFACT_OUTPUT = ARTIFACT_DIR / "Smart_Agriculture_IoT_Research_Report.pdf"

# Verify Images
IMG_MODALITY = BASE_DIR / "modality_accuracy_comparison.png"
IMG_MODEL_COMP = BASE_DIR / "ai_model_comparison.png"
IMG_LATENCY = BASE_DIR / "latency_comparison.png"
IMG_BANDWIDTH = BASE_DIR / "bandwidth_savings.png"
IMG_SUBMERSION = BASE_DIR / "hardware_test_submersion.png"
IMG_DASHBOARD = BASE_DIR / "dashboard_live_view.png"
IMG_PLANNER = BASE_DIR / "crop_planner_gap_analysis.png"
IMG_THINGSPEAK = BASE_DIR / "thingspeak_cloud_mirror.png"
IMG_IIIT_LOGO = BASE_DIR / "iiit_dharwad_logo.png"

image_map = {
    "__IMG_MODALITY__": f"file://{IMG_MODALITY}",
    "__IMG_MODEL_COMP__": f"file://{IMG_MODEL_COMP}",
    "__IMG_LATENCY__": f"file://{IMG_LATENCY}",
    "__IMG_BANDWIDTH__": f"file://{IMG_BANDWIDTH}",
    "__IMG_SUBMERSION__": f"file://{IMG_SUBMERSION}",
    "__IMG_DASHBOARD__": f"file://{IMG_DASHBOARD}",
    "__IMG_PLANNER__": f"file://{IMG_PLANNER}",
    "__IMG_THINGSPEAK__": f"file://{IMG_THINGSPEAK}",
    "__IIIT_LOGO__": f"file://{IMG_IIIT_LOGO}",
}

for key, uri in image_map.items():
    file_path = Path(uri.replace("file://", ""))
    if not file_path.exists():
        print(f"Error: Missing image {file_path}")
        sys.exit(1)

print("All 8 required figures verified.")

def build_pdf():
    print(f"Reading HTML template from {HTML_TEMPLATE}...")
    with open(HTML_TEMPLATE, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Substitute image URIs
    for placeholder, uri in image_map.items():
        html_content = html_content.replace(placeholder, uri)

    print(f"Writing PDF to {PDF_OUTPUT}...")
    with open(PDF_OUTPUT, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

    if pisa_status.err:
        print(f"❌ Error during PDF rendering: {pisa_status.err}")
        sys.exit(1)

    # Check page count with pypdf
    reader = PdfReader(str(PDF_OUTPUT))
    num_pages = len(reader.pages)
    file_size_mb = os.path.getsize(PDF_OUTPUT) / (1024 * 1024)
    print(f"✅ Generated PDF successfully!")
    print(f"📄 Total Page Count: {num_pages} pages")
    print(f"💾 File Size: {file_size_mb:.2f} MB")

    if num_pages < 20:
        print(f"⚠️ Warning: Page count is {num_pages}, which is less than the requested 20 pages!")
    else:
        print(f"🎉 SUCCESS: Page count {num_pages} >= 20 pages satisfied!")

    # Copy to user Downloads
    shutil.copyfile(PDF_OUTPUT, DOWNLOADS_OUTPUT)
    print(f"📋 Copied to User Downloads: {DOWNLOADS_OUTPUT}")

    # Copy to artifact directory
    shutil.copyfile(PDF_OUTPUT, ARTIFACT_OUTPUT)
    print(f"📋 Copied to Artifacts: {ARTIFACT_OUTPUT}")

if __name__ == "__main__":
    build_pdf()
