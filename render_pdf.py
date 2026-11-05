#!/usr/bin/env python3
import os
import re
from pathlib import Path
import markdown
from xhtml2pdf import pisa

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "report.md"
PDF_PATH = BASE_DIR / "Smart_Agriculture_IoT_Research_Report.pdf"

def build_pdf():
    print(f"Reading markdown from {MD_PATH}...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML
    # Replace unicode box-drawing characters with standard ASCII for crisp Courier rendering
    md_text = md_text.replace('│', '|').replace('─', '-').replace('┌', '+').replace('┐', '+').replace('└', '+').replace('┘', '+').replace('▼', 'v').replace('•', '*')
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'sane_lists'])

    # Fix relative image paths to absolute file:/// URIs for xhtml2pdf
    def img_replace(match):
        src = match.group(1)
        if not src.startswith("http") and not src.startswith("file://"):
            full_img = BASE_DIR / src
            if full_img.exists():
                return f'src="file://{full_img}"'
        return f'src="{src}"'

    html_body = re.sub(r'src=["\']([^"\']+)["\']', img_replace, html_body)

    # Wrap in styled HTML document
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: a4 portrait;
    margin: 1.8cm 1.5cm 2.0cm 1.5cm;
    @frame footer_frame {{
      -pdf-frame-content: footer_content;
      left: 1.5cm; width: 18cm; top: 27.6cm; height: 1.0cm;
    }}
  }}
  body {{
    font-family: Helvetica, Arial, sans-serif;
    color: #1e293b;
    font-size: 8.5pt;
    line-height: 1.45;
  }}
  h1 {{
    font-size: 15pt;
    font-weight: bold;
    color: #14532d;
    text-align: center;
    margin-top: 0;
    margin-bottom: 8px;
    line-height: 1.25;
  }}
  h2 {{
    font-size: 11pt;
    font-weight: bold;
    color: #0f172a;
    border-bottom: 1.5px solid #16a34a;
    padding-bottom: 2px;
    margin-top: 14px;
    margin-bottom: 6px;
  }}
  h3 {{
    font-size: 9.5pt;
    font-weight: bold;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
  }}
  h4 {{
    font-size: 8.5pt;
    font-weight: bold;
    color: #475569;
    margin-top: 8px;
    margin-bottom: 3px;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 6px;
    text-align: justify;
  }}
  ul, ol {{
    margin-top: 2px;
    margin-bottom: 6px;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 3px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 6px;
    margin-bottom: 10px;
    font-size: 7.5pt;
  }}
  th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: bold;
    border: 0.5px solid #94a3b8;
    padding: 4px 5px;
    text-align: left;
  }}
  td {{
    border: 0.5px solid #cbd5e1;
    padding: 3.5px 5px;
  }}
  pre {{
    font-family: Courier, monospace;
    font-size: 7pt;
    background-color: #f8fafc;
    border: 0.5px solid #cbd5e1;
    padding: 5px;
    margin: 5px 0;
  }}
  code {{
    font-family: Courier, monospace;
    font-size: 7.5pt;
    background-color: #f1f5f9;
  }}
  blockquote {{
    border-left: 2.5px solid #16a34a;
    background-color: #f0fdf4;
    padding: 4px 8px;
    margin: 6px 0;
  }}
  img {{
    width: 90%;
    margin-left: auto;
    margin-right: auto;
    margin-top: 6px;
    margin-bottom: 6px;
    text-align: center;
  }}
  hr {{
    border: 0;
    border-top: 0.5px solid #cbd5e1;
    margin: 10px 0;
  }}
  .meta-box {{
    text-align: center;
    font-size: 8pt;
    color: #475569;
    margin-bottom: 12px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
  }}
  #footer_content {{
    font-size: 7pt;
    color: #64748b;
    border-top: 0.5px solid #cbd5e1;
    padding-top: 3px;
  }}
</style>
</head>
<body>

{html_body}

<div id="footer_content">
  <table style="width: 100%; border: 0; margin: 0; font-size: 7pt;">
    <tr>
      <td style="border: 0; text-align: left; color: #64748b;">Adaptive Multimodal Edge-Cloud IoT Framework for Precision Agriculture</td>
      <td style="border: 0; text-align: right; color: #64748b;">Page <pdf:pageNumber/> of <pdf:pageCount/></td>
    </tr>
  </table>
</div>

</body>
</html>
"""

    print(f"Generating PDF: {PDF_PATH}...")
    with open(PDF_PATH, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        print(f"❌ Error occurred during PDF generation: {pisa_status.err}")
    else:
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"✅ PDF generated successfully: {PDF_PATH} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    build_pdf()
