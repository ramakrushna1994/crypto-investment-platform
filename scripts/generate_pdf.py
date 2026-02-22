"""Simple Markdown -> PDF generator using ReportLab.

Writes `docs/codebase_explanation.pdf` from `docs/codebase_explanation.md`.
"""
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap
import os


def md_to_pdf(md_path, pdf_path, page_size=A4, font_name='Helvetica', font_size=10, margin=40):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    width, height = page_size
    c = canvas.Canvas(pdf_path, pagesize=page_size)
    x = margin
    y = height - margin
    wrapper = textwrap.TextWrapper(width=100)
    c.setFont(font_name, font_size)

    for line in lines:
        if not line.strip():
            y -= font_size * 0.8
            if y < margin:
                c.showPage(); c.setFont(font_name, font_size); y = height - margin
            continue

        # If markdown heading, make it slightly larger
        if line.startswith('#'):
            heading_level = len(line) - len(line.lstrip('#'))
            text = line.lstrip('#').strip()
            c.setFont(font_name, min(14, font_size + (4 - heading_level)))
            wrapped = wrapper.wrap(text)
            for w in wrapped:
                c.drawString(x, y, w)
                y -= font_size * 1.4
                if y < margin:
                    c.showPage(); c.setFont(font_name, font_size); y = height - margin
            c.setFont(font_name, font_size)
            continue

        wrapped = wrapper.wrap(line)
        for w in wrapped:
            c.drawString(x, y, w)
            y -= font_size * 1.2
            if y < margin:
                c.showPage(); c.setFont(font_name, font_size); y = height - margin

    c.save()


if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(__file__), '..')
    md = os.path.abspath(os.path.join(BASE, 'docs', 'codebase_explanation.md'))
    pdf = os.path.abspath(os.path.join(BASE, 'docs', 'codebase_explanation.pdf'))
    print(f"Reading: {md}")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    md_to_pdf(md, pdf)
    print(f"Wrote PDF to: {pdf}")
