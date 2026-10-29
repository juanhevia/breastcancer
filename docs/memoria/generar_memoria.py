"""Genera el DOCX editable desde memoria.md y figuras PNG de datos verificados.

Requiere python-docx, pandas, numpy y matplotlib. No entrena ni consulta test.
"""
from pathlib import Path
import argparse
import json
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def figures():
    """La evidencia se prepara con app.exportar_v8; no sobrescribirla con v7."""
    datos = ROOT / 'app/static/evidencia/v8/datos.json'
    if not datos.exists():
        raise FileNotFoundError('Falta evidencia final v8: ejecutar python -m app.exportar_v8 --origen <carpeta models>')
    evidencia = json.loads(datos.read_text())
    (HERE / 'fuentes.json').write_text(json.dumps({
        'modelo': 'v8', 'sha256_checkpoint': evidencia['sha256'],
        'unidad_auc': 'paciente', 'fuentes': evidencia['fuentes']
    }, ensure_ascii=False, indent=2) + '\n')


def runs(p, text):
    for i, bit in enumerate(re.split(r"\*\*(.*?)\*\*", text)):
        run = p.add_run(bit)
        run.bold = bool(i % 2)


def table(doc, lines):
    rows = [[c.strip() for c in s.strip().strip("|").split("|")] for s in lines]
    rows = [r for r in rows if not all(re.fullmatch(r"[:\- ]+", c) for c in r)]
    n = len(rows[0])
    widths = {3: [8.0, 4.25, 4.25], 4: [4.2, 4.1, 4.1, 4.1],
              5: [4.6, 3.5, 2.8, 2.8, 2.8], 2: [6.0, 10.5]}[n]
    if rows[0][0] == "Fold externo":
        widths = [3.3, 4.5, 8.7]
    if rows[0][0].startswith("Cohorte"):
        widths = [5, 2.8, 2.8, 2.8, 3.1]
    t = doc.add_table(rows=1, cols=n)
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Cm(width)
    for idx, values in enumerate(rows):
        cells = t.rows[0].cells if idx == 0 else t.add_row().cells
        trpr = cells[0]._tc.getparent().get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if idx == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for j, (cell, value) in enumerate(zip(cells, values)):
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            borders = OxmlElement("w:tcBorders")
            for edge in ["top", "left", "bottom", "right"]:
                b = OxmlElement("w:" + edge)
                for k, v in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
                    b.set(qn("w:" + k), v)
                borders.append(b)
            tcpr.append(borders)
            margins = OxmlElement("w:tcMar")
            for edge in ["top", "left", "bottom", "right"]:
                m = OxmlElement("w:" + edge)
                m.set(qn("w:w"), "80")
                m.set(qn("w:type"), "dxa")
                margins.append(m)
            tcpr.append(margins)
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "DBE5EF" if idx == 0 else ("F6F7F9" if idx % 2 == 0 else "FFFFFF"))
            tcpr.append(shade)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 or n == 2 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(value)
            run.font.size = Pt(9)
            run.bold = idx == 0
            run.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
    sec.left_margin, sec.right_margin = Cm(2.25), Cm(2.25)
    sec.header_distance, sec.footer_distance = Cm(.8), Cm(.8)
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3", "Caption", "Header", "Footer"]:
        st = doc.styles[name]
        st.font.name = "Arial"
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.widow_control = True
    # La plantilla instalada puede incluir una línea azul en Title.
    for node in list(doc.styles.element.iter(qn("w:pBdr"))):
        node.getparent().remove(node)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(7)
    doc.styles["Title"].font.size = Pt(24)
    doc.styles["Title"].font.bold = True
    doc.styles["Title"].paragraph_format.space_after = Pt(12)
    for name, size in [("Heading 1", 16), ("Heading 2", 12)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].font.bold = True
        doc.styles[name].paragraph_format.space_before = Pt(10)
        doc.styles[name].paragraph_format.space_after = Pt(8)
        doc.styles[name].paragraph_format.keep_with_next = True
    doc.styles["Caption"].font.size = Pt(9)
    doc.styles["Caption"].font.italic = False
    doc.styles["Caption"].paragraph_format.space_after = Pt(8)
    sec.header.paragraphs[0].text = "BreastDCEDL  ·  Memoria del proyecto"
    sec.header.paragraphs[0].style = "Header"
    sec.header.paragraphs[0].runs[0].font.size = Pt(8)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Resultados finales v8 · Octubre 2026     |     ").font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.author = "Juan Hevia"
    doc.core_properties.title = "Predicción de respuesta patológica completa mediante resonancia magnética de mama"
    doc.core_properties.subject = "CNN propias y evaluación por paciente en BreastDCEDL"
    lines = (HERE / "memoria.md").read_text().splitlines()
    i, in_code, next_page = 0, False, False
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- page -->":
            next_page = True
            i += 1
            continue
        elif line.startswith("```"):
            in_code = not in_code
        elif in_code:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(line)
            r.font.name, r.font.size = "Courier New", Pt(8)
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            table(doc, lines[i:j])
            i = j
            continue
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            p = doc.add_paragraph()
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_after = Pt(4)
            p.add_run().add_picture(str(HERE / m.group(2)), width=Cm(13.0 if any(n in m.group(2) for n in ("roc_test", "matriz_confusion_test", "calibracion_test")) else 16.5))
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("# "):
            p = doc.add_paragraph(line[2:], "Title")
        elif line.startswith("## "):
            p = doc.add_paragraph(line[3:], "Heading 1")
        elif line.startswith("### "):
            p = doc.add_paragraph(line[4:], "Heading 2")
        else:
            p = doc.add_paragraph(style="Caption" if re.match(r"(?:Tabla|Figura) \d+\.", line) else "Normal")
            runs(p, line)
            if re.match(r"\d+\. ", line):
                p.paragraph_format.space_after = Pt(2)
        if next_page and not line.startswith("```"):
            p.paragraph_format.page_break_before = True
            next_page = False
        i += 1
    for node in list(doc.element.iter(qn("w:pBdr"))):
        node.getparent().remove(node)
    target = HERE / "Memoria_BreastDCEDL_Juan_Hevia.docx"
    doc.save(target)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo-documento", action="store_true", help="Usar las figuras PNG existentes")
    args = parser.parse_args()
    if not args.solo_documento:
        figures()
    build()
