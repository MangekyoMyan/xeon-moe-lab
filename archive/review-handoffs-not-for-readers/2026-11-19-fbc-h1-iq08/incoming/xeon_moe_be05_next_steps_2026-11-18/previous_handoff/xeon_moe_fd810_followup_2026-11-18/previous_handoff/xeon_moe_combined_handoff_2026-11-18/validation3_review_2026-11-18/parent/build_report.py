"""Create a Japanese PDF and complete text companion from report content.

Uses local Noto CJK outlines, converted to a small TrueType subset for ReportLab.
Does not execute repository or model code.
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont as SourceFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#007F7A")
INK = colors.HexColor("#263747")
MUTED = colors.HexColor("#566777")
LIGHT = colors.HexColor("#EFF5F7")


def make_font(source, destination, characters, name):
    source = SourceFont(source, fontNumber=0)
    source_map = source.getBestCmap()
    points = set(map(ord, characters)) | set(range(32, 127))
    points.discard(10)
    points.discard(13)
    missing = sorted(cp for cp in points if cp not in source_map)
    if missing:
        raise ValueError(f"Unmapped report glyphs: {missing}")
    mapping = {cp: source_map[cp] for cp in points}
    selected = {".notdef", *mapping.values()}
    order = [g for g in source.getGlyphOrder() if g in selected]
    glyphset = source.getGlyphSet()
    glyphs = {}
    for glyphname in order:
        pen = TTGlyphPen(glyphset)
        convert = Cu2QuPen(pen, max_err=1.0, reverse_direction=True)
        glyphset[glyphname].draw(convert)
        glyphs[glyphname] = pen.glyph()
    builder = FontBuilder(source["head"].unitsPerEm, isTTF=True)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap(mapping)
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({g: source["hmtx"].metrics[g] for g in order})
    builder.setupHorizontalHeader(ascent=source["hhea"].ascent, descent=source["hhea"].descent)
    builder.setupNameTable({"familyName": name, "styleName": "Regular", "uniqueFontIdentifier": name, "fullName": name, "psName": name, "version": "Version 1.0"})
    os2 = source["OS/2"]
    builder.setupOS2(sTypoAscender=os2.sTypoAscender, sTypoDescender=os2.sTypoDescender, sTypoLineGap=os2.sTypoLineGap, usWinAscent=os2.usWinAscent, usWinDescent=os2.usWinDescent)
    builder.setupPost()
    builder.setupMaxp()
    builder.save(destination)
    source.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deliverable", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("numbers", type=Path)
    args = ap.parse_args()
    out, work = args.deliverable.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    obj = json.loads((out / "REPORT_CONTENT.json").read_text())
    nums = json.loads(args.numbers.read_text())
    all_text = json.dumps(obj, ensure_ascii=False) + "xeon-moe-lab 調査報告 2026-11-18 JST / 0123456789"
    for style in ("Regular", "Bold"):
        target = work / f"NotoJPReview-{style}.ttf"
        make_font(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{style}.ttc", target, all_text, f"NotoJPReview{style}")
        pdfmetrics.registerFont(TTFont(f"JP{style}", str(target)))
    pdfmetrics.registerFontFamily("JPRegular", normal="JPRegular", bold="JPBold", italic="JPRegular", boldItalic="JPBold")
    chart_dir = out / "charts"
    chart_dir.mkdir(exist_ok=True)
    values = [r["gain_pct"] for r in nums["timing"]]
    labels = ["JA schedule", "EN schedule", "Python: initial", "Python: extra repeat"]
    fig, ax = plt.subplots(figsize=(7.6, 2.6), layout="constrained")
    bars = ax.barh(range(4), values, color=["#007F7A", "#007F7A", "#007F7A", "#A4B6C5"], height=.56)
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor("#62798D")
    ax.set_yticks(range(4), labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 6.1)
    ax.set_xticks(range(7), [f"{n}%" for n in range(7)])
    ax.set_xlabel("P speed increase over F, fixed-input decoding", fontsize=10)
    ax.xaxis.grid(True, color="#dce4e9", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(value+.08, bar.get_y()+bar.get_height()/2, f"+{value:.2f}%", va="center", fontsize=11, color="#17324D")
    fig.savefig(chart_dir / "performance_gain.png", dpi=220, facecolor="white")
    fig.savefig(chart_dir / "performance_gain.svg", facecolor="white")
    plt.close(fig)

    styles = {
        "kicker": ParagraphStyle("kicker", fontName="JPBold", fontSize=8.7, leading=13, textColor=TEAL, spaceAfter=5, wordWrap="CJK"),
        "title": ParagraphStyle("title", fontName="JPBold", fontSize=22, leading=30, textColor=NAVY, spaceAfter=12, wordWrap="CJK"),
        "lead": ParagraphStyle("lead", fontName="JPBold", fontSize=11.5, leading=19, textColor=NAVY, spaceAfter=15, wordWrap="CJK"),
        "p": ParagraphStyle("p", fontName="JPRegular", fontSize=10.1, leading=16.5, textColor=INK, spaceAfter=10, wordWrap="CJK"),
        "h": ParagraphStyle("h", fontName="JPBold", fontSize=12.0, leading=18, textColor=TEAL, spaceBefore=5, spaceAfter=7, wordWrap="CJK"),
        "note": ParagraphStyle("note", fontName="JPRegular", fontSize=9.0, leading=14.5, textColor=MUTED, wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontName="JPRegular", fontSize=9.4, leading=14.5, textColor=INK, wordWrap="CJK"),
        "th": ParagraphStyle("th", fontName="JPBold", fontSize=9.2, leading=14, textColor=colors.white, wordWrap="CJK"),
        "source": ParagraphStyle("source", fontName="JPRegular", fontSize=8.5, leading=12.5, textColor=MUTED, spaceAfter=5, wordWrap="CJK"),
    }
    doc = SimpleDocTemplate(str(out / "USER_REPORT_JA.pdf"), pagesize=A4, rightMargin=47, leftMargin=47, topMargin=53, bottomMargin=53, title=obj["title"], author="Review prepared for MangekyoMyan", subject="validation3 review and GPT-6.2 Astra subagent trial")
    width = A4[0] - 94
    page_sources = [p["sources"] for p in obj["pages"]]
    def decorate(canvas, document):
        page = document.page
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.setFont("JPBold", 8.5)
        canvas.drawString(47, A4[1]-27, "xeon-moe-lab  /  validation3")
        canvas.setFont("JPRegular", 8)
        canvas.drawRightString(A4[0]-47, A4[1]-27, obj["date"])
        canvas.setStrokeColor(colors.HexColor("#D7E2E8"))
        canvas.line(47, A4[1]-36, A4[0]-47, A4[1]-36)
        canvas.line(47, 40, A4[0]-47, 40)
        canvas.setFont("JPRegular", 7.4)
        canvas.setFillColor(MUTED)
        if page <= len(page_sources):
            footer = Paragraph(escape(page_sources[page-1]), ParagraphStyle("footer", fontName="JPRegular", fontSize=7.3, leading=10, textColor=MUTED, wordWrap="CJK"))
            fw, fh = footer.wrap(width-40, 28)
            footer.drawOn(canvas, 47, 33-fh)
        canvas.setFont("JPRegular", 8)
        canvas.drawRightString(A4[0]-47, 22, f"{page} / {len(obj['pages'])}")
        canvas.restoreState()

    story = []
    text_lines = [obj["title"], obj["date"], "対象commit: " + obj["commit"], ""]
    for index, page in enumerate(obj["pages"], 1):
        if index > 1:
            story.append(PageBreak())
        story.append(Paragraph(escape(page["kicker"]), styles["kicker"]))
        story.append(Paragraph(escape(page["title"]), styles["title"]))
        story.append(Paragraph(escape(page["lead"]), styles["lead"]))
        text_lines += [f"{index}. {page['title']}", page["lead"], ""]
        for block in page["blocks"]:
            kind = block["type"]
            if kind in ("p", "h"):
                story.append(Paragraph(escape(block["text"]), styles[kind]))
                text_lines += [block["text"], ""]
            elif kind == "note":
                table = Table([[Paragraph(escape(block["text"]), styles["note"])]], colWidths=[width])
                table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT), ("BOX",(0,0),(-1,-1),.4,colors.HexColor("#D9E7EA")), ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9)]))
                story += [table, Spacer(1, 10)]
                text_lines += ["注: " + block["text"], ""]
            elif kind == "table":
                rows = [[Paragraph(escape(str(v)), styles["th"]) for v in block["headers"]]]
                rows += [[Paragraph(escape(str(v)), styles["cell"]) for v in row] for row in block["rows"]]
                table = Table(rows, colWidths=[width*v for v in block["widths"]], repeatRows=1, hAlign="LEFT")
                ts = [("BACKGROUND",(0,0),(-1,0),NAVY),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("LINEBELOW",(0,0),(-1,0),.8,NAVY)]
                for ri in range(1,len(rows)):
                    ts.append(("BACKGROUND",(0,ri),(-1,ri),LIGHT if ri%2 else colors.white))
                    ts.append(("LINEBELOW",(0,ri),(-1,ri),.35,colors.HexColor("#D7E2E8")))
                table.setStyle(TableStyle(ts))
                story += [table, Spacer(1,12)]
                text_lines += [" | ".join(block["headers"])] + [" | ".join(row) for row in block["rows"]] + [""]
            elif kind == "chart":
                story += [Image(str(chart_dir / "performance_gain.png"), width=width, height=width*2.6/7.6), Spacer(1,8)]
                text_lines += ["図: Pの速度増加率（F比）", "日本語 +5.0703%、英語 +1.7987%、Python初回 +0.8016%、Python探索的追試 +0.8077%。", ""]
            elif kind == "source":
                key = block["key"]
                source = obj["sources"][key]
                title = f"[{key}] {source['title']}"
                if "url" in source:
                    line = f'<link href="{escape(source["url"], {chr(34): "&quot;"})}" color="#007F7A">{escape(title)}</link>'
                else:
                    line = escape(title)
                if "file" in source:
                    line += "<br/>" + escape("同梱: " + source["file"])
                story.append(Paragraph(line, styles["source"]))
                text_lines += [title, source.get("url", ""), "同梱: " + source["file"] if "file" in source else "", ""]
        text_lines += ["出典: " + page["sources"], ""]
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    (out / "USER_REPORT_JA.txt").write_text("\n".join(text_lines), encoding="utf-8")
    print(json.dumps({"pdf": str(out / "USER_REPORT_JA.pdf"), "text": str(out / "USER_REPORT_JA.txt"), "expected_pages": len(obj["pages"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
