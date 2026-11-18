#!/usr/bin/env python3
"""Render the Japanese trial report with embedded, locally subsetted Noto fonts."""
import argparse
import json
import pathlib
from xml.sax.saxutils import escape

from fontTools import subset
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont as FontToolsFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether


def make_font(source, output, text, family):
    font = FontToolsFont(str(source), fontNumber=0, lazy=True)
    opts = subset.Options()
    opts.layout_features = []
    sub = subset.Subsetter(options=opts)
    sub.populate(text=text)
    sub.subset(font)
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(glyph_set)
        converter = Cu2QuPen(pen, max_err=1.0, reverse_direction=True)
        glyph_set[name].draw(converter)
        glyphs[name] = pen.glyph()
    fb = FontBuilder(font['head'].unitsPerEm, isTTF=True)
    fb.setupGlyphOrder(font.getGlyphOrder())
    fb.setupCharacterMap(font.getBestCmap())
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(font['hmtx'].metrics)
    hh = font['hhea']
    fb.setupHorizontalHeader(ascent=hh.ascent, descent=hh.descent, lineGap=hh.lineGap)
    fb.setupNameTable({'familyName': family, 'styleName': 'Regular', 'uniqueFontIdentifier': family,
                       'fullName': family, 'psName': family,
                       'copyright': font['name'].getDebugName(0) or 'Noto CJK, SIL Open Font License'})
    os2 = font['OS/2']
    fb.setupOS2(sTypoAscender=os2.sTypoAscender, sTypoDescender=os2.sTypoDescender,
                sTypoLineGap=os2.sTypoLineGap, usWinAscent=os2.usWinAscent, usWinDescent=os2.usWinDescent)
    fb.setupPost()
    fb.setupMaxp()
    fb.save(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('content', type=pathlib.Path)
    ap.add_argument('pdf', type=pathlib.Path)
    ap.add_argument('--scratch', type=pathlib.Path, default=pathlib.Path('parent_evaluation/pdf_work'))
    a = ap.parse_args()
    data = json.loads(a.content.read_text())
    a.scratch.mkdir(exist_ok=True, parents=True)
    text = json.dumps(data, ensure_ascii=False) + '0123456789 /｜独立探索・検証記録2026-11-18参考'
    for weight, name in [('Regular','NotoTrial'), ('Bold','NotoTrialBold')]:
        out = a.scratch/f'{name}.ttf'
        make_font(pathlib.Path(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc'), out, text, name)
        pdfmetrics.registerFont(TTFont(name, str(out)))
    pdfmetrics.registerFontFamily('NotoTrial', normal='NotoTrial', bold='NotoTrialBold')
    ink = colors.HexColor('#173343')
    teal = colors.HexColor('#176d76')
    muted = colors.HexColor('#526776')
    pale = colors.HexColor('#f1f6f8')
    styles = {
        'body': ParagraphStyle('Body',fontName='NotoTrial',fontSize=9.3,leading=15.3,textColor=ink,wordWrap='CJK',spaceAfter=8),
        'small': ParagraphStyle('Small',fontName='NotoTrial',fontSize=7.8,leading=12.4,textColor=muted,wordWrap='CJK',spaceAfter=6),
        'title': ParagraphStyle('Title',fontName='NotoTrialBold',fontSize=25,leading=34,textColor=ink,wordWrap='CJK',spaceAfter=16),
        'h1': ParagraphStyle('H1',fontName='NotoTrialBold',fontSize=18,leading=26,textColor=ink,wordWrap='CJK',spaceAfter=13),
        'h2': ParagraphStyle('H2',fontName='NotoTrialBold',fontSize=11.8,leading=18,textColor=teal,wordWrap='CJK',spaceBefore=8,spaceAfter=7),
        'cell': ParagraphStyle('Cell',fontName='NotoTrial',fontSize=8.2,leading=12.8,textColor=ink,wordWrap='CJK'),
        'head': ParagraphStyle('Head',fontName='NotoTrialBold',fontSize=8.2,leading=12.8,textColor=colors.white,wordWrap='CJK'),
    }
    def para(s, style='body'):
        return Paragraph(escape(s).replace('\n','<br/>'), styles[style])
    story=[]
    # SimpleDocTemplate's frame adds 6 pt of internal padding at each side.
    width=A4[0]-84-12
    for pi,page in enumerate(data['pages']):
        if pi: story.append(PageBreak())
        if page.get('kicker'):
            story.append(para(page['kicker'],'small'))
        story.append(para(page['title'],'title' if pi==0 else 'h1'))
        for block in page['blocks']:
            kind=block['type']
            if kind in ('body','small','h2'):
                story.append(para(block['text'],kind))
            elif kind=='callout':
                box=Table([[para(block['text'])]],colWidths=[width])
                box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),pale),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#d9e5eb')),
                                        ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
                story += [box,Spacer(1,11)]
            elif kind=='table':
                cells=[[para(str(c),'head' if ri==0 else 'cell') for c in row] for ri,row in enumerate([block['headers']]+block['rows'])]
                shares=block.get('widths',[1]*len(block['headers']))
                tw=[width*x/sum(shares) for x in shares]
                tab=Table(cells,colWidths=tw,repeatRows=1,hAlign='LEFT')
                tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,pale]),
                                        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                                        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#d9e5eb'))]))
                story += [tab,Spacer(1,10)]
            elif kind=='image':
                im=Image(block['path']);im.drawHeight=im.imageHeight*width/im.imageWidth;im.drawWidth=width
                story += [im,Spacer(1,6)]
                if block.get('caption'):story.append(para(block['caption'],'small'))
            else:raise ValueError(kind)
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#d9e5eb'));c.line(42,40,A4[0]-42,40)
        c.setFont('NotoTrial',7.3);c.setFillColor(muted)
        c.drawString(42,27,'2026-11-18  |  独立探索・検証記録')
        c.drawRightString(A4[0]-42,27,str(doc.page))
    doc=SimpleDocTemplate(str(a.pdf),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=53,
                          title=data.get('title','6.2 Astra / max 独立探索試行'),author='ChatGPT',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(a.pdf)


if __name__=='__main__':main()
