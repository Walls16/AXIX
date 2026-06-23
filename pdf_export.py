"""
AXIX Ω∞ — PDF Report Generator v2
Supports portrait (technical) and landscape (institutional) formats.
"""
from reportlab.lib.pagesizes import A4, landscape as RL_LANDSCAPE
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak, KeepTogether,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as scipy_stats

# ─── PALETTE ─────────────────────────────────────────────────────────────────
C_BG    = colors.HexColor("#0a0e14")
C_SURF  = colors.HexColor("#0d1117")
C_SURF2 = colors.HexColor("#10161e")
C_BORD  = colors.HexColor("#21262d")
C_GREEN = colors.HexColor("#00cc66")
C_GREENB= colors.HexColor("#00ff88")
C_TEXT  = colors.HexColor("#c9d1d9")
C_TEXTB = colors.HexColor("#e6edf3")
C_TEXTD = colors.HexColor("#8b949e")
C_RED   = colors.HexColor("#f85149")
C_ORG   = colors.HexColor("#f0883e")
C_BLUE  = colors.HexColor("#58a6ff")
C_PURP  = colors.HexColor("#bc8cff")
C_MUTED = colors.HexColor("#30363d")

# ─── STYLES ──────────────────────────────────────────────────────────────────
def S(name, **kw):
    defaults = dict(fontName="Helvetica", fontSize=8, textColor=C_TEXT,
                    spaceAfter=2, spaceBefore=0, leading=12)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

def make_styles():
    return {
        "badge":    S("badge", fontName="Helvetica-Bold", fontSize=7,
                      textColor=C_GREENB, spaceAfter=0),
        "h1":       S("h1", fontName="Helvetica-Bold", fontSize=20,
                      textColor=C_TEXTB, leading=24, spaceAfter=2),
        "h2":       S("h2", fontName="Helvetica-Bold", fontSize=13,
                      textColor=C_TEXTB, leading=16, spaceAfter=2),
        "sec":      S("sec", fontName="Helvetica-Bold", fontSize=6.5,
                      textColor=C_GREEN, spaceAfter=4, spaceBefore=8,
                      leading=9),
        "sub":      S("sub", fontName="Helvetica", fontSize=7.5,
                      textColor=C_TEXTD, leading=11),
        "body":     S("body", fontName="Helvetica", fontSize=8,
                      textColor=C_TEXT, leading=13),
        "mono":     S("mono", fontName="Courier", fontSize=7.5,
                      textColor=C_TEXT, leading=12),
        "mono_b":   S("mono_b", fontName="Courier-Bold", fontSize=8,
                      textColor=C_TEXTB, leading=12),
        "caption":  S("caption", fontName="Helvetica", fontSize=7,
                      textColor=C_TEXTD, leading=10, alignment=TA_CENTER),
        "disc":     S("disc", fontName="Helvetica", fontSize=6,
                      textColor=C_MUTED, leading=9, alignment=TA_CENTER),
        "interp":   S("interp", fontName="Courier", fontSize=7,
                      textColor=C_TEXT, leading=11),
    }

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def img(raw, w, h):
    return RLImage(BytesIO(raw), width=w, height=h)

def hr():
    return HRFlowable(width="100%", thickness=0.3, color=C_BORD,
                      spaceAfter=5, spaceBefore=5)

def _base_table_style():
    return [
        ("FONTNAME",     (0,0),(-1,-1), "Courier"),
        ("FONTSIZE",     (0,0),(-1,-1), 7.5),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("GRID",         (0,0),(-1,-1), 0.3, C_BORD),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[C_SURF, C_SURF2]),
    ]

def metrics_table(m, s, col_w):
    last = m["last_price"]
    axix = m["axix"]
    is_up = m["period_return"] >= 0
    rows = [
        [Paragraph("PRECIO ACTUAL",    s["sub"]),
         Paragraph("CAMBIO PERÍODO",   s["sub"]),
         Paragraph("VOL ANUALIZADA",   s["sub"]),
         Paragraph("EXP. DE HURST",    s["sub"]),
         Paragraph("AXIX SCORE",       s["sub"]),
         Paragraph("SEÑAL",            s["sub"])],
        [Paragraph(f'<b>${last:,.2f}</b>',                       s["mono_b"]),
         Paragraph(f'<font color="{"#00cc66" if is_up else "#f85149"}"><b>{m["period_return"]:+.2f}%</b></font>', s["mono_b"]),
         Paragraph(f'<b>{m["vol_annual"]*100:.1f}%</b>',         s["mono_b"]),
         Paragraph(f'<b>{m["hurst"]:.3f}</b>',                   s["mono_b"]),
         Paragraph(f'<font color="{axix["color"]}"><b>{axix["score"]:.0f}/100</b></font>', s["mono_b"]),
         Paragraph(f'<font color="{axix["color"]}"><b>{axix["signal"]}</b></font>', s["mono_b"])],
    ]
    t = Table(rows, colWidths=[col_w/6]*6)
    t.setStyle(TableStyle(_base_table_style() + [
        ("BACKGROUND",(0,0),(-1,-1), C_SURF),
    ]))
    return t

def levels_table(m, s, col_w):
    lv   = m["levels"]
    last = m["last_price"]
    data = [
        ("Resistencia fuerte", lv["strong_resistance"], "#f85149"),
        ("Resistencia",        lv["resistance"],        "#f0883e"),
        ("Zona equilibrio",    lv["equilibrium"],       "#58a6ff"),
        ("Soporte",            lv["support"],           "#3fb950"),
        ("Soporte fuerte",     lv["strong_support"],    "#00ff88"),
    ]
    rows = [[
        Paragraph(name, s["mono"]),
        Paragraph(f'<font color="{c}"><b>{(p/last-1)*100:+.2f}%</b></font>', s["mono_b"]),
        Paragraph(f'<b>${p:,.2f}</b>', s["mono_b"]),
    ] for name, p, c in data]
    t = Table(rows, colWidths=[col_w*0.45, col_w*0.25, col_w*0.30])
    t.setStyle(TableStyle(_base_table_style()))
    return t

def scenarios_table(m, s, col_w):
    rows = [
        [Paragraph("Expansión alcista  (+10%)",  s["mono"]),
         Paragraph(f'<font color="#00cc66"><b>{m["p_bull"]:.1f}%</b></font>', s["mono_b"])],
        [Paragraph("Consolidación lateral",       s["mono"]),
         Paragraph(f'<font color="#58a6ff"><b>{m["p_lateral"]:.1f}%</b></font>', s["mono_b"])],
        [Paragraph("Corrección bajista  (−5%)", s["mono"]),
         Paragraph(f'<font color="#f85149"><b>{m["p_bear"]:.1f}%</b></font>', s["mono_b"])],
    ]
    t = Table(rows, colWidths=[col_w*0.65, col_w*0.35])
    t.setStyle(TableStyle(_base_table_style()))
    return t

def mc_proj_table(m, s, col_w):
    last = m["last_price"]
    rows = [
        [Paragraph("P10 — pesimista",  s["mono"]),
         Paragraph(f'<font color="#f85149"><b>${m["proj_p10"][-1]:,.2f}</b></font>', s["mono_b"]),
         Paragraph(f'{(m["proj_p10"][-1]/last-1)*100:+.1f}%', s["mono"])],
        [Paragraph("P50 — mediana",    s["mono"]),
         Paragraph(f'<b>${m["proj_p50"][-1]:,.2f}</b>', s["mono_b"]),
         Paragraph(f'{(m["proj_p50"][-1]/last-1)*100:+.1f}%', s["mono"])],
        [Paragraph("P90 — optimista",  s["mono"]),
         Paragraph(f'<font color="#00ff88"><b>${m["proj_p90"][-1]:,.2f}</b></font>', s["mono_b"]),
         Paragraph(f'{(m["proj_p90"][-1]/last-1)*100:+.1f}%', s["mono"])],
    ]
    t = Table(rows, colWidths=[col_w*0.40, col_w*0.35, col_w*0.25])
    t.setStyle(TableStyle(_base_table_style()))
    return t

def stats_table(m, s, col_w):
    kurt = float(scipy_stats.kurtosis(m["log_returns"]))
    skew = float(scipy_stats.skew(m["log_returns"]))
    data = [
        ("Máximo histórico", f"${m['max_price']:,.2f}"),
        ("Mínimo histórico", f"${m['min_price']:,.2f}"),
        ("Media del período", f"${m['mean_price']:,.2f}"),
        ("Desv. estándar",   f"${m['std_price']:,.2f}"),
        ("Curtosis",         f"{kurt:.3f}"),
        ("Asimetría",        f"{skew:.3f}"),
        ("Energía topofísica", f"{m['energy']:.2f}%"),
        ("Coherencia geom.", f"{m['coherence']:.1f}%"),
    ]
    rows = [[Paragraph(k, s["mono"]), Paragraph(f"<b>{v}</b>", s["mono_b"])]
            for k, v in data]
    t = Table(rows, colWidths=[col_w*0.55, col_w*0.45])
    t.setStyle(TableStyle(_base_table_style()))
    return t

def yoy_pdf_table(yoy_df, s, col_w):
    if yoy_df.empty:
        return Paragraph("Sin datos anuales suficientes.", s["sub"])
    header = ["Año","Apertura","Cierre","Máximo","Mínimo","Retorno %","Vol %"]
    rows = [[Paragraph(h, s["mono_b"]) for h in header]]
    for yr, row in yoy_df.iterrows():
        ret   = row["Retorno %"]
        color = "#00cc66" if ret >= 0 else "#f85149"
        rows.append([
            Paragraph(str(int(yr)), s["mono"]),
            Paragraph(f"${row['Apertura']:,.2f}", s["mono"]),
            Paragraph(f"${row['Cierre']:,.2f}",   s["mono"]),
            Paragraph(f"${row['Máximo']:,.2f}",   s["mono"]),
            Paragraph(f"${row['Mínimo']:,.2f}",   s["mono"]),
            Paragraph(f'<font color="{color}"><b>{ret:+.2f}%</b></font>', s["mono_b"]),
            Paragraph(f"{row['Vol anual %']:.2f}%", s["mono"]),
        ])
    cw = col_w / 7
    t  = Table(rows, colWidths=[cw]*7)
    t.setStyle(TableStyle(_base_table_style() + [
        ("BACKGROUND",  (0,0),(-1,0),  C_SURF),
        ("FONTNAME",    (0,0),(-1,0),  "Courier-Bold"),
        ("TEXTCOLOR",   (0,0),(-1,0),  C_GREEN),
    ]))
    return t

def backtest_table(bt, s, col_w):
    if not bt:
        return Paragraph("Datos insuficientes para backtesting.", s["sub"])
    data = [
        ("Ventanas analizadas",   str(bt["windows_tested"])),
        ("Resistencia tocada %",  f"{bt['resistance_hit_pct']:.1f}%"),
        ("Soporte mantenido %",   f"{bt['support_held_pct']:.1f}%"),
        ("Períodos de look-fwd",  str(bt["lookforward_bars"])),
    ]
    rows = [[Paragraph(k, s["mono"]), Paragraph(f"<b>{v}</b>", s["mono_b"])]
            for k, v in data]
    t = Table(rows, colWidths=[col_w*0.60, col_w*0.40])
    t.setStyle(TableStyle(_base_table_style()))
    return t

def axix_components_table(axix, s, col_w):
    rows = [[Paragraph(k, s["mono"]),
             Paragraph(f'<font color="{axix["color"]}"><b>{v:.1f}</b></font>', s["mono_b"])]
            for k, v in axix["components"].items()]
    t = Table(rows, colWidths=[col_w*0.60, col_w*0.40])
    t.setStyle(TableStyle(_base_table_style()))
    return t

def interp_block(m, geo, s):
    lv = m["levels"]
    lines = [
        "1. ESTADO GEOMÉTRICO DEL ESPACIO DE PRECIO",
        f"   Señal: {m['axix']['signal']}  |  Score: {m['axix']['score']:.0f}/100  |  Hurst: {m['hurst']:.3f}",
        f"   {'Tendencia persistente' if m['hurst']>0.55 else 'Reversión a la media' if m['hurst']<0.45 else 'Caminata aleatoria'}",
        "",
        "2. NIVELES TOPOFÍSICOS CLAVE",
        f"   Res. fuerte: ${lv['strong_resistance']:,.2f}  |  Res.: ${lv['resistance']:,.2f}",
        f"   Equilibrio:  ${lv['equilibrium']:,.2f}",
        f"   Soporte:     ${lv['support']:,.2f}  |  Sop. fuerte: ${lv['strong_support']:,.2f}",
        "",
        "3. PROYECCIÓN 12 MESES (Monte Carlo · 2,000 trayectorias)",
        f"   P10: ${m['proj_p10'][-1]:,.2f}  |  P50: ${m['proj_p50'][-1]:,.2f}  |  P90: ${m['proj_p90'][-1]:,.2f}",
        "",
        "4. PROBABILIDADES DE ESCENARIO (90 días)",
        f"   Alcista: {m['p_bull']:.1f}%  |  Lateral: {m['p_lateral']:.1f}%  |  Bajista: {m['p_bear']:.1f}%",
    ]
    if geo:
        lines += ["", f"5. GEOMETRÍA GUE  |  Calidad ajuste Riemann: {geo['gue_fit_quality']}  |  KS: {geo['wd_ks_stat']:.4f}"]
    lines += ["", "NOTA: Modelo experimental. No constituye asesoría financiera. AXIX Ω∞"]
    return [Paragraph(ln if ln else " ", s["interp"]) for ln in lines]

# ─── PAGE CALLBACKS ───────────────────────────────────────────────────────────
def _on_page(canvas, doc, page_w, page_h, ticker, ts):
    canvas.saveState()
    canvas.setFillColor(C_BG)
    canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
    canvas.setStrokeColor(C_BORD)
    canvas.setLineWidth(0.3)
    canvas.line(12*mm, 12*mm, page_w-12*mm, 12*mm)
    canvas.setFont("Helvetica", 6)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(12*mm, 8*mm, f"AXIX Ω∞  ·  {ticker}  ·  Modelo experimental  ·  No asesoría financiera")
    canvas.drawRightString(page_w-12*mm, 8*mm, f"Pág. {doc.page}  ·  {ts}")
    canvas.restoreState()

# ─── MAIN EXPORT FUNCTION ─────────────────────────────────────────────────────
def generate_pdf_report(
    ticker, period, interval, metrics, yoy_df, backtest, geo,
    imgs: dict, landscape: bool = False, pro_mode: bool = True,
) -> bytes:

    buf      = BytesIO()
    page_sz  = RL_LANDSCAPE(A4) if landscape else A4
    page_w, page_h = page_sz
    ts       = datetime.now().strftime("%d/%m/%Y %H:%M")

    lm = rm = 12*mm
    tm = bm = 18*mm
    uw = page_w - lm - rm   # usable width

    doc = SimpleDocTemplate(buf, pagesize=page_sz,
                            leftMargin=lm, rightMargin=rm,
                            topMargin=tm, bottomMargin=bm,
                            title=f"AXIX Topofísico · {ticker}",
                            author="AXIX Ω∞")

    s    = make_styles()
    m    = metrics
    axix = m["axix"]
    story = []

    on_page = lambda c, d: _on_page(c, d, page_w, page_h, ticker, ts)

    # ── COVER HEADER ─────────────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("AXIX Ω∞", s["badge"]),
        [Paragraph(f"<b>{ticker}</b>  —  Análisis Topofísico Financiero", s["h1"]),
         Paragraph(f"Período: {period}  ·  Intervalo: {interval}  ·  {ts}  ·  {'Pro' if pro_mode else 'Retail'}", s["sub"])],
    ]], colWidths=[22*mm, uw-22*mm])
    hdr.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story += [hdr, HRFlowable(width="100%", thickness=0.4, color=C_GREEN,
                               spaceAfter=8, spaceBefore=4)]

    # ── METRICS ROW ──────────────────────────────────────────────────────────
    story += [Paragraph("MÉTRICAS CLAVE", s["sec"]), metrics_table(m, s, uw), Spacer(1,6)]

    # Min/max inline
    story.append(Paragraph(
        f"Máx. histórico: <b>${m['max_price']:,.2f}</b>  ·  "
        f"Mín. histórico: <b>${m['min_price']:,.2f}</b>  ·  "
        f"Media: <b>${m['mean_price']:,.2f}</b>  ·  "
        f"Desv. est.: <b>${m['std_price']:,.2f}</b>  ·  "
        f"Energía topofísica: <b>{m['energy']:.2f}%</b>",
        s["sub"]
    ))
    story.append(hr())

    # ── MAIN CHART ───────────────────────────────────────────────────────────
    story.append(Paragraph("HISTÓRICO + PROYECCIÓN TOPOFÍSICA 12 MESES", s["sec"]))
    main_h = uw * 560/1400
    story.append(img(imgs["main"], uw, main_h))
    story.append(Paragraph(
        "Precio histórico (verde) · Proyección mediana (blanco) · Banda P10–P90 · "
        "Densidad informacional Riemann (panel inferior)", s["caption"]))
    story.append(hr())

    # ── LEVELS + SCENARIOS + MC in 3 columns ─────────────────────────────────
    cw3 = uw / 3 - 3*mm
    col1 = [Paragraph("NIVELES TOPOFÍSICOS", s["sec"]),
             levels_table(m, s, cw3)]
    col2 = [Paragraph("ESCENARIOS 90 DÍAS", s["sec"]),
             scenarios_table(m, s, cw3),
             Spacer(1,6),
             Paragraph("PROYECCIÓN 12 MESES", s["sec"]),
             mc_proj_table(m, s, cw3)]
    col3 = [Paragraph("ESTADÍSTICAS", s["sec"]),
             stats_table(m, s, cw3)]

    three = Table([[col1, col2, col3]], colWidths=[cw3, cw3, cw3])
    three.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(1,-1),6),
        ("LEFTPADDING",(1,0),(2,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(three)
    story.append(hr())

    # ── AXIX SCORE + RADAR ───────────────────────────────────────────────────
    story.append(Paragraph("AXIX SIGNAL SCORE", s["sec"]))
    gauge_h = uw/2 * 260/600
    score_row = Table([[
        img(imgs["gauge"], uw/2-4*mm, gauge_h),
        [Paragraph("COMPONENTES", s["sec"]),
         axix_components_table(axix, s, uw/2-4*mm)],
    ]], colWidths=[uw/2-4*mm, uw/2-4*mm])
    score_row.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(0,-1),6),
        ("LEFTPADDING",(1,0),(1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(score_row)
    story.append(hr())

    # ── PAGE BREAK → PAGE 2 ──────────────────────────────────────────────────
    story.append(PageBreak())

    # ── MONTE CARLO + DIST + VOL ─────────────────────────────────────────────
    story.append(Paragraph("ANÁLISIS PROBABILÍSTICO", s["sec"]))
    mc_h   = uw * 300/1100
    story.append(img(imgs["mc"], uw, mc_h))
    story.append(Paragraph(
        "2,000 trayectorias Monte Carlo · Proceso log-normal calibrado · Bandas P10/P50/P90",
        s["caption"]))
    story.append(Spacer(1, 6))

    half = uw/2 - 3*mm
    dist_h = half * 280/950
    vol_h  = half * 280/950
    two = Table([[img(imgs["dist"], half, dist_h), img(imgs["vol"], half, vol_h)]],
                colWidths=[half, half])
    two.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(0,-1),4),
        ("LEFTPADDING",(1,0),(1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(two)
    story.append(Paragraph(
        "Izq: Distribución de retornos logarítmicos vs normal teórica  ·  "
        "Der: Volatilidad rodante anualizada 21d y 63d", s["caption"]))
    story.append(hr())

    # ── YOY TABLE ────────────────────────────────────────────────────────────
    story.append(Paragraph("RETORNO AÑO A AÑO — COMPARATIVA HISTÓRICA", s["sec"]))
    if imgs.get("yoy"):
        yoy_chart_h = uw * 280/1100
        story.append(img(imgs["yoy"], uw, yoy_chart_h))
    story.append(Spacer(1,4))
    story.append(yoy_pdf_table(yoy_df, s, uw))
    story.append(hr())

    # ── PRO SECTIONS ─────────────────────────────────────────────────────────
    if pro_mode:
        story.append(PageBreak())

        # GUE geometry
        if imgs.get("gue"):
            story.append(Paragraph("GEOMETRÍA TOPOFÍSICA — GUE / RIEMANN", s["sec"]))
            story.append(Paragraph(
                "Comparación entre espaciados normalizados de retornos y la distribución "
                "Wigner-Dyson del Gaussian Unitary Ensemble (GUE), análoga al espaciado "
                f"de ceros no triviales de ζ(s).  Calidad ajuste: {geo['gue_fit_quality']}  "
                f"·  KS stat: {geo['wd_ks_stat']:.4f}", s["body"]))
            gue_h = uw * 270/900
            story.append(img(imgs["gue"], uw, gue_h))
            story.append(hr())

        # Backtesting
        if imgs.get("bt") and backtest:
            story.append(Paragraph("BACKTESTING DE NIVELES TOPOFÍSICOS", s["sec"]))
            story.append(backtest_table(backtest, s, uw/2))
            bt_h = uw * 270/900
            story.append(img(imgs["bt"], uw, bt_h))
            story.append(hr())

    # ── INTERPRETATION ───────────────────────────────────────────────────────
    story.append(Paragraph("INTERPRETACIÓN TOPOFÍSICA COMPLETA", s["sec"]))
    bg_box = Table(
        [[block] for block in interp_block(m, geo if pro_mode else None, s)],
        colWidths=[uw]
    )
    bg_box.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),C_SURF),
        ("LEFTPADDING",(0,0),(-1,-1),10),
        ("RIGHTPADDING",(0,0),(-1,-1),10),
        ("TOPPADDING",(0,0),(-1,-1),2),
        ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ("GRID",(0,0),(-1,-1),0.2,C_BORD),
    ]))
    story.append(bg_box)
    story.append(Spacer(1,8))

    # ── DISCLAIMER ───────────────────────────────────────────────────────────
    story.append(hr())
    story.append(Paragraph(
        f"AXIX Ω∞ — Sistema Topofísico Financiero  ·  Modelo experimental, no asesoría financiera  ·  "
        f"Ticker: {ticker}  ·  Período: {period}  ·  Intervalo: {interval}  ·  "
        f"Generado: {ts}  ·  Datos: Yahoo Finance",
        s["disc"]
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()
