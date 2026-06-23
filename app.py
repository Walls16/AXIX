import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from engine import compute_full_metrics, yoy_table, backtest_levels, topophysical_geometry
from charts import (
    chart_main, chart_yoy, chart_rolling_vol, chart_distribution,
    chart_montecarlo, chart_axix_gauge, chart_axix_radar,
    chart_gue_spacings, chart_acf, chart_multi_overlay, chart_backtest,
    chart_comparison_overlay, chart_comparison_metrics, chart_comparison_coverage,
)
from comparison import run_all_comparisons, backtest_comparison
from pdf_export import generate_pdf_report

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AXIX · Sistema Topofísico Financiero",
    page_icon="∞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.stApp{background:#0a0e14;color:#c9d1d9;}
.main .block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1440px;}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
[data-testid="stSidebar"] h3{color:#00ff88;font-family:monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;}
[data-testid="stMetric"]{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 14px;}
[data-testid="stMetricLabel"] p{font-size:.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:.08em;}
[data-testid="stMetricValue"]{font-family:monospace!important;font-size:1.2rem!important;color:#e6edf3!important;}
[data-testid="stMetricDelta"] p{font-family:monospace!important;font-size:.72rem!important;}
.axix-header{display:flex;align-items:center;gap:14px;padding:10px 0 8px;border-bottom:1px solid #21262d;margin-bottom:4px;}
.axix-badge{background:#0d1117;border:1px solid #00ff88;color:#00ff88;font-family:monospace;font-size:.78rem;font-weight:500;padding:4px 10px;border-radius:4px;letter-spacing:.1em;}
.axix-title{font-size:.95rem;font-weight:500;color:#e6edf3;}
.axix-sub{font-size:.68rem;color:#8b949e;margin-top:1px;}
.sec{font-size:.62rem;letter-spacing:.14em;text-transform:uppercase;color:#00ff88;font-family:monospace;margin-bottom:6px;padding-bottom:3px;border-bottom:1px solid #21262d;}
.score-card{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:12px 16px;text-align:center;}
.tag-sr{color:#f85149;background:#3d0f0f;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:.64rem;}
.tag-r{color:#f0883e;background:#3d1f07;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:.64rem;}
.tag-eq{color:#58a6ff;background:#0d2044;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:.64rem;}
.tag-s{color:#3fb950;background:#0d2b16;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:.64rem;}
.tag-ss{color:#00ff88;background:#002e17;padding:1px 6px;border-radius:3px;font-family:monospace;font-size:.64rem;}
.stButton>button{background:#00cc66!important;color:#0a0e14!important;border:none!important;font-family:monospace!important;font-weight:500!important;border-radius:6px!important;}
.stButton>button:hover{background:#00ff88!important;box-shadow:0 0 16px rgba(0,255,136,.2)!important;}
.stTabs [data-baseweb="tab-list"]{background:#0d1117;border-bottom:1px solid #21262d;gap:0;}
.stTabs [data-baseweb="tab"]{font-family:monospace;font-size:.7rem;color:#8b949e;padding:7px 14px;background:transparent;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:#00ff88!important;border-bottom:2px solid #00ff88!important;background:transparent!important;}
.stSelectbox>div>div,.stTextInput>div>div>input{background:#161b22!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:6px!important;}
.stDataFrame{background:#0d1117;}
.stDownloadButton>button{background:#161b22!important;color:#00ff88!important;border:1px solid #00ff88!important;font-family:monospace!important;font-size:.72rem!important;}
hr{border-color:#21262d!important;}
.mode-badge-retail{display:inline-block;background:#0d2044;color:#58a6ff;border:1px solid #1f4080;border-radius:4px;padding:2px 8px;font-size:.65rem;font-family:monospace;letter-spacing:.06em;}
.mode-badge-pro{display:inline-block;background:#002e17;color:#00ff88;border:1px solid #006633;border-radius:4px;padding:2px 8px;font-size:.65rem;font-family:monospace;letter-spacing:.06em;}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [("results", None), ("multi_dfs", {}), ("pdf_ready", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="axix-badge">AXIX</span>', unsafe_allow_html=True)
    st.markdown("### Activo principal")
    ticker = st.text_input(
        "Ticker", value="SPY", label_visibility="collapsed",
        help="SPY · AAPL · BTC-USD · NVDA · GLD · TSLA · QQQ · EURUSD=X · ^GSPC · AMXL.MX"
    ).strip().upper()
    st.caption("Escribe cualquier ticker de Yahoo Finance y presiona Analizar.")

    st.divider()
    st.markdown("### Período & Intervalo")
    period_opts = {
        "3 meses":("3mo","1d"), "6 meses":("6mo","1d"),
        "1 año":("1y","1d"),    "2 años":("2y","1wk"),
        "5 años":("5y","1wk"),  "10 años":("10y","1mo"),
        "Máx. histórico":("max","1mo"),
    }
    period_lbl = st.selectbox("Período", list(period_opts.keys()), index=2, label_visibility="collapsed")
    period, def_iv = period_opts[period_lbl]
    iv_opts = {"Diario":"1d","Semanal":"1wk","Mensual":"1mo"}
    iv_lbl  = st.selectbox("Intervalo", list(iv_opts.keys()),
                            index=list(iv_opts.values()).index(def_iv), label_visibility="collapsed")
    interval = iv_opts[iv_lbl]

    st.divider()
    st.markdown("### Comparativa multi-activo")
    multi_raw = st.text_input("Tickers adicionales (separados por coma)",
                              placeholder="AAPL, QQQ", label_visibility="collapsed")

    st.divider()
    st.markdown("### Modo de visualización")
    ui_mode = st.radio("Modo", ["Retail", "Pro"], horizontal=True, label_visibility="collapsed")

    st.divider()
    st.markdown("### Exportar PDF")
    pdf_fmt = st.radio("Formato PDF", ["Técnico (portrait)", "Institucional (landscape)"],
                       label_visibility="collapsed")

    st.divider()
    analyze_btn = st.button("→ ANALIZAR", type="primary", use_container_width=True)

    st.markdown("""
<div style='font-size:.62rem;color:#30363d;line-height:1.7;margin-top:8px;'>
AXIX · Marco experimental<br>No es asesoría financiera.<br>
Datos: Yahoo Finance
</div>""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
mode_badge = f'<span class="mode-badge-{"pro" if ui_mode=="Pro" else "retail"}">{"PRO" if ui_mode=="Pro" else "RETAIL"}</span>'
st.markdown(f"""
<div class="axix-header">
  <span class="axix-badge">AXIX</span>
  <div>
    <div class="axix-title">Sistema Topofísico Financiero &nbsp;{mode_badge}</div>
    <div class="axix-sub">Análisis geométrico · Proyección probabilística · Cualquier activo</div>
  </div>
</div>""", unsafe_allow_html=True)

# ─── FETCH & COMPUTE ─────────────────────────────────────────────────────────
def fetch_df(tkr, per, iv):
    df = yf.download(tkr, period=per, interval=iv, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()

if analyze_btn:
    with st.spinner(f"Descargando {ticker}…"):
        try:
            df = fetch_df(ticker, period, interval)
            if df.empty:
                st.error(f"No se encontraron datos para **{ticker}**.")
                st.stop()
            if len(df) < 30:
                st.warning("Pocos datos. Considera un período mayor.")

            with st.spinner("Computando geometría topofísica…"):
                m   = compute_full_metrics(df)
                yoy = yoy_table(df)
                bt  = backtest_levels(df)
                geo = topophysical_geometry(m["log_returns"])
                with st.spinner("Calculando modelos de comparación…"):
                    cmp_models = run_all_comparisons(m["prices"])
                    cmp_bt     = backtest_comparison(m["prices"], None)

            # Multi-asset
            multi_dfs = {ticker: df}
            if multi_raw.strip():
                for t2 in [x.strip().upper() for x in multi_raw.split(",") if x.strip()]:
                    try:
                        d2 = fetch_df(t2, period, interval)
                        if not d2.empty:
                            multi_dfs[t2] = d2
                    except:
                        pass

            st.session_state.results = dict(
                df=df, m=m, yoy=yoy, bt=bt, geo=geo,
                ticker=ticker, period=period_lbl, interval=iv_lbl,
                multi_dfs=multi_dfs,
                cmp_models=cmp_models, cmp_bt=cmp_bt,
            )
            st.session_state.pdf_ready = False
            st.success(f"✓ {ticker} · {len(df):,} observaciones · {period_lbl} · {iv_lbl}")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

# ─── DISPLAY ─────────────────────────────────────────────────────────────────
r = st.session_state.results
if not r:
    st.markdown("""
<div style="text-align:center;padding:70px 20px;color:#30363d;">
  <div style="font-size:3rem;margin-bottom:10px;">∞</div>
  <div style="font-size:.88rem;color:#8b949e;">Ingresa un ticker y presiona → ANALIZAR</div>
  <div style="font-size:.7rem;color:#30363d;margin-top:6px;">
    SPY · AAPL · BTC-USD · NVDA · GLD · TSLA · EURUSD=X · ^GSPC · GC=F · AMXL.MX
  </div>
</div>""", unsafe_allow_html=True)
    st.stop()

df, m, yoy, bt, geo = r["df"], r["m"], r["yoy"], r["bt"], r["geo"]
cmp_models = r.get("cmp_models", {})
cmp_bt     = r.get("cmp_bt", None)
t, multi_dfs = r["ticker"], r["multi_dfs"]

# ── TOP METRICS ──────────────────────────────────────────────────────────────
axix = m["axix"]
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Precio",         f"${m['last_price']:,.2f}")
c2.metric("Cambio período", f"{m['period_return']:+.2f}%", delta=f"{m['period_return']:+.2f}%")
c3.metric("Vol anual",      f"{m['vol_annual']*100:.1f}%")
c4.metric("Hurst",          f"{m['hurst']:.3f}",
          help="<0.5 reversión · =0.5 random · >0.5 tendencia")
c5.metric("AXIX Score",     f"{axix['score']:.0f}/100")
c6.metric("Señal",          axix["signal"])

# Máx / Mín históricos
st.markdown(f"""
<div style="display:flex;gap:12px;margin:8px 0 4px;flex-wrap:wrap;">
  <span style="font-size:.72rem;font-family:monospace;color:#8b949e;">
     MÁX HISTÓRICO <b style="color:#00ff88;">${m['max_price']:,.2f}</b>
  </span>
  <span style="font-size:.72rem;font-family:monospace;color:#8b949e;">
     MÍN HISTÓRICO <b style="color:#f85149;">${m['min_price']:,.2f}</b>
  </span>
  <span style="font-size:.72rem;font-family:monospace;color:#8b949e;">
    ∅ MEDIA <b style="color:#58a6ff;">${m['mean_price']:,.2f}</b>
  </span>
  <span style="font-size:.72rem;font-family:monospace;color:#8b949e;">
    σ DESV. EST. <b style="color:#f0883e;">${m['std_price']:,.2f}</b>
  </span>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── TABS ──────────────────────────────────────────────────────────────────────
tabs_retail = ["Precio & Proyección", "Estadísticas", "Multi-activo", "Comparativa", "Reporte"]
tabs_pro    = ["Precio & Proyección", "Estadísticas", "Geometría", "Multi-activo", "Backtesting", "Comparativa", "Reporte"]
tab_labels  = tabs_pro if ui_mode == "Pro" else tabs_retail
tabs        = st.tabs(tab_labels)

tab_map = {lbl: tab for lbl, tab in zip(tab_labels, tabs)}

# ── TAB 1: PRECIO & PROYECCIÓN ───────────────────────────────────────────────
with tab_map["Precio & Proyección"]:
    st.plotly_chart(chart_main(df, m, t), use_container_width=True)

    col_a, col_b, col_c = st.columns([1.2, 1, 1])

    with col_a:
        st.markdown('<p class="sec">Niveles topofísicos clave</p>', unsafe_allow_html=True)
        lv = m["levels"]
        rows_html = ""
        for name, price, cls in [
            ("Resistencia fuerte", lv["strong_resistance"], "tag-sr"),
            ("Resistencia",        lv["resistance"],        "tag-r"),
            ("Zona de equilibrio", lv["equilibrium"],       "tag-eq"),
            ("Soporte",            lv["support"],           "tag-s"),
            ("Soporte fuerte",     lv["strong_support"],    "tag-ss"),
        ]:
            pct = (price / m["last_price"] - 1) * 100
            rows_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:5px 0;border-bottom:1px solid #21262d;font-size:.74rem;">
              <span>{name}</span>
              <span style="display:flex;gap:8px;align-items:center;">
                <span class="{cls}">{pct:+.1f}%</span>
                <span style="font-family:monospace;color:#e6edf3;">${price:,.2f}</span>
              </span>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="sec">Probabilidad de escenarios (90d)</p>', unsafe_allow_html=True)
        for label, pct, color in [
            ("Expansión alcista (+10%)",  m["p_bull"],    "#00cc66"),
            ("Consolidación lateral",     m["p_lateral"], "#58a6ff"),
            ("Corrección bajista (−5%)", m["p_bear"],    "#f85149"),
        ]:
            st.markdown(f"""
            <div style="margin:5px 0;">
              <div style="display:flex;justify-content:space-between;font-size:.72rem;margin-bottom:2px;">
                <span>{label}</span>
                <span style="font-family:monospace;color:{color};">{pct:.1f}%</span>
              </div>
              <div style="background:#161b22;border-radius:3px;height:5px;">
                <div style="width:{pct:.1f}%;background:{color};opacity:.75;height:5px;border-radius:3px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown('<p class="sec">Proyección Monte Carlo 12m</p>', unsafe_allow_html=True)
        last = m["last_price"]
        for label, val, color in [
            ("P90 (optimista)",  m["proj_p90"][-1], "#00ff88"),
            ("P50 (mediana)",    m["proj_p50"][-1], "#e6edf3"),
            ("P10 (pesimista)",  m["proj_p10"][-1], "#f85149"),
        ]:
            pct = (val/last-1)*100
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                        border-bottom:1px solid #21262d;font-size:.74rem;">
              <span style="color:#8b949e;">{label}</span>
              <span style="font-family:monospace;color:{color};">${val:,.2f}
                <span style="font-size:.65rem;color:{color};">({pct:+.1f}%)</span>
              </span>
            </div>""", unsafe_allow_html=True)

    st.plotly_chart(chart_montecarlo(m, t), use_container_width=True)

# ── TAB 2: ESTADÍSTICAS ───────────────────────────────────────────────────────
with tab_map["Estadísticas"]:
    # AXIX Score
    col_g, col_r = st.columns([1, 1])
    with col_g:
        st.markdown('<p class="sec">AXIX Signal Score</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_axix_gauge(axix), use_container_width=True)
    with col_r:
        st.markdown('<p class="sec">Dimensiones topofísicas</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_axix_radar(axix), use_container_width=True)

    st.divider()

    col_d, col_v = st.columns(2)
    with col_d:
        st.plotly_chart(chart_distribution(m, t), use_container_width=True)
    with col_v:
        st.plotly_chart(chart_rolling_vol(df, m, t), use_container_width=True)

    st.divider()

    # Stats summary table
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<p class="sec">Estadísticas descriptivas del período</p>', unsafe_allow_html=True)
        stats_data = {
            "Métrica": ["Precio actual","Precio inicial","Máximo histórico","Mínimo histórico",
                        "Media del período","Desv. estándar","Retorno total","Vol anualizada",
                        "Curtosis","Asimetría","Exp. de Hurst","Energía topofísica"],
            "Valor": [
                f"${m['last_price']:,.2f}", f"${m['first_price']:,.2f}",
                f"${m['max_price']:,.2f}",  f"${m['min_price']:,.2f}",
                f"${m['mean_price']:,.2f}", f"${m['std_price']:,.2f}",
                f"{m['period_return']:+.2f}%", f"{m['vol_annual']*100:.2f}%",
                f"{m['kurt']:.3f}", f"{m['skew']:.3f}",
                f"{m['hurst']:.3f}", f"{m['energy']:.2f}%",
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True,
                     column_config={"Métrica": st.column_config.TextColumn(width="medium"),
                                    "Valor":   st.column_config.TextColumn(width="medium")})

    with col_s2:
        st.markdown('<p class="sec">Retorno año a año</p>', unsafe_allow_html=True)
        if not yoy.empty:
            st.plotly_chart(chart_yoy(yoy), use_container_width=True)

    if not yoy.empty:
        st.markdown('<p class="sec">Tabla comparativa histórica año a año</p>', unsafe_allow_html=True)

        def color_ret(val):
            try:
                v = float(str(val).replace("%",""))
                return f"color: {'#00cc66' if v > 0 else '#f85149'}"
            except:
                return ""

        styled = (
            yoy.style
            .applymap(color_ret, subset=["Retorno %"])
            .format({
                "Apertura": "${:,.2f}", "Cierre": "${:,.2f}",
                "Máximo":   "${:,.2f}", "Mínimo": "${:,.2f}",
                "Retorno %":"{:+.2f}%", "Vol anual %": "{:.2f}%",
            })
            .set_properties(**{
                "background-color": "#0d1117",
                "color": "#c9d1d9",
                "border-color": "#21262d",
                "font-family": "monospace",
                "font-size": "12px",
            })
        )
        st.dataframe(styled, use_container_width=True)

# ── TAB GEOMETRÍA ──────────────────────────────────────────────────
if ui_mode == "Pro":
    with tab_map["Geometría"]:
        st.markdown('<p class="sec">Geometría GUE — Espaciados de Riemann</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:.72rem;color:#8b949e;margin-bottom:8px;line-height:1.7;">
        Comparación entre la distribución de espaciados normalizados de retornos y la distribución
        Wigner-Dyson del <b style="color:#bc8cff;">Gaussian Unitary Ensemble (GUE)</b>, análoga al espaciado
        de los ceros no triviales de la función ζ de Riemann.<br>
        Calidad de ajuste GUE: <b style="color:#00ff88;">{geo['gue_fit_quality']}</b> 
        (KS stat: {geo['wd_ks_stat']:.4f})
        </div>""", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(chart_gue_spacings(geo, t), use_container_width=True)
        with col_g2:
            st.plotly_chart(chart_acf(geo, t), use_container_width=True)

        st.divider()
        st.markdown('<p class="sec">Componentes AXIX Score — Detalle</p>', unsafe_allow_html=True)
        comp_df = pd.DataFrame([
            {"Dimensión": k, "Score": v, "Descripción": desc}
            for (k, v), desc in zip(axix["components"].items(), [
                "Inverso de volatilidad normalizada. Alta estabilidad = score alto.",
                "Proximidad del Hurst a zona de tendencia óptima (0.60–0.70).",
                "Dirección del momentum de 21 días normalizado.",
                "Inverso de la autocorrelación de retornos al cuadrado.",
                "Proximidad de la energía topofísica al nivel óptimo (15%).",
            ])
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True,
                     column_config={
                         "Score": st.column_config.ProgressColumn(
                             "Score", min_value=0, max_value=100, format="%.1f")
                     })

        st.divider()
        st.markdown('<p class="sec">Interpretación topofísica completa</p>', unsafe_allow_html=True)
        lv = m["levels"]
        interp = f"""1. ESTADO GEOMÉTRICO DEL ESPACIO DE PRECIO
   {t} muestra señal dominante de {axix['signal']} (Score AXIX: {axix['score']:.0f}/100).
   Exponente de Hurst: {m['hurst']:.3f} — {'tendencia persistente' if m['hurst']>0.55 else 'reversión a la media' if m['hurst']<0.45 else 'caminata aleatoria'}.
   Ajuste GUE (Riemann): calidad {geo['gue_fit_quality']}.

2. ZONAS DE COHERENCIA Y DENSIDAD INFORMACIONAL
   Resistencia fuerte: ${lv['strong_resistance']:,.2f} ({(lv['strong_resistance']/m['last_price']-1)*100:+.1f}%)
   Resistencia:        ${lv['resistance']:,.2f} ({(lv['resistance']/m['last_price']-1)*100:+.1f}%)
   Equilibrio actual:  ${lv['equilibrium']:,.2f}
   Soporte:            ${lv['support']:,.2f} ({(lv['support']/m['last_price']-1)*100:+.1f}%)
   Soporte fuerte:     ${lv['strong_support']:,.2f} ({(lv['strong_support']/m['last_price']-1)*100:+.1f}%)

3. CAMINO DE MÍNIMA ENERGÍA — 12 MESES
   P50 (mediana):  ${m['proj_p50'][-1]:,.2f} ({(m['proj_p50'][-1]/m['last_price']-1)*100:+.1f}%)
   Banda P10–P90: ${m['proj_p10'][-1]:,.2f} — ${m['proj_p90'][-1]:,.2f}

4. PROBABILIDADES DE ESCENARIO (90 DÍAS)
   Expansión alcista (+10%):  {m['p_bull']:.1f}%
   Consolidación lateral:      {m['p_lateral']:.1f}%
   Corrección bajista (−5%): {m['p_bear']:.1f}%

5. NOTA METODOLÓGICA
   Modelo experimental. Distribución GUE análoga al espaciado de ceros de Riemann.
   No constituye asesoría financiera. AXIX."""
        st.code(interp, language=None)

# ── TAB MULTI-ACTIVO ─────────────────────────────────────────────────────────
with tab_map["Multi-activo"]:
    if len(multi_dfs) > 1:
        st.plotly_chart(chart_multi_overlay(multi_dfs), use_container_width=True)

        st.markdown('<p class="sec">Tabla comparativa de métricas</p>', unsafe_allow_html=True)
        comp_rows = []
        for tkr2, df2 in multi_dfs.items():
            try:
                m2 = compute_full_metrics(df2)
                comp_rows.append({
                    "Ticker":       tkr2,
                    "Precio":       f"${m2['last_price']:,.2f}",
                    "Retorno %":    f"{m2['period_return']:+.2f}%",
                    "Vol anual %":  f"{m2['vol_annual']*100:.1f}%",
                    "Hurst":        f"{m2['hurst']:.3f}",
                    "AXIX Score":   f"{m2['axix']['score']:.0f}",
                    "Señal":        m2['axix']['signal'],
                })
            except:
                pass
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Agrega tickers en el panel lateral (campo 'Comparativa multi-activo') para ver el overlay.")

# ── TAB BACKTESTING (Pro only) ────────────────────────────────────────────────
if ui_mode == "Pro":
    with tab_map["Backtesting"]:
        if bt:
            st.plotly_chart(chart_backtest(bt, t), use_container_width=True)
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Ventanas analizadas", str(bt["windows_tested"]))
            col_b2.metric("Resistencia tocada", f"{bt['resistance_hit_pct']:.1f}%")
            col_b3.metric("Soporte mantenido",  f"{bt['support_held_pct']:.1f}%")
            st.info(f"""
**Metodología de backtesting:**
Para cada una de las **{bt['windows_tested']} ventanas** históricas se calcularon los niveles de
resistencia (+18% vol) y soporte (−15% vol) usando únicamente datos anteriores a esa ventana.
Se evaluó si el precio tocó la resistencia o mantuvo el soporte en los siguientes
**{bt['lookforward_bars']} periodos**.

Un porcentaje alto indica que los niveles topofísicos actuaron históricamente como zonas relevantes.
""")
        else:
            st.warning("Se necesitan al menos 100 observaciones para el backtesting.")


# ── TAB COMPARATIVA ──────────────────────────────────────────────────────────
with tab_map["Comparativa"]:
    st.markdown("""
    <div style="font-size:.76rem;color:#8b949e;line-height:1.8;margin-bottom:12px;">
    Comparación automática entre el modelo topofísico <b style="color:#00ff88;">AXIX</b>
    y 4 modelos clásicos de proyección de precio. El backtesting walk-forward evalúa
    qué modelo tiene menor error y mayor acierto direccional en el histórico real del activo.
    </div>""", unsafe_allow_html=True)

    if cmp_models:
        # Overlay chart
        st.plotly_chart(chart_comparison_overlay(df, m, cmp_models, t),
                        use_container_width=True)

        # Model descriptions
        st.markdown('<p class="sec">Modelos incluidos en la comparativa</p>',
                    unsafe_allow_html=True)
        desc_cols = st.columns(len(cmp_models) + 1)
        axix_desc = [("AXIX", "#00ff88",
                      "Geometría inspirada en ceros de Riemann. "
                      "Monte Carlo calibrado con exponente de Hurst. "
                      "Caminos de mínima energía topofísica.")]
        model_descs = axix_desc + [(v["name"], v["color"], v["description"])
                                    for v in cmp_models.values()]
        for col, (name, color, desc) in zip(desc_cols, model_descs):
            col.markdown(f"""
            <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;
                        padding:10px;height:100%;">
              <div style="font-family:monospace;font-size:.72rem;color:{color};
                          font-weight:500;margin-bottom:4px;">{name}</div>
              <div style="font-size:.68rem;color:#8b949e;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Backtest results
        if cmp_bt is not None and not cmp_bt.empty:
            st.markdown('<p class="sec">Backtesting walk-forward — resultados cuantitativos</p>',
                        unsafe_allow_html=True)

            col_t, col_s = st.columns([1.2, 1])
            with col_t:
                # Highlight best model
                best = cmp_bt.iloc[0]["Modelo"]
                st.dataframe(
                    cmp_bt[["Modelo","MAE %","RMSE %","Dir. Acc. %","Cobertura %","Ventanas"]],
                    hide_index=True, use_container_width=True,
                    column_config={
                        "MAE %":       st.column_config.NumberColumn(format="%.2f%%"),
                        "RMSE %":      st.column_config.NumberColumn(format="%.2f%%"),
                        "Dir. Acc. %": st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%.1f%%"),
                        "Cobertura %": st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%.1f%%"),
                    }
                )
                st.caption(f"Menor Score Cmp = mejor rendimiento global. "
                           f"Mejor modelo en este período: **{best}**")

            with col_s:
                st.plotly_chart(chart_comparison_coverage(cmp_bt),
                                use_container_width=True)

            st.plotly_chart(chart_comparison_metrics(cmp_bt),
                            use_container_width=True)

            # Interpretation
            st.markdown('<p class="sec">Lectura de resultados</p>',
                        unsafe_allow_html=True)
            axix_row = cmp_bt[cmp_bt["Modelo"] == "AXIX"]
            if not axix_row.empty:
                ar = axix_row.iloc[0]
                rank_row = cmp_bt.reset_index(drop=True)
                axix_rank = rank_row[rank_row["Modelo"]=="AXIX"].index[0] + 1
                total_models = len(cmp_bt)
                rank_txt = (
                    "**Mejor modelo** en este activo y período."
                    if axix_rank == 1 else
                    f"Posición **{axix_rank} de {total_models}** modelos."
                )
                st.info(f"""
**AXIX en este backtesting:**
{rank_txt}
- MAE: **{ar['MAE %']:.2f}%** del precio promedio
- Dirección correcta: **{ar['Dir. Acc. %']:.1f}%** de las ventanas
- Cobertura de bandas: **{ar['Cobertura %']:.1f}%** del tiempo el precio estuvo dentro de la banda

*Los resultados varían por activo, período e intervalo. El backtesting pasado no garantiza rendimiento futuro.*
""")
        else:
            st.info("Se necesitan al menos 150 observaciones para el backtesting comparativo.")
    else:
        st.warning("Ejecuta un análisis para ver la comparativa de modelos.")

# ── TAB REPORTE / PDF ─────────────────────────────────────────────────────────
with tab_map["Reporte"]:
    st.markdown('<p class="sec">Generar reporte PDF</p>', unsafe_allow_html=True)

    landscape = "Institucional" in pdf_fmt
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.markdown(f"""
        <div style="font-size:.76rem;color:#8b949e;line-height:1.8;">
        <b style="color:#e6edf3;">Formato seleccionado:</b> {pdf_fmt}<br>
        El reporte incluye: métricas clave · niveles topofísicos · proyección Monte Carlo ·
        distribución de retornos · volatilidad rodante · tabla año a año · AXIX Score ·
        {'geometría GUE · backtesting · ' if ui_mode=='Pro' else ''}interpretación completa.
        </div>""", unsafe_allow_html=True)

    if st.button("Generar PDF", type="secondary"):
        with st.spinner("Renderizando gráficas y generando PDF…"):
            try:
                def to_png(fig, w, h):
                    return fig.to_image(format="png", width=w, height=h, scale=1.5)

                imgs = {
                    "main":  to_png(chart_main(df, m, t), 1400, 560),
                    "mc":    to_png(chart_montecarlo(m, t), 1100, 300),
                    "dist":  to_png(chart_distribution(m, t), 950, 280),
                    "vol":   to_png(chart_rolling_vol(df, m, t), 950, 280),
                    "gauge": to_png(chart_axix_gauge(axix), 600, 260),
                    "radar": to_png(chart_axix_radar(axix), 600, 260),
                    "yoy":   to_png(chart_yoy(yoy), 1100, 280) if not yoy.empty else None,
                    "gue":   to_png(chart_gue_spacings(geo, t), 900, 270) if ui_mode=="Pro" else None,
                    "bt":    to_png(chart_backtest(bt, t), 900, 270) if ui_mode=="Pro" and bt else None,
                }

                pdf_bytes = generate_pdf_report(
                    ticker=t, period=r["period"], interval=r["interval"],
                    metrics=m, yoy_df=yoy, backtest=bt, geo=geo,
                    imgs=imgs, landscape=landscape, pro_mode=(ui_mode=="Pro"),
                )

                fname = f"AXIX_{t}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.download_button("⬇  Descargar PDF", data=pdf_bytes,
                                   file_name=fname, mime="application/pdf")
                st.success(f"✓ PDF generado — {len(pdf_bytes)//1024} KB")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")
