"""
AXIX Ω∞ — Chart library v2
All Plotly figures, fully dark-themed.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

BG   = "#0a0e14"
SURF = "#0d1117"
GRID = "#161b22"
BORD = "#21262d"
TEXT = "#8b949e"
TEXTB= "#e6edf3"
GREEN= "#00cc66"
GREENB="#00ff88"
RED  = "#f85149"
ORG  = "#f0883e"
BLUE = "#58a6ff"
PURP = "#bc8cff"

BASE = dict(
    paper_bgcolor=BG, plot_bgcolor=SURF,
    font=dict(family="monospace", color=TEXT, size=10),
    margin=dict(l=52, r=18, t=36, b=40),
)

def _apply(fig, extra=None):
    cfg = dict(**BASE)
    if extra:
        cfg.update(extra)
    fig.update_layout(**cfg)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BORD, showgrid=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BORD, showgrid=True)
    return fig


# ─── 1. MAIN PRICE + PROJECTION ──────────────────────────────────────────────

def chart_main(df, m, ticker):
    prices = m["prices"]
    dates  = df.index.tolist()
    n      = len(prices)
    future = pd.bdate_range(start=dates[-1], periods=253)[1:]
    p10, p50, p90 = m["proj_p10"], m["proj_p50"], m["proj_p90"]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.82, 0.18],
                        shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines",
        name="Precio", line=dict(color=GREEN, width=1.6)), row=1, col=1)

    # Projection band
    fig.add_trace(go.Scatter(
        x=list(future)+list(future[::-1]),
        y=list(p90)+list(p10[::-1]),
        fill="toself", fillcolor="rgba(0,204,102,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Banda P10–P90"), row=1, col=1)
    fig.add_trace(go.Scatter(x=future, y=p50, mode="lines",
        name="Proyección mediana",
        line=dict(color="rgba(255,255,255,0.55)", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=future, y=p90, mode="lines", showlegend=False,
        line=dict(color="rgba(0,255,136,0.35)", width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=future, y=p10, mode="lines", showlegend=False,
        line=dict(color="rgba(248,81,73,0.35)", width=0.8)), row=1, col=1)

    # Levels
    for key, color, label in [
        ("strong_resistance", RED,   "Res. fuerte"),
        ("resistance",        ORG,   "Resistencia"),
        ("support",           GREEN, "Soporte"),
        ("strong_support",    GREENB,"Sop. fuerte"),
    ]:
        lvl = m["levels"][key]
        fig.add_hline(y=lvl, line=dict(color=color, width=0.7, dash="dot"),
            annotation_text=f" {label} ${lvl:,.2f}",
            annotation_font_size=8, annotation_font_color=color, row=1, col=1)

    # Density heatmap row
    fig.add_trace(go.Bar(x=dates, y=m["density"],
        marker=dict(color=m["density"],
            colorscale=[[0,SURF],[0.4,"#003d1f"],[0.7,"#006633"],[1,GREENB]],
            showscale=False),
        name="Densidad Riemann", showlegend=True), row=2, col=1)

    _apply(fig, dict(
        title=dict(text=f"<b>{ticker}</b>  ·  Análisis Topofísico  ·  Proyección 12m",
            font=dict(size=13, color=TEXTB), x=0),
        height=560,
        legend=dict(orientation="h", y=1.02, x=0,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    fig.update_yaxes(tickformat="$,.0f", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(showticklabels=False, row=2, col=1, gridcolor=GRID)
    return fig


# ─── 2. YEAR-OVER-YEAR BAR ───────────────────────────────────────────────────

def chart_yoy(yoy_df):
    if yoy_df.empty:
        return go.Figure()
    years  = yoy_df.index.tolist()
    rets   = yoy_df["Retorno %"].tolist()
    colors = [GREEN if r >= 0 else RED for r in rets]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(y) for y in years], y=rets,
        marker_color=colors, name="Retorno anual %",
        text=[f"{r:+.1f}%" for r in rets], textposition="outside",
        textfont=dict(size=8, color=TEXTB)))
    fig.add_hline(y=0, line=dict(color=BORD, width=0.8))
    _apply(fig, dict(
        title=dict(text="Retorno anual histórico (%)",
            font=dict(size=12, color=TEXTB), x=0),
        height=280, yaxis=dict(ticksuffix="%", gridcolor=GRID),
        showlegend=False,
    ))
    return fig


# ─── 3. ROLLING VOLATILITY ───────────────────────────────────────────────────

def chart_rolling_vol(df, m, ticker):
    prices = pd.Series(m["prices"], index=df.index)
    lr = np.log(prices / prices.shift(1))
    v21 = lr.rolling(21).std() * np.sqrt(252) * 100
    v63 = lr.rolling(63).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=v21, name="Vol 21d",
        line=dict(color=GREEN, width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=v63, name="Vol 63d",
        line=dict(color=BLUE, width=1.2, dash="dot")))
    _apply(fig, dict(
        title=dict(text=f"Volatilidad rodante anualizada · {ticker}",
            font=dict(size=12, color=TEXTB), x=0),
        height=280, yaxis=dict(ticksuffix="%", gridcolor=GRID),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ─── 4. RETURNS DISTRIBUTION ─────────────────────────────────────────────────

def chart_distribution(m, ticker):
    lr  = m["log_returns"] * 100
    mu, sigma = np.mean(lr), np.std(lr)
    x   = np.linspace(lr.min(), lr.max(), 300)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=lr, nbinsx=80, histnorm="probability density",
        name="Retornos log",
        marker=dict(color="rgba(0,204,102,0.4)", line=dict(color=SURF, width=0.3))))
    fig.add_trace(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sigma),
        mode="lines", name="Normal teórica",
        line=dict(color=ORG, width=1.5, dash="dot")))
    _apply(fig, dict(
        title=dict(text=f"Distribución de retornos · {ticker}",
            font=dict(size=12, color=TEXTB), x=0),
        height=280,
        xaxis=dict(title="Retorno log (%)", gridcolor=GRID),
        yaxis=dict(title="Densidad", gridcolor=GRID),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ─── 5. MONTE CARLO ──────────────────────────────────────────────────────────

def chart_montecarlo(m, ticker):
    p10, p50, p90 = m["proj_p10"], m["proj_p50"], m["proj_p90"]
    x = list(range(len(p50)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p90)+list(p10[::-1]),
        fill="toself", fillcolor="rgba(0,204,102,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="Banda P10–P90"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", name="Mediana",
        line=dict(color=GREENB, width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=p90, mode="lines", name="P90",
        line=dict(color="rgba(0,255,136,0.4)", width=0.8, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=p10, mode="lines", name="P10",
        line=dict(color="rgba(248,81,73,0.4)", width=0.8, dash="dot")))
    _apply(fig, dict(
        title=dict(text=f"Monte Carlo · {ticker} · 252 días / 2,000 trayectorias",
            font=dict(size=12, color=TEXTB), x=0),
        height=300,
        yaxis=dict(tickformat="$,.0f", gridcolor=GRID),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ─── 6. AXIX SCORE GAUGE ─────────────────────────────────────────────────────

def chart_axix_gauge(axix: dict):
    score = axix["score"]
    color = axix["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(font=dict(color=color, size=36, family="monospace"),
                    suffix=""),
        delta=dict(reference=50, increasing=dict(color=GREEN),
                   decreasing=dict(color=RED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0.5,
                      tickcolor=BORD, tickfont=dict(size=9, color=TEXT)),
            bar=dict(color=color, thickness=0.22),
            bgcolor=SURF,
            borderwidth=0,
            steps=[
                dict(range=[0,  35], color="#1a0a0a"),
                dict(range=[35, 55], color="#0d1a0d"),
                dict(range=[55, 75], color="#0a1520"),
                dict(range=[75,100], color="#002e17"),
            ],
            threshold=dict(line=dict(color=color, width=2), value=score),
        ),
        title=dict(text=f"<b>AXIX Score</b><br><span style='font-size:12px;color:{color}'>{axix['signal']}</span>",
                   font=dict(color=TEXTB, size=14)),
        domain=dict(x=[0,1], y=[0,1]),
    ))
    _apply(fig, dict(height=260, margin=dict(l=20, r=20, t=40, b=10)))
    return fig


# ─── 7. AXIX SCORE RADAR ─────────────────────────────────────────────────────

def chart_axix_radar(axix: dict):
    cats  = list(axix["components"].keys())
    vals  = list(axix["components"].values())
    cats  += [cats[0]]
    vals  += [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(0,204,102,0.12)",
        line=dict(color=GREEN, width=1.5),
        name="AXIX",
    ))
    _apply(fig, dict(
        height=260,
        polar=dict(
            bgcolor=SURF,
            radialaxis=dict(range=[0,100], visible=True, gridcolor=BORD,
                            tickfont=dict(size=8, color=TEXT), showline=False),
            angularaxis=dict(gridcolor=BORD, tickfont=dict(size=9, color=TEXTB)),
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    ))
    return fig


# ─── 8. GUE GEOMETRY ─────────────────────────────────────────────────────────

def chart_gue_spacings(geo: dict, ticker: str):
    sp   = geo["norm_spacings"]
    x    = np.linspace(0, 3, 300)
    wigner = (np.pi / 2) * x * np.exp(-np.pi * x**2 / 4)  # Wigner-Dyson
    fig  = go.Figure()
    fig.add_trace(go.Histogram(x=sp, nbinsx=60, histnorm="probability density",
        name="Espaciados reales",
        marker=dict(color="rgba(188,140,255,0.45)", line=dict(color=SURF, width=0.3))))
    fig.add_trace(go.Scatter(x=x, y=wigner, mode="lines", name="Wigner-Dyson (GUE)",
        line=dict(color=PURP, width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=np.exp(-x), mode="lines", name="Poisson (random)",
        line=dict(color=RED, width=1.2, dash="dot")))
    _apply(fig, dict(
        title=dict(text=f"Geometría GUE · espaciados normalizados · {ticker}",
            font=dict(size=12, color=TEXTB), x=0),
        height=270,
        xaxis=dict(title="s (espaciado normalizado)", gridcolor=GRID),
        yaxis=dict(title="Densidad", gridcolor=GRID),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


def chart_acf(geo: dict, ticker: str):
    lags = list(range(1, len(geo["acf"]) + 1))
    acf  = geo["acf"]
    ci   = 1.96 / np.sqrt(200)
    fig  = go.Figure()
    fig.add_hrect(y0=-ci, y1=ci, fillcolor="rgba(88,166,255,0.06)",
                  line_width=0, annotation_text="IC 95%",
                  annotation_font_size=8, annotation_font_color=BLUE)
    fig.add_trace(go.Bar(x=lags, y=acf,
        marker_color=[GREEN if a > 0 else RED for a in acf],
        name="Autocorrelación"))
    fig.add_hline(y=0, line=dict(color=BORD, width=0.8))
    _apply(fig, dict(
        title=dict(text=f"Autocorrelación de retornos (lags 1–20) · {ticker}",
            font=dict(size=12, color=TEXTB), x=0),
        height=250,
        xaxis=dict(title="Lag", gridcolor=GRID, dtick=2),
        yaxis=dict(title="ACF", gridcolor=GRID),
        showlegend=False,
    ))
    return fig


# ─── 9. MULTI-ASSET OVERLAY ──────────────────────────────────────────────────

def chart_multi_overlay(dfs: dict):
    """dfs: {ticker: DataFrame with Close}"""
    fig = go.Figure()
    palette = [GREEN, BLUE, ORG, PURP, RED]
    for i, (tkr, df) in enumerate(dfs.items()):
        prices = df["Close"].values.astype(float)
        norm   = (prices / prices[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=df.index.tolist(), y=norm,
            mode="lines", name=tkr,
            line=dict(color=palette[i % len(palette)], width=1.5),
        ))
    fig.add_hline(y=0, line=dict(color=BORD, width=0.6, dash="dot"))
    _apply(fig, dict(
        title=dict(text="Comparativa multi-activo (retorno normalizado %)",
            font=dict(size=12, color=TEXTB), x=0),
        height=340,
        yaxis=dict(ticksuffix="%", gridcolor=GRID),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    ))
    return fig


# ─── 10. BACKTEST LEVELS ─────────────────────────────────────────────────────

def chart_backtest(bt: dict, ticker: str):
    if not bt:
        return go.Figure()
    labels  = ["Resistencia tocada", "Soporte mantenido"]
    values  = [bt["resistance_hit_pct"], bt["support_held_pct"]]
    colors  = [ORG, GREEN]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=TEXTB),
        width=0.4,
    ))
    fig.add_hline(y=50, line=dict(color=BORD, width=0.8, dash="dot"),
                  annotation_text="50%", annotation_font_size=8)
    _apply(fig, dict(
        title=dict(text=f"Backtesting de niveles topofísicos · {ticker} · {bt['windows_tested']} ventanas",
            font=dict(size=12, color=TEXTB), x=0),
        height=270,
        yaxis=dict(ticksuffix="%", range=[0, 110], gridcolor=GRID),
        showlegend=False,
    ))
    return fig


# ─── COMPARISON CHARTS ───────────────────────────────────────────────────────

def chart_comparison_overlay(df, m, models: dict, ticker: str):
    """All projection models overlaid on the same chart."""
    prices = m["prices"]
    dates  = df.index.tolist()
    future = pd.bdate_range(start=dates[-1], periods=253)[1:]
    last   = prices[-1]

    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines",
        name="Precio histórico",
        line=dict(color="#e6edf3", width=1.8)))

    # AXIX projection (P50)
    p50 = m["proj_p50"]
    p10 = m["proj_p10"]
    p90 = m["proj_p90"]
    fig.add_trace(go.Scatter(
        x=list(future)+list(future[::-1]),
        y=list(p90)+list(p10[::-1]),
        fill="toself", fillcolor="rgba(0,204,102,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="AXIX banda"))
    fig.add_trace(go.Scatter(x=future, y=p50, mode="lines",
        name="AXIX Ω∞ (P50)",
        line=dict(color="#00ff88", width=2.0)))

    # Other models
    for key, model in models.items():
        h = min(len(model["proj"]), len(future))
        fig.add_trace(go.Scatter(
            x=future[:h], y=model["proj"][:h],
            mode="lines", name=model["name"],
            line=dict(color=model["color"], width=1.3, dash="dot")))
        # Bands (faint) — proper hex → rgba conversion
        def _hex_rgba(hx, a=0.07):
            h = hx.lstrip("#")
            return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
        fill_c = _hex_rgba(model["color"]) if model["color"].startswith("#") else "rgba(128,128,128,0.07)"
        fig.add_trace(go.Scatter(
            x=list(future[:h]) + list(future[:h][::-1]),
            y=list(model["upper"][:h]) + list(model["lower"][:h][::-1]),
            fill="toself",
            fillcolor=fill_c,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ))

    _apply(fig, dict(
        title=dict(text=f"<b>{ticker}</b>  ·  Comparativa de modelos de proyección",
            font=dict(size=13, color=TEXTB), x=0),
        height=520,
        legend=dict(orientation="h", y=1.02, x=0,
            font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickformat="$,.0f", gridcolor=GRID),
    ))
    return fig


def chart_comparison_metrics(comp_df: pd.DataFrame):
    """Bar chart comparing MAE% and Direction Accuracy across models."""
    if comp_df.empty:
        return go.Figure()

    models   = comp_df["Modelo"].tolist()
    mae_vals = comp_df["MAE %"].tolist()
    dir_vals = comp_df["Dir. Acc. %"].tolist()

    pal = {"AXIX":"#00ff88","DRIFT":"#f0883e","GARCH":"#58a6ff",
           "BOLL":"#bc8cff","EMA":"#3fb950"}
    colors_bar = [pal.get(m, "#8b949e") for m in models]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["MAE % (menor = mejor)", "Dirección correcta % (mayor = mejor)"])

    fig.add_trace(go.Bar(x=models, y=mae_vals,
        marker_color=colors_bar,
        text=[f"{v:.2f}%" for v in mae_vals], textposition="outside",
        textfont=dict(size=9, color=TEXTB), name="MAE %"), row=1, col=1)

    fig.add_trace(go.Bar(x=models, y=dir_vals,
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in dir_vals], textposition="outside",
        textfont=dict(size=9, color=TEXTB),
        showlegend=False, name="Dir %"), row=1, col=2)

    fig.add_hline(y=50, row=1, col=2,
        line=dict(color=BORD, width=0.8, dash="dot"),
        annotation_text="50%", annotation_font_size=8)

    _apply(fig, dict(
        title=dict(text="Backtesting walk-forward — comparativa de modelos",
            font=dict(size=12, color=TEXTB), x=0),
        height=320, showlegend=False,
    ))
    fig.update_annotations(font_color=TEXT, font_size=9)
    return fig


def chart_comparison_coverage(comp_df: pd.DataFrame):
    """Scatter: coverage vs direction accuracy — ideal = top right."""
    if comp_df.empty:
        return go.Figure()

    pal = {"AXIX":"#00ff88","DRIFT":"#f0883e","GARCH":"#58a6ff",
           "BOLL":"#bc8cff","EMA":"#3fb950"}
    fig = go.Figure()

    for _, row in comp_df.iterrows():
        m = row["Modelo"]
        fig.add_trace(go.Scatter(
            x=[row["Cobertura %"]], y=[row["Dir. Acc. %"]],
            mode="markers+text",
            text=[m], textposition="top center",
            textfont=dict(size=10, color=pal.get(m,"#8b949e")),
            marker=dict(size=14, color=pal.get(m,"#8b949e"),
                        line=dict(color="#0a0e14", width=1.5)),
            name=m, showlegend=False,
        ))

    # Ideal corner annotation
    fig.add_annotation(x=95, y=95, text="Ideal →", showarrow=False,
        font=dict(size=8, color=TEXT))

    _apply(fig, dict(
        title=dict(text="Cobertura vs Dirección — espacio de calidad de modelos",
            font=dict(size=12, color=TEXTB), x=0),
        height=300,
        xaxis=dict(title="Cobertura % (precio dentro de bandas)", gridcolor=GRID,
                   range=[0, 110]),
        yaxis=dict(title="Dirección correcta %", gridcolor=GRID, range=[0, 110]),
    ))
    return fig
