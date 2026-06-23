# AXIX Ω∞ — Sistema Topofísico Financiero v2

Plataforma de análisis topofísico para cualquier activo de Yahoo Finance.

## Instalación local

```bash
unzip axix_topofisico.zip && cd axix_topofisico
pip install -r requirements.txt
streamlit run app.py
```

Abre en http://localhost:8501

## Deploy gratuito (Streamlit Cloud)

1. Sube la carpeta a GitHub (puede ser privado)
2. Ve a https://share.streamlit.io → conecta GitHub → selecciona `app.py`
3. Deploy → URL pública en ~2 min

## Qué incluye v2

### Análisis principal
- Datos reales via yfinance (acciones, ETFs, crypto, divisas, índices)
- Precio histórico + proyección topofísica 12 meses (Monte Carlo P10/P50/P90)
- Niveles de soporte/resistencia geométricos
- Mapa de densidad informacional Riemann

### AXIX Signal Score (0–100)
Score compuesto de 5 dimensiones topofísicas:
estabilidad, memoria (Hurst), momentum, coherencia, energía

### Estadísticas completas
- Máx/Mín/Media/Desv históricos
- Tabla año a año (YOY) con retorno y volatilidad por año
- Distribución de retornos vs normal teórica
- Volatilidad rodante 21d / 63d
- Curtosis, asimetría, exponente de Hurst

### Modo Pro (extras)
- Geometría GUE / Riemann: espaciados normalizados vs Wigner-Dyson
- Autocorrelación de retornos (lags 1–20)
- Backtesting de niveles: qué % de veces los niveles funcionaron históricamente
- Radar de componentes AXIX

### Comparativa multi-activo
Overlay normalizado de hasta 3+ tickers simultáneos con tabla comparativa

### Exportación PDF
- Formato técnico portrait (research note)  
- Formato institucional landscape (pitch deck)
- Incluye todas las gráficas + tablas + interpretación completa

## Tickers soportados

| Tipo | Ejemplos |
|---|---|
| Acciones US | AAPL, NVDA, TSLA, MSFT, AMZN, META |
| ETFs | SPY, QQQ, GLD, TLT |
| Crypto | BTC-USD, ETH-USD, SOL-USD |
| Divisas | EURUSD=X, USDJPY=X |
| Índices | ^GSPC, ^DJI, ^IXIC |
| Materias primas | GC=F, CL=F, SI=F |
| México | AMXL.MX, GFNORTEO.MX, WALMEX*.MX |

## Estructura

```
axix_topofisico/
├── app.py          # UI Streamlit principal
├── engine.py       # Motor de cómputo topofísico
├── charts.py       # Todas las gráficas Plotly
├── pdf_export.py   # Generador PDF (portrait + landscape)
├── requirements.txt
└── .streamlit/config.toml
```

> Modelo experimental · No es asesoría financiera · AXIX Ω∞
