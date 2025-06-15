# -*- coding: utf-8 -*-
"""
Streamlit App · EMSEC × EMTEC Heatmap Dashboard (v2)
===================================================
* Author   : Generated via ChatGPT — refined to meet **UI 요건 정의.pdf** (시각화 01).
* Purpose  : Interactive matrix-style heatmap exploring corporate data by
              EMSEC (Industry) × EMTEC (Technology) with rich metric controls,
              subtotals, and a 10-step colour scale.
* Run      :
      pip install streamlit pandas numpy plotly
      streamlit run streamlit_heatmap_app.py
"""

import streamlit as st
st.set_page_config(page_title="EMSEC × EMTEC Dashboard", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Callable

# ---------------------------------------------------------------------------
# 1. Synthetic Data Generator (≈1,000 rows) with 'Unclassified' categories
@st.cache_data(show_spinner=False)
def load_data(seed: int = 42, n: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = ["한국", "미국", "일본", "Unclassified"]
    markets_by_country = {
        "한국": ["KOSPI", "KOSDAQ"],
        "미국": ["NYSE", "NASDAQ"],
        "일본": ["PRIME", "STANDARD"],
        "Unclassified": ["Unclassified"]
    }
    sectors    = [f"Sector_{i}" for i in range(1,6)] + ["Unclassified"]
    industries = [f"Industry_{i}" for i in range(1,11)] + ["Unclassified"]
    subinds    = [f"SubInd_{i}" for i in range(1,16)] + ["Unclassified"]
    themes     = [f"Theme_{i}" for i in range(1,5)] + ["Unclassified"]
    techs      = [f"Tech_{i}" for i in range(1,9)] + ["Unclassified"]
    subtechs   = [f"SubTech_{i}" for i in range(1,13)] + ["Unclassified"]

    def pick(lst):
        return rng.choice(lst)

    rows = []
    for _ in range(n):
        country = pick(countries)
        market  = pick(markets_by_country[country])
        sector, industry, subind = pick(sectors), pick(industries), pick(subinds)
        theme, tech, subtech    = pick(themes), pick(techs), pick(subtechs)
        row = dict(
            Country=country, Market=market,
            Sector=sector, Industry=industry, Sub_industry=subind,
            Theme=theme, Technology=tech, Sub_Technology=subtech,
            PER=rng.normal(15,5) if rng.random()>0.07 else np.nan,
            PBR=rng.normal(1.5,0.4),
            EV_EBITDA=rng.normal(10,3),
            ROE=rng.normal(0.12,0.05),
            EBIT=rng.normal(80000,30000),
            EBITDA=rng.normal(110000,35000),
            Sales=rng.normal(500000,200000),
            Assets=rng.normal(1000000,400000),
            Book=rng.normal(600000,250000),
            Liabilities=rng.normal(400000,180000),
            Net_Income=rng.normal(50000,20000)
        )
        if rng.random()<0.05:
            row["Net_Income"] = -abs(row["Net_Income"])
        if rng.random()<0.03:
            row["Sales"] = 0
        rows.append(row)
    return pd.DataFrame(rows)

DF_RAW = load_data()

# ---------------------------------------------------------------------------
# 2. Sidebar: Filters & Controls

# 2-1 Stock Exchange
st.sidebar.header("상장시장")
country_opt = st.sidebar.selectbox("국가", ["전체"] + sorted(DF_RAW.Country.unique()))
market_pool = (
    sorted(DF_RAW.Market.unique())
    if country_opt=="전체"
    else sorted(DF_RAW.loc[DF_RAW.Country==country_opt, "Market"].unique())
)
market_opts = st.sidebar.multiselect("시장", market_pool, default=market_pool)

# 2-2 EMSEC (행)
st.sidebar.markdown("---")
st.sidebar.subheader("EMSEC · 행(Row)")
sector_sel = st.sidebar.selectbox("Sector", ["전체"]+sorted(DF_RAW.Sector.unique()))
industry_pool = (
    sorted(DF_RAW.Industry.unique())
    if sector_sel=="전체"
    else sorted(DF_RAW.loc[DF_RAW.Sector==sector_sel, "Industry"].unique())
)
industry_sel = st.sidebar.selectbox("Industry", ["전체"]+industry_pool)
subind_pool = (
    sorted(DF_RAW.Sub_industry.unique())
    if industry_sel=="전체"
    else sorted(DF_RAW.loc[DF_RAW.Industry==industry_sel, "Sub_industry"].unique())
)
rows_sel = st.sidebar.multiselect("Sub-industry", subind_pool, default=subind_pool)

# EMSEC 전용 필터 (key 추가)
exclude_unclassified_emsec = st.sidebar.checkbox(
    "Unclassified 제외", True, key="exclude_uc_emsec"
)
exclude_empty_emsec = st.sidebar.checkbox(
    "Empty(0/NaN) 제외", True, key="exclude_empty_emsec"
)

# 2-3 EMTEC (열)
st.sidebar.markdown("---")
st.sidebar.subheader("EMTEC · 열(Column)")
theme_sel = st.sidebar.selectbox("Theme", ["전체"]+sorted(DF_RAW.Theme.unique()))
tech_pool = (
    sorted(DF_RAW.Technology.unique())
    if theme_sel=="전체"
    else sorted(DF_RAW.loc[DF_RAW.Theme==theme_sel, "Technology"].unique())
)
tech_sel = st.sidebar.selectbox("Technology", ["전체"]+tech_pool)
subtech_pool = (
    sorted(DF_RAW.Sub_Technology.unique())
    if tech_sel=="전체"
    else sorted(DF_RAW.loc[DF_RAW.Technology==tech_sel, "Sub_Technology"].unique())
)
cols_sel = st.sidebar.multiselect("Sub-Technology", subtech_pool, default=subtech_pool)

# EMTEC 전용 필터 (key 추가)
exclude_unclassified_emtec = st.sidebar.checkbox(
    "Unclassified 제외", True, key="exclude_uc_emtec"
)
exclude_empty_emtec = st.sidebar.checkbox(
    "Empty(0/NaN) 제외", True, key="exclude_empty_emtec"
)

# 2-4 Metric Controls
st.sidebar.markdown("---")
st.sidebar.subheader("계측값 선택")
group_sel = st.sidebar.selectbox("그룹", ["기업", "비교가치 멀티플", "재무비율"])

if group_sel=="기업":
    corp_first = st.sidebar.selectbox("선택1", ["기업수", "0이하 비율"])
    if corp_first=="기업수":
        corp_mode = st.sidebar.selectbox("선택2", ["결측 포함", "결측 비율"])
        metric_main = "기업수"; metric_mode = corp_mode; base_col = None
    else:
        base_col_map = {"순이익":"Net_Income","EBITDA":"EBITDA","매출":"Sales","자산총":"Assets","순자산":"Book"}
        corp_base = st.sidebar.selectbox("선택2", list(base_col_map.keys()))
        corp_mode = st.sidebar.selectbox("선택3", ["결측 포함", "결측 제외"])
        metric_main = "0이하비율"; metric_mode = corp_mode; base_col = base_col_map[corp_base]
    agg_func = None; filter_option = None; allow_subtotal = True

elif group_sel=="비교가치 멀티플":
    metric_main = st.sidebar.selectbox("멀티플", ["PER","PBR","EV_EBITDA"])
    agg_func = st.sidebar.selectbox("집계", ["AVG","HRM","MED","AGG"])
    filter_option = st.sidebar.selectbox("데이터 필터", ["전체","최상위 10% 제외","0이하 제외","모두 제외"])
    allow_subtotal = False; base_col = None

else:
    metric_main = st.sidebar.selectbox("비율",
        ["ROE","영업이익률","EBITDA/Sales","총자산이익률","자산회전율","자기자본비율","부채비율"]
    )
    agg_func = st.sidebar.selectbox("집계", ["AVG","HRM","MED","AGG"])
    filter_option = st.sidebar.selectbox("데이터 필터", ["전체","0이하 제외"])
    allow_subtotal = True; base_col = None

# ---------------------------------------------------------------------------
# 3. Data Filtering
DF = DF_RAW.copy()
if country_opt!="전체": DF = DF[DF.Country==country_opt]
DF = DF[DF.Market.isin(market_opts)]
DF = DF[DF.Sub_industry.isin(rows_sel)]
DF = DF[DF.Sub_Technology.isin(cols_sel)]

# EMSEC Unclassified 제외
if exclude_unclassified_emsec:
    for col in ["Sector","Industry","Sub_industry"]:
        DF = DF[DF[col]!="Unclassified"]

# EMTEC Unclassified 제외
if exclude_unclassified_emtec:
    for col in ["Theme","Technology","Sub_Technology"]:
        DF = DF[DF[col]!="Unclassified"]

# Empty(0/NaN) 제외
if (exclude_empty_emsec or exclude_empty_emtec) and not (group_sel=="기업" and metric_main=="0이하비율"):
    if group_sel!="기업":
        DF = DF.replace({0:np.nan}).dropna(subset=[metric_main])

# ---------------------------------------------------------------------------
# 4. Aggregation Utilities
def harmonic_mean(s):
    arr = s.dropna()[s>0]
    return len(arr)/np.sum(1/arr) if len(arr)>0 else np.nan

def apply_filter(s):
    s2 = s.dropna()
    if filter_option=="최상위 10% 제외":
        k=int(len(s2)*0.1); s2=s2.sort_values().iloc[:-k] if k>0 else s2
    if filter_option=="0이하 제외":
        s2=s2[s2>0]
    if filter_option=="모두 제외":
        k=int(len(s2)*0.1); s2=s2.sort_values().iloc[:-k] if k>0 else s2; s2=s2[s2>0]
    return s2

def aggregator(s):
    if group_sel=="기업":
        if metric_main=="기업수":
            return len(s) if metric_mode=="결측 포함" else s.isna().sum()/len(s) if len(s) else np.nan
        arr = pd.to_numeric(s, errors='coerce')
        if metric_mode=="결측 제외": arr = arr.dropna()
        total = len(arr) if metric_mode=="결측 제외" else len(s)
        return (arr<=0).sum()/total if total else np.nan

    arr = apply_filter(s)
    if len(arr)==0: return np.nan
    if agg_func=="AVG": return arr.mean()
    if agg_func=="MED": return arr.median()
    if agg_func=="HRM": return harmonic_mean(arr)
    return arr.sum()

# ---------------------------------------------------------------------------
# 5. Pivot & Heatmap
values_col = base_col if (group_sel=="기업" and metric_main=="0이하비율") else (metric_main if group_sel!="기업" else "Country")
pivot = pd.pivot_table(
    DF, values=values_col,
    index="Sub_industry", columns="Sub_Technology",
    aggfunc=aggregator, fill_value=np.nan, observed=True
)

if allow_subtotal and not pivot.empty:
    pivot.loc["Subtotal"] = pivot.mean(numeric_only=True)
    pivot["Subtotal"]   = pivot.mean(axis=1, numeric_only=True)

st.title("EMSEC × EMTEC Heatmap")
bc = " > ".join([
    country_opt if country_opt!="전체" else "ALL",
    sector_sel  if sector_sel!="전체"  else "ALL",
    theme_sel   if theme_sel!="전체"   else "ALL"
])
st.markdown(f"<div style='font-size:0.9rem;color:#666;'>선택 경로 • {bc}</div>", unsafe_allow_html=True)
if pivot.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
    st.stop()

COLOR_SCALE = [
    "#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6",
    "#4292c6","#2171b5","#08519c","#08306b","#041e42"
]
z = pivot.values.astype(float)

if group_sel=="기업":
    if metric_main=="기업수" and metric_mode=="결측 포함":
        textfmt = lambda v: f"{v:,.0f}" if not np.isnan(v) else ""
    else:
        textfmt = lambda v: f"{v*100:.1f}%" if not np.isnan(v) else ""
elif group_sel=="비교가치 멀티플":
    textfmt = lambda v: f"{v:,.2f}" if not np.isnan(v) else ""
else:
    textfmt = lambda v: f"{v*100:.1f}%" if not np.isnan(v) else ""

fig = go.Figure(data=go.Heatmap(
    z=z, x=pivot.columns, y=pivot.index,
    colorscale=list(zip(np.linspace(0,1,len(COLOR_SCALE)), COLOR_SCALE)),
    hovertemplate="<b>%{y}</b> / %{x}<br>값: %{z:.3f}<extra></extra>",
    colorbar=dict(title=metric_main)
))
fig.update_traces(
    text=[[textfmt(val) for val in row] for row in z],
    texttemplate="%{text}", textfont=dict(color="white")
)
fig.update_layout(height=650, margin=dict(l=50, r=50, t=50, b=50))

st.plotly_chart(fig, use_container_width=True)
st.caption(f"총 행 개수: {len(DF):,}")
