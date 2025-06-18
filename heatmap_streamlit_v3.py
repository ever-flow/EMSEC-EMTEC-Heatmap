# -*- coding: utf-8 -*-

import streamlit as st
st.set_page_config(page_title="EMSEC × EMTEC Dashboard", layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Callable

# ---------------------------------------------------------------------------
# 1. Synthetic Data Generator (≈1,000 rows) with 'Unclassified' categories
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(seed: int = 42, n: int = 1_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = ["한국", "미국", "일본", "Unclassified"]
    markets_by_country = {
        "한국": ["KOSPI", "KOSDAQ"],
        "미국": ["NYSE", "NASDAQ"],
        "일본": ["PRIME", "STANDARD"],
        "Unclassified": ["Unclassified"],
    }
    sectors = [f"Sector_{i}" for i in range(1, 6)] + ["Unclassified"]
    industries = [f"Industry_{i}" for i in range(1, 11)] + ["Unclassified"]
    subinds = [f"SubInd_{i}" for i in range(1, 16)] + ["Unclassified"]
    themes = [f"Theme_{i}" for i in range(1, 5)] + ["Unclassified"]
    techs = [f"Tech_{i}" for i in range(1, 9)] + ["Unclassified"]
    subtechs = [f"SubTech_{i}" for i in range(1, 13)] + ["Unclassified"]

    def pick(lst):
        return rng.choice(lst)

    rows = []
    for _ in range(n):
        country = pick(countries)
        market = pick(markets_by_country[country])
        sector, industry, subind = pick(sectors), pick(industries), pick(subinds)
        theme, tech, subtech = pick(themes), pick(techs), pick(subtechs)
        row = dict(
            Country=country,
            Market=market,
            Sector=sector,
            Industry=industry,
            Sub_industry=subind,
            Theme=theme,
            Technology=tech,
            Sub_Technology=subtech,
            PER=rng.normal(15, 5) if rng.random() > 0.07 else np.nan,
            PBR=rng.normal(1.5, 0.4),
            EV_EBITDA=rng.normal(10, 3),
            ROE=rng.normal(0.12, 0.05),
            EBIT=rng.normal(80_000, 30_000),
            EBITDA=rng.normal(110_000, 35_000),
            Sales=rng.normal(500_000, 200_000),
            Assets=rng.normal(1_000_000, 400_000),
            Book=rng.normal(600_000, 250_000),
            Liabilities=rng.normal(400_000, 180_000),
            Net_Income=rng.normal(50_000, 20_000),
        )
        if rng.random() < 0.05:
            row["Net_Income"] = -abs(row["Net_Income"])
        if rng.random() < 0.03:
            row["Sales"] = 0
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["Market"] != "Unclassified"].copy()
    df.loc[df["Country"] == "Unclassified", "Country"] = np.nan
    return df

DF_RAW = load_data()

# ---------------------------------------------------------------------------
# 2. Sidebar: Filters & Controls
# ---------------------------------------------------------------------------
st.sidebar.header("상장시장")
market_options = [
    "전체",
    "한국 전체",
    "KOSPI",
    "KOSDAQ",
    "미국 전체",
    "NYSE",
    "NASDAQ",
    "일본 전체",
    "PRIME",
    "STANDARD",
]
market_sel = st.sidebar.selectbox("", market_options, label_visibility="collapsed")

country_filter, market_filter = None, None
if "한국" in market_sel:
    country_filter = "한국"
if "미국" in market_sel:
    country_filter = "미국"
if "일본" in market_sel:
    country_filter = "일본"
if market_sel not in ["전체", "한국 전체", "미국 전체", "일본 전체"]:
    market_filter = market_sel

st.sidebar.markdown("---")
st.sidebar.subheader("EMSEC · 행(Row)")
st.sidebar.text("Sector")
sector_sel = st.sidebar.selectbox("", ["전체"] + sorted(DF_RAW.Sector.unique()), label_visibility="collapsed", key="sector_sel")
industry_pool = (
    sorted(DF_RAW.Industry.unique())
    if sector_sel == "전체"
    else sorted(DF_RAW.loc[DF_RAW.Sector == sector_sel, "Industry"].unique())
)
st.sidebar.text("Industry")
industry_sel = st.sidebar.selectbox("", ["전체"] + industry_pool, label_visibility="collapsed", key="industry_sel")
exclude_unclassified_emsec = st.sidebar.checkbox("Unclassified 제외", True, key="exclude_uc_emsec")
exclude_empty_emsec = st.sidebar.checkbox("Empty(0/NaN) 제외", True, key="exclude_empty_emsec")

st.sidebar.markdown("---")
st.sidebar.subheader("EMTEC · 열(Column)")
st.sidebar.text("Theme")
theme_sel = st.sidebar.selectbox("", ["전체"] + sorted(DF_RAW.Theme.unique()), label_visibility="collapsed", key="theme_sel")
tech_pool = (
    sorted(DF_RAW.Technology.unique())
    if theme_sel == "전체"
    else sorted(DF_RAW.loc[DF_RAW.Theme == theme_sel, "Technology"].unique())
)
st.sidebar.text("Technology")
tech_sel = st.sidebar.selectbox("", ["전체"] + tech_pool, label_visibility="collapsed", key="tech_sel")
exclude_unclassified_emtec = st.sidebar.checkbox("Unclassified 제외", True, key="exclude_uc_emtec")
exclude_empty_emtec = st.sidebar.checkbox("Empty(0/NaN) 제외", True, key="exclude_empty_emtec")

st.sidebar.markdown("---")
st.sidebar.subheader("계측값 선택")
group_sel = st.sidebar.selectbox("", ["기업", "비교가치 멀티플", "재무비율"], label_visibility="collapsed")
metric_main, metric_mode, base_col, agg_func, filter_option = None, None, None, None, None
allow_subtotal = True

if group_sel == "기업":
    corp_first = st.sidebar.selectbox("", ["기업수", "0이하 비율"], label_visibility="collapsed")
    if corp_first == "기업수":
        metric_main = "기업수"
        metric_mode = st.sidebar.selectbox("", ["결측 포함", "결측 비율"], label_visibility="collapsed")
    else:
        metric_main = "0이하비율"
        base_col_map = {
            "순이익": "Net_Income",
            "EBITDA": "EBITDA",
            "매출": "Sales",
            "자산총계": "Assets",
            "순자산": "Book",
        }
        corp_base = st.sidebar.selectbox("", list(base_col_map.keys()), label_visibility="collapsed")
        base_col = base_col_map[corp_base]
        metric_mode = st.sidebar.selectbox("", ["결측 포함", "결측 제외"], label_visibility="collapsed")
elif group_sel == "비교가치 멀티플":
    metric_main = st.sidebar.selectbox("", ["PER", "PBR", "EV_EBITDA"], label_visibility="collapsed")
    agg_func = st.sidebar.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed")
    filter_option = st.sidebar.selectbox("", ["전체", "최상위 10% 제외", "0이하 제외", "모두 제외"], label_visibility="collapsed")
    allow_subtotal = agg_func == "AGG"
else:
    metric_main = st.sidebar.selectbox(
        "",
        ["ROE", "영업이익률", "EBITDA/Sales", "총자산이익률", "자산회전율", "자기자본비율", "부채비율"],
        label_visibility="collapsed",
    )
    agg_func = st.sidebar.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed")
    filter_option = st.sidebar.selectbox("", ["전체", "0이하 제외"], label_visibility="collapsed")
    allow_subtotal = agg_func == "AGG"

# ---------------------------------------------------------------------------
# 3. Data Processing and Filtering
# ---------------------------------------------------------------------------
DF = DF_RAW.copy()
if country_filter:
    DF = DF[DF.Country == country_filter]
if market_filter:
    DF = DF[DF.Market == market_filter]
if sector_sel != "전체":
    DF = DF[DF.Sector == sector_sel]
if industry_sel != "전체":
    DF = DF[DF.Industry == industry_sel]
if exclude_unclassified_emsec:
    DF = DF[(DF.Sector != "Unclassified") & (DF.Industry != "Unclassified") & (DF.Sub_industry != "Unclassified")]
if theme_sel != "전체":
    DF = DF[DF.Theme == theme_sel]
if tech_sel != "전체":
    DF = DF[DF.Technology == tech_sel]
if exclude_unclassified_emtec:
    DF = DF[(DF.Theme != "Unclassified") & (DF.Technology != "Unclassified") & (DF.Sub_Technology != "Unclassified")]

def calculate_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    sales = df_calc["Sales"].replace(0, np.nan)
    assets = df_calc["Assets"].replace(0, np.nan)
    book = df_calc["Book"].replace(0, np.nan)
    df_calc["영업이익률"] = df_calc["EBIT"] / sales
    df_calc["EBITDA/Sales"] = df_calc["EBITDA"] / sales
    df_calc["총자산이익률"] = df_calc["Net_Income"] / assets
    df_calc["자산회전율"] = sales / assets
    df_calc["자기자본비율"] = book / assets
    df_calc["부채비율"] = df_calc["Liabilities"] / book
    return df_calc

if group_sel == "재무비율":
    DF = calculate_financial_ratios(DF)

if exclude_empty_emsec or exclude_empty_emtec:
    if group_sel != "기업":
        metric_col_to_check = base_col if base_col else metric_main
        if metric_col_to_check in DF.columns:
            DF[metric_col_to_check] = DF[metric_col_to_check].replace(0, np.nan)
            DF = DF.dropna(subset=[metric_col_to_check])

# ---------------------------------------------------------------------------
# 4. Aggregation Utilities
# ---------------------------------------------------------------------------
def harmonic_mean(s: pd.Series) -> float:
    arr = s.dropna()
    arr = arr[arr > 0]
    return len(arr) / np.sum(1.0 / arr) if len(arr) > 0 else np.nan

def apply_filter(s: pd.Series, filter_opt: str) -> pd.Series:
    s2 = s.dropna()
    if filter_opt == "최상위 10% 제외":
        k = int(len(s2) * 0.1)
        s2 = s2.sort_values().iloc[:-k] if k > 0 else s2
    elif filter_opt == "0이하 제외":
        s2 = s2[s2 > 0]
    elif filter_opt == "모두 제외":
        k = int(len(s2) * 0.1)
        s2 = s2.sort_values().iloc[:-k] if k > 0 else s2
        s2 = s2[s2 > 0]
    return s2

def aggregator(s: pd.Series) -> float:
    if group_sel == "기업":
        if metric_main == "기업수":
            return len(s) if metric_mode == "결측 포함" else s.isna().sum() / len(s) if len(s) > 0 else np.nan
        arr = pd.to_numeric(s, errors="coerce")
        if metric_mode == "결측 제외":
            arr = arr.dropna()
        total = len(arr)
        return (arr <= 0).sum() / total if total > 0 else np.nan
    arr = apply_filter(s, filter_option)
    if len(arr) == 0:
        return np.nan
    if agg_func == "AVG":
        return arr.mean()
    if agg_func == "MED":
        return arr.median()
    if agg_func == "HRM":
        return harmonic_mean(arr)
    if agg_func == "AGG":
        return arr.sum()
    return np.nan

# ---------------------------------------------------------------------------
# 5. Pivot & Heatmap
# ---------------------------------------------------------------------------
st.title("EMSEC × EMTEC Heatmap")
crumbs = [
    market_sel if market_sel != "전체" else "ALL",
    sector_sel if sector_sel != "전체" else "ALL",
    theme_sel if theme_sel != "전체" else "ALL",
]
st.markdown(f"<div style='font-size:0.9rem;color:#666;'>선택 경로 • {' > '.join(crumbs)}</div>", unsafe_allow_html=True)

values_col = base_col if group_sel == "기업" and metric_main == "0이하비율" else metric_main if group_sel != "기업" else "Country"
if group_sel != "기업" and values_col not in DF.columns:
    st.warning(f"선택된 '{values_col}' 지표를 계산할 수 없습니다. 데이터 필터링 조건을 확인해주세요.")
    st.stop()
if DF.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
    st.stop()

pivot_main = pd.pivot_table(
    DF, values=values_col, index="Sub_industry", columns="Sub_Technology",
    aggfunc=aggregator, fill_value=np.nan, observed=True
)
pivot_counts = pd.pivot_table(DF, index="Sub_industry", columns="Sub_Technology", aggfunc="size",
                              fill_value=0, observed=True)
if pivot_main.empty:
    st.warning("피벗 테이블을 생성할 수 없습니다. 조건에 맞는 데이터가 부족합니다.")
    st.stop()

x_labels = pivot_main.columns.tolist()
y_labels = pivot_main.index.tolist()
z_data = pivot_main.values
z_main = z_data
custom_main = pivot_counts.reindex(index=y_labels, columns=x_labels).fillna(0).values

z_sub = z_grd = custom_sub = custom_grd = None
z_combined = z_main

if allow_subtotal:
    row_subtotals = DF.groupby("Sub_industry")[values_col].agg(aggregator).reindex(y_labels)
    col_subtotals = DF.groupby("Sub_Technology")[values_col].agg(aggregator).reindex(x_labels)
    grand_subtotal = aggregator(DF[values_col])
    row_counts = DF.groupby("Sub_industry").size().reindex(y_labels)
    col_counts = DF.groupby("Sub_Technology").size().reindex(x_labels)
    grand_count = len(DF)

    z_sub = np.full((len(y_labels) + 1, len(x_labels) + 1), np.nan)
    z_sub[:-1, -1] = row_subtotals.values
    z_sub[-1, :-1] = col_subtotals.values
    z_grd = np.full((len(y_labels) + 1, len(x_labels) + 1), np.nan)
    z_grd[-1, -1] = grand_subtotal

    z_main_resized = np.full_like(z_sub, np.nan)
    z_main_resized[:-1, :-1] = z_main
    z_main = z_main_resized

    x_labels.append("Subtotal")
    y_labels.append("Subtotal")

    z_combined = np.where(np.isnan(z_main), z_sub, z_main)
    z_combined[-1, -1] = grand_subtotal

    custom_main_resized = np.full_like(z_main, np.nan)
    custom_main_resized[:-1, :-1] = custom_main
    custom_main = custom_main_resized

    custom_sub = np.full_like(z_sub, np.nan)
    custom_sub[:-1, -1] = row_counts.values
    custom_sub[-1, :-1] = col_counts.values
    custom_grd = np.full_like(z_grd, np.nan)
    custom_grd[-1, -1] = grand_count

if group_sel == "기업" and metric_main == "기업수" and metric_mode == "결측 포함":
    fmt = lambda v: f"{v:,.0f}" if pd.notna(v) else ""
elif group_sel == "기업" or (group_sel == "재무비율" and agg_func != "AGG"):
    fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
else:
    fmt = lambda v: f"{v:,.2f}" if pd.notna(v) else ""
text_labels = [[fmt(val) for val in row] for row in z_combined]

# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------
COLOR_SCALE_MAIN = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5",
                    "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#0d47a1"]
COLOR_SCALE_SUBTOTAL = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
COLOR_GRAND_TOTAL = [[0, "#a6a6a6"], [1, "#a6a6a6"]]

fig = go.Figure()

# TRACE 1: Main data (bottom layer)
fig.add_trace(
    go.Heatmap(
        z=z_main,
        x=x_labels,
        y=y_labels,
        colorscale=COLOR_SCALE_MAIN,
        colorbar=dict(title=metric_main),
        hovertemplate="<b>%{y}</b> / %{x}<br>값: %{z:.3f}<br>기업수: %{customdata:.0f}<extra></extra>",
        customdata=custom_main,
        xgap=1,
        ygap=1,
        name="Data",
        hoverongaps=False,
    )
)

# TRACE 2: Subtotals (middle layer)
if allow_subtotal and z_sub is not None:
    fig.add_trace(
        go.Heatmap(
            z=z_sub,
            x=x_labels,
            y=y_labels,
            colorscale=COLOR_SCALE_SUBTOTAL,
            showscale=False,
            hovertemplate="<b>%{y}</b> / %{x}<br><b>Subtotal</b>: %{z:.3f}<br>기업수: %{customdata:.0f}<extra></extra>",
            customdata=custom_sub,
            xgap=1,
            ygap=1,
            name="Subtotal",
            hoverongaps=False,
        )
    )

# TRACE 3: Grand Total (top layer)
if allow_subtotal and z_grd is not None:
    fig.add_trace(
        go.Heatmap(
            z=z_grd,
            x=x_labels,
            y=y_labels,
            colorscale=COLOR_GRAND_TOTAL,
            showscale=False,
            hovertemplate="<b>Grand Total</b><br>값: %{z:.3f}<br>기업수: %{customdata:.0f}<extra></extra>",
            customdata=custom_grd,
            xgap=1,
            ygap=1,
            name="Grand Total",
            hoverongaps=False,
        )
    )

annotations = []
for n, row in enumerate(z_combined):
    for m, val in enumerate(row):
        if pd.notna(val):
            is_grand = allow_subtotal and n == len(y_labels) - 1 and m == len(x_labels) - 1
            is_sub = allow_subtotal and (n == len(y_labels) - 1 or m == len(x_labels) - 1)
            color = "white" if (is_grand or not is_sub) else "black"
            annotations.append(
                go.layout.Annotation(
                    text=text_labels[n][m],
                    x=x_labels[m],
                    y=y_labels[n],
                    xref="x1",
                    yref="y1",
                    showarrow=False,
                    font=dict(color=color),
                )
            )

fig.update_layout(
    annotations=annotations,
    height=max(650, len(y_labels) * 35),
    margin=dict(l=50, r=50, t=80, b=50),
    xaxis=dict(showgrid=False, side="top"),
    yaxis=dict(showgrid=False, autorange="reversed"),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"전체 기업 수: {len(DF):,}")
