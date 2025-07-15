# -*- coding: utf-8 -*-
"""
Streamlit · EMSEC × EMTEC 〈시각화 02 – 규모 변수〉
ver 1.8 · 2025-07-10
(Subtotal Tooltip 0 표기 버그 수정 – text 배열 주입)

주요 변경
─────────────────────────────────────────────────────────
1. 각 Heatmap Trace에 text 배열(콤마 포맷)을 주입하고 hovertemplate에서 %{text} 사용
2. hoverinfo="skip" 대신 hovertemplate 지정으로 x/y 라벨 유지
3. 기타 로직·UI 동일
"""

###############################################################################
# 0. 라이브러리 & 페이지
###############################################################################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
from typing import List

st.set_page_config(
    page_title="EMSEC × EMTEC 규모 시각화",
    layout="wide",
    initial_sidebar_state="expanded",
)

###############################################################################
# 1. 환율 & 보조 함수
###############################################################################
EXCHANGE_RATES = {
    '한국': 1380.0, '미국': 1.0, '일본': 157.0, 'Unclassified': 1.0
}

def convert_to_usd(val, country):
    return np.nan if pd.isna(val) else val / EXCHANGE_RATES.get(country, 1.0)

def parse_emtec_list(txt: str) -> List[str]:
    try:
        return [] if pd.isna(txt) or txt in ('[]', '') else ast.literal_eval(txt)
    except Exception:
        return []

###############################################################################
# 2. 데이터 로드 & 전처리
###############################################################################
@st.cache_data(show_spinner=True)
def load_raw() -> pd.DataFrame:
    src = "C:/Users/gimyo/OneDrive/Desktop/heatmap_data_with_SE_v2.xlsx"
    df  = pd.read_excel(src, sheet_name="Sheet1")

    emsec_cols = [f"EMSEC{i}" for i in range(1, 6)]
    df = df[df[emsec_cols].notna().any(axis=1)].copy()

    if "Company" not in df.columns:
        df["Company"] = df["ticker"]

    mk2cty = {
        "KOSPI":"한국","KOSDAQ":"한국","KOSDAQ GLOBAL":"한국",
        "NASDAQ":"미국",
        "Prime (Domestic Stocks)":"일본",
        "Standard (Domestic Stocks)":"일본",
        "Prime (Foreign Stocks)":"일본"
    }
    df["Country"] = df["market"].map(mk2cty).fillna("Unclassified")
    df["Market"]  = df["market"].replace("KOSDAQ GLOBAL","KOSDAQ")
    return df

def create_multi_class(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _,r in df.iterrows():
        emsecs=[{"Sector":r[f"EMSEC{i}_Sector"],
                 "Industry":r[f"EMSEC{i}_Industry"],
                 "Sub_industry":r[f"EMSEC{i}"]}
                for i in range(1,6) if pd.notna(r[f"EMSEC{i}"])]
        l1,l2,l3 = map(parse_emtec_list,
                       (r["EMTEC_LEVEL1"],r["EMTEC_LEVEL2"],r["EMTEC_LEVEL3"]))
        emtecs=[]
        if l1:
            for t1 in l1:
                if l2:
                    for t2 in l2:
                        if l3:
                            emtecs += [{"Theme":t1,"Technology":t2,"Sub_Technology":t3} for t3 in l3]
                        else:
                            emtecs.append({"Theme":t1,"Technology":t2,"Sub_Technology":"Unclassified"})
                else:
                    emtecs.append({"Theme":t1,"Technology":"Unclassified","Sub_Technology":"Unclassified"})
        if not emtecs:
            emtecs=[{"Theme":"Unclassified","Technology":"Unclassified","Sub_Technology":"Unclassified"}]

        for e in emsecs:
            for t in emtecs:
                new=r.copy()
                for k,v in {**e,**t}.items():
                    new[k]=v
                rows.append(new)
    return pd.DataFrame(rows)

def add_fin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    periods=["LTM","LTM-1","LTM-2","LTM-3"]
    if "Market Cap (2024-12-31)" in df.columns:
        df["Market Cap (2024-12-31)_USD"]=df.apply(
            lambda x:convert_to_usd(x["Market Cap (2024-12-31)"],x["Country"]),axis=1)
    for p in periods:
        for base in ("Revenue","Total Assets"):
            col=f"{base} ({p})"
            if col in df.columns:
                df[f"{col}_USD"]=df.apply(lambda x:convert_to_usd(x[col],x["Country"]),axis=1)
    return df

@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    raw  = load_raw()
    usd  = add_fin_metrics(raw)
    multi= create_multi_class(usd)
    dfs=[]
    for yr in ["LTM","LTM-1","LTM-2","LTM-3"]:
        tmp = multi.copy()
        tmp["Year"]=yr
        dfs.append(tmp)
    return pd.concat(dfs,ignore_index=True)

DF_ALL = load_data()

###############################################################################
# 3. UI (사이드바)
###############################################################################
with st.sidebar:
    year_sel = st.selectbox("기준 연도", ["LTM","LTM-1","LTM-2","LTM-3"])
    market_sel = st.selectbox(
        "상장시장",
        ["전체","한국 전체","KOSPI","KOSDAQ",
         "미국 전체","NASDAQ",
         "일본 전체","Prime (Domestic Stocks)",
         "Standard (Domestic Stocks)","Prime (Foreign Stocks)"]
    )

    country_filter = market_filter = None
    if market_sel == "전체":
        pass
    elif "전체" in market_sel:
        country_filter = market_sel.split()[0]
    else:
        country_filter = (
            "한국" if market_sel in ("KOSPI","KOSDAQ") else
            "미국" if market_sel == "NASDAQ" else
            "일본"
        )
        market_filter = market_sel

    class_type = st.radio("분류 체계", ["EMSEC","EMTEC"], horizontal=True)

    if class_type == "EMSEC":
        sectors = sorted([s for s in DF_ALL.Sector.dropna().unique() if s != 'Unclassified'])
        sector_sel = st.selectbox("Sector", ["전체"] + sectors)

        if sector_sel != "전체":
            indus = sorted(
                DF_ALL.loc[DF_ALL.Sector == sector_sel, "Industry"]
                      .dropna().unique()
            )
            industry_sel = st.selectbox("Industry", ["전체"] + indus)
        else:
            industry_sel = "전체"

        row_level = (
            "Sector" if sector_sel == "전체" else
            "Industry" if industry_sel == "전체" else
            "Sub_industry"
        )
    else:
        themes = sorted([t for t in DF_ALL.Theme.dropna().unique() if t != 'Unclassified'])
        theme_sel = st.selectbox("Theme", ["전체"] + themes)

        if theme_sel != "전체":
            techs = sorted(
                DF_ALL.loc[DF_ALL.Theme == theme_sel, "Technology"]
                      .dropna().unique()
            )
            tech_sel = st.selectbox("Technology", ["전체"] + techs)
        else:
            tech_sel = "전체"

        row_level = (
            "Theme" if theme_sel == "전체" else
            "Technology" if tech_sel == "전체" else
            "Sub_Technology"
        )

    metric_base = {
        "시가총액": "Market Cap (2024-12-31)_USD",
        "자산총계": "Total Assets ({})_USD",
        "매출액"  : "Revenue ({})_USD",
    }
    metric_name = st.selectbox("계측값", list(metric_base.keys()), key="metric_name")
    metric_col = (
        metric_base[metric_name] if metric_name == "시가총액"
        else metric_base[metric_name].format(year_sel)
    )

###############################################################################
# 4. 데이터 필터
###############################################################################
DF = DF_ALL[DF_ALL.Year == year_sel].copy()

if country_filter: DF = DF[DF.Country == country_filter]
if market_filter:  DF = DF[DF.Market  == market_filter]

if class_type == "EMSEC":
    if sector_sel   != "전체": DF = DF[DF.Sector   == sector_sel]
    if industry_sel != "전체": DF = DF[DF.Industry == industry_sel]
else:
    if theme_sel != "전체": DF = DF[DF.Theme      == theme_sel]
    if tech_sel  != "전체": DF = DF[DF.Technology  == tech_sel]

if metric_col not in DF.columns:
    st.error(f"'{metric_name}' 열이 없습니다.")
    st.stop()

DF["metric_bil"] = DF[metric_col] / 1e9
DF = DF[DF["metric_bil"].notna()]
DF = DF[DF["metric_bil"] >= 0]
if not DF.empty:
    max_th = DF["metric_bil"].quantile(0.999)
    DF = DF[DF["metric_bil"] <= max_th]

if DF.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
    st.stop()

###############################################################################
# 5. 구간(edge) 생성
###############################################################################
valid_vals = DF["metric_bil"]
vl_max = valid_vals.max() if not valid_vals.empty else 0

def make_edges(max_val: float) -> List[float]:
    base=[10,30,60,100,300,600]
    edges=[0]
    if max_val <= 0:
        edges += base[:1]
    else:
        exp=0
        while True:
            factor = 10 ** exp
            for b in base:
                edge = b * factor
                if edge > max_val:
                    edges = sorted(set(edges))
                    return edges + [np.inf]
                edges.append(edge)
            exp += 1
    edges = sorted(set(edges))
    return edges + [np.inf]

bin_edges  = make_edges(vl_max)
bin_labels = ["0~"] + [f"{int(e):,}~" for e in bin_edges[1:-1]]
DF["metric_bin"] = pd.cut(DF["metric_bil"], bins=bin_edges,
                          labels=bin_labels, right=False)

###############################################################################
# 6. 피벗집계
###############################################################################
pivot = (DF.groupby([row_level,"metric_bin"])["Company"]
         .nunique().unstack(fill_value=0)
         .reindex(columns=bin_labels,fill_value=0).astype(int))

if pivot.empty:
    st.warning("조건에 맞는 데이터가 없어 집계표를 생성할 수 없습니다.")
    st.stop()
    
subtotal = pd.DataFrame(pivot.sum()).T
subtotal.index=["Subtotal"]
pivot_full = pd.concat([subtotal,pivot])

###############################################################################
# 7. 히트맵 (text 배열 주입으로 Tooltip 0 문제 해결)
###############################################################################
rows = pivot_full.index.tolist()

# Helper: 숫자를 '1,234' 형식 문자열로 변환
fmt = np.vectorize(lambda x: "" if np.isnan(x) else f"{int(x):,}")

# Trace 1: Subtotal (Greys)
z_sub = pivot_full.astype(float).copy()
z_sub.iloc[1:] = np.nan                      # 데이터 행 투명
text_sub = fmt(z_sub.values)

heat_sub = go.Heatmap(
    z=z_sub.values, x=bin_labels, y=rows,
    colorscale="Greys", xgap=0, ygap=0,
    showscale=False,
    text=text_sub,
    hovertemplate="Subtotal/%{x}<br>기업수 %{text}<extra></extra>",
    zmin=0, zmax=np.nanmax(z_sub.values)
)

# Trace 2: 데이터 (Greens)
z_data = pivot_full.astype(float).copy()
z_data.iloc[0] = np.nan                      # Subtotal 행 투명
text_data = fmt(z_data.values)

heat_data = go.Heatmap(
    z=z_data.values, x=bin_labels, y=rows,
    colorscale="Greens", xgap=0, ygap=0,
    colorbar=dict(title="기업 수"),
    text=text_data,
    hovertemplate="%{y}/%{x}<br>기업수 %{text}<extra></extra>",
    zmin=0, zmax=np.nanmax(z_data.values)
)

fig = go.Figure([heat_sub, heat_data])

# 주석 추가
for r_idx, row_name in enumerate(rows):
    for c_idx, col_name in enumerate(bin_labels):
        val = pivot_full.iat[r_idx, c_idx]
        if val:
            fig.add_annotation(
                x=col_name, y=row_name, text=f"{val:,}",
                showarrow=False, font=dict(color="black", size=12)
            )

fig.update_layout(
    height=max(600, 35 * len(rows)),
    margin=dict(l=40, r=40, t=150, b=40),
    title=dict(
        text=f"{metric_name} 분포 (단위: billion(10억) USD, {year_sel})",
        x=0.5, xanchor="center", yanchor="top", pad=dict(t=20)
    ),
    xaxis=dict(side="top"),
    yaxis=dict(autorange="reversed")
)

fig.update_yaxes(tickvals=rows, ticktext=rows)
st.plotly_chart(fig, use_container_width=True)

###############################################################################
# 8. 부가 정보
###############################################################################
st.caption(
    f"총 고유 기업 수: {DF['Company'].nunique():,}"
    f" | 구간 수: {len(bin_labels)}"
)
with st.expander("📋 원본 집계표 보기", False):
    st.dataframe(pivot_full, use_container_width=True)
