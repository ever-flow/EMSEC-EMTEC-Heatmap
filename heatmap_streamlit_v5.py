# -*- coding: utf-8 -*-
"""Streamlit · EMSEC × EMTEC Heatmap Dashboard (v2.7)
초기: 2025-06-18 — 사이드바 최적화 2차
* '필터' expander 삭제 → 필터 컨트롤을 사이드바에 바로 배치
* 메뉴 expander 미사용 상태 유지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Callable

st.set_page_config(
    page_title="EMSEC × EMTEC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

###############################################################################
# 0. Global CSS (간격‧폰트)
###############################################################################
st.markdown(
    """
    <style>
        /* Sidebar spacing */
        [data-testid="stSidebar"] section div {margin-top:2px;margin-bottom:2px}
        /* Selectbox label 폰트 & 간격 */
        label {font-size:0.82rem;margin-bottom:0rem;}
        /* Grand-total cell 텍스트 색상 → 흰 */
        g[class*="hoverlayer"] text {color:white !important}
    </style>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# 1. Synthetic Data Generator (≈1 000 rows)
###############################################################################
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
        row = dict(
            Country=country,
            Market=market,
            Sector=pick(sectors),
            Industry=pick(industries),
            Sub_industry=pick(subinds),
            Theme=pick(themes),
            Technology=pick(techs),
            Sub_Technology=pick(subtechs),
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

DF_RAW = load_data(n = 10000)

###############################################################################
# 2. UI — Tabs & Sidebar Filters
###############################################################################
main_tab, other1_tab, other2_tab = st.tabs(["Heatmap", "(WIP) 시각화A", "(WIP) 시각화B"])

with main_tab:
    # --- Sidebar Filters (expander 제거) ---
    with st.sidebar:
        # 상장시장
        st.markdown("**상장시장**")
        market_options = [
            "전체",
            "한국 전체", "KOSPI", "KOSDAQ",
            "미국 전체", "NYSE", "NASDAQ",
            "일본 전체", "PRIME", "STANDARD",
        ]
        market_sel = st.selectbox("", market_options, label_visibility="collapsed")

        country_filter = market_filter = None
        if "한국" in market_sel:
            country_filter = "한국"
        elif "미국" in market_sel:
            country_filter = "미국"
        elif "일본" in market_sel:
            country_filter = "일본"
        if market_sel not in ["전체", "한국 전체", "미국 전체", "일본 전체"]:
            market_filter = market_sel

        # EMSEC Row
        st.markdown("**Sector > Industry**")
        sector_sel = st.selectbox("", ["전체"] + sorted(DF_RAW.Sector.unique()),
                                  label_visibility="collapsed", key="sector_sel")
        industry_pool = (
            sorted(DF_RAW.Industry.unique()) if sector_sel == "전체"
            else sorted(DF_RAW.loc[DF_RAW.Sector == sector_sel, "Industry"].unique())
        )
        industry_sel = st.selectbox("", ["전체"] + industry_pool,
                                    label_visibility="collapsed", key="industry_sel")
        exclude_unclassified_emsec = st.checkbox("Unclassified 제외", True, key="exclude_uc_emsec")
        exclude_empty_emsec = st.checkbox("Empty(0/NaN) 제외", True, key="exclude_empty_emsec")

        # EMTEC Column
        st.markdown("**Theme > Technology**")
        theme_sel = st.selectbox("", ["전체"] + sorted(DF_RAW.Theme.unique()),
                                 label_visibility="collapsed", key="theme_sel")
        tech_pool = (
            sorted(DF_RAW.Technology.unique()) if theme_sel == "전체"
            else sorted(DF_RAW.loc[DF_RAW.Theme == theme_sel, "Technology"].unique())
        )
        tech_sel = st.selectbox("", ["전체"] + tech_pool,
                                label_visibility="collapsed", key="tech_sel")
        exclude_unclassified_emtec = st.checkbox("Unclassified 제외", True, key="exclude_uc_emtec")
        exclude_empty_emtec = st.checkbox("Empty(0/NaN) 제외", True, key="exclude_empty_emtec")

        # Metric 선택
        st.markdown("**계측값 선택**")
        group_sel = st.selectbox("", ["기업", "비교가치 멀티플", "재무비율"],
                                 label_visibility="collapsed")

        metric_main = metric_mode = base_col = agg_func = filter_option = None
        allow_subtotal = True

        if group_sel == "기업":
            corp_first = st.selectbox("", ["기업수", "0이하 비율"], label_visibility="collapsed")
            if corp_first == "기업수":
                metric_main = "기업수"
                metric_mode = st.selectbox("", ["결측 포함", "결측 비율"], label_visibility="collapsed")
            else:
                metric_main = "0이하비율"
                base_col_map = {
                    "순이익": "Net_Income",
                    "EBITDA": "EBITDA",
                    "매출": "Sales",
                    "자산총계": "Assets",
                    "순자산": "Book",
                }
                corp_base = st.selectbox("", list(base_col_map.keys()), label_visibility="collapsed")
                base_col = base_col_map[corp_base]
                metric_mode = st.selectbox("", ["결측 포함", "결측 제외"], label_visibility="collapsed")
        elif group_sel == "비교가치 멀티플":
            metric_main = st.selectbox("", ["PER", "PBR", "EV_EBITDA"], label_visibility="collapsed")
            agg_func = st.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed")
            filter_option = st.selectbox("", ["전체", "최상위 10% 제외", "0이하 제외", "모두 제외"],
                                         label_visibility="collapsed")
            allow_subtotal = agg_func == "AGG"
        else:
            metric_main = st.selectbox("", [
                "ROE", "영업이익률", "EBITDA/Sales", "총자산이익률",
                "자산회전율", "자기자본비율", "부채비율",
            ], label_visibility="collapsed")
            agg_func = st.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed")
            filter_option = st.selectbox("", ["전체", "0이하 제외"], label_visibility="collapsed")
            allow_subtotal = agg_func == "AGG"

    ###############################################################################
    # 3. Data Processing ---------------------------------------------------------
    ###############################################################################
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
        DF = DF[(DF.Sector != "Unclassified") & (DF.Industry != "Unclassified") &
                (DF.Sub_industry != "Unclassified")]
    if theme_sel != "전체":
        DF = DF[DF.Theme == theme_sel]
    if tech_sel != "전체":
        DF = DF[DF.Technology == tech_sel]
    if exclude_unclassified_emtec:
        DF = DF[(DF.Theme != "Unclassified") & (DF.Technology != "Unclassified") &
                (DF.Sub_Technology != "Unclassified")]

    def calc_ratios(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        sales = d["Sales"].replace(0, np.nan)
        assets = d["Assets"].replace(0, np.nan)
        book = d["Book"].replace(0, np.nan)
        d["영업이익률"] = d["EBIT"] / sales
        d["EBITDA/Sales"] = d["EBITDA"] / sales
        d["총자산이익률"] = d["Net_Income"] / assets
        d["자산회전율"] = sales / assets
        d["자기자본비율"] = book / assets
        d["부채비율"] = d["Liabilities"] / book
        return d

    if group_sel == "재무비율":
        DF = calc_ratios(DF)

    if exclude_empty_emsec or exclude_empty_emtec:
        if group_sel != "기업":
            chk_col = base_col if base_col else metric_main
            DF[chk_col] = DF[chk_col].replace(0, np.nan)
            DF = DF.dropna(subset=[chk_col])

    ###############################################################################
    # 4. Aggregation helpers -----------------------------------------------------
    ###############################################################################
    def harmonic_mean(s: pd.Series) -> float:
        arr = s.dropna()
        arr = arr[arr > 0]
        return len(arr) / np.sum(1.0 / arr) if len(arr) else np.nan

    def apply_filter(s: pd.Series, opt: str) -> pd.Series:
        s2 = s.dropna()
        if opt == "최상위 10% 제외":
            k = int(len(s2) * 0.1)
            s2 = s2.sort_values().iloc[:-k] if k else s2
        elif opt == "0이하 제외":
            s2 = s2[s2 > 0]
        elif opt == "모두 제외":
            k = int(len(s2) * 0.1)
            s2 = s2.sort_values().iloc[:-k] if k else s2
            s2 = s2[s2 > 0]
        return s2

    def aggregator(s: pd.Series) -> float:
        if group_sel == "기업":
            if metric_main == "기업수":
                return len(s) if metric_mode == "결측 포함" else s.isna().sum() / len(s) if len(s) else np.nan
            arr = pd.to_numeric(s, errors="coerce")
            arr = arr if metric_mode == "결측 포함" else arr.dropna()
            return (arr <= 0).sum() / len(arr) if len(arr) else np.nan
        arr = apply_filter(s, filter_option)
        if not len(arr):
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

    ###############################################################################
    # 5. Pivot & Sub-/Grand-Totals ---------------------------------------------
    ###############################################################################
    values_col = (
        base_col if group_sel == "기업" and metric_main == "0이하비율" else
        metric_main if group_sel != "기업" else "Country"
    )
    if group_sel != "기업" and values_col not in DF.columns:
        st.warning(f"'{values_col}' 지표를 계산할 수 없습니다.")
        st.stop()
    if DF.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
        st.stop()

    pivot_main = pd.pivot_table(
        DF,
        values=values_col,
        index="Sub_industry",
        columns="Sub_Technology",
        aggfunc=aggregator,
        fill_value=np.nan,
        observed=True,
    )
    pivot_counts = pd.pivot_table(
        DF, index="Sub_industry", columns="Sub_Technology",
        aggfunc="size", fill_value=0, observed=True
    )

    if pivot_main.empty:
        st.warning("피벗 테이블을 생성할 수 없습니다.")
        st.stop()

    x_orig, y_orig = pivot_main.columns.tolist(), pivot_main.index.tolist()
    z_core = pivot_main.values
    cnt_core = pivot_counts.reindex(index=y_orig, columns=x_orig).fillna(0).values

    # Sub-/Grand-Totals 계산
    if allow_subtotal:
        row_tot = DF.groupby("Sub_industry")[values_col].agg(aggregator).reindex(y_orig)
        col_tot = DF.groupby("Sub_Technology")[values_col].agg(aggregator).reindex(x_orig)
        grand_tot = aggregator(DF[values_col])
        row_cnt = DF.groupby("Sub_industry").size().reindex(y_orig)
        col_cnt = DF.groupby("Sub_Technology").size().reindex(x_orig)
        grand_cnt = len(DF)

        # 새 행·열 삽입 (좌/상)
        x_labels = ["Subtotal"] + x_orig
        y_labels = ["Subtotal"] + y_orig
        size = (len(y_labels), len(x_labels))

        z_main = np.full(size, np.nan)
        z_main[1:, 1:] = z_core
        cnt_main = np.full(size, np.nan)
        cnt_main[1:, 1:] = cnt_core

        z_sub = np.full(size, np.nan)
        z_sub[0, 1:] = col_tot.values
        z_sub[1:, 0] = row_tot.values
        cnt_sub = np.full(size, np.nan)
        cnt_sub[0, 1:] = col_cnt.values
        cnt_sub[1:, 0] = row_cnt.values

        z_grd = np.full(size, np.nan)
        z_grd[0, 0] = grand_tot
        cnt_grd = np.full(size, np.nan)
        cnt_grd[0, 0] = grand_cnt

        z_comb = np.where(np.isnan(z_main), z_sub, z_main)
        z_comb[0, 0] = grand_tot
    else:
        x_labels, y_labels = x_orig, y_orig
        z_main, cnt_main = z_core, cnt_core
        z_comb = z_main
        z_sub = z_grd = cnt_sub = cnt_grd = None

    # 값 포맷
    if group_sel == "기업" and metric_main == "기업수" and metric_mode == "결측 포함":
        fmt = lambda v: f"{v:,.0f}" if pd.notna(v) else ""
    elif group_sel == "기업" or (group_sel == "재무비율" and (agg_func != "AGG")):
        fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
    else:
        fmt = lambda v: f"{v:,.2f}" if pd.notna(v) else ""
    txt = [[fmt(v) for v in row] for row in z_comb]

    ###############################################################################
    # 6. Plot --------------------------------------------------------------------
    ###############################################################################
    st.markdown("#### EMSEC × EMTEC Heatmap")
    crumbs = [
        market_sel if market_sel != "전체" else "ALL",
        sector_sel if sector_sel != "전체" else "ALL",
        theme_sel if theme_sel != "전체" else "ALL",
    ]
    st.markdown(
        f"<div style='font-size:0.85rem;color:#666;'>선택 경로 • {' > '.join(crumbs)}</div>",
        unsafe_allow_html=True,
    )

    MAIN_CS = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5",
               "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#0d47a1"]
    SUB_CS = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
    GT_CS = [[0, "#000000"], [1, "#000000"]]

    fig = go.Figure()
    # Main
    fig.add_trace(
        go.Heatmap(
            z=z_main, x=x_labels, y=y_labels,
            colorscale=MAIN_CS, colorbar=dict(title=metric_main),
            customdata=cnt_main,
            hovertemplate="<b>%{y}</b> / %{x}<br>값: %{z:.3f}<br>기업수: %{customdata:.0f}<extra></extra>",
            xgap=1, ygap=1, hoverongaps=False,
        )
    )
    # Subtotals
    if allow_subtotal:
        fig.add_trace(
            go.Heatmap(
                z=z_sub, x=x_labels, y=y_labels,
                colorscale=SUB_CS, showscale=False,
                customdata=cnt_sub,
                hovertemplate="<b>%{y}</b> / %{x}<br><b>Subtotal</b>: %{z:.3f}<br>기업수: %{customdata:.0f}<extra></extra>",
                xgap=1, ygap=1, hoverongaps=False,
            )
        )
        # Grand total
        fig.add_trace(
            go.Heatmap(
                z=z_grd, x=x_labels, y=y_labels,
                colorscale=GT_CS, showscale=False,
                customdata=cnt_grd,
                hovertemplate="<b>Grand Total</b><br>값: %{z:.3f}<br>기업수: %{customdata:.0f}<extra></extra>",
                xgap=1, ygap=1, hoverongaps=False,
            )
        )

    # Annotations
    annotations = []
    for r_idx, row in enumerate(z_comb):
        for c_idx, v in enumerate(row):
            if pd.isna(v):
                continue
            is_grand = allow_subtotal and r_idx == 0 and c_idx == 0
            is_sub = allow_subtotal and (r_idx == 0 or c_idx == 0)
            color = "white" if is_grand else ("black" if is_sub else "white")
            annotations.append(
                go.layout.Annotation(
                    text=txt[r_idx][c_idx],
                    x=x_labels[c_idx], y=y_labels[r_idx],
                    xref="x1", yref="y1", showarrow=False,
                    font=dict(color=color)
                )
            )

    fig.update_layout(
    annotations=annotations,
    height=max(650, len(y_labels) * 35),
    margin=dict(l=40, r=40, t=20, b=40),
    xaxis=dict(side="top", showgrid=False),
    yaxis=dict(
        autorange="reversed",
        showgrid=False,
        categoryorder="array",        # 순서를 배열 그대로 사용
        categoryarray=y_labels        # y_labels 순서로 카테고리 지정
    ),
    showlegend=False,
    )


    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"최종 필터링 후 데이터 행 개수: {len(DF):,}")

with other1_tab:
    st.info("추가 시각화 A — 준비 중입니다.")
with other2_tab:
    st.info("추가 시각화 B — 준비 중입니다.")
