# -*- coding: utf-8 -*-
"""Streamlit · EMSEC × EMTEC Heatmap Dashboard (v3.6 - Refined Metrics & Logic)
수정일: 2025-07-08
* ROE를 일반적인 방법으로 계산 (Net Income / Equity)
* 환율 적용하여 모든 절대값 지표를 USD로 통일
* calculation.txt의 AGG 계산법 통합 및 사용자 선택에 따른 집계 개선
* 요청사항 반영:
    1. 재무비율/멀티플 하위 필터 옵션('0이하 제거', '이상치 제외') 제거
    2. 재무비율에 '시가총액/매출액', '시가총액/영업이익' 변수 추가
    3. '기업수' 로직 구체화 (결측 포함/미포함/비율)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ast
from typing import Callable, List

st.set_page_config(
    page_title="EMSEC × EMTEC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
###############################################################################
# 0. Global CSS
###############################################################################
st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
)
###############################################################################
# 1. Currency Exchange Rates (2024-12-31 기준)
###############################################################################
EXCHANGE_RATES = {
    '한국': 1380.0,    # KRW/USD (2024-12-31 기준)
    '미국': 1.0,       # USD/USD
    '일본': 157.0,     # JPY/USD (2024-12-31 기준)
    'Unclassified': 1.0
}

def convert_to_usd(value, country):
    """해당 국가 화폐를 USD로 환산"""
    if pd.isna(value) or pd.isna(country):
        return value
    rate = EXCHANGE_RATES.get(country, 1.0)
    return value / rate

###############################################################################
# 2. Data Processing with Currency Conversion
###############################################################################
@st.cache_data(show_spinner=True)
def load_real_data() -> pd.DataFrame:
    """실제 데이터를 로드하고 전처리하는 함수"""
    df = pd.read_excel('data/heatmap_data_with_SE_v2.xlsx', sheet_name='Sheet1')
    # EMSEC1~5가 모두 결측인 기업 제외
    emsec_cols = [f'EMSEC{i}' for i in range(1, 6)]
    has_emsec = df[emsec_cols].notna().any(axis=1)
    df = df[has_emsec].copy()
    # 'Company' 열이 없으면 'ticker'로 생성
    if 'Company' not in df.columns:
        df['Company'] = df['ticker']
    # 시장별 국가 매핑
    market_country_map = {
        'KOSPI': '한국', 'KOSDAQ': '한국', 'KOSDAQ GLOBAL': '한국',
        'NASDAQ': '미국',
        'Prime (Domestic Stocks)': '일본', 'Standard (Domestic Stocks)': '일본',
        'Prime (Foreign Stocks)': '일본'
    }
    df['Country'] = df['market'].map(market_country_map).fillna('Unclassified')
    df['Market'] = df['market'].replace('KOSDAQ GLOBAL', 'KOSDAQ')  # KOSDAQ GLOBAL을 KOSDAQ으로 통합
    return df

def parse_emtec_list(emtec_str: str) -> List[str]:
    """EMTEC 리스트 문자열을 파싱하는 함수"""
    try:
        if pd.isna(emtec_str) or emtec_str == '[]':
            return []
        return ast.literal_eval(emtec_str)
    except:
        return []

def create_multiple_classification_data(df: pd.DataFrame) -> pd.DataFrame:
    """EMSEC과 EMTEC의 중복 분류를 처리하여 확장된 데이터 생성"""
    expanded_rows = []
    for idx, row in df.iterrows():
        # EMSEC 데이터 수집 (EMSEC1~5, 모든 분류 포함)
        emsec_classifications = []
        for i in range(1, 6):
            emsec_col = f'EMSEC{i}'
            sector_col = f'EMSEC{i}_Sector'
            industry_col = f'EMSEC{i}_Industry'
            if pd.notna(row[emsec_col]):
                emsec_classifications.append({
                    'sector': row[sector_col],
                    'industry': row[industry_col],
                    'sub_industry': row[emsec_col]
                })
        # EMTEC 데이터 수집
        emtec_level1 = parse_emtec_list(row['EMTEC_LEVEL1'])
        emtec_level2 = parse_emtec_list(row['EMTEC_LEVEL2'])
        emtec_level3 = parse_emtec_list(row['EMTEC_LEVEL3'])
        # EMTEC 조합 생성
        emtec_combinations = []
        if emtec_level1:
            for level1 in emtec_level1:
                if emtec_level2:
                    for level2 in emtec_level2:
                        if emtec_level3:
                            for level3 in emtec_level3:
                                emtec_combinations.append({
                                    'theme': level1,
                                    'technology': level2,
                                    'sub_technology': level3
                                })
                        else:
                            emtec_combinations.append({
                                'theme': level1,
                                'technology': level2,
                                'sub_technology': 'Unclassified'
                            })
                else:
                    emtec_combinations.append({
                        'theme': level1,
                        'technology': 'Unclassified',
                        'sub_technology': 'Unclassified'
                    })
        if not emtec_combinations:
            emtec_combinations = [{
                'theme': 'Unclassified',
                'technology': 'Unclassified',
                'sub_technology': 'Unclassified'
            }]
        # 모든 EMSEC × EMTEC 조합에 대해 행 생성
        for emsec in emsec_classifications:
            for emtec in emtec_combinations:
                new_row = row.copy()
                new_row['Sector'] = emsec['sector']
                new_row['Industry'] = emsec['industry']
                new_row['Sub_industry'] = emsec['sub_industry']
                new_row['Theme'] = emtec['theme']
                new_row['Technology'] = emtec['technology']
                new_row['Sub_Technology'] = emtec['sub_technology']
                expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

def calculate_financial_metrics_with_currency_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """재무 지표 계산 및 환율 적용"""
    df = df.copy()
    # 0으로 나누기 방지를 위한 함수
    def safe_divide(numerator, denominator):
        return numerator / denominator.replace(0, np.nan)
    # 절대값 지표들 (환율 적용 대상)
    absolute_value_columns = [
        'Market Cap (2024-12-31)', 'Enterprise Value (FQ0)'
    ]
    periods = ['LTM', 'LTM-1', 'LTM-2', 'LTM-3']
    # 시점별 절대값 지표들
    for period in periods:
        absolute_value_columns.extend([
            f'Revenue ({period})', f'EBIT ({period})', f'Net Income ({period})',
            f'Total Assets ({period})', f'Equity ({period})', f'Total Liabilities ({period})',
            f'Net Debt ({period})', f'Depreciation ({period})', f'Dividends ({period})',
            f'Net Income After Minority ({period})'
        ])
    # 환율 적용하여 USD로 통일
    for col in absolute_value_columns:
        if col in df.columns:
            df[f'{col}_USD'] = df.apply(lambda row: convert_to_usd(row[col], row['Country']), axis=1)
    # 각 시점별로 계산 (USD 기준)
    for period in periods:
        market_cap_col = 'Market Cap (2024-12-31)_USD'
        # EBITDA 계산 (USD)
        ebit_col_usd = f'EBIT ({period})_USD'
        depreciation_col_usd = f'Depreciation ({period})_USD'
        if ebit_col_usd in df.columns and depreciation_col_usd in df.columns:
            df[f'EBITDA ({period})_USD'] = df[ebit_col_usd] + df[depreciation_col_usd]
        # Enterprise Value 보완 (USD)
        net_debt_col_usd = f'Net Debt ({period})_USD'
        if market_cap_col in df.columns and net_debt_col_usd in df.columns:
            if 'Enterprise Value (FQ0)_USD' not in df.columns:
                df['Enterprise Value (FQ0)_USD'] = df[market_cap_col] + df[net_debt_col_usd].fillna(0)
            else:
                mask = df['Enterprise Value (FQ0)_USD'].isna()
                df.loc[mask, 'Enterprise Value (FQ0)_USD'] = df.loc[mask, market_cap_col] + df.loc[mask, net_debt_col_usd].fillna(0)
        # PER 계산 (USD 기준)
        if market_cap_col in df.columns:
            net_income_col_usd = f'Net Income ({period})_USD'
            if net_income_col_usd in df.columns:
                df[f'PER ({period})'] = safe_divide(df[market_cap_col], df[net_income_col_usd])
        # PBR 계산 (USD 기준)
        if market_cap_col in df.columns:
            equity_col_usd = f'Equity ({period})_USD'
            if equity_col_usd in df.columns:
                df[f'PBR ({period})'] = safe_divide(df[market_cap_col], df[equity_col_usd])
        # EV/EBITDA 계산 (USD 기준)
        ev_col_usd = 'Enterprise Value (FQ0)_USD'
        ebitda_col_usd = f'EBITDA ({period})_USD'
        if ev_col_usd in df.columns and ebitda_col_usd in df.columns:
            df[f'EV_EBITDA ({period})'] = safe_divide(df[ev_col_usd], df[ebitda_col_usd])
        # [NEW] 시가총액/매출액, 시가총액/영업이익 추가
        revenue_col_usd = f'Revenue ({period})_USD'
        if market_cap_col in df.columns and revenue_col_usd in df.columns:
            df[f'시가총액/매출액 ({period})'] = safe_divide(df[market_cap_col], df[revenue_col_usd])
        if market_cap_col in df.columns and ebit_col_usd in df.columns:
            df[f'시가총액/영업이익 ({period})'] = safe_divide(df[market_cap_col], df[ebit_col_usd])

    # 일반적인 ROE 계산 (Net Income / Equity) - 비율이므로 환율 무관
    for period in periods:
        ni_col = f'Net Income ({period})'
        eq_col = f'Equity ({period})'
        # Net Income After Minority가 있으면 우선 사용
        ni_after_minority_col = f'Net Income After Minority ({period})'
        if ni_after_minority_col in df.columns:
            ni_col = ni_after_minority_col
        if ni_col in df.columns and eq_col in df.columns:
            df[f'ROE ({period})'] = safe_divide(df[ni_col], df[eq_col])
    # 기타 재무 비율 계산 (비율이므로 환율 무관)
    for period in periods:
        ebit_col = f'EBIT ({period})'
        revenue_col = f'Revenue ({period})'
        assets_col = f'Total Assets ({period})'
        equity_col = f'Equity ({period})'
        liab_col = f'Total Liabilities ({period})'

        if ebit_col in df.columns and revenue_col in df.columns:
            df[f'영업이익률 ({period})'] = safe_divide(df[ebit_col], df[revenue_col])
        if f'EBITDA ({period})_USD' in df.columns and revenue_col in df.columns:
            df[f'EBITDA/Sales ({period})'] = safe_divide(df.get(f'EBIT ({period})', 0) + df.get(f'Depreciation ({period})', 0), df[revenue_col])
        if f'Net Income ({period})' in df.columns and assets_col in df.columns:
            df[f'총자산이익률 ({period})'] = safe_divide(df[f'Net Income ({period})'], df[assets_col])
        if revenue_col in df.columns and assets_col in df.columns:
            df[f'자산회전율 ({period})'] = safe_divide(df[revenue_col], df[assets_col])
        if equity_col in df.columns and assets_col in df.columns:
            df[f'자기자본비율 ({period})'] = safe_divide(df[equity_col], df[assets_col])
        if liab_col in df.columns and equity_col in df.columns:
            df[f'부채비율 ({period})'] = safe_divide(df[liab_col], df[equity_col])
    return df

def prepare_streamlit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit에서 사용할 수 있는 형태로 데이터 준비"""
    years = ['LTM', 'LTM-1', 'LTM-2', 'LTM-3']
    final_rows = []
    for year in years:
        year_df = df.copy()
        year_df['Year'] = year
        # 해당 연도의 재무 데이터 매핑
        financial_mappings = {
            f'PER ({year})': 'PER',
            f'PBR ({year})': 'PBR',
            f'EV_EBITDA ({year})': 'EV_EBITDA',
            f'ROE ({year})': 'ROE',
            f'영업이익률 ({year})': '영업이익률',
            f'EBITDA/Sales ({year})': 'EBITDA/Sales',
            f'총자산이익률 ({year})': '총자산이익률',
            f'자산회전율 ({year})': '자산회전율',
            f'자기자본비율 ({year})': '자기자본비율',
            f'부채비율 ({year})': '부채비율',
            f'시가총액/매출액 ({year})': '시가총액/매출액',
            f'시가총액/영업이익 ({year})': '시가총액/영업이익',
            f'Net Income ({year})_USD': 'Net_Income',
            f'EBITDA ({year})_USD': 'EBITDA',
            f'Revenue ({year})_USD': 'Sales',
            f'Total Assets ({year})_USD': 'Assets',
            f'Equity ({year})_USD': 'Book'
        }
        for old_col, new_col in financial_mappings.items():
            if old_col in year_df.columns:
                year_df[new_col] = year_df[old_col]
        final_rows.append(year_df)
    result_df = pd.concat(final_rows, ignore_index=True)
    return result_df

@st.cache_data(show_spinner=True)
def load_processed_data() -> pd.DataFrame:
    """전체 데이터 처리 파이프라인"""
    # 1. 원본 데이터 로드
    raw_df = load_real_data()
    # 2. 재무 지표 계산 및 환율 적용
    df_with_metrics = calculate_financial_metrics_with_currency_conversion(raw_df)
    
    # [NEW] 재무지표 결측 기업 식별 로직 추가
    # 재무지표 열 정의: 'EMSEC', 'EMTEC', 'ticker', 'market' 등 비재무적 식별자 제외
    non_financial_keywords = ['EMSEC', 'EMTEC', 'ticker', 'market', 'Country', 'Market', 'name', 'Company']
    financial_cols = [
        col for col in df_with_metrics.columns
        if not any(keyword in col for keyword in non_financial_keywords)
    ]
    # 기업별(ticker)로 재무지표에 결측치가 하나라도 있는지 확인
    company_has_missing = df_with_metrics.groupby('ticker')[financial_cols].apply(lambda x: x.isnull().values.any())
    df_with_metrics['has_missing_financials'] = df_with_metrics['ticker'].map(company_has_missing)

    # 3. 중복 분류 데이터 생성
    df_expanded = create_multiple_classification_data(df_with_metrics)
    # 4. Streamlit 형태로 변환
    final_df = prepare_streamlit_data(df_expanded)
    return final_df

# 실제 데이터 로드
DF_RAW = load_processed_data()
###############################################################################
# 3. Accurate Aggregation Functions
###############################################################################
def compute_aggregate(sub_df, metric, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col):
    """calculation.txt 기반의 집계 함수 (필터 옵션 제거)"""
    if group_sel == "기업":
        if metric_main == "기업수":
            # [NEW] 기업수 계산 로직 수정
            total_unique_companies = sub_df['Company'].nunique()
            if total_unique_companies == 0:
                return 0 if metric_mode != "결측 비율" else np.nan

            if metric_mode == "결측 포함":
                return total_unique_companies
            elif metric_mode == "결측 미포함":
                # 재무지표 결측이 없는 고유 기업 수
                return sub_df.loc[~sub_df['has_missing_financials'], 'Company'].nunique()
            elif metric_mode == "결측 비율":
                # 재무지표 결측이 있는 고유 기업의 비율
                missing_unique_companies = sub_df.loc[sub_df['has_missing_financials'], 'Company'].nunique()
                return missing_unique_companies / total_unique_companies if total_unique_companies > 0 else np.nan

        elif metric_main == "0이하비율":
            arr = pd.to_numeric(sub_df[base_col], errors="coerce")
            if metric_mode == "결측 제외":
                arr = arr.dropna()
            return (arr <= 0).sum() / len(arr) if len(arr) > 0 else np.nan
    else:
        if agg_func == "AGG":
            mc_col = 'Market Cap (2024-12-31)_USD'
            if mc_col not in sub_df.columns or sub_df[mc_col].isna().all():
                return np.nan
            mc = sub_df[mc_col]
            q1 = mc.quantile(0.25)
            q3 = mc.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 2 * iqr
            upper = q3 + 2 * iqr
            filtered = sub_df[(mc >= lower) & (mc <= upper)]
            if filtered.empty:
                return np.nan

            if metric in ['PER', 'PBR', 'EV_EBITDA']:
                if metric == 'PER':
                    num = filtered['Market Cap (2024-12-31)_USD'].sum()
                    den = filtered['Net_Income'].sum()
                elif metric == 'PBR':
                    num = filtered['Market Cap (2024-12-31)_USD'].sum()
                    den = filtered['Book'].sum()
                elif metric == 'EV_EBITDA':
                    num = filtered['Enterprise Value (FQ0)_USD'].sum()
                    den = filtered['EBITDA'].sum()
                return num / den if den != 0 else np.nan
            else:
                arr = filtered[metric].dropna()
                return arr.sum() if len(arr) > 0 else np.nan
        else: # AVG, HRM, MED
            arr = sub_df[metric].dropna()
            if len(arr) == 0:
                return np.nan
            if agg_func == "AVG":
                return arr.mean()
            elif agg_func == "MED":
                return arr.median()
            elif agg_func == "HRM":
                # 0 또는 음수 값으로 인한 오류 방지
                arr = arr[arr > 0]
                return len(arr) / (1.0 / arr).sum() if len(arr) > 0 else np.nan

###############################################################################
# 4. UI
###############################################################################
main_tab, other1_tab, other2_tab = st.tabs(["Heatmap", "(WIP) 시각화A", "(WIP) 시각화B"])
with main_tab:
    with st.sidebar:
        st.markdown("**기준 연도**")
        year_sel = st.selectbox("", ["LTM", "LTM-1", "LTM-2", "LTM-3"], label_visibility="collapsed")
        st.markdown("**상장시장**")
        market_options = [
            "전체",
            "한국 전체", "KOSPI", "KOSDAQ",
            "미국 전체", "NASDAQ",
            "일본 전체", "Prime (Domestic Stocks)", "Standard (Domestic Stocks)", "Prime (Foreign Stocks)",
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
        available_sectors = sorted([s for s in DF_RAW.Sector.unique() if pd.notna(s) and s != 'Unclassified'])
        sector_sel = st.selectbox("", ["전체"] + available_sectors,
                                  label_visibility="collapsed", key="sector_sel")
        if sector_sel == "전체":
            industry_pool = sorted([i for i in DF_RAW.Industry.unique() if pd.notna(i) and i != 'Unclassified'])
        else:
            industry_pool = sorted([i for i in DF_RAW.loc[DF_RAW.Sector == sector_sel, "Industry"].unique()
                                  if pd.notna(i) and i != 'Unclassified'])
        industry_sel = st.selectbox("", ["전체"] + industry_pool,
                                    label_visibility="collapsed", key="industry_sel")
        # EMTEC Column
        st.markdown("**Theme > Technology**")
        available_themes = sorted([t for t in DF_RAW.Theme.unique() if pd.notna(t) and t != 'Unclassified'])
        theme_sel = st.selectbox("", ["전체"] + available_themes,
                                 label_visibility="collapsed", key="theme_sel")
        if theme_sel == "전체":
            tech_pool = sorted([t for t in DF_RAW.Technology.unique() if pd.notna(t) and t != 'Unclassified'])
        else:
            tech_pool = sorted([t for t in DF_RAW.loc[DF_RAW.Theme == theme_sel, "Technology"].unique()
                              if pd.notna(t) and t != 'Unclassified'])
        tech_sel = st.selectbox("", ["전체"] + tech_pool,
                                label_visibility="collapsed", key="tech_sel")
        # Metric 선택
        st.markdown("**계측값 선택**")
        group_sel = st.selectbox("", ["기업", "비교가치 멀티플", "재무비율"],
                                 label_visibility="collapsed")
        metric_main = metric_mode = base_col = agg_func = None
        allow_subtotal = True
        if group_sel == "기업":
            corp_first = st.selectbox("", ["기업수", "0이하 비율"], label_visibility="collapsed")
            if corp_first == "기업수":
                metric_main = "기업수"
                # [NEW] 기업수 옵션 변경
                metric_mode = st.selectbox("", ["결측 포함", "결측 미포함", "결측 비율"], label_visibility="collapsed")
            else: # 0이하 비율
                metric_main = "0이하비율"
                base_col_map = {
                    "순이익": "Net_Income", "EBITDA": "EBITDA", "매출": "Sales",
                    "자산총계": "Assets", "순자산": "Book",
                }
                corp_base = st.selectbox("", list(base_col_map.keys()), label_visibility="collapsed")
                base_col = base_col_map[corp_base]
                metric_mode = st.selectbox("", ["결측 포함", "결측 제외"], label_visibility="collapsed")
        elif group_sel == "비교가치 멀티플":
            metric_main = st.selectbox("", ["PER", "PBR", "EV_EBITDA"], label_visibility="collapsed")
            agg_func = st.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed")
            # [REMOVED] 필터 옵션 제거
            allow_subtotal = agg_func == "AGG"
        else: # 재무비율
            # [NEW] 재무비율 변수 추가
            metric_main = st.selectbox("", [
                "ROE", "영업이익률", "EBITDA/Sales", "총자산이익률",
                "자산회전율", "자기자본비율", "부채비율",
                "시가총액/매출액", "시가총액/영업이익"
            ], label_visibility="collapsed")
            agg_func = st.selectbox("", ["AVG", "HRM", "MED", "AGG"], label_visibility="collapsed")
            # [REMOVED] 필터 옵션 제거
            allow_subtotal = agg_func == "AGG"
    ###############################################################################
    # 5. Data Processing & Visualization
    ###############################################################################
    DF = DF_RAW[DF_RAW.Year == year_sel].copy()
    # 필터 적용
    if country_filter:
        DF = DF[DF.Country == country_filter]
    if market_filter:
        DF = DF[DF.Market == market_filter]
    if sector_sel != "전체":
        DF = DF[DF.Sector == sector_sel]
    if industry_sel != "전체":
        DF = DF[DF.Industry == industry_sel]
    if theme_sel != "전체":
        DF = DF[DF.Theme == theme_sel]
    if tech_sel != "전체":
        DF = DF[DF.Technology == tech_sel]
    # 인덱스 결정
    if sector_sel == "전체": row_index = "Sector"
    elif industry_sel == "전체": row_index = "Industry"
    else: row_index = "Sub_industry"
    if theme_sel == "전체": col_index = "Theme"
    elif tech_sel == "전체": col_index = "Technology"
    else: col_index = "Sub_Technology"
    
    values_col = (
        base_col if group_sel == "기업" and metric_main == "0이하비율" else
        metric_main if group_sel != "기업" else "Company"
    )
    if group_sel != "기업" and values_col not in DF.columns:
        st.warning(f"'{values_col}' 지표를 계산할 수 없습니다. 데이터나 설정을 확인해주세요.")
        st.stop()
    if DF.empty:
        st.warning("조건에 맞는 데이터가 없습니다.")
        st.stop()
    # Pivot tables
    pivot_main = DF.groupby([row_index, col_index]).apply(
        lambda g: compute_aggregate(g, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)
    ).unstack(fill_value=np.nan)
    pivot_counts = DF.groupby([row_index, col_index])['Company'].nunique().unstack(fill_value=0)
    if pivot_main.empty:
        st.warning("피벗 테이블을 생성할 수 없습니다.")
        st.stop()
    x_orig, y_orig = pivot_main.columns.tolist(), pivot_main.index.tolist()
    z_core = pivot_main.values
    cnt_core = pivot_counts.reindex(index=y_orig, columns=x_orig).fillna(0).values
    # Sub-/Grand-Totals 계산
    if allow_subtotal:
        row_tot = DF.groupby(row_index).apply(lambda g: compute_aggregate(g, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)).reindex(y_orig)
        col_tot = DF.groupby(col_index).apply(lambda g: compute_aggregate(g, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)).reindex(x_orig)
        grand_tot = compute_aggregate(DF, values_col, agg_func, year_sel, group_sel, metric_main, metric_mode, base_col)
        row_cnt = DF.groupby(row_index)['Company'].nunique().reindex(y_orig)
        col_cnt = DF.groupby(col_index)['Company'].nunique().reindex(x_orig)
        grand_cnt = DF['Company'].nunique()
        
        x_labels = ["Subtotal"] + x_orig
        y_labels = ["Subtotal"] + y_orig
        size = (len(y_labels), len(x_labels))
        z_main, cnt_main = np.full(size, np.nan), np.full(size, 0)
        z_main[1:, 1:], cnt_main[1:, 1:] = z_core, cnt_core
        z_sub, cnt_sub = np.full(size, np.nan), np.full(size, 0)
        z_sub[0, 1:], z_sub[1:, 0] = col_tot.values, row_tot.values
        cnt_sub[0, 1:], cnt_sub[1:, 0] = col_cnt.values, row_cnt.values
        z_grd, cnt_grd = np.full(size, np.nan), np.full(size, 0)
        z_grd[0, 0], cnt_grd[0, 0] = grand_tot, grand_cnt
        
        z_comb = np.where(np.isnan(z_main), z_sub, z_main)
        z_comb[0, 0] = grand_tot
    else:
        x_labels, y_labels = x_orig, y_orig
        z_main, cnt_main = z_core, cnt_core
        z_comb, z_sub, z_grd, cnt_sub, cnt_grd = z_main, None, None, None, None
    # 값 포맷팅 로직 수정
    if group_sel == "기업":
        if metric_main == "기업수":
            if metric_mode in ["결측 포함", "결측 미포함"]:
                fmt = lambda v: f"{v:,.0f}" if pd.notna(v) else ""
            else:  # 결측 비율
                fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
        else:  # 0이하비율
            fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
    elif metric_main in ["PER", "PBR", "EV_EBITDA", "시가총액/매출액", "시가총액/영업이익", "자산회전율"]:
        fmt = lambda v: f"{v:,.2f}x" if pd.notna(v) else ""
    elif metric_main in ["ROE", "영업이익률", "EBITDA/Sales", "총자산이익률", "자기자본비율", "부채비율"] and agg_func != 'AGG':
        fmt = lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
    else:  # AGG 집계값 (주로 절대값)
        fmt = lambda v: f"${v:,.0f}" if pd.notna(v) else ""

    txt = [[fmt(v) for v in row] for row in z_comb]
    # 플롯 생성
    st.markdown("#### EMSEC × EMTEC Heatmap")
    crumbs = [
        year_sel,
        market_sel if market_sel != "전체" else "ALL",
        sector_sel if sector_sel != "전체" else "ALL",
        theme_sel if theme_sel != "전체" else "ALL",
    ]
    st.markdown(f"선택 경로 • {' > '.join(crumbs)} • 절대값 지표: USD 통일", unsafe_allow_html=True)
    MAIN_CS = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#0d47a1"]
    SUB_CS = ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252"]
    GT_CS = [[0, "#000000"], [1, "#000000"]]
    fig = go.Figure()
    # Main Heatmap
    fig.add_trace(go.Heatmap(z=z_main, x=x_labels, y=y_labels, colorscale=MAIN_CS, colorbar=dict(title=metric_main), customdata=cnt_main, hovertemplate="%{y} / %{x}<br>값: %{z:,.3f}<br>기업수: %{customdata:.0f}", xgap=1, ygap=1, hoverongaps=False))
    if allow_subtotal:
        fig.add_trace(go.Heatmap(z=z_sub, x=x_labels, y=y_labels, colorscale=SUB_CS, showscale=False, customdata=cnt_sub, hovertemplate="%{y} / %{x}<br>Subtotal: %{z:,.3f}<br>기업수: %{customdata:.0f}", xgap=1, ygap=1, hoverongaps=False))
        fig.add_trace(go.Heatmap(z=z_grd, x=x_labels, y=y_labels, colorscale=GT_CS, showscale=False, customdata=cnt_grd, hovertemplate="Grand Total<br>값: %{z:,.3f}<br>기업수: %{customdata:.0f}", xgap=1, ygap=1, hoverongaps=False))
    # Annotations
    annotations = []
    for r_idx, row in enumerate(z_comb):
        for c_idx, v in enumerate(row):
            if pd.isna(v): continue
            is_grand = allow_subtotal and r_idx == 0 and c_idx == 0
            is_sub = allow_subtotal and (r_idx == 0 or c_idx == 0) and not is_grand
            color = "white" if is_grand else "black"
            annotations.append(go.layout.Annotation(text=txt[r_idx][c_idx], x=x_labels[c_idx], y=y_labels[r_idx], xref="x1", yref="y1", showarrow=False, font=dict(color=color)))
    fig.update_layout(annotations=annotations, height=max(650, len(y_labels) * 35), margin=dict(l=40, r=40, t=40, b=40), xaxis=dict(side="top", showgrid=False), yaxis=dict(autorange="reversed", showgrid=False, categoryorder="array", categoryarray=y_labels), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    # 추가 정보 표시
    unique_companies = DF['Company'].nunique()
    total_classifications = len(DF)
    st.caption(f"고유 기업 수: {unique_companies:,} | 총 분류 조합 수: {total_classifications:,} | 절대값 지표: USD 기준")
    st.caption("참고: '기업수'에서 '결측 미포함'은 재무지표에 결측치가 없는 기업 수, \n\
               '결측 비율'은 재무지표에 결측치가 하나라도 있는 고유 기업의 비율을 의미합니다. \n\
               '0이하 비율'은 해당 지표가 0 이하인 기업의 비율을 나타냅니다.")

with other1_tab:
    st.info("추가 시각화 A — 준비 중입니다.")
with other2_tab:
    st.info("추가 시각화 B — 준비 중입니다.")
