import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import time
import datetime
import io
import re
import google.generativeai as genai

# --- 1. 頁面配置與主題強制設定 ---
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")

# CSS 注入：強制淺色模式、美化卡片、適應手機 RWD
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* === 強制淺色主題 (解決暗黑模式看不清問題) === */
    [data-testid="stAppViewContainer"], html, body {
        background-color: #F8F9FA !important;
        color: #212529 !important;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, p, span, div, label { color: #212529 !important; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; }
    
    /* 修正輸入框 */
    .stTextInput input, .stNumberInput input, .stSelectbox div {
        color: #212529 !important;
        background-color: #FFFFFF !important;
    }

    /* === 自定義 KPI 卡片 (資產透視專用) === */
    .custom-kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 12px;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease;
    }
    .custom-kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.05); }
    .kpi-label { font-size: 14px; color: #6C757D; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; }
    .kpi-val-usd { font-size: 24px; font-weight: 800; color: #212529; line-height: 1.1; }
    .kpi-val-twd { font-size: 15px; color: #888; font-weight: 500; margin-top: 5px; }
    .delta-text { font-size: 14px; font-weight: 700; margin-top: 8px; padding: 2px 8px; border-radius: 4px; width: fit-content; }
    .pos { color: #D32F2F; background-color: rgba(211, 47, 47, 0.1); }
    .neg { color: #2E7D32; background-color: rgba(46, 125, 50, 0.1); }

    /* === 策略訊號卡片 === */
    .strategy-card {
        padding: 18px; 
        border-radius: 12px; 
        margin-bottom: 15px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        background-color: white;
        border: 1px solid #E9ECEF;
    }
    .strategy-title { margin: 0; color: #495057 !important; font-weight: 700; font-size: 15px; }
    .strategy-signal { margin: 8px 0; font-weight: 800; font-size: 20px; }
    .strategy-desc { font-size: 13px; color: #868E96 !important; margin: 0; }

    /* AI 分析盒 */
    .ai-box {
        background-color: #F0F4F8;
        border-left: 5px solid #4285F4;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        color: #212529 !important;
        font-size: 15px;
        line-height: 1.6;
    }

    @media (max-width: 640px) {
        .kpi-val-usd { font-size: 20px !important; }
        .kpi-val-twd { font-size: 14px !important; }
        .custom-kpi-card { padding: 15px !important; min-height: 110px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 連線與初始化 ---
@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

def init_gemini():
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return True
    return False

# --- 3. 資料處理函式 ---
def standardize_symbol(symbol):
    s = str(symbol).replace("'", "").strip().upper()
    if s.isdigit():
        if len(s) == 3: return "00" + s 
        if len(s) == 2: return "00" + s 
        if len(s) < 4: return s.zfill(4)
    return s

def standardize_date(date_val):
    try:
        if pd.isna(date_val) or str(date_val).strip() == "": return None
        if isinstance(date_val, (int, float)):
            dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=date_val)
            return dt.strftime("%Y-%m-%d")
        dt = pd.to_datetime(str(date_val).replace('.', '-').replace('/', '-'))
        return dt.strftime("%Y-%m-%d")
    except: return None

def is_tw_stock(symbol):
    s = str(symbol).upper()
    return s.isdigit() or ".TW" in s

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        ticker = yf.Ticker("TWD=X")
        hist = ticker.history(period="1d")
        return hist['Close'].iloc[-1] if not hist.empty else 32.5
    except: return 32.5

def get_stock_info_extended(symbol):
    try:
        clean = standardize_symbol(symbol)
        q_sym = f"{clean}.TW" if clean.isdigit() else clean
        stock = yf.Ticker(q_sym)
        info = stock.info
        name = info.get('longName') or info.get('shortName') or clean
        fund = {'pe': info.get('trailingPE'), 'yield': info.get('dividendYield', 0)*100, 
                'pb': info.get('priceToBook'), 'roe': info.get('returnOnEquity', 0)*100, 'beta': info.get('beta')}
        return q_sym, name, fund
    except: return symbol, symbol, {}

# --- 4. 技術分析與 AI 診斷 ---
def calculate_technicals(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + std20*2
    df['BB_Lower'] = df['MA20'] - std20*2
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    low_min = df['Low'].rolling(9).min(); high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    k, d = 50, 50; k_l, d_l = [], []
    for r in rsv:
        if pd.isna(r): k_l.append(k); d_l.append(d)
        else: k = (2/3)*k + (1/3)*r; d = (2/3)*d + (1/3)*k; k_l.append(k); d_l.append(d)
    df['K'], df['D'] = k_l, d_l
    exp1 = df['Close'].ewm(span=12).mean(); exp2 = df['Close'].ewm(span=26).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
    return df

def ask_gemini_analyst(symbol, name, data_summary):
    try:
        prompt = f"你是一位專業分析師。請分析標的：{symbol} {name}。數據：最新收盤{data_summary['close']:.2f}, RSI {data_summary['rsi']:.1f}, KD(K) {data_summary['k']:.1f}。請用繁體中文給出買進/減持/持有的具體建議與原因（約120字）。"
        model_names = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-pro']
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(model_name=m_name)
                response = model.generate_content(prompt)
                if response and response.text: return f"{response.text}\n\n(Engine: {m_name})"
            except: continue
        return "AI 暫時無法連線，請檢查 API Key 或稍後再試。"
    except Exception as e: return f"AI 異常: {str(e)}"

def analyze_full_signal(symbol):
    q_sym, name, fund = get_stock_info_extended(symbol)
    df = yf.Ticker(q_sym).history(period="1y")
    if len(df)<60: return None, None, None
    df = calculate_technicals(df); last = df.iloc[-1]
    metrics = {"close": last['Close'], "rsi": last['RSI'], "k": last['K'], "d": last['D'], "ma20": last['MA20'], "ma60": last['MA60']}
    st_sig = {"txt": "🔴 短線買進", "col": "#D32F2F", "desc": "KD金叉+強勁動能"} if last['K']>last['D'] and last['Close']>last['MA5'] else {"txt": "🟠 持有/觀望", "col": "#FF9800", "desc": "等待攻擊訊號"}
    mt_sig = {"txt": "🔴 波段看多", "col": "#D32F2F", "desc": "站穩月線"} if last['Close']>last['MA20'] else {"txt": "🟢 波段看空", "col": "#2E7D32", "desc": "趨勢轉弱"}
    lt_sig = {"txt": "🟠 長線持有", "col": "#FF9800", "desc": "多頭格局"} if last['Close']>last['MA60'] else {"txt": "🟢 避開觀望", "col": "#2E7D32", "desc": "長線空頭"}
    return df, {"st": st_sig, "mt": mt_sig, "lt": lt_sig, "metrics": metrics, "fund": fund, "name": name, "symbol": q_sym}, None

# --- 5. 檔案與資產邏輯 ---
def load_data():
    try:
        client = init_connection(); spreadsheet = client.open("TradeLog")
        try: tw = pd.DataFrame(spreadsheet.worksheet("TW_Trades").get_all_records()); tw['Market'] = 'TW'
        except: tw = pd.DataFrame()
        try: us = pd.DataFrame(spreadsheet.worksheet("US_Trades").get_all_records()); us['Market'] = 'US'
        except: us = pd.DataFrame()
        return pd.concat([tw, us], ignore_index=True)
    except: return pd.DataFrame()

def calculate_full_portfolio(df, rate):
    portfolio = {}
    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))
    df = df.sort_values('日期')
    for _, row in df.iterrows():
        sym = standardize_symbol(row['代號'])
        if sym not in portfolio: portfolio[sym] = {'Name': row['名稱'], 'Qty': 0, 'Cost': 0, 'Realized': 0, 'IsUS': not is_tw_stock(sym)}
        p = portfolio[sym]; qty = safe_float(row['股數']); price = safe_float(row['價格']); fees = safe_float(row['手續費']); tax = safe_float(row['交易稅'])
        if "買" in str(row['類別']): p['Cost'] += (qty*price+fees); p['Qty'] += qty
        elif "賣" in str(row['類別']): 
            if p['Qty']>0: cost_sold = (p['Cost']/p['Qty'])*qty; p['Realized'] += (qty*price-fees-tax) - cost_sold; p['Qty'] -= qty; p['Cost'] -= cost_sold
        elif "息" in str(row['類別']): p['Realized'] += price
    
    # 抓現價
    active_syms = [s for s, v in portfolio.items() if v['Qty'] > 0]
    prices = {}
    if active_syms:
        qs = [f"{s}.TW" if is_tw_stock(s) and s.isdigit() else s for s in active_syms]
        data = yf.Tickers(" ".join(qs))
        for i, s in enumerate(active_syms):
            try: prices[s] = data.tickers[qs[i]].history(period="1d")['Close'].iloc[-1]
            except: prices[s] = 0
            
    res, t_twd, t_usd = [], {'mkt':0, 'unreal':0, 'real':0}, {'mkt':0, 'unreal':0, 'real':0}
    for s, v in portfolio.items():
        cp = prices.get(s, 0); mkt = v['Qty']*cp; unreal = mkt - v['Cost'] if v['Qty']>0 else 0
        if v['IsUS']: 
            t_usd['mkt']+=mkt; t_usd['unreal']+=unreal; t_usd['real']+=v['Realized']
            t_twd['mkt']+=mkt*rate; t_twd['unreal']+=unreal*rate; t_twd['real']+=v['Realized']*rate
        else:
            t_twd['mkt']+=mkt; t_twd['unreal']+=unreal; t_twd['real']+=v['Realized']
        if v['Qty']!=0 or v['Realized']!=0:
            res.append({"代號":s,"名稱":v['Name'],"庫存":v['Qty'],"現價":cp,"市值":mkt,"未實現":unreal,"已實現+息":v['Realized'],"IsUS":v['IsUS']})
    return pd.DataFrame(res), {"twd": t_twd, "usd": t_usd}, None

# --- 6. 主介面 ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易", "📥 匯入", "📊 趨勢戰情", "💰 資產透視"])

with tab3:
    st.markdown("### 🔍 個股診斷與 AI 分析")
    manual = st.text_input("輸入代號 (如 2330 或 AAPL)", "")
    if manual:
        with st.spinner("資料抓取中..."):
            hist, ana, _ = analyze_full_signal(manual)
        if hist is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{ana['metrics']['close']:.2f}")
            c2.metric("RSI", f"{ana['metrics']['rsi']:.1f}")
            c3.metric("K值", f"{ana['metrics']['k']:.1f}")
            c4.metric("vs 大盤", f"{ana['metrics']['perf_stock']:.1f}%", f"{ana['metrics']['perf_diff']:+.1f}%")
            
            if init_gemini():
                if st.button("🤖 呼叫 AI 分析師"):
                    with st.spinner("AI 分析中..."):
                        res = ask_gemini_analyst(ana['symbol'], ana['name'], ana['metrics'])
                        st.markdown(f'<div class="ai-box"><b>🤖 AI 分析觀點：</b><br>{res}</div>', unsafe_allow_html=True)
            
            s1, s2, s3 = st.columns(3)
            for col, key, title in zip([s1, s2, s3], ['st', 'mt', 'lt'], ['⚡ 短期', '🌊 中期', '🏔️ 長期']):
                with col: st.markdown(f'<div class="strategy-card" style="border-left:5px solid {ana[key]["col"]}"><div class="strategy-title">{title}</div><div class="strategy-signal" style="color:{ana[key]["col"]}">{ana[key]["txt"]}</div><div class="strategy-desc">{ana[key]["desc"]}</div></div>', unsafe_allow_html=True)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='MACD'), row=2, col=1)
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 💰 資產透視 (雙幣別)")
    rate = get_exchange_rate(); df_raw = load_data()
    
    def render_kpi(label, usd_val, twd_val, delta=None):
        d_html = f'<div class="delta-text {"pos" if delta>0 else "neg"}">{"↑" if delta>0 else "↓"} {abs(delta):.1f}%</div>' if delta is not None else ""
        st.markdown(f'<div class="custom-kpi-card"><div class="kpi-label">{label}</div><div class="kpi-val-usd">US$ {usd_val:,.0f}</div><div class="kpi-val-twd">≈ NT$ {twd_val:,.0f}</div>{d_html}</div>', unsafe_allow_html=True)

    if not df_raw.empty:
        p_df, totals, _ = calculate_full_portfolio(df_raw, rate)
        k1, k2, k3, k4 = st.columns(4)
        with k1: render_kpi("總市值", totals['usd']['mkt'], totals['twd']['mkt'])
        with k2: 
            d = (totals['usd']['unreal']/totals['usd']['mkt']*100) if totals['usd']['mkt']>0 else 0
            render_kpi("未實現損益", totals['usd']['unreal'], totals['twd']['unreal'], delta=d)
        with k3: render_kpi("已實現+息", totals['usd']['real'], totals['twd']['real'])
        with k4: render_kpi("總損益", totals['usd']['unreal']+totals['usd']['real'], totals['twd']['unreal']+totals['twd']['real'])
        
        st.subheader("📋 資產明細表")
        display_df = p_df.copy()
        for col in ['市值', '未實現', '已實現+息']:
            display_df[col] = display_df.apply(lambda r: f"${r[col]:,.0f} / NT${r[col]*rate:,.0f}" if r['IsUS'] else f"{r[col]:,.0f}", axis=1)
        st.dataframe(display_df.drop(columns=['IsUS']), use_container_width=True)
