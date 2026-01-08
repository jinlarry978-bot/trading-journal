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

# --- 1. 頁面配置與 CSS ---
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    [data-testid="stAppViewContainer"], html, body {
        background-color: #F8F9FA !important;
        color: #212529 !important;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, p, span, div, label { color: #212529 !important; }
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
    }
    .kpi-label { font-size: 14px; color: #6C757D; font-weight: 600; margin-bottom: 6px; }
    .kpi-val-usd { font-size: 24px; font-weight: 800; color: #212529; line-height: 1.1; }
    .kpi-val-twd { font-size: 15px; color: #888; font-weight: 500; margin-top: 5px; }
    .delta-text { font-size: 14px; font-weight: 700; margin-top: 8px; padding: 2px 8px; border-radius: 4px; width: fit-content; }
    .pos { color: #D32F2F; background-color: rgba(211, 47, 47, 0.1); }
    .neg { color: #2E7D32; background-color: rgba(46, 125, 50, 0.1); }
    .ai-box { background-color: #F0F4F8; border-left: 5px solid #4285F4; padding: 15px; border-radius: 5px; margin-top: 20px; color: #212529 !important; }
    .strategy-card { padding: 18px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #E9ECEF; background-color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心工具函式 (修復 NameError 關鍵) ---
def safe_float(val):
    try:
        if pd.isna(val) or str(val).strip() == "": return 0.0
        return float(val)
    except: return 0.0

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

# --- 3. 連線與 AI ---
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

def ask_gemini_analyst(symbol, name, data_summary):
    try:
        prompt = f"分析標的：{symbol} {name}。數據：收盤{data_summary['close']:.2f}, RSI {data_summary['rsi']:.1f}。請用繁體中文給出建議（約100字）。"
        for m_name in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-pro']:
            try:
                model = genai.GenerativeModel(model_name=m_name)
                response = model.generate_content(prompt)
                if response.text: return f"{response.text}\n\n(分析引擎: {m_name})"
            except: continue
        return "AI 目前無法回應。"
    except: return "AI 啟動異常。"

# --- 4. 資料庫與運算 ---
def load_data():
    try:
        client = init_connection(); spreadsheet = client.open("TradeLog")
        try: 
            tw = pd.DataFrame(spreadsheet.worksheet("TW_Trades").get_all_records())
            if not tw.empty: tw['Market'] = 'TW'
        except: tw = pd.DataFrame()
        try: 
            us = pd.DataFrame(spreadsheet.worksheet("US_Trades").get_all_records())
            if not us.empty: us['Market'] = 'US'
        except: us = pd.DataFrame()
        return pd.concat([tw, us], ignore_index=True)
    except: return pd.DataFrame()

def save_data(row_data):
    try:
        client = init_connection(); spreadsheet = client.open("TradeLog")
        sheet = spreadsheet.worksheet("TW_Trades" if is_tw_stock(row_data[2]) else "US_Trades")
        sheet.append_row(row_data); st.cache_data.clear(); return True
    except: return False

def calculate_full_portfolio(df, rate):
    portfolio = {}
    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))
    df = df.sort_values('日期')
    for _, row in df.iterrows():
        sym = standardize_symbol(row['代號'])
        if sym not in portfolio: portfolio[sym] = {'Name': row['名稱'], 'Qty': 0, 'Cost': 0, 'Realized': 0, 'IsUS': not is_tw_stock(sym)}
        p = portfolio[sym]; qty = safe_float(row['股數']); price = safe_float(row['價格']); f = safe_float(row['手續費']); t = safe_float(row['交易稅'])
        type_str = str(row['類別'])
        if "買" in type_str: p['Cost'] += (qty*price+f); p['Qty'] += qty
        elif "賣" in type_str:
            if p['Qty']>0: 
                avg = p['Cost']/p['Qty']; cost_sold = avg*qty
                p['Realized'] += (qty*price-f-t) - cost_sold; p['Qty'] -= qty; p['Cost'] -= cost_sold
        elif "息" in type_str: p['Realized'] += price
            
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

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try: return yf.Ticker("TWD=X").history(period="1d")['Close'].iloc[-1]
    except: return 32.5

def calculate_technicals(df):
    df['MA5'] = df['Close'].rolling(5).mean(); df['MA20'] = df['Close'].rolling(20).mean(); df['MA60'] = df['Close'].rolling(60).mean()
    std20 = df['Close'].rolling(20).std(); df['BB_Upper'] = df['MA20'] + std20*2; df['BB_Lower'] = df['MA20'] - std20*2
    delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    low_min = df['Low'].rolling(9).min(); high_max = df['High'].rolling(9).max(); rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    k, d = 50, 50; k_l, d_l = [], []
    for r in rsv:
        if pd.isna(r): k_l.append(k); d_l.append(d)
        else: k = (2/3)*k + (1/3)*r; d = (2/3)*d + (1/3)*k; k_l.append(k); d_l.append(d)
    df['K'], df['D'] = k_l, d_l
    return df

def analyze_full_signal(symbol):
    try:
        clean = standardize_symbol(symbol); q_sym = f"{clean}.TW" if clean.isdigit() else clean
        stock = yf.Ticker(q_sym); info = stock.info; df = stock.history(period="1y")
        if len(df)<60: return None, None, None
        df = calculate_technicals(df); last = df.iloc[-1]
        name = info.get('longName') or info.get('shortName') or clean
        metrics = {"close": last['Close'], "rsi": last['RSI'], "k": last['K'], "d": last['D'], "ma20": last['MA20'], "ma60": last['MA60']}
        st_sig = {"txt": "🔴 短線看多", "col": "#D32F2F", "desc": "KD金叉"} if last['K']>last['D'] else {"txt": "🟠 震盪整理", "col": "#FF9800", "desc": "等待方向"}
        lt_sig = {"txt": "🟢 趨勢向下", "col": "#2E7D32", "desc": "季線之下"} if last['Close']<last['MA60'] else {"txt": "🟠 長線偏多", "col": "#FF9800", "desc": "季線支撐"}
        return df, {"st": st_sig, "lt": lt_sig, "metrics": metrics, "name": name, "symbol": q_sym}, None
    except: return None, None, None

# --- 5. 主介面 ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易", "📥 匯入", "📊 趨勢戰情", "💰 資產透視"])

with tab1:
    with st.form("trade_form"):
        st.subheader("新增單筆交易")
        c1, c2, c3 = st.columns(3)
        itype = c1.selectbox("類別", ["買入", "賣出", "股息"])
        idate = c2.date_input("日期")
        isym = c3.text_input("代號")
        c4, c5, c6, c7 = st.columns(4)
        iqty = c4.number_input("股數", step=1.0)
        iprice = c5.number_input("價格", step=0.1)
        ifee = c6.number_input("手續費", value=0.0)
        itax = c7.number_input("交易稅", value=0.0)
        submit = st.form_submit_button("送出交易")
        if submit and isym:
            _, name, _ = get_stock_info_extended(isym)
            amt = -(iqty*iprice+ifee) if "買" in itype else (iqty*iprice-ifee-itax) if "賣" in itype else iprice
            if save_data([str(idate), itype, standardize_symbol(isym), name, iprice, iqty, ifee, itax, amt]):
                st.success("已儲存交易")

with tab3:
    st.markdown("### 🔍 個股診斷與 AI 分析")
    manual = st.text_input("輸入搜尋代號", "")
    if manual:
        with st.spinner("資料抓取中..."):
            hist, ana, _ = analyze_full_signal(manual)
        if hist is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("收盤", f"{ana['metrics']['close']:.2f}"); c2.metric("RSI", f"{ana['metrics']['rsi']:.1f}"); c3.metric("K值", f"{ana['metrics']['k']:.1f}")
            if init_gemini():
                if st.button("🤖 呼叫 AI"):
                    with st.spinner("AI 思考中..."):
                        st.markdown(f'<div class="ai-box">{ask_gemini_analyst(ana["symbol"], ana["name"], ana["metrics"])}</div>', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            for col, key, title in zip([s1, s2], ['st', 'lt'], ['短期', '長期']):
                with col: st.markdown(f'<div class="strategy-card" style="border-left:5px solid {ana[key]["col"]}"><div style="color:#666">{title}</div><div style="font-size:20px;color:{ana[key]["col"]}">{ana[key]["txt"]}</div><div>{ana[key]["desc"]}</div></div>', unsafe_allow_html=True)
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(height=450, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 💰 資產透視 (雙幣別)")
    rate = get_exchange_rate(); df_raw = load_data()
    if not df_raw.empty:
        p_df, totals, _ = calculate_full_portfolio(df_raw, rate)
        def render_kpi(label, usd, twd, d=None):
            dt = f'<div class="delta-text {"pos" if d>0 else "neg"}">{"↑" if d>0 else "↓"} {abs(d):.1f}%</div>' if d else ""
            st.markdown(f'<div class="custom-kpi-card"><div class="kpi-label">{label}</div><div class="kpi-val-usd">US$ {usd:,.0f}</div><div class="kpi-val-twd">≈ NT$ {twd:,.0f}</div>{dt}</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1: render_kpi("總市值", totals['usd']['mkt'], totals['twd']['mkt'])
        with k2: render_kpi("未實現", totals['usd']['unreal'], totals['twd']['unreal'], d=(totals['usd']['unreal']/totals['usd']['mkt']*100 if totals['usd']['mkt']>0 else 0))
        with k3: render_kpi("已實現", totals['usd']['real'], totals['twd']['real'])
        with k4: render_kpi("總損益", totals['usd']['unreal']+totals['usd']['real'], totals['twd']['unreal']+totals['twd']['real'])
        st.dataframe(p_df, use_container_width=True)
