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

# --- 1. 頁面設定 ---
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")

# --- 2. 核心 CSS 修復 (強制淺色模式 + 防止數字切斷 + 手機優化) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* 強制淺色主題：解決暗黑模式看不清問題 */
    [data-testid="stAppViewContainer"], html, body {
        background-color: #F8F9FA !important;
        color: #212529 !important;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, p, span, div, label { color: #212529 !important; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; }

    /* 自定義 KPI 卡片：解決數字太長被切斷問題，改為上下分層 */
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
    
    /* 漲跌文字 */
    .delta-text { font-size: 14px; font-weight: 700; margin-top: 8px; padding: 2px 8px; border-radius: 4px; width: fit-content; }
    .pos { color: #D32F2F; background-color: rgba(211, 47, 47, 0.1); } /* 紅漲 */
    .neg { color: #2E7D32; background-color: rgba(46, 125, 50, 0.1); } /* 綠跌 */

    /* 策略卡片 */
    .strategy-card { padding: 18px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #E9ECEF; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    
    /* AI 分析區塊 */
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

# --- 3. 初始化與連線 ---
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

# --- 4. 核心 AI 分析函式 (防 404 回退機制版) ---
def ask_gemini_analyst(symbol, name, data_summary):
    try:
        prompt = f"""
        你是一位專業投資分析師。請分析以下標的並提供繁體中文建議（約120字）：
        股票：{symbol} {name}
        最新收盤：{data_summary['close']:.2f}
        技術指標：RSI {data_summary['rsi']:.1f}, KD(K) {data_summary['k']:.1f}
        均線位置：月線 {data_summary['ma20']:.2f}, 季線 {data_summary['ma60']:.2f}
        
        請給出「買進/減持/持有」的操作建議與簡短原因。
        """

        # 嘗試模型名單 (由新到舊嘗試，解決 404 找不到模型問題)
        model_names = [
            'gemini-2.0-flash-exp', 
            'gemini-1.5-flash', 
            'gemini-1.5-pro',
            'gemini-pro'
        ]
        
        last_err = ""
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(model_name=m_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    return f"{response.text}\n\n(分析引擎: {m_name})"
            except Exception as e:
                last_err = str(e)
                continue
        
        return f"AI 連線失敗。嘗試了所有模型皆回傳錯誤：{last_err}"
    except Exception as e:
        return f"AI 啟動異常：{str(e)}"

# --- 5. 資料處理與標準化 (保持核心邏輯) ---
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

# (此處略過 load_data, save_data, calculate_technicals, calculate_full_portfolio 等計算邏輯，請沿用 V8.1 的程式碼區塊)
# 為確保代碼能跑，這裡放入 calculate_technicals 簡化版與 analyze_full_signal
def calculate_technicals(df):
    df['MA5'] = df['Close'].rolling(5).mean(); df['MA20'] = df['Close'].rolling(20).mean(); df['MA60'] = df['Close'].rolling(60).mean()
    std20 = df['Close'].rolling(20).std(); df['BB_Upper'] = df['MA20'] + std20*2; df['BB_Lower'] = df['MA20'] - std20*2
    delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    low_min = df['Low'].rolling(9).min(); high_max = df['High'].rolling(9).max(); rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    k, d = 50, 50; k_l, d_l = [], []
    for r in rsv:
        if pd.isna(r): k_l.append(50); d_l.append(50)
        else: k = (2/3)*k + (1/3)*r; d = (2/3)*d + (1/3)*k; k_l.append(k); d_l.append(d)
    df['K'], df['D'] = k_l, d_l
    exp1 = df['Close'].ewm(span=12).mean(); exp2 = df['Close'].ewm(span=26).mean()
    df['MACD_Hist'] = (exp1 - exp2) - (exp1 - exp2).ewm(span=9).mean()
    return df

def analyze_full_signal(symbol):
    q_sym, name, fund = get_stock_info_extended(symbol)
    df = yf.Ticker(q_sym).history(period="1y")
    if len(df)<60: return None, None, None
    df = calculate_technicals(df); last = df.iloc[-1]
    metrics = {"close": last['Close'], "rsi": last['RSI'], "k": last['K'], "d": last['D'], "ma20": last['MA20'], "ma60": last['MA60']}
    # 策略判定
    st_sig = {"txt": "🔴 短線買進", "col": "#D32F2F", "desc": "站上5日線+KD金叉"} if last['Close']>last['MA5'] and last['K']>last['D'] else {"txt": "🟠 持有/觀望", "col": "#FF9800", "desc": "整理中"}
    mt_sig = {"txt": "🔴 波段看多", "col": "#D32F2F", "desc": "站穩月線"} if last['Close']>last['MA20'] else {"txt": "🟢 波段看空", "col": "#2E7D32", "desc": "跌破月線"}
    lt_sig = {"txt": "🟠 長線持有", "col": "#FF9800", "desc": "季線之上"} if last['Close']>last['MA60'] else {"txt": "🟢 趨勢轉弱", "col": "#2E7D32", "desc": "跌破生命線"}
    analysis = {"st": st_sig, "mt": mt_sig, "lt": lt_sig, "metrics": metrics, "fund": fund, "name": name, "symbol": q_sym}
    return df, analysis, None

# --- 6. 主介面邏輯 (Tab 4 重點修復) ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易", "📥 匯入", "📊 趨勢戰情", "💰 資產透視"])

with tab3:
    st.markdown("### 🔍 個股全方位診斷")
    manual = st.text_input("搜尋代號 (例如 2330)", "")
    if manual:
        with st.spinner("分析中..."):
            hist, ana, _ = analyze_full_signal(manual)
        if hist is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"{ana['metrics']['close']:.2f}")
            c2.metric("RSI", f"{ana['metrics']['rsi']:.1f}")
            c3.metric("K值", f"{ana['metrics']['k']:.1f}")
            c4.metric("D值", f"{ana['metrics']['d']:.1f}")
            
            # AI 按鈕
            if init_gemini():
                if st.button("🤖 呼叫 AI 分析師 (Gemini)"):
                    with st.spinner("AI 正在思考..."):
                        ai_res = ask_gemini_analyst(ana['symbol'], ana['name'], ana['metrics'])
                        st.markdown(f'<div class="ai-box"><b>🤖 AI 分析觀點：</b><br>{ai_res}</div>', unsafe_allow_html=True)
            
            # 策略卡片
            s1, s2, s3 = st.columns(3)
            for col, key, title in zip([s1, s2, s3], ['st', 'mt', 'lt'], ['⚡ 短期', '🌊 中期', '🏔️ 長期']):
                with col: st.markdown(f'<div class="strategy-card" style="border-left:5px solid {ana[key]["col"]}"><div class="strategy-title">{title}</div><div class="strategy-signal" style="color:{ana[key]["col"]}">{ana[key]["txt"]}</div><div class="strategy-desc">{ana[key]["desc"]}</div></div>', unsafe_allow_html=True)
            
            # 圖表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='月線', line=dict(color='#FF9800')), row=1, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='MACD'), row=2, col=1)
            fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 💰 資產透視")
    # 此處需加入您的 load_data 與 calculate_full_portfolio 呼叫
    # 以下為自定義 KPI 卡片渲染邏輯
    def render_kpi(label, usd_val, twd_val, delta=None):
        d_html = f'<div class="delta-text {"pos" if delta>0 else "neg"}">{"↑" if delta>0 else "↓"} {abs(delta):.1f}%</div>' if delta is not None else ""
        st.markdown(f"""
            <div class="custom-kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-val-usd">US$ {usd_val:,.0f}</div>
                <div class="kpi-val-twd">≈ NT$ {twd_val:,.0f}</div>
                {d_html}
            </div>
        """, unsafe_allow_html=True)

    # 範例渲染 (實際運行時請換成您的 totals 數據)
    k1, k2, k3, k4 = st.columns(4)
    with k1: render_kpi("總市值", 34357, 1080803)
    with k2: render_kpi("未實現損益", -623, -19587, delta=-1.8)
    with k3: render_kpi("已實現+股息", -26096, -820924)
    with k4: render_kpi("總損益", -26719, -840517)

# (其餘 Tab 1, 2 功能請保持不變)
