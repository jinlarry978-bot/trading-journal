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

# --- 1. 頁面設定 ---
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")

# CSS 優化：讓指標卡片更立體，策略區塊更明顯
st.markdown("""
    <style>
    .stApp {background-color: #F5F7F9;}
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .strategy-card {
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 連線設定 ---
SHEET_TW = "TW_Trades"
SHEET_US = "US_Trades"

# 內建熱門股字典 (Fallback 用)
KNOWN_STOCKS = {
    '0050': '元大台灣50', '0056': '元大高股息', '00878': '國泰永續高股息', 
    '00929': '復華台灣科技優息', '00919': '群益台灣精選高息', '006208': '富邦台50',
    '00940': '元大台灣價值高息', '00939': '統一台灣高息動能',
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2303': '聯電',
    '2881': '富邦金', '2882': '國泰金', '2891': '中信金', '2886': '兆豐金',
    '2884': '玉山金', '2412': '中華電', '1101': '台泥', '2002': '中鋼',
    '2603': '長榮', '2609': '陽明', '2615': '萬海', '3231': '緯創', '2382': '廣達'
}

@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

def is_tw_stock(symbol):
    symbol = str(symbol).upper().strip()
    if ".TW" in symbol or symbol.isdigit(): return True
    return False

def load_data():
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        try:
            tw_data = spreadsheet.worksheet(SHEET_TW).get_all_records()
            df_tw = pd.DataFrame(tw_data)
            if not df_tw.empty: df_tw['Market'] = 'TW'
        except: df_tw = pd.DataFrame()

        try:
            us_data = spreadsheet.worksheet(SHEET_US).get_all_records()
            df_us = pd.DataFrame(us_data)
            if not df_us.empty: df_us['Market'] = 'US'
        except: df_us = pd.DataFrame()

        df_all = pd.concat([df_tw, df_us], ignore_index=True)
        return df_all
    except Exception as e: return pd.DataFrame()

def save_data(row_data):
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        symbol = row_data[2]
        target_sheet = SHEET_TW if is_tw_stock(symbol) else SHEET_US
        sheet = spreadsheet.worksheet(target_sheet)
        sheet.append_row(row_data)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"寫入失敗: {e}")
        return False

# --- 工具函數 ---
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
        date_str = str(date_val).strip()
        if isinstance(date_val, (pd.Timestamp, datetime.date, datetime.datetime)):
            return date_val.strftime("%Y-%m-%d")
        date_str = date_str.replace('.', '-').replace('/', '-')
        if '-' in date_str:
            parts = date_str.split('-')
            if len(parts) == 3:
                y, m, d = parts
                if len(y) <= 3 and int(y) < 1900: 
                    y = str(int(y) + 1911)
                    date_str = f"{y}-{m}-{d}"
        dt = pd.to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except: return None

# 直通寫入模式 (不去重)
def batch_save_data_smart(rows, market_type):
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        target_sheet_name = SHEET_TW if market_type == 'TW' else SHEET_US
        sheet = spreadsheet.worksheet(target_sheet_name)
        if rows:
            sheet.append_rows(rows)
            st.cache_data.clear()
            return True, len(rows), 0
        else: return True, 0, 0
    except Exception as e:
        st.error(f"批次寫入錯誤: {e}")
        return False, 0, 0

# --- 3. 股票資訊 (擴充基本面數據) ---
def get_stock_info_extended(symbol):
    """
    獲取股票資訊，包含更多基本面數據 (PB, ROE, Beta)
    """
    try:
        clean_symbol = standardize_symbol(symbol)
        if clean_symbol.isdigit(): query_symbol = f"{clean_symbol}.TW"
        else: query_symbol = clean_symbol
            
        stock = yf.Ticker(query_symbol)
        
        # 預設值
        name = clean_symbol
        if clean_symbol in KNOWN_STOCKS: name = KNOWN_STOCKS[clean_symbol]
        
        info = {}
        try:
            info = stock.info
            # 優先使用 Yahoo 的名稱，若無則用字典或代號
            api_name = info.get('longName') or info.get('shortName')
            if api_name: name = api_name
        except: pass
        
        # 安全獲取數據的 Helper
        def get_val(key, default=None):
            return info.get(key, default)

        fundamentals = {
            'pe': get_val('trailingPE'),
            'yield': get_val('dividendYield'),
            'pb': get_val('priceToBook'),         # 股價淨值比
            'roe': get_val('returnOnEquity'),     # 股東權益報酬率
            'beta': get_val('beta'),              # 波動率 (Beta)
            'marketCap': get_val('marketCap')     # 市值
        }
        
        # 格式化百分比
        if fundamentals['yield']: fundamentals['yield'] *= 100
        if fundamentals['roe']: fundamentals['roe'] *= 100
            
        return query_symbol, name, fundamentals
    except: 
        return symbol, symbol, {}

# --- 4. 技術分析與策略 ---
def calculate_technicals(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 布林通道
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std20 * 2)
    df['BB_Lower'] = df['MA20'] - (std20 * 2)
    
    # 成交量均線
    df['VolMA5'] = df['Volume'].rolling(window=5).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # KD
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    k_list, d_list = [], []
    k, d = 50, 50
    for rsv in df['RSV']:
        if pd.isna(rsv): k_list.append(50); d_list.append(50)
        else:
            k = (2/3) * k + (1/3) * rsv
            d = (2/3) * d + (1/3) * k
            k_list.append(k); d_list.append(d)
    df['K'] = k_list
    df['D'] = d_list
    return df

def analyze_full_signal(symbol):
    try:
        # 1. 獲取個股資料
        q_sym, name, fund = get_stock_info_extended(symbol)
        stock = yf.Ticker(q_sym)
        df = stock.history(period="1y")
        
        if len(df) < 60: return None, None, None
        
        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        # 2. 獲取比較基準 (0050.TW) 進行績效 PK
        try:
            benchmark = yf.Ticker("0050.TW").history(period="1y")['Close']
            # 計算近一年報酬率
            stock_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            bench_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
            perf_diff = stock_ret - bench_ret
        except:
            stock_ret, bench_ret, perf_diff = 0, 0, 0

        # --- 策略判斷 ---
        close = last['Close']
        ma5, ma20, ma60 = last['MA5'], last['MA20'], last['MA60']
        rsi, k, d = last['RSI'], last['K'], last['D']
        macd_hist = last['MACD_Hist']
        vol, vol_ma5 = last['Volume'], last['VolMA5']
        
        # 短期策略
        if close > ma5 and k > d and vol > vol_ma5:
            st_sig = {"txt": "🔴 短線買進", "col": "#D32F2F", "desc": "站上5日線+帶量+KD金叉"}
        elif rsi < 25:
            st_sig = {"txt": "🔴 搶反彈", "col": "#D32F2F", "desc": "RSI嚴重超賣(<25)"}
        elif close < ma5 and k < d:
            st_sig = {"txt": "🟢 短線賣出", "col": "#2E7D32", "desc": "跌破5日線+KD死叉"}
        elif rsi > 80:
            st_sig = {"txt": "🟢 獲利了結", "col": "#2E7D32", "desc": "RSI過熱(>80)"}
        else:
            st_sig = {"txt": "🟠 持有/觀望", "col": "#FF9800", "desc": "短期震盪整理"}

        # 中期策略
        if close > ma20 and macd_hist > 0:
            mt_sig = {"txt": "🔴 波段買進", "col": "#D32F2F", "desc": "站穩月線+MACD多頭"}
        elif close < ma20 and macd_hist < 0:
            mt_sig = {"txt": "🟢 波段賣出", "col": "#2E7D32", "desc": "跌破月線+MACD空頭"}
        elif close > ma20:
            mt_sig = {"txt": "🟠 續抱", "col": "#FF9800", "desc": "股價於月線之上"}
        else:
            mt_sig = {"txt": "⚪ 弱勢整理", "col": "#666666", "desc": "股價受制於月線"}

        # 長期策略
        is_bull_align = ma5 > ma20 and ma20 > ma60
        if close > ma60 and is_bull_align:
            lt_sig = {"txt": "🔴 長線加碼", "col": "#D32F2F", "desc": "均線多頭排列"}
        elif close > ma60:
            lt_sig = {"txt": "🟠 長期持有", "col": "#FF9800", "desc": "長線趨勢向上"}
        elif close < ma60:
            lt_sig = {"txt": "🟢 趨勢轉空", "col": "#2E7D32", "desc": "跌破季線(生命線)"}
        else:
            lt_sig = {"txt": "⚪ 盤整", "col": "#666666", "desc": "季線附近震盪"}

        analysis = {
            "st": st_sig, "mt": mt_sig, "lt": lt_sig,
            "metrics": {
                "close": close, "rsi": rsi, "k": k, "d": d,
                "perf_stock": stock_ret, "perf_bench": bench_ret, "perf_diff": perf_diff
            },
            "fund": fund
        }
        return df, analysis, benchmark
    except: return None, None, None

# --- 5. 資產計算 ---
def get_sort_rank(t_type):
    t_type = str(t_type)
    if "Buy" in t_type or "買" in t_type or "配股" in t_type: return 1
    if "Sell" in t_type or "賣" in t_type: return 2
    return 3

def calculate_full_portfolio(df):
    portfolio = {}
    monthly_pnl = {}
    
    # 日期標準化與排序
    df['日期'] = df['日期'].apply(standardize_date)
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce') 
    df = df.dropna(subset=['日期'])
    df['Rank'] = df['類別'].apply(get_sort_rank)
    df = df.sort_values(by=['日期', 'Rank'])
    
    for _, row in df.iterrows():
        sym = standardize_symbol(row['代號'])
        name = row['名稱']
        qty = safe_float(row['股數'])
        price = safe_float(row['價格'])
        fees = safe_float(row['手續費'])
        tax = safe_float(row['交易稅'])
        t_type = str(row['類別'])
        date_str = row['日期'].strftime("%Y-%m")
        
        if sym not in portfolio:
            portfolio[sym] = {'Name': name, 'Qty': 0, 'Cost': 0, 'Realized': 0, 'Div': 0}
        if date_str not in monthly_pnl: monthly_pnl[date_str] = 0
            
        p = portfolio[sym]
        
        is_buy = any(x in t_type for x in ["Buy", "買"])
        is_sell = any(x in t_type for x in ["Sell", "賣"])
        is_div = any(x in t_type for x in ["Dividend", "股息", "配息"])
        
        if is_buy:
            p['Cost'] += (qty * price) + fees
            p['Qty'] += qty
        elif is_sell:
            if p['Qty'] > 0:
                avg_cost = p['Cost'] / p['Qty']
                cost_sold = avg_cost * qty
                revenue = (qty * price) - fees - tax
                profit = revenue - cost_sold
                p['Realized'] += profit
                monthly_pnl[date_str] += profit
                p['Qty'] -= qty
                p['Cost'] -= cost_sold
            else:
                revenue = (qty * price) - fees - tax
                p['Realized'] += revenue
                monthly_pnl[date_str] += revenue
                p['Qty'] -= qty
        elif is_div:
            p['Div'] += price
            monthly_pnl[date_str] += price
            p['Qty'] += qty

    active_syms = [s for s, v in portfolio.items() if v['Qty'] > 0]
    curr_prices = {}
    if active_syms:
        try:
            q_list = []
            for s in active_syms:
                if s.isdigit(): q_list.append(f"{s}.TW")
                else: q_list.append(s)
            
            data = yf.Tickers(" ".join(q_list))
            for i, s in enumerate(active_syms):
                try:
                    h = data.tickers[q_list[i]].history(period="1d")
                    curr_prices[s] = h['Close'].iloc[-1] if not h.empty else 0
                except: curr_prices[s] = 0
        except: pass
        
    res = []
    tot_mkt, tot_unreal, tot_real = 0, 0, 0
    
    for sym, v in portfolio.items():
        cp = curr_prices.get(sym, 0)
        if abs(v['Qty']) < 0.001: v['Qty'] = 0
        
        mkt = v['Qty'] * cp
        unreal = mkt - v['Cost'] if v['Qty'] > 0 else 0
        
        tot_mkt += mkt
        tot_unreal += unreal
        tot_real += (v['Realized'] + v['Div'])
        
        if v['Qty'] != 0 or v['Realized']!=0 or v['Div']!=0:
            res.append({
                "代號": sym, "名稱": v['Name'], "庫存": v['Qty'], "均價": v['Cost']/v['Qty'] if v['Qty']>0 else 0,
                "現價": cp, "市值": mkt, "未實現": unreal, "已實現+息": v['Realized']+v['Div']
            })
            
    m_df = pd.DataFrame(list(monthly_pnl.items()), columns=['Month', 'PnL']).sort_values('Month')
    return pd.DataFrame(res), tot_mkt, tot_unreal, tot_real, m_df

def convert_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- 6. 主程式 ---
st.title("💎 專業投資戰情室 Pro")
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易", "📥 匯入", "📊 趨勢戰情", "💰 資產透視"])

# Tab 1: 單筆輸入
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("新增交易")
        itype = st.selectbox("類別", ["買入 (Buy)", "賣出 (Sell)", "股息 (Dividend)"])
        idate = st.date_input("日期")
        isym = st.text_input("代號", placeholder="台股2330, 美股AAPL")
        
        name = "..."
        rsym = isym
        if isym: 
            check_sym = standardize_symbol(isym)
            # 使用擴充版獲取資訊
            rsym, name, _ = get_stock_info_extended(check_sym)
        
        st.info(f"股票: **{name}**")
        
        iqty = st.number_input("股數 (或配股數)", min_value=0.0, step=100.0)
        iprice = st.number_input("價格 (或現金股息總額)", min_value=0.0, step=0.1)
        ifees = st.number_input("手續費", min_value=0.0)
        itax = st.number_input("交易稅", min_value=0.0)
        
        tot = -(iqty*iprice+ifees) if "買" in itype else (iqty*iprice-ifees-itax) if "賣" in itype else iprice
        st.metric("總金額", f"${tot:,.0f}")
        
        if st.button("送出", type="primary"):
            type_val = "買入" if "買" in itype else "賣出" if "賣" in itype else "股息"
            clean_sym = rsym.replace('.TW', '') 
            clean_sym = standardize_symbol(clean_sym)
            std_date = standardize_date(idate)
            if save_data([std_date, type_val, clean_sym, name, iprice, iqty, ifees, itax, tot]): 
                st.success("已儲存")

# Tab 2: 匯入
with tab2:
    st.markdown("### 📥 批次匯入")
    template_data = {
        "日期": ["2024-01-01", "2024-06-15"], 
        "類別": ["買入", "股息"], 
        "代號": ["0050", "2330"],
        "名稱": ["元大台灣50", "台積電"], 
        "價格": [150, 5000], "股數": [1000, 0], "手續費": [20, 10], "交易稅": [0, 0]
    }
    st.download_button("📥 下載範本", convert_to_excel(pd.DataFrame(template_data)), "template.xlsx")
    
    uploaded_file = st.file_uploader("上傳檔案", type=["csv", "xlsx"])
    if uploaded_file and st.button("開始匯入"):
        try:
            if uploaded_file.name.endswith('.csv'):
                df_u = pd.read_csv(uploaded_file, dtype={'代號': str})
            else:
                df_u = pd.read_excel(uploaded_file, dtype={'代號': str})
            
            df_u = df_u.dropna(how='all')
            df_u['日期'] = df_u['日期'].apply(standardize_date)
            df_u = df_u.dropna(subset=['日期'])
            
            tw_rows, us_rows = [], []
            bar = st.progress(0.0)
            total = len(df_u)
            
            for i, (index, r) in enumerate(df_u.iterrows()):
                clean_sym = standardize_symbol(r['代號'])
                excel_name = str(r.get('名稱', '')).strip()
                if excel_name and excel_name.lower() != 'nan': name = excel_name
                else:
                    _, name, _ = get_stock_info_extended(clean_sym)
                
                tt = "買入" if any(x in str(r['類別']) for x in ["Buy","買"]) else "賣出" if any(x in str(r['類別']) for x in ["Sell","賣"]) else "股息"
                q, p, f, t = safe_float(r['股數']), safe_float(r['價格']), safe_float(r['手續費']), safe_float(r['交易稅'])
                amt = -(q*p+f) if "買" in tt else (q*p-f-t) if "賣" in tt else p
                
                row = [str(r['日期']), tt, clean_sym, name, p, q, f, t, amt]
                if is_tw_stock(clean_sym): tw_rows.append(row)
                else: us_rows.append(row)
                
                if total > 0: bar.progress(min((i+1)/total, 1.0))
            
            msg = ""
            if tw_rows:
                _, added_tw, _ = batch_save_data_smart(tw_rows, 'TW')
                msg += f"🇹🇼 台股: {added_tw} 筆。 "
            if us_rows:
                _, added_us, _ = batch_save_data_smart(us_rows, 'US')
                msg += f"🇺🇸 美股: {added_us} 筆。"
            st.success(f"匯入完成！ {msg}")
        except Exception as e: st.error(f"匯入失敗: {str(e)}")

# Tab 3: 全方位戰情 (包含體質分析 + 大盤 PK)
with tab3:
    st.markdown("### 🔍 個股全方位診斷")
    market_filter = st.radio("選擇市場", ["全部", "台股 (TW)", "美股 (US)"], horizontal=True)
    df_raw = load_data()
    if not df_raw.empty:
        if "台股" in market_filter: df_raw = df_raw[df_raw['Market'] == 'TW']
        elif "美股" in market_filter: df_raw = df_raw[df_raw['Market'] == 'US']
        
        inventory = {}
        names = {}
        for _, row in df_raw.iterrows():
            sym = standardize_symbol(row['代號'])
            tt = str(row['類別'])
            q = safe_float(row['股數'])
            if "買" in tt or "Buy" in tt or "配股" in tt: inventory[sym] = inventory.get(sym, 0) + q
            elif "賣" in tt or "Sell" in tt: inventory[sym] = inventory.get(sym, 0) - q
            names[sym] = row['名稱']
        
        # 只顯示庫存 > 0
        active_list = [f"{k} {names[k]}" for k, v in inventory.items() if v > 0.1]
        col_sel, col_search = st.columns([1, 1])
        with col_sel:
            sel = st.selectbox("庫存快選", active_list) if active_list else None
        with col_search:
            manual = st.text_input("或搜尋代號", placeholder="例如 2330")
        
        target = manual if manual else (sel.split()[0] if sel else None)
        
        if target:
            with st.spinner("AI 正在進行六大體質分析與大盤比對..."):
                hist, ana, benchmark_hist = analyze_full_signal(target)
            
            if hist is not None:
                # 1. 價格與基本指標
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("股價", f"{ana['metrics']['close']:.2f}")
                m2.metric("RSI (14)", f"{ana['metrics']['rsi']:.1f}")
                m3.metric("KD (K值)", f"{ana['metrics']['k']:.1f}")
                
                # 績效 PK (個股 vs 大盤)
                diff = ana['metrics']['perf_diff']
                diff_color = "normal" if diff == 0 else ("inverse" if diff < 0 else "normal") # 正數綠色(台股紅漲綠跌邏輯相反，這裡Streamlit預設紅漲)
                # Streamlit metric delta: 紅是負，綠是正。台股習慣紅漲。
                # 這裡顯示年報酬率
                m4.metric("近一年報酬 vs 0050", 
                          f"{ana['metrics']['perf_stock']:.1f}%", 
                          delta=f"{diff:+.1f}% (領先)" if diff>0 else f"{diff:+.1f}% (落後)")

                # 2. 策略訊號卡片
                st.write("")
                s1, s2, s3 = st.columns(3)
                for col, key, title in zip([s1, s2, s3], ['st', 'mt', 'lt'], ['⚡ 短期 (5日線)', '🌊 中期 (月線)', '🏔️ 長期 (季線)']):
                    with col:
                        st.markdown(f"""
                        <div class="strategy-card" style="border-left:5px solid {ana[key]['col']};">
                            <h4 style="margin:0; color:#333;">{title}</h4>
                            <h3 style="margin:5px 0; color:{ana[key]['col']};">{ana[key]['txt']}</h3>
                            <p style="font-size:13px; color:#666; margin:0;">{ana[key]['desc']}</p>
                        </div>""", unsafe_allow_html=True)

                # 3. 六大體質雷達 (使用 Metrics 呈現比較清晰)
                st.subheader("🏥 體質健檢 (Fundamentals)")
                f = ana['fund']
                f1, f2, f3, f4, f5 = st.columns(5)
                f1.metric("本益比 P/E", f"{f.get('pe', 0):.1f}" if f.get('pe') else "N/A")
                f2.metric("股價淨值比 P/B", f"{f.get('pb', 0):.2f}" if f.get('pb') else "N/A")
                f3.metric("殖利率 Yield", f"{f.get('yield', 0):.2f}%" if f.get('yield') else "N/A")
                f4.metric("ROE", f"{f.get('roe', 0):.1f}%" if f.get('roe') else "N/A")
                f5.metric("波動率 Beta", f"{f.get('beta', 0):.2f}" if f.get('beta') else "N/A")
                
                # 4. 圖表區
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
                # K線 + 布林
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], increasing_line_color='#D32F2F', decreasing_line_color='#2E7D32', name='K線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Upper'], line=dict(color='rgba(0,100,255,0.3)', width=1), name='布林上軌'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_Lower'], line=dict(color='rgba(0,100,255,0.3)', width=1), name='布林下軌', fill='tonexty', fillcolor='rgba(0,100,255,0.05)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='#FF9800'), name='月線'), row=1, col=1)
                # KD
                fig.add_trace(go.Scatter(x=hist.index, y=hist['K'], line=dict(color='#9C27B0'), name='K'), row=2, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['D'], line=dict(color='#E91E63'), name='D'), row=2, col=1)
                # MACD
                colors = ['#D32F2F' if v >= 0 else '#2E7D32' for v in hist['MACD_Hist']]
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], marker_color=colors, name='MACD'), row=3, col=1)
                
                fig.update_layout(height=800, template="plotly_white", xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("查無資料或代號錯誤")

# Tab 4: 資產透視
with tab4:
    st.markdown("### 💰 資產透視")
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        view_filter = st.radio("顯示市場", ["全部", "台股僅見", "美股僅見"], horizontal=True)
    with filter_col2:
        st.write("")
        st.write("") 
        show_only_held = st.checkbox("只顯示目前持倉 (隱藏已出清)", value=False)
    
    df_raw = load_data()
    if not df_raw.empty:
        if "台股" in view_filter: df_raw = df_raw[df_raw['Market'] == 'TW']
        elif "美股" in view_filter: df_raw = df_raw[df_raw['Market'] == 'US']
        
        if not df_raw.empty:
            p_df, t_mkt, t_unreal, t_real, m_df = calculate_full_portfolio(df_raw)
            if show_only_held: p_df = p_df[p_df['庫存'] > 0]
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("總市值", f"${t_mkt:,.0f}")
            k2.metric("未實現損益", f"${t_unreal:,.0f}", delta=f"{(t_unreal/t_mkt*100):.1f}%" if t_mkt>0 else "0%", delta_color="normal")
            k3.metric("已實現+股息", f"${t_real:,.0f}")
            k4.metric("總損益", f"${(t_unreal+t_real):,.0f}")
            st.markdown("---")
            
            g1, g2 = st.columns([1, 1])
            with g1:
                if not p_df.empty and p_df[p_df['市值']>0].shape[0] > 0:
                    fig_pie = px.pie(p_df[p_df['市值']>0], values='市值', names='名稱', hole=0.4, title="現有持倉分佈")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else: st.info("無持倉市值")
            with g2:
                if not m_df.empty:
                    m_df['Color'] = m_df['PnL'].apply(lambda x: '#D32F2F' if x >= 0 else '#2E7D32')
                    fig_bar = px.bar(m_df, x='Month', y='PnL', text_auto='.0s', title="每月已實現損益")
                    fig_bar.update_traces(marker_color=m_df['Color'])
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("📋 資產明細表")
            if not p_df.empty:
                st.dataframe(p_df.style.format("{:,.0f}", subset=["庫存", "市值", "未實現", "已實現+息"]).format("{:.2f}", subset=["均價", "現價"]).map(lambda x: 'color: #D32F2F; font-weight:bold' if x > 0 else 'color: #2E7D32; font-weight:bold', subset=['未實現']), use_container_width=True)
            else: st.info("無符合條件資料")
        else: st.info("該市場無資料")
    else: st.info("資料庫無資料")
