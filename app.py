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
    div[data-testid="stMetricLabel"] p {font-size: 14px; color: #666;}
    div[data-testid="stMetricValue"] {font-size: 24px !important; font-weight: 700 !important;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 連線設定 ---
SHEET_TW = "TW_Trades"
SHEET_US = "US_Trades"

# 內建熱門股字典
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

# --- 核心更新：代號標準化函數 (支援去除單引號與正確補零) ---
def standardize_symbol(symbol):
    """
    將代號標準化：
    1. 移除單引號 ' (處理 Excel 文字格式 '00878)
    2. 移除空白
    3. 針對純數字補零：
       - 878 -> 00878 (3碼補2個0)
       - 50  -> 0050  (2碼補2個0)
       - 2330 -> 2330 (4碼不變)
    """
    # 1. 強制轉字串並移除單引號與空白
    s = str(symbol).replace("'", "").strip().upper()
    
    # 2. 如果剩下的是純數字，進行智慧補零
    if s.isdigit():
        if len(s) == 3: return "00" + s # 878 -> 00878
        if len(s) == 2: return "00" + s # 50 -> 0050
        if len(s) < 4: return s.zfill(4) # 其他少於4碼的預設補零
        
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

# --- 3. 股票資訊 (使用標準化函數) ---
def get_stock_info(symbol):
    try:
        # 使用新的標準化邏輯 (去除單引號 + 補零)
        clean_symbol = standardize_symbol(symbol)
        
        if clean_symbol.isdigit(): query_symbol = f"{clean_symbol}.TW"
        else: query_symbol = clean_symbol
            
        if clean_symbol in KNOWN_STOCKS:
            return query_symbol, KNOWN_STOCKS[clean_symbol], 0, 0
            
        stock = yf.Ticker(query_symbol)
        try:
            info = stock.info
            name = info.get('longName') or info.get('shortName') or clean_symbol
            pe = info.get('trailingPE', 0)
            yield_rate = info.get('dividendYield', 0)
            if yield_rate: yield_rate *= 100
        except:
            name = clean_symbol
            pe = 0
            yield_rate = 0
        return query_symbol, name, pe, yield_rate
    except: return symbol, "查無名稱", 0, 0

# --- 4. 技術分析 ---
def calculate_technicals(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
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
        clean_sym = standardize_symbol(symbol)
        
        if clean_sym.isdigit(): query_symbol = f"{clean_sym}.TW"
        else: query_symbol = clean_sym
            
        stock = yf.Ticker(query_symbol)
        df = stock.history(period="1y")
        if len(df) < 60: return None, None
        
        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        score = 0
        reasons = []
        if last['Close'] > last['MA20']: score += 1; reasons.append("站上月線")
        if last['MA20'] > last['MA60']: score += 1; reasons.append("均線多頭排列")
        if last['RSI'] < 30: score += 1; reasons.append("RSI超賣")
        elif last['RSI'] > 70: score -= 1; reasons.append("RSI超買")
        if last['MACD_Hist'] > 0 and df.iloc[-2]['MACD_Hist'] < 0: score += 2; reasons.append("MACD 金叉")
        if last['K'] < 20 and last['K'] > last['D']: score += 1; reasons.append("KD 低檔金叉")
        
        if score >= 3: signal, color = "強勢買進 🔥", "#D32F2F"
        elif score >= 1: signal, color = "偏多操作 📈", "#E65100"
        elif score <= -2: signal, color = "建議賣出 📉", "#2E7D32"
        else: signal, color = "區間震盪 ☁️", "#666666"
        
        try:
            info = stock.info
            pe = info.get('trailingPE', 0)
            yield_rate = info.get('dividendYield', 0)
            if yield_rate: yield_rate *= 100
        except: pe = 0; yield_rate = 0
        
        analysis = {
            "signal": signal, "color": color, "reasons": reasons,
            "close": last['Close'], "rsi": last['RSI'], "k": last['K'], "d": last['D'],
            "pe": pe, "yield": yield_rate
        }
        return df, analysis
    except: return None, None

# --- 5. 資產計算 ---
def safe_float(val):
    try:
        if pd.isna(val) or val == "": return 0.0
        return float(val)
    except: return 0.0

def get_sort_rank(t_type):
    t_type = str(t_type)
    if "Buy" in t_type or "買" in t_type or "配股" in t_type: return 1
    if "Sell" in t_type or "賣" in t_type: return 2
    return 3

def calculate_full_portfolio(df):
    portfolio = {}
    monthly_pnl = {}
    
    df['日期'] = df['日期'].apply(standardize_date)
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce') 
    df = df.dropna(subset=['日期'])
    
    df['Rank'] = df['類別'].apply(get_sort_rank)
    df = df.sort_values(by=['日期', 'Rank'])
    
    for _, row in df.iterrows():
        # 這裡也要應用標準化，確保資料庫取出的代號被正確補零
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

# Tab 1: 單筆
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
            rsym, name, _, _ = get_stock_info(check_sym)
        
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
                st.success(f"已儲存至 {'台股' if is_tw_stock(rsym) else '美股'} 分頁")

# Tab 2: 匯入
with tab2:
    st.markdown("### 📥 批次匯入 (優先使用檔案名稱)")
    
    template_data = {
        "日期": ["2024-01-01", "2024-02-01", "2024-07-15", "2024-08-20", "2024-09-01"], 
        "類別": ["買入", "賣出", "股息", "股息", "股息"], 
        "代號": ["0050", "0050", "2330", "2884", "2317"],
        "名稱": ["元大台灣50", "元大台灣50", "台積電", "玉山金", "鴻海"], 
        "價格": [150, 160, 5000, 0, 2000],   
        "股數": [1000, 500, 0, 50, 20],      
        "手續費": [20, 20, 10, 0, 0], 
        "交易稅": [0, 100, 0, 0, 0]
    }
    
    st.download_button(
        label="📥 下載 Excel 完整範本 (.xlsx)",
        data=convert_to_excel(pd.DataFrame(template_data)),
        file_name="trade_template_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
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
            
            tw_rows = []
            us_rows = []
            bar = st.progress(0.0)
            status = st.empty()
            total = len(df_u)
            
            for i, (index, r) in enumerate(df_u.iterrows()):
                # 使用標準化函數處理代號
                clean_sym = standardize_symbol(r['代號'])
                
                excel_name = str(r.get('名稱', '')).strip()
                if excel_name and excel_name.lower() != 'nan':
                    name = excel_name
                else:
                    query_sym = f"{clean_sym}.TW" if clean_sym.isdigit() else clean_sym
                    _, name, _, _ = get_stock_info(query_sym)
                
                tt_raw = str(r['類別'])
                tt = "買入" if any(x in tt_raw for x in ["Buy","買"]) else "賣出" if any(x in tt_raw for x in ["Sell","賣"]) else "股息"
                
                q = safe_float(r['股數'])
                p = safe_float(r['價格'])
                f = safe_float(r['手續費'])
                t = safe_float(r['交易稅'])
                
                amt = -(q*p+f) if "買" in tt else (q*p-f-t) if "賣" in tt else p
                
                row_data = [str(r['日期']), tt, clean_sym, name, p, q, f, t, amt]
                
                if is_tw_stock(clean_sym): tw_rows.append(row_data)
                else: us_rows.append(row_data)
                
                if total > 0:
                    val = (i + 1) / total
                    if val > 1.0: val = 1.0
                    bar.progress(val)
                
                status.text(f"處理中: {clean_sym} - {name}")
            
            msg = ""
            if tw_rows:
                _, added_tw, dup_tw = batch_save_data_smart(tw_rows, 'TW')
                msg += f"🇹🇼 台股: 新增 {added_tw} 筆。 "
            if us_rows:
                _, added_us, dup_us = batch_save_data_smart(us_rows, 'US')
                msg += f"🇺🇸 美股: 新增 {added_us} 筆。"
            
            if not tw_rows and not us_rows:
                st.warning("無有效資料匯入。")
            else:
                st.success(f"匯入完成！ {msg}")
            
        except Exception as e: st.error(f"匯入失敗: {str(e)}")

# Tab 3 (保持不變)
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
            if "買" in tt or "Buy" in tt or "股" in tt: inventory[sym] = inventory.get(sym, 0) + q
            elif "賣" in tt or "Sell" in tt: inventory[sym] = inventory.get(sym, 0) - q
            names[sym] = row['名稱']
        active_list = [f"{k} {names[k]}" for k, v in inventory.items() if v > 0.1]
        col_sel, col_search = st.columns([1, 1])
        with col_sel:
            sel = st.selectbox("庫存快選", active_list) if active_list else None
        with col_search:
            manual = st.text_input("或搜尋代號", placeholder="例如 2330")
        target = manual if manual else (sel.split()[0] if sel else None)
        if target:
            with st.spinner("分析中..."):
                hist, ana = analyze_full_signal(target)
            if hist is not None:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("股價", f"{ana['close']:.2f}")
                m2.metric("RSI", f"{ana['rsi']:.1f}")
                m3.metric("本益比", f"{ana['pe']:.1f}" if ana['pe'] else "-")
                m4.metric("殖利率", f"{ana['yield']:.2f}%" if ana['yield'] else "-")
                st.markdown(f"""<div style="background-color:white; padding:10px; border-radius:10px; border:1px solid #ddd; text-align:center; margin-bottom:10px;"><span style="color:{ana['color']}; font-size:24px; font-weight:bold;">{ana['signal']}</span><br><span style="font-size:14px; color:#555;">{' / '.join(ana['reasons'])}</span></div>""", unsafe_allow_html=True)
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], increasing_line_color='#D32F2F', decreasing_line_color='#2E7D32', name='K線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='#FF9800'), name='MA20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['K'], line=dict(color='#9C27B0'), name='K'), row=2, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['D'], line=dict(color='#E91E63'), name='D'), row=2, col=1)
                colors = ['#D32F2F' if v >= 0 else '#2E7D32' for v in hist['MACD_Hist']]
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], marker_color=colors, name='MACD'), row=3, col=1)
                fig.update_layout(height=700, template="plotly_white", xaxis_rangeslider_visible=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("查無資料，請檢查代號是否正確。")

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
                else: st.info("目前無持倉市值可畫圖")
            with g2:
                if not m_df.empty:
                    m_df['Color'] = m_df['PnL'].apply(lambda x: '#D32F2F' if x >= 0 else '#2E7D32')
                    fig_bar = px.bar(m_df, x='Month', y='PnL', text_auto='.0s', title="每月已實現損益")
                    fig_bar.update_traces(marker_color=m_df['Color'])
                    st.plotly_chart(fig_bar, use_container_width=True)
            st.subheader("📋 資產明細表")
            if not p_df.empty:
                st.dataframe(p_df.style.format("{:,.0f}", subset=["庫存", "市值", "未實現", "已實現+息"]).format("{:.2f}", subset=["均價", "現價"]).map(lambda x: 'color: #D32F2F; font-weight:bold' if x > 0 else 'color: #2E7D32; font-weight:bold', subset=['未實現']), use_container_width=True)
            else: st.info("沒有符合條件的持倉資料。")
        else: st.info("該市場目前無任何交易紀錄")
    else: st.info("資料庫尚無資料")
