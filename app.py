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

# --- 1. 頁面設定 (專業亮色寬版) ---
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")

# CSS: 優化指標卡片與圖表背景，使其更像券商軟體
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
SHEET_NAME = "TradeLog"

@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

def load_data():
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except: return pd.DataFrame()

def save_data(row):
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        sheet.append_row(row)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(str(e))
        return False

def batch_save_data(rows):
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        sheet.append_rows(rows)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(str(e))
        return False

# --- 3. 強化版股票資訊 (含基本面) ---
@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        symbol = str(symbol).strip()
        if symbol.isdigit() and len(symbol) < 4: symbol = symbol.zfill(4)
        query_symbol = f"{symbol}.TW" if symbol.isdigit() else symbol
        
        stock = yf.Ticker(query_symbol)
        info = stock.info
        name = info.get('longName', symbol)
        
        # 嘗試抓取基本面數據
        pe = info.get('trailingPE', 0)
        yield_rate = info.get('dividendYield', 0)
        if yield_rate: yield_rate *= 100 # 轉百分比
        
        return query_symbol, name, pe, yield_rate
    except:
        return symbol, "查無名稱", 0, 0

# --- 4. 專業技術分析 (含 KD, MACD) ---
def calculate_technicals(df):
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # KD (Stochastic Oscillator) (9, 3, 3)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    # 遞迴計算 K, D
    k_list, d_list = [], []
    k, d = 50, 50 # 初始值
    for rsv in df['RSV']:
        if pd.isna(rsv): 
            k_list.append(50); d_list.append(50)
        else:
            k = (2/3) * k + (1/3) * rsv
            d = (2/3) * d + (1/3) * k
            k_list.append(k); d_list.append(d)
    df['K'] = k_list
    df['D'] = d_list
    
    return df

def analyze_full_signal(symbol):
    try:
        sym = str(symbol).strip()
        if sym.isdigit() and len(sym) < 4: sym = sym.zfill(4)
        if sym.isdigit(): sym += ".TW"
        
        stock = yf.Ticker(sym)
        df = stock.history(period="1y") # 抓一年資料畫圖較好看
        if len(df) < 60: return None, {}, 0, 0
        
        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        # 綜合訊號判斷
        score = 0
        reasons = []
        
        # 1. 均線
        if last['Close'] > last['MA20']: score += 1; reasons.append("站上月線")
        if last['MA20'] > last['MA60']: score += 1; reasons.append("均線多頭排列")
        
        # 2. RSI
        if last['RSI'] < 30: score += 1; reasons.append("RSI超賣(反彈機會)")
        elif last['RSI'] > 70: score -= 1; reasons.append("RSI超買(過熱)")
        
        # 3. MACD
        if last['MACD_Hist'] > 0 and df.iloc[-2]['MACD_Hist'] < 0: score += 2; reasons.append("MACD 黃金交叉")
        
        # 4. KD
        if last['K'] > last['D'] and last['K'] < 80: score += 1
        if last['K'] < 20 and last['K'] > last['D']: score += 1; reasons.append("KD 低檔金叉")
        
        # 結論
        if score >= 3: signal, color = "強勢買進 🔥", "#D32F2F"
        elif score >= 1: signal, color = "偏多操作 📈", "#E65100"
        elif score <= -2: signal, color = "建議賣出 📉", "#2E7D32"
        else: signal, color = "區間震盪 ☁️", "#666666"
        
        # 抓取即時基本面
        _, _, pe, yield_rate = get_stock_info(sym.split('.')[0])
        
        analysis = {
            "signal": signal, "color": color, "reasons": reasons,
            "close": last['Close'], "rsi": last['RSI'], "k": last['K'], "d": last['D'],
            "pe": pe, "yield": yield_rate
        }
        return df, analysis
    except Exception as e:
        return None, {}, 0, 0

# --- 5. 核心：資產地圖計算 ---
def get_holdings_map(df):
    holdings = {} # {Symbol: Name}
    df = df.sort_values(by='Date')
    
    # 建立目前庫存表
    inventory = {}
    for _, row in df.iterrows():
        sym = str(row['Symbol']).strip()
        if sym.isdigit() and len(sym) < 4: sym = sym.zfill(4)
        name = row['Name']
        qty = float(row['Quantity'])
        t_type = row['Type']
        
        if sym not in inventory: inventory[sym] = 0
        if "Buy" in t_type or "Dividend" in t_type: inventory[sym] += qty
        elif "Sell" in t_type: inventory[sym] -= qty
        
        if name and name != "查無名稱": holdings[sym] = name

    # 只回傳庫存 > 0
    return {k: v for k, v in holdings.items() if inventory.get(k, 0) > 0.1}

def calculate_full_portfolio(df):
    portfolio = {}
    monthly_pnl = {} # { "2024-01": 5000 }
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    
    for _, row in df.iterrows():
        sym = str(row['Symbol']).strip()
        if sym.isdigit() and len(sym) < 4: sym = sym.zfill(4)
        
        name = row['Name']
        qty = float(row['Quantity'])
        price = float(row['Price'])
        fees = float(row['Fees'])
        tax = float(row['Tax'])
        t_type = row['Type']
        date_str = row['Date'].strftime("%Y-%m")
        
        if sym not in portfolio:
            portfolio[sym] = {'Name': name, 'Qty': 0, 'Cost': 0, 'Realized': 0, 'Div': 0}
        
        if date_str not in monthly_pnl: monthly_pnl[date_str] = 0
            
        p = portfolio[sym]
        
        if "Buy" in t_type:
            p['Cost'] += (qty * price) + fees
            p['Qty'] += qty
        elif "Sell" in t_type:
            # 實現損益計算
            if p['Qty'] > 0:
                avg_cost = p['Cost'] / p['Qty']
                cost_sold = avg_cost * qty
                revenue = (qty * price) - fees - tax
                profit = revenue - cost_sold
                
                p['Realized'] += profit
                monthly_pnl[date_str] += profit
                
                p['Qty'] -= qty
                p['Cost'] -= cost_sold
        elif "Dividend" in t_type:
            p['Div'] += price
            monthly_pnl[date_str] += price # 股息算當月獲利
            p['Qty'] += qty

    # 抓現價
    active_syms = [s for s, v in portfolio.items() if v['Qty'] > 0]
    curr_prices = {}
    if active_syms:
        try:
            q_list = [f"{s}.TW" if s.isdigit() else s for s in active_syms]
            data = yf.Tickers(" ".join(q_list))
            for i, s in enumerate(active_syms):
                try:
                    h = data.tickers[q_list[i]].history(period="1d")
                    curr_prices[s] = h['Close'].iloc[-1] if not h.empty else 0
                except: curr_prices[s] = 0
        except: pass
        
    # 匯總數據
    res = []
    tot_mkt = 0
    tot_unreal = 0
    tot_real = 0
    
    for sym, v in portfolio.items():
        cp = curr_prices.get(sym, 0)
        if abs(v['Qty']) < 0.001: v['Qty'] = 0
        
        mkt = v['Qty'] * cp
        unreal = mkt - v['Cost'] if v['Qty'] > 0 else 0
        
        tot_mkt += mkt
        tot_unreal += unreal
        tot_real += (v['Realized'] + v['Div'])
        
        if v['Qty'] > 0 or v['Realized']!=0 or v['Div']!=0:
            res.append({
                "代號": sym, "名稱": v['Name'], "庫存": v['Qty'], "均價": v['Cost']/v['Qty'] if v['Qty']>0 else 0,
                "現價": cp, "市值": mkt, "未實現": unreal, "已實現+息": v['Realized']+v['Div']
            })
            
    # 整理月損益圖表資料
    m_df = pd.DataFrame(list(monthly_pnl.items()), columns=['Month', 'PnL']).sort_values('Month')
    
    return pd.DataFrame(res), tot_mkt, tot_unreal, tot_real, m_df

# --- 6. 主程式 ---
st.title("💎 專業投資戰情室 Pro")
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易", "📥 匯入", "📊 趨勢戰情", "💰 資產透視"])

# Tab 1: 單筆 (維持簡潔)
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("新增交易")
        itype = st.selectbox("方向", ["Buy", "Sell", "Dividend"])
        idate = st.date_input("日期")
        isym = st.text_input("代號", placeholder="2330")
        
        name = "..."
        rsym = isym
        if isym: rsym, name, _, _ = get_stock_info(isym)
        st.info(f"股票: **{name}**")
        
        iqty = st.number_input("股數", min_value=0.0, step=1000.0)
        iprice = st.number_input("價格", min_value=0.0, step=0.1)
        ifees = st.number_input("手續費", min_value=0.0)
        itax = st.number_input("稅", min_value=0.0)
        
        tot = -(iqty*iprice+ifees) if "Buy" in itype else (iqty*iprice-ifees-itax) if "Sell" in itype else iprice
        st.metric("總金額", f"${tot:,.0f}")
        
        if st.button("送出", type="primary"):
            if save_data([str(idate), itype, rsym, name, iprice, iqty, ifees, itax, tot]): st.success("已儲存")

# Tab 2: 匯入 (維持)
with tab2:
    st.markdown("### 📥 批次匯入")
    # (省略部分重複代碼，邏輯同前一版)
    uploaded_file = st.file_uploader("上傳 CSV (格式同前)", type=["csv"])
    if uploaded_file and st.button("開始匯入"):
        try:
            df_u = pd.read_csv(uploaded_file, dtype={'Symbol': str})
            rows = []
            bar = st.progress(0)
            for i, r in df_u.iterrows():
                rs = str(r['Symbol']).strip()
                if rs.isdigit() and len(rs)<4: rs = rs.zfill(4)
                rsym, name, _, _ = get_stock_info(rs)
                tt = str(r['Type']).capitalize()
                q, p, f, t = float(r['Quantity']), float(r['Price']), float(r['Fees']), float(r['Tax'])
                amt = -(q*p+f) if "Buy" in tt else (q*p-f-t) if "Sell" in tt else p
                rows.append([str(r['Date']), tt, rsym, name, p, q, f, t, amt])
                bar.progress((i+1)/len(df_u))
            if batch_save_data(rows): st.success("匯入成功")
        except Exception as e: st.error(str(e))

# Tab 3: 趨勢戰情 (大幅升級)
with tab3:
    st.markdown("### 🔍 個股全方位診斷")
    df_raw = load_data()
    holdings = get_holdings_map(df_raw) if not df_raw.empty else {}
    
    col_sel, col_search = st.columns([1, 1])
    with col_sel:
        opts = [f"{k} {v}" for k, v in holdings.items()]
        sel = st.selectbox("庫存快選", opts) if opts else None
    with col_search:
        manual = st.text_input("或搜尋代號", placeholder="例如 2330")
    
    target = manual if manual else (sel.split()[0] if sel else "2330")
    
    if target:
        with st.spinner("正在進行多維度技術分析..."):
            hist, ana = analyze_full_signal(target)
        
        if hist is not None:
            # 1. 戰情儀表板 (Metric Dashboard)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("即時股價", f"{ana['close']:.2f}")
            m2.metric("RSI (14)", f"{ana['rsi']:.1f}")
            m3.metric("本益比 P/E", f"{ana['pe']:.1f}" if ana['pe'] else "-")
            m4.metric("殖利率 Yield", f"{ana['yield']:.2f}%" if ana['yield'] else "-")
            
            # 2. AI 訊號區
            st.markdown(f"""
            <div style="background-color:white; padding:15px; border-radius:10px; border:1px solid #ddd; text-align:center; margin-bottom:20px;">
                <span style="color:#666; font-size:16px;">AI 綜合評級</span><br>
                <span style="color:{ana['color']}; font-size:32px; font-weight:bold;">{ana['signal']}</span>
                <br><span style="font-size:14px; color:#555;">{' / '.join(ana['reasons'])}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. 專業圖表 (K線 + MA + KD + MACD)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)
            
            # 主圖 (K線 + MA)
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                increasing_line_color='#D32F2F', decreasing_line_color='#2E7D32', name='K線'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='#FF9800', width=1), name='月線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='#2196F3', width=1), name='季線'), row=1, col=1)
            
            # 副圖1 (KD)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['K'], line=dict(color='#9C27B0', width=1), name='K值'), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['D'], line=dict(color='#E91E63', width=1), name='D值'), row=2, col=1)
            
            # 副圖2 (MACD)
            colors = ['#D32F2F' if v >= 0 else '#2E7D32' for v in hist['MACD_Hist']]
            fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], marker_color=colors, name='MACD柱'), row=3, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='#FF9800', width=1), name='DIF'), row=3, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal_Line'], line=dict(color='#2196F3', width=1), name='MACD'), row=3, col=1)
            
            fig.update_layout(height=800, template="plotly_white", xaxis_rangeslider_visible=False, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# Tab 4: 資產透視 (新增圓餅圖與長條圖)
with tab4:
    st.markdown("### 💰 資產透視")
    with st.spinner("計算庫存與損益中..."):
        df_raw = load_data()
        if not df_raw.empty:
            p_df, t_mkt, t_unreal, t_real, m_df = calculate_full_portfolio(df_raw)
            
            # 1. 核心 KPI
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("總市值", f"${t_mkt:,.0f}")
            k2.metric("未實現損益", f"${t_unreal:,.0f}", delta=f"{(t_unreal/t_mkt*100):.1f}%" if t_mkt>0 else "0%", delta_color="normal")
            k3.metric("已實現+股息", f"${t_real:,.0f}")
            k4.metric("總損益", f"${(t_unreal+t_real):,.0f}")
            
            st.markdown("---")
            
            # 2. 圖表分析區 (Asset Pie + Monthly Bar)
            g1, g2 = st.columns([1, 1])
            
            with g1:
                st.subheader("📊 持倉分布 (市值)")
                if not p_df[p_df['市值']>0].empty:
                    fig_pie = px.pie(p_df[p_df['市值']>0], values='市值', names='名稱', hole=0.4)
                    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else: st.info("無持倉市值")
            
            with g2:
                st.subheader("📅 每月損益 (已實現)")
                if not m_df.empty:
                    # 顏色：賺紅賠綠
                    m_df['Color'] = m_df['PnL'].apply(lambda x: '#D32F2F' if x >= 0 else '#2E7D32')
                    fig_bar = px.bar(m_df, x='Month', y='PnL', text_auto='.0s')
                    fig_bar.update_traces(marker_color=m_df['Color'])
                    fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else: st.info("尚無已實現損益")
            
            # 3. 詳細表格
            st.subheader("📋 庫存明細表")
            if not p_df.empty:
                st.dataframe(
                    p_df.style.format("{:,.0f}", subset=["庫存", "市值", "未實現", "已實現+息"])
                    .format("{:.2f}", subset=["均價", "現價"])
                    .map(lambda x: 'color: #D32F2F; font-weight:bold' if x > 0 else 'color: #2E7D32; font-weight:bold', subset=['未實現']),
                    use_container_width=True
                )
        else:
            st.info("無資料，請先輸入交易")
