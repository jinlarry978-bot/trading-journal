import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import time

# --- 1. 頁面設定 (改回標準亮色) ---
st.set_page_config(page_title="專業投資戰情室", layout="wide", page_icon="📈")

# CSS 微調：只優化卡片邊框，保持白底黑字的高對比度
st.markdown("""
    <style>
    /* 調整指標卡片 (Metric) 增加邊框與陰影，讓它在白底中突顯 */
    div[data-testid="stMetric"] {
        background-color: #F0F2F6; /* 淺灰背景 */
        border: 1px solid #D6D6D6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 讓指標數值更大更清楚 */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 常數與連線 ---
SHEET_NAME = "TradeLog"

@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

# --- 3. 資料庫操作 ---
def load_data():
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def save_data(row_data):
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        sheet.append_row(row_data)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"寫入失敗: {e}")
        return False

def batch_save_data(rows_data):
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        sheet.append_rows(rows_data)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"批次寫入失敗: {e}")
        return False

# --- 4. 輔助函數 ---
@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        symbol = str(symbol).strip()
        if symbol.isdigit() and len(symbol) < 4: symbol = symbol.zfill(4)
        query_symbol = f"{symbol}.TW" if symbol.isdigit() else symbol
        
        stock = yf.Ticker(query_symbol)
        info = stock.info
        name = info.get('longName', symbol)
        return query_symbol, name
    except:
        return symbol, "查無名稱"

def analyze_signal(symbol):
    try:
        symbol = str(symbol).strip()
        if symbol.isdigit() and len(symbol) < 4: symbol = symbol.zfill(4)
        if symbol.isdigit(): symbol += ".TW"
            
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if len(df) < 60: return None, "資料不足"
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        last = df.iloc[-1]
        # 台股習慣：紅漲綠跌
        signal, color = "觀望整理", "#555555" # 灰色
        
        if last['Close'] > last['MA20'] > last['MA60']: 
            signal, color = "強勢多頭 🔥", "#D32F2F" # 深紅 (漲)
        elif last['Close'] < last['MA20'] < last['MA60']: 
            signal, color = "空頭走勢 🔻", "#2E7D32" # 深綠 (跌)
        elif last['RSI'] < 25: 
            signal, color = "超賣反彈機會 ⤴️", "#F57C00" # 橘色
        elif last['RSI'] > 75: 
            signal, color = "超買過熱警示 ⚠️", "#2E7D32" # 綠色警示
        
        return df, {"signal": signal, "color": color, "rsi": last['RSI'], "close": last['Close']}
    except Exception as e:
        return None, str(e)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 5. 核心：庫存計算 ---
def get_current_holdings_map(df):
    portfolio_qty = {}
    portfolio_name = {}
    
    df = df.sort_values(by='Date')
    
    for _, row in df.iterrows():
        sym = str(row['Symbol']).strip()
        if sym.isdigit() and len(sym) < 4: sym = sym.zfill(4)
        
        name = row['Name']
        qty = float(row['Quantity'])
        t_type = row['Type']
        
        if sym not in portfolio_qty: 
            portfolio_qty[sym] = 0
            portfolio_name[sym] = name 
        
        if name and name != "查無名稱":
            portfolio_name[sym] = name

        if "Buy" in t_type or "Dividend" in t_type:
            portfolio_qty[sym] += qty
        elif "Sell" in t_type:
            portfolio_qty[sym] -= qty
            
    active_holdings = {}
    for sym, qty in portfolio_qty.items():
        if qty > 0.1:
            active_holdings[sym] = portfolio_name.get(sym, sym)
    return active_holdings

def calculate_portfolio_full(df):
    portfolio = {}
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
        
        if sym not in portfolio:
            portfolio[sym] = {'Name': name, 'Qty': 0, 'Total_Cost': 0, 'Realized_PnL': 0, 'Dividend': 0}
        p = portfolio[sym]
        
        if "Buy" in t_type:
            p['Total_Cost'] += (qty * price) + fees
            p['Qty'] += qty
        elif "Sell" in t_type:
            if p['Qty'] > 0:
                avg_cost = p['Total_Cost'] / p['Qty']
                cost_of_sold = avg_cost * qty
                revenue = (qty * price) - fees - tax
                p['Realized_PnL'] += (revenue - cost_of_sold)
                p['Qty'] -= qty
                p['Total_Cost'] -= cost_of_sold
            else:
                p['Realized_PnL'] += (qty * price) - fees - tax
                p['Qty'] -= qty
        elif "Dividend" in t_type:
            p['Dividend'] += price
            p['Qty'] += qty

    results = []
    tickers_list = [s for s, v in portfolio.items() if v['Qty'] > 0]
    
    current_prices = {}
    if tickers_list:
        try:
            query_list = [f"{s}.TW" if s.isdigit() else s for s in tickers_list]
            tickers_str = " ".join(query_list)
            data = yf.Tickers(tickers_str)
            for i, sym in enumerate(tickers_list):
                try:
                    q_sym = query_list[i]
                    hist = data.tickers[q_sym].history(period="1d")
                    current_prices[sym] = hist['Close'].iloc[-1] if not hist.empty else 0
                except: current_prices[sym] = 0
        except: pass

    total_mkt, total_unreal, total_real = 0, 0, 0
    for sym, v in portfolio.items():
        curr_price = current_prices.get(sym, 0)
        if abs(v['Qty']) < 0.001: v['Qty'] = 0
        
        mkt_val = v['Qty'] * curr_price
        unreal = mkt_val - v['Total_Cost'] if v['Qty'] > 0 else 0
        
        total_mkt += mkt_val
        total_unreal += unreal
        total_real += (v['Realized_PnL'] + v['Dividend'])
        
        if v['Qty'] > 0 or v['Realized_PnL'] != 0 or v['Dividend'] != 0:
            results.append({
                "代號": sym, "名稱": v['Name'], "庫存股數": v['Qty'], 
                "平均成本": v['Total_Cost']/v['Qty'] if v['Qty']>0 else 0,
                "現價": curr_price, "市值": mkt_val, "未實現損益": unreal,
                "已實現+股息": v['Realized_PnL'] + v['Dividend']
            })
            
    return pd.DataFrame(results), total_mkt, total_unreal, total_real

# --- 6. 主程式介面 ---
st.title("📈 專業投資戰情室")
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易錄入", "📥 大量匯入", "📊 持股訊號", "💰 資產庫存"])

# === Tab 1: 單筆錄入 ===
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("新增單筆")
        input_type = st.selectbox("類別", ["Buy", "Sell", "Dividend"])
        tx_date = st.date_input("日期")
        symbol_input = st.text_input("代號", placeholder="例如 2330")
        
        stock_name = "..."
        real_symbol = symbol_input
        if symbol_input:
            real_symbol, stock_name = get_stock_info(symbol_input)
        st.info(f"股票: **{stock_name}**")
        
        qty = st.number_input("股數", min_value=0.0, step=1000.0)
        price = st.number_input("價格/股息總額", min_value=0.0, step=0.1)
        fees = st.number_input("手續費", min_value=0.0)
        tax = st.number_input("交易稅", min_value=0.0)
        
        total = -(qty*price+fees) if input_type=="Buy" else (qty*price-fees-tax) if input_type=="Sell" else price
        st.metric("預估金額", f"${total:,.0f}")

        if st.button("寫入", type="primary"):
            row = [str(tx_date), input_type, real_symbol, stock_name, price, qty, fees, tax, total]
            if save_data(row): st.success("已儲存！")

# === Tab 2: 批次匯入 ===
with tab2:
    st.header("📥 批次匯入")
    template_data = {"Date": ["2024-01-01"], "Type": ["Buy"], "Symbol": ["0050"], "Price": [150], "Quantity": [1000], "Fees": [20], "Tax": [0]}
    st.download_button("下載範本", convert_df(pd.DataFrame(template_data)), "template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("上傳 CSV", type=["csv"])
    if uploaded_file and st.button("開始匯入"):
        try:
            df_up = pd.read_csv(uploaded_file, dtype={'Symbol': str})
            rows = []
            progress = st.progress(0)
            status = st.empty()
            
            for i, row in df_up.iterrows():
                r_sym = str(row['Symbol']).strip()
                if r_sym.isdigit() and len(r_sym)<4: r_sym = r_sym.zfill(4)
                real_sym, name = get_stock_info(r_sym)
                
                t_type = str(row['Type']).capitalize()
                q, p, f, t = float(row['Quantity']), float(row['Price']), float(row['Fees']), float(row['Tax'])
                amt = -(q*p+f) if "Buy" in t_type else (q*p-f-t) if "Sell" in t_type else p
                
                rows.append([str(row['Date']), t_type, real_sym, name, p, q, f, t, amt])
                progress.progress((i+1)/len(df_up))
                status.text(f"處理中: {name}")
                time.sleep(0.1)
            
            if batch_save_data(rows): st.success(f"匯入 {len(rows)} 筆！")
        except Exception as e: st.error(f"錯誤: {e}")

# === Tab 3: 持股訊號 ===
with tab3:
    st.header("🔍 持股技術診斷")
    
    df_sig = load_data()
    
    if not df_sig.empty:
        holdings_map = get_current_holdings_map(df_sig)
        
        if holdings_map:
            # 顯示格式： 0050 元大台灣50
            options = [f"{sym} {name}" for sym, name in holdings_map.items()]
            selected_option = st.selectbox("選擇庫存股票", options)
            selected_symbol = selected_option.split()[0]
            
            st.markdown("---")
            manual_search = st.text_input("或查詢其他代號", placeholder="輸入代號")
            target_stock = manual_search if manual_search else selected_symbol
            
            if target_stock:
                display_name = holdings_map.get(target_stock, target_stock)
                with st.spinner(f"分析中: {target_stock} ..."):
                    hist, ana = analyze_signal(target_stock)
                
                if hist is not None:
                    # 指標卡片 (背景淺灰，字體黑)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("即時股價", f"{ana['close']:.2f}")
                    c2.metric("RSI (14)", f"{ana['rsi']:.1f}")
                    
                    # 訊號燈 (白底，字體帶顏色)
                    c3.markdown(f"""
                        <div style="background-color:white; padding:10px; border:1px solid #ddd; border-radius:5px; text-align:center;">
                            <p style="color:#666; font-size:14px; margin:0;">AI 建議</p>
                            <p style="color:{ana['color']}; font-size:24px; font-weight:bold; margin:0;">{ana['signal']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 亮色版 K 線圖
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    # 紅漲綠跌 K 線
                    fig.add_trace(go.Candlestick(
                        x=hist.index, 
                        open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                        increasing_line_color='#D32F2F', decreasing_line_color='#2E7D32', # 台股紅漲綠跌
                        name='K線'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], line=dict(color='#FF9800', width=1), name='20MA'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA60'], line=dict(color='#2196F3', width=1), name='60MA'), row=1, col=1)
                    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color='#9E9E9E', name='成交量'), row=2, col=1)
                    
                    # 亮色圖表主題
                    fig.update_layout(
                        title=f"{target_stock} K線圖",
                        height=550, 
                        template="plotly_white", # 改為亮色主題
                        xaxis_rangeslider_visible=False, 
                        showlegend=False,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("無庫存")
    else:
        st.warning("無資料")

# === Tab 4: 資產庫存 (顏色修正) ===
with tab4:
    st.header("💰 資產庫存")
    with st.spinner("計算中..."):
        df_raw = load_data()
        if not df_raw.empty:
            p_df, t_mkt, t_unreal, t_real = calculate_portfolio_full(df_raw)
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("總市值", f"${t_mkt:,.0f}")
            # 台股習慣：紅賺(+) 綠賠(-)
            k2.metric("未實現損益", f"${t_unreal:,.0f}", delta=f"{(t_unreal/t_mkt*100):.1f}%" if t_mkt>0 else "0%", delta_color="normal")
            k3.metric("已實現+股息", f"${t_real:,.0f}")
            k4.metric("總損益", f"${(t_unreal+t_real):,.0f}")
            
            st.markdown("---")
            if not p_df.empty:
                # 表格字體顏色修正 (紅賺綠賠)
                st.dataframe(
                    p_df.style.format({
                        "庫存股數": "{:,.0f}", "平均成本": "{:.2f}", "現價": "{:.2f}",
                        "市值": "{:,.0f}", "未實現損益": "{:,.0f}", "已實現+股息": "{:,.0f}"
                    }).map(lambda x: 'color: #D32F2F; font-weight: bold' if x > 0 else 'color: #2E7D32; font-weight: bold', subset=['未實現損益']), 
                    use_container_width=True
                )
            else:
                st.info("目前無持倉")
        else:
            st.info("無資料")
