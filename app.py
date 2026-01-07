import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import datetime

# --- 頁面設定 (專業黑底風格) ---
st.set_page_config(page_title="專業投資戰情室", layout="wide", page_icon="📈")
st.markdown("""
    <style>
    .stMetric {background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

# --- 常數設定 ---
SHEET_NAME = "TradeLog"

# --- 連接 Google Sheets ---
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

# --- 股票資訊獲取 (含快取) ---
@st.cache_data(ttl=3600)  # 快取1小時，避免一直請求
def get_stock_info(symbol):
    try:
        # 台股代號若未加 .TW 自動補上 (簡單判斷)
        if symbol.isdigit() and len(symbol) == 4:
            symbol = f"{symbol}.TW"
        
        stock = yf.Ticker(symbol)
        info = stock.info
        # 嘗試獲取中文名稱 (Yahoo Finance 有時只給英文，這裡做簡單處理)
        name = info.get('longName', symbol)
        return symbol, name
    except:
        return symbol, "查無名稱"

# --- 技術分析訊號判斷 ---
def analyze_signal(symbol):
    try:
        if symbol.isdigit(): symbol += ".TW"
        stock = yf.Ticker(symbol)
        # 抓取過去 100 天數據
        df = stock.history(period="6mo")
        
        if len(df) < 60: return None, "資料不足"
        
        # 計算指標
        df['MA20'] = df['Close'].rolling(window=20).mean() # 月線
        df['MA60'] = df['Close'].rolling(window=60).mean() # 季線
        
        # RSI 計算 (簡單版)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        last_close = df['Close'].iloc[-1]
        last_ma20 = df['MA20'].iloc[-1]
        last_ma60 = df['MA60'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        
        # 訊號邏輯
        signal = "觀望 (Neutral)"
        color = "gray"
        
        # 多頭排列
        if last_close > last_ma20 > last_ma60:
            signal = "強勢多頭 (Strong Buy)"
            color = "green"
        elif last_close < last_ma20 < last_ma60:
            signal = "空頭走勢 (Bearish)"
            color = "red"
        elif last_rsi < 30:
            signal = "超賣區 (反彈機會)"
            color = "orange"
        elif last_rsi > 70:
            signal = "超買區 (回檔風險)"
            color = "red"
            
        return df, {"signal": signal, "color": color, "rsi": last_rsi, "close": last_close}
    except Exception as e:
        return None, str(e)

# --- 主程式介面 ---
st.title("📈 專業投資戰情室")

# 建立分頁
tab1, tab2, tab3 = st.tabs(["📝 交易錄入", "📊 訊號分析", "🗃️ 資產明細"])

# === Tab 1: 交易錄入 ===
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("新增紀錄")
        input_type = st.selectbox("交易類別", ["買入股票 (Buy)", "賣出股票 (Sell)", "股息收入 (Dividend)"])
        
        tx_date = st.date_input("日期")
        symbol_input = st.text_input("股票代號 (例: 2330)", placeholder="輸入後按Enter自動抓名稱")
        
        # 自動抓取名稱邏輯
        stock_name = "等待輸入..."
        real_symbol = symbol_input
        if symbol_input:
            real_symbol, stock_name = get_stock_info(symbol_input)
        
        st.info(f"股票名稱: **{stock_name}**")
        
        qty = 0.0
        price = 0.0
        fees = 0.0
        tax = 0.0
        cash_div = 0.0
        
        # 根據不同類別顯示不同輸入框
        if "Buy" in input_type:
            qty = st.number_input("購買股數", min_value=1, step=1000)
            price = st.number_input("成交單價", min_value=0.0, step=0.1)
            fees = st.number_input("手續費", min_value=0)
            total = -(qty * price + fees) # 買入為流出資金
            st.metric("預估交割金額", f"${total:,.0f}")
            
        elif "Sell" in input_type:
            qty = st.number_input("賣出股數", min_value=1, step=1000)
            price = st.number_input("成交單價", min_value=0.0, step=0.1)
            fees = st.number_input("手續費", min_value=0)
            tax = st.number_input("交易稅", min_value=0)
            total = (qty * price - fees - tax) # 賣出為流入資金
            st.metric("預估入帳金額", f"${total:,.0f}")
            
        elif "Dividend" in input_type:
            qty = st.number_input("配股數量 (股)", min_value=0.0)
            cash_div = st.number_input("配息金額 (元)", min_value=0.0)
            total = cash_div
            st.metric("總股息收入", f"${total:,.0f}")

        if st.button("確認寫入資料庫", type="primary"):
            if not symbol_input:
                st.error("請輸入股票代號")
            else:
                # 準備資料
                row = [
                    str(tx_date), 
                    input_type.split()[0], # 只存 '買入股票' 等簡短字
                    real_symbol, 
                    stock_name, 
                    price if "Dividend" not in input_type else cash_div, 
                    qty, 
                    fees, 
                    tax, 
                    total
                ]
                if save_data(row):
                    st.success(f"已儲存 {stock_name} 的交易紀錄！")

# === Tab 2: 訊號分析 (看盤軟體風格) ===
with tab2:
    st.header("🔍 個股趨勢診斷")
    target_stock = st.text_input("輸入代號查看 K 線與訊號", value="2330")
    
    if target_stock:
        with st.spinner("正在進行技術分析運算..."):
            hist_df, analysis = analyze_signal(target_stock)
            
        if hist_df is not None:
            # 顯示訊號燈
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("目前股價", f"{analysis['close']:.2f}")
            s_col2.metric("RSI (14)", f"{analysis['rsi']:.1f}")
            s_col3.markdown(f"#### 系統建議: <span style='color:{analysis['color']}'>{analysis['signal']}</span>", unsafe_allow_html=True)
            
            # 繪製專業 K 線圖
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # K線
            fig.add_trace(go.Candlestick(x=hist_df.index,
                            open=hist_df['Open'], high=hist_df['High'],
                            low=hist_df['Low'], close=hist_df['Close'], name='K線'), row=1, col=1)
            
            # 均線
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA20'], line=dict(color='orange', width=1), name='月線 (20MA)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA60'], line=dict(color='blue', width=1), name='季線 (60MA)'), row=1, col=1)

            # 成交量
            fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], marker_color='gray', name='成交量'), row=2, col=1)

            fig.update_layout(title=f"{target_stock} 技術分析圖表", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("無法取得資料，請確認代號是否正確 (台股建議加 .TW，系統已嘗試自動加入)")

# === Tab 3: 資產明細 ===
with tab3:
    st.subheader("🗃️ 歷史交易流水帳")
    df = load_data()
    if not df.empty:
        # 簡單整理顯示
        st.dataframe(df, use_container_width=True)
        
        # 簡單統計
        st.markdown("---")
        total_in = df[df['Total_Amt'] > 0]['Total_Amt'].sum()
        total_out = df[df['Total_Amt'] < 0]['Total_Amt'].sum()
        st.metric("淨現金流 (已實現+股息-投入成本)", f"${total_in + total_out:,.0f}", delta_color="normal")
    else:
        st.info("目前無交易紀錄")
