import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import time

# --- 1. 頁面設定 (必須放在第一行) ---
st.set_page_config(page_title="專業投資戰情室", layout="wide", page_icon="📈")
st.markdown("""
    <style>
    .stMetric {background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. 常數與連線設定 ---
SHEET_NAME = "TradeLog"

@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

# --- 3. 資料庫操作函數 ---
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

# --- 4. 輔助函數 (抓股名、技術分析、CSV下載) ---
@st.cache_data(ttl=3600)
def get_stock_info(symbol):
    try:
        # 強制轉為字串並去除空白
        symbol = str(symbol).strip()
        
        # 智慧補零：如果是純數字且長度小於4，補0 (例如 50 -> 0050)
        if symbol.isdigit() and len(symbol) < 4:
            symbol = symbol.zfill(4)
            
        # 台股代號若未加 .TW 自動補上
        query_symbol = symbol
        if symbol.isdigit():
            query_symbol = f"{symbol}.TW"
        
        stock = yf.Ticker(query_symbol)
        info = stock.info
        name = info.get('longName', symbol)
        return query_symbol, name # 回傳完整的代號(含.TW)與名稱
    except:
        return symbol, "查無名稱"

def analyze_signal(symbol):
    try:
        if str(symbol).isdigit(): symbol += ".TW"
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
        
        last_close = df['Close'].iloc[-1]
        last_ma20 = df['MA20'].iloc[-1]
        last_ma60 = df['MA60'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        
        signal = "觀望 (Neutral)"
        color = "gray"
        
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

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 5. 主程式介面 ---
st.title("📈 專業投資戰情室")

# 關鍵：這裡必須先定義 tab1 到 tab4，程式才知道它們是誰
tab1, tab2, tab3, tab4 = st.tabs(["📝 交易錄入", "📥 大量匯入", "📊 訊號分析", "🗃️ 資產明細"])

# === Tab 1: 單筆錄入 ===
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("新增單筆紀錄")
        input_type = st.selectbox("交易類別", ["Buy", "Sell", "Dividend"])
        tx_date = st.date_input("日期")
        symbol_input = st.text_input("股票代號 (例: 2330)", placeholder="輸入後按Enter")
        
        stock_name = "等待輸入..."
        real_symbol = symbol_input
        if symbol_input:
            real_symbol, stock_name = get_stock_info(symbol_input)
        st.info(f"股票名稱: **{stock_name}**")
        
        qty = st.number_input("股數/配股", min_value=0.0, step=1000.0)
        price = st.number_input("單價/配息金額", min_value=0.0, step=0.1)
        fees = st.number_input("手續費", min_value=0.0)
        tax = st.number_input("交易稅", min_value=0.0)
        
        total = 0.0
        if input_type == "Buy":
            total = -(qty * price + fees)
        elif input_type == "Sell":
            total = (qty * price - fees - tax)
        elif input_type == "Dividend":
            total = price 
            
        st.metric("預估金額", f"${total:,.0f}")

        if st.button("確認寫入", type="primary"):
            if not symbol_input:
                st.error("請輸入股票代號")
            else:
                row = [str(tx_date), input_type, real_symbol, stock_name, price, qty, fees, tax, total]
                if save_data(row):
                    st.success(f"已儲存 {stock_name} 的交易紀錄！")

# === Tab 2: 大量匯入 (含自動補零修正) ===
with tab2:
    st.header("📥 批次匯入交易紀錄")
    st.markdown("""
    **使用說明：**
    1. 請下載範本 CSV 檔案。
    2. **股票代號若為 0050，Excel 轉 CSV 變 50 沒關係，系統會自動補零。**
    3. Type 請填: `Buy`, `Sell`, 或 `Dividend`。
    """)
    
    template_data = {
        "Date": ["2024-01-01", "2024-02-01"],
        "Type": ["Buy", "Sell"],
        "Symbol": ["0050", "2330"],
        "Price": [150, 600],
        "Quantity": [1000, 2000],
        "Fees": [20, 100],
        "Tax": [0, 300]
    }
    template_df = pd.DataFrame(template_data)
    st.download_button(
        label="📥 下載 CSV 範本",
        data=convert_df(template_df),
        file_name="trade_template.csv",
        mime="text/csv",
    )
    
    uploaded_file = st.file_uploader("上傳您的 CSV 檔案", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # 讀取 CSV (強制代號轉字串)
            df_upload = pd.read_csv(uploaded_file, dtype={'Symbol': str})
            
            st.write("預覽上傳資料：")
            st.dataframe(df_upload.head())
            
            if st.button("🚀 開始匯入資料庫"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                rows_to_upload = []
                total_rows = len(df_upload)
                
                for index, row in df_upload.iterrows():
                    # 智慧補零邏輯
                    raw_symbol = str(row['Symbol']).strip()
                    if raw_symbol.isdigit() and len(raw_symbol) < 4:
                        raw_symbol = raw_symbol.zfill(4)
                    
                    real_symbol, stock_name = get_stock_info(raw_symbol)
                    
                    t_type = str(row['Type']).capitalize()
                    qty = float(row['Quantity'])
                    price = float(row['Price'])
                    fees = float(row['Fees'])
                    tax = float(row['Tax'])
                    
                    total_amt = 0.0
                    if "Buy" in t_type:
                        total_amt = -(qty * price + fees)
                    elif "Sell" in t_type:
                        total_amt = (qty * price - fees - tax)
                    elif "Dividend" in t_type:
                        total_amt = price 
                        
                    record = [
                        str(row['Date']),
                        t_type,
                        real_symbol,
                        stock_name,
                        price,
                        qty,
                        fees,
                        tax,
                        total_amt
                    ]
                    rows_to_upload.append(record)
                    
                    progress = (index + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"正在處理: {stock_name} ({index+1}/{total_rows})")
                    time.sleep(0.1)
                
                if batch_save_data(rows_to_upload):
                    st.success(f"🎉 成功匯入 {len(rows_to_upload)} 筆交易！")
                    st.balloons()
                
        except Exception as e:
            st.error(f"檔案處理失敗，請檢查格式: {e}")

# === Tab 3: 訊號分析 ===
with tab3:
    st.header("🔍 個股趨勢診斷")
    target_stock = st.text_input("輸入代號", value="2330")
    if target_stock:
        with st.spinner("分析中..."):
            hist_df, analysis = analyze_signal(target_stock)
        if hist_df is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("股價", f"{analysis['close']:.2f}")
            c2.metric("RSI", f"{analysis['rsi']:.1f}")
            c3.markdown(f"#### <span style='color:{analysis['color']}'>{analysis['signal']}</span>", unsafe_allow_html=True)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'], low=hist_df['Low'], close=hist_df['Close'], name='K線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA20'], line=dict(color='orange'), name='20MA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['MA60'], line=dict(color='blue'), name='60MA'), row=1, col=1)
            fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], name='量'), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# === Tab 4: 資產明細 ===
with tab4:
    st.subheader("🗃️ 交易紀錄")
    df = load_data()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        t_in = df[df['Total_Amt'] > 0]['Total_Amt'].sum()
        t_out = df[df['Total_Amt'] < 0]['Total_Amt'].sum()
        st.metric("淨現金流", f"${t_in + t_out:,.0f}")
    else:
        st.info("無資料")
