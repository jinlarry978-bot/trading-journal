import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- 頁面設定 ---
st.set_page_config(page_title="專業交易員日誌", layout="wide")
st.title("📈 Pro Trading Journal (Google Sheets 連線版)")

# --- 設定常數 ---
SHEET_NAME = "TradeLog"  # 您的 Google Sheet 名稱

# --- 連接 Google Sheets 的函數 ---
@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # 從 Secrets 讀取金鑰
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    return client

# --- 讀取資料函數 ---
def load_data():
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        data = sheet.get_all_records()
        if not data:
            return pd.DataFrame(columns=[
                'Date', 'Symbol', 'Type', 'Entry_Price', 
                'Exit_Price', 'Quantity', 'Fees', 'Strategy', 'Notes', 'Status'
            ])
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"⚠️ 連線錯誤: {e}")
        st.info("請檢查：1. Google Sheet 是否命名為 TradeLog？ 2. 是否已共用給機器人 Email？ 3. Secrets 是否設定正確？")
        return pd.DataFrame()

# --- 寫入資料函數 ---
def save_data(row_data):
    try:
        client = init_connection()
        sheet = client.open(SHEET_NAME).sheet1
        sheet.append_row(row_data)
        st.cache_data.clear() # 清除快取
        return True
    except Exception as e:
        st.error(f"寫入錯誤: {e}")
        return False

# --- 主程式 ---
df = load_data()

# 資料型態轉換
if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])
    cols_to_num = ['Entry_Price', 'Exit_Price', 'Quantity', 'Fees']
    for col in cols_to_num:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 側邊欄輸入
st.sidebar.header("📝 新增交易")
with st.sidebar.form("entry_form"):
    date = st.date_input("進場日期")
    symbol = st.text_input("股票代號").upper()
    trade_type = st.selectbox("方向", ["Long", "Short"])
    entry_price = st.number_input("進場價", min_value=0.0, step=0.1)
    exit_price = st.number_input("出場價 (持倉填0)", min_value=0.0, step=0.1)
    qty = st.number_input("股數", min_value=1, step=100)
    fees = st.number_input("手續費", min_value=0.0, step=1.0)
    strategy = st.selectbox("策略", ["Breakout", "Pullback", "Reversal", "Trend", "Other"])
    status = st.selectbox("狀態", ["Closed", "Open"])
    notes = st.text_area("筆記")
    
    submitted = st.form_submit_button("☁️ 上傳至雲端")

    if submitted:
        # 準備寫入的資料
        row_data = [
            str(date), symbol, trade_type, entry_price, exit_price, 
            qty, fees, strategy, notes, status
        ]
        if save_data(row_data):
            st.success("✅ 資料已成功寫入 Google Sheet！")
            st.rerun()

# 分析儀表板
if not df.empty and 'Closed' in df['Status'].values:
    closed_trades = df[df['Status'] == 'Closed'].copy()
    
    if not closed_trades.empty:
        # 計算損益
        closed_trades['PnL'] = closed_trades.apply(
            lambda x: ((x['Exit_Price'] - x['Entry_Price']) * x['Quantity'] - x['Fees']) if x['Type'] == 'Long' 
            else ((x['Entry_Price'] - x['Exit_Price']) * x['Quantity'] - x['Fees']), axis=1
        )
        
        # 指標
        total_pnl = closed_trades['PnL'].sum()
        win_rate = (len(closed_trades[closed_trades['PnL'] > 0]) / len(closed_trades)) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 總損益", f"${total_pnl:,.0f}")
        col2.metric("🎯 勝率", f"{win_rate:.1f}%")
        col3.metric("📦 交易筆數", len(closed_trades))
        
        st.markdown("---")
        
        # 繪圖
        closed_trades = closed_trades.sort_values(by='Date')
        closed_trades['Cum_PnL'] = closed_trades['PnL'].cumsum()
        fig = px.line(closed_trades, x='Date', y='Cum_PnL', title="資金權益曲線", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("交易紀錄表")
        st.dataframe(df)
    else:
        st.info("尚無已平倉 (Closed) 的資料。")
else:
    st.info("👋 資料庫是空的，請輸入第一筆交易測試看看！")
