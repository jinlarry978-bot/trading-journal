import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 頁面設定 ---
st.set_page_config(page_title="專業交易員日誌", layout="wide")
st.title("📈 Pro Trading Journal & Analytics")

# --- 1. 資料處理核心 ---
# 定義資料庫檔案名稱
DATA_FILE = "trades.csv"

# 載入資料函數
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            return pd.read_csv(DATA_FILE)
        except Exception as e:
            st.error(f"讀取資料庫失敗: {e}")
            return pd.DataFrame()
    else:
        # 若檔案不存在，回傳空的 DataFrame 結構
        return pd.DataFrame(columns=[
            'Date', 'Symbol', 'Type', 'Entry_Price', 
            'Exit_Price', 'Quantity', 'Fees', 'Strategy', 'Notes', 'Status'
        ])

# 讀取資料
df = load_data()

# 確保 Date 欄位是日期格式
if not df.empty:
    df['Date'] = pd.to_datetime(df['Date'])

# --- Side Bar: 交易錄入介面 ---
st.sidebar.header("📝 新增交易紀錄")

with st.sidebar.form("entry_form"):
    date = st.date_input("進場日期")
    symbol = st.text_input("股票代號 (Symbol)").upper()
    trade_type = st.selectbox("交易方向", ["Long (做多)", "Short (做空)"])
    entry_price = st.number_input("進場價格", min_value=0.0, step=0.1, format="%.2f")
    exit_price = st.number_input("出場價格 (若持倉中填0)", min_value=0.0, step=0.1, format="%.2f")
    qty = st.number_input("股數 (Quantity)", min_value=1, step=100)
    fees = st.number_input("總手續費 (Fees)", min_value=0.0, step=1.0, format="%.2f")
    strategy = st.selectbox("使用策略", ["Breakout (突破)", "Pullback (回檔)", "Reversal (反轉)", "Trend Follow (順勢)", "Other"])
    status = st.selectbox("狀態", ["Closed (已平倉)", "Open (持倉中)"])
    notes = st.text_area("交易筆記 (進場理由/檢討)")
    
    submitted = st.form_submit_button("💾 儲存交易")

    if submitted:
        # 處理資料格式
        type_val = "Long" if "Long" in trade_type else "Short"
        status_val = "Closed" if "Closed" in status else "Open"
        
        new_data = {
            'Date': date, 'Symbol': symbol, 'Type': type_val,
            'Entry_Price': entry_price, 'Exit_Price': exit_price,
            'Quantity': qty, 'Fees': fees, 'Strategy': strategy,
            'Notes': notes, 'Status': status_val
        }
        
        # 將新資料合併並存檔
        new_df = pd.DataFrame([new_data])
        # 處理日期格式以確保寫入 CSV 正確
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("✅ 交易已成功儲存！請點擊右上角 Rerun 更新數據。")

# --- 2. 趨勢自動分析邏輯 ---
if not df.empty and 'Closed' in df['Status'].values:
    # 過濾出已平倉交易進行分析
    closed_trades = df[df['Status'] == 'Closed'].copy()
    
    if not closed_trades.empty:
        # 計算單筆損益 (PnL)
        # Long: (Exit - Entry) * Qty - Fees
        # Short: (Entry - Exit) * Qty - Fees
        closed_trades['PnL'] = closed_trades.apply(
            lambda x: ((x['Exit_Price'] - x['Entry_Price']) * x['Quantity'] - x['Fees']) if x['Type'] == 'Long' 
            else ((x['Entry_Price'] - x['Exit_Price']) * x['Quantity'] - x['Fees']), axis=1
        )
        
        # 累積損益 (Equity Curve)
        closed_trades = closed_trades.sort_values(by='Date')
        closed_trades['Cumulative_PnL'] = closed_trades['PnL'].cumsum()

        # --- 3. 儀表板顯示 ---
        
        # KPI 指標列
        total_pnl = closed_trades['PnL'].sum()
        win_count = len(closed_trades[closed_trades['PnL'] > 0])
        total_count = len(closed_trades)
        win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
        
        # 獲利因子 (Profit Factor)
        gross_profit = closed_trades[closed_trades['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(closed_trades[closed_trades['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

        # KPI 顯示
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 總損益 (Net PnL)", f"${total_pnl:,.0f}")
        col2.metric("🎯 勝率 (Win Rate)", f"{win_rate:.1f}%")
        col3.metric("📊 獲利因子 (PF)", f"{profit_factor:.2f}")
        col4.metric("📝 總交易數", total_count)

        st.markdown("---")

        # 圖表區
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 資金權益曲線 (Equity Curve)")
            if len(closed_trades) > 0:
                fig_equity = px.line(closed_trades, x='Date', y='Cumulative_PnL', markers=True, title="帳戶淨值走勢")
                st.plotly_chart(fig_equity, use_container_width=True)

        with col_chart2:
            st.subheader("🧠 策略績效分析")
            if len(closed_trades) > 0:
                strategy_perf = closed_trades.groupby('Strategy')['PnL'].sum().reset_index()
                fig_strategy = px.bar(strategy_perf, x='Strategy', y='PnL', color='PnL', title="各策略損益比較")
                st.plotly_chart(fig_strategy, use_container_width=True)

        # 詳細數據表 (顯示所有資料，包含持倉)
        st.subheader("🗃️ 詳細交易紀錄")
        
        # 格式化顯示
        display_df = df.sort_values(by='Date', ascending=False).copy()
        display_df['Date'] = display_df['Date'].dt.date # 只顯示日期
        st.dataframe(display_df, use_container_width=True)
        
    else:
        st.info("尚無「已平倉 (Closed)」的交易紀錄。")
else:
    st.info("👋 歡迎使用！目前沒有交易資料。請從左側側邊欄輸入您的第一筆交易。")
    st.markdown("""
    **快速開始指南：**
    1. 在左側填寫交易資訊。
    2. 如果是正在持有的股票，狀態選 **Open**。
    3. 如果已經賣出，狀態選 **Closed** 並填寫出場價。
    4. 系統會自動計算 **Closed** 狀態的損益並繪圖。
    """)
