# === Tab 2: 大量匯入 (修正版) ===
with tab2:
    st.header("📥 批次匯入交易紀錄")
    st.markdown("""
    **使用說明：**
    1. 請下載範本 CSV 檔案。
    2. **股票代號若為 0050，Excel 可能會顯示 50，不用擔心，上傳後系統會自動補 0。**
    3. Type 請填: `Buy`, `Sell`, 或 `Dividend`。
    """)
    
    # 產生範本供下載
    template_data = {
        "Date": ["2024-01-01", "2024-02-01"],
        "Type": ["Buy", "Sell"],
        "Symbol": ["0050", "2330"], # 範本直接用字串格式
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
            # 修正 1: 強制將 Symbol 欄位讀取為字串 (避免 0050 變 50)
            df_upload = pd.read_csv(uploaded_file, dtype={'Symbol': str})
            
            st.write("預覽上傳資料：")
            st.dataframe(df_upload.head())
            
            if st.button("🚀 開始匯入資料庫"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                rows_to_upload = []
                total_rows = len(df_upload)
                
                # 遍歷每一行進行處理
                for index, row in df_upload.iterrows():
                    # 修正 2: 智慧補零邏輯
                    raw_symbol = str(row['Symbol']).strip()
                    
                    # 如果是純數字且長度小於 4 (例如 "50" 或 "56")，自動補成 "0050", "0056"
                    if raw_symbol.isdigit() and len(raw_symbol) < 4:
                        raw_symbol = raw_symbol.zfill(4)
                    
                    # 抓取名稱
                    real_symbol, stock_name = get_stock_info(raw_symbol)
                    
                    # 判斷交易類型與計算金額
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
