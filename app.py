import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf
import datetime
import io
import json
import time
import google.generativeai as genai

# --- 1. 頁面配置 ---
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    [data-testid="stAppViewContainer"], html, body {
        background-color: #F8F9FA !important;
        color: #212529 !important;
        font-family: 'Inter', sans-serif;
    }
    .kpi-card {
        background-color: white; border: 1px solid #ddd; padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;
    }
    .metric-val { font-size: 22px; font-weight: bold; color: #333; }
    .metric-lbl { font-size: 14px; color: #666; }
    </style>
""", unsafe_allow_html=True)

# --- 2. 核心工具函式 ---

def safe_float(val):
    try:
        if pd.isna(val) or str(val).strip() == "": return 0.0
        return float(str(val).replace(',', ''))
    except: return 0.0

def standardize_symbol(symbol):
    """
    處理代號邏輯：
    1. 強制轉字串
    2. 去除前後空白
    3. 若為純數字且長度為3 (Excel有時會把0050存成50)，嘗試補0 (不完全可靠，建議Excel端設定為文字格式)
    4. 00919, 0050 保持原樣
    """
    s = str(symbol).replace("'", "").strip().upper()
    # 簡單補零邏輯，針對常見台股狀況
    if s.isdigit():
        # 如果是 50 -> 0050, 919 -> 00919 (假設台股ETF多為4-5碼)
        # 但這裡為了精確，主要依賴 Excel 匯入時指定 dtype=str
        pass 
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
    # 簡單判斷：純數字 (台股) 或有 .TW
    return s.isdigit() or ".TW" in s

def get_full_name(symbol):
    """取得 代號+中文名稱"""
    clean = standardize_symbol(symbol)
    q_sym = f"{clean}.TW" if clean.isdigit() else clean
    try:
        stock = yf.Ticker(q_sym)
        # 嘗試抓取各種名稱欄位
        name = stock.info.get('shortName') or stock.info.get('longName') or clean
        return f"{clean} {name}"
    except:
        return f"{clean}"

def fetch_name_only(symbol):
    """交易錄入自動帶出名稱用"""
    if not symbol: return ""
    clean = standardize_symbol(symbol)
    q_sym = f"{clean}.TW" if clean.isdigit() else clean
    try:
        stock = yf.Ticker(q_sym)
        return stock.info.get('shortName') or stock.info.get('longName') or ""
    except:
        return ""

# --- 3. 連線設定 ---

@st.cache_resource
def init_connection():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    if "gcp_service_account" not in st.secrets:
        st.error("❌ 未設定 Secrets: gcp_service_account")
        st.stop()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

def init_gemini():
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return True
    return False

# --- 4. 資料存取 ---

@st.cache_data(ttl=60)
def load_data():
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        dfs = []
        for sheet_name in ["TW_Trades", "US_Trades"]:
            try:
                ws = spreadsheet.worksheet(sheet_name)
                recs = ws.get_all_records()
                # 強制將 '代號' 轉為字串，避免 0050 變成 50
                if recs:
                    d = pd.DataFrame(recs)
                    d['代號'] = d['代號'].astype(str)
                    dfs.append(d)
            except: pass
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        return pd.DataFrame()

def save_data(row_data):
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        # 根據代號決定分頁
        sheet_name = "TW_Trades" if is_tw_stock(row_data[2]) else "US_Trades"
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except:
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=10)
            sheet.append_row(["日期", "類別", "代號", "名稱", "價格", "股數", "手續費", "交易稅", "總金額"])
        sheet.append_row(row_data)
        st.cache_data.clear()
        return True
    except: return False

def batch_save_data_xlsx(df):
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        tw_rows, us_rows = [], []
        
        for _, r in df.iterrows():
            sym = standardize_symbol(r['代號']) # 保持 00919
            row = [
                standardize_date(r['日期']), r['類別'], sym, r['名稱'],
                safe_float(r['價格']), safe_float(r['股數']), 
                safe_float(r['手續費']), safe_float(r['交易稅']), safe_float(r['總金額'])
            ]
            if is_tw_stock(sym): tw_rows.append(row)
            else: us_rows.append(row)
            
        if tw_rows:
            sheet = spreadsheet.worksheet("TW_Trades")
            sheet.append_rows(tw_rows)
        if us_rows:
            sheet = spreadsheet.worksheet("US_Trades")
            sheet.append_rows(us_rows)
        
        st.cache_data.clear()
        return len(tw_rows) + len(us_rows)
    except Exception as e:
        st.error(f"寫入失敗: {e}")
        return 0

# --- 5. 計算邏輯 (核心) ---

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        h = yf.Ticker("TWD=X").history(period="1d")
        return h['Close'].iloc[-1] if not h.empty else 32.5
    except: return 32.5

def calculate_portfolio(df, rate):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    
    portfolio = {}
    history = [] # 儲存所有交易紀錄，包含已實現損益計算
    
    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))
    df = df.sort_values('日期')
    
    for _, row in df.iterrows():
        sym = standardize_symbol(row['代號'])
        name = row.get('名稱', sym)
        full_display_name = f"{sym} {name}" # 02. 代號+中文名稱
        
        if sym not in portfolio:
            portfolio[sym] = {
                'DisplayName': full_display_name,
                'Symbol': sym,
                'Qty': 0, 'TotalCost': 0, 'Realized': 0, 
                'IsUS': not is_tw_stock(sym)
            }
        p = portfolio[sym]
        
        act = str(row['類別'])
        q = safe_float(row['股數'])
        pr = safe_float(row['價格'])
        f = safe_float(row['手續費'])
        t = safe_float(row['交易稅'])
        
        # 紀錄單筆
        trade_pl = 0
        
        if "買" in act:
            cost = q * pr + f
            p['Qty'] += q
            p['TotalCost'] += cost
        elif "賣" in act and p['Qty'] > 0:
            avg_cost = p['TotalCost'] / p['Qty']
            cost_sold = avg_cost * q
            revenue = q * pr - f - t
            trade_pl = revenue - cost_sold
            
            p['Realized'] += trade_pl
            p['Qty'] -= q
            p['TotalCost'] -= cost_sold
        elif "現金" in act: # 股息
            trade_pl = pr # 假設填入的是總金額
            p['Realized'] += trade_pl
            
        history.append({
            'DisplayName': full_display_name,
            '日期': row['日期'],
            '類別': act,
            '股數': q,
            '價格': pr,
            '單筆損益': trade_pl if ("賣" in act or "現金" in act) else 0,
            'IsUS': p['IsUS']
        })

    # 計算現價與市值
    active_syms = [s for s, v in portfolio.items() if v['Qty'] > 0]
    prices = {}
    if active_syms:
        qs = [f"{s}.TW" if is_tw_stock(s) and s.isdigit() else s for s in active_syms]
        try:
            data = yf.Tickers(" ".join(qs))
            for i, s in enumerate(active_syms):
                try:
                    h = data.tickers[qs[i]].history(period="1d")
                    prices[s] = h['Close'].iloc[-1]
                except: prices[s] = 0
        except: pass
        
    res = []
    for s, v in portfolio.items():
        if v['Qty'] > 0:
            cp = prices.get(s, 0)
            mkt = v['Qty'] * cp
            unreal = mkt - v['TotalCost']
            ret = (unreal / v['TotalCost'] * 100) if v['TotalCost'] > 0 else 0
            
            res.append({
                '顯示名稱': v['DisplayName'],
                '代號': s,
                'IsUS': v['IsUS'],
                '持有股數': v['Qty'],
                '平均單價': v['TotalCost'] / v['Qty'],
                '投入成本': v['TotalCost'],
                '目前現價': cp,
                '目前市值': mkt,
                '未實現損益': unreal,
                '損益率%': ret
            })
            
    return pd.DataFrame(res), pd.DataFrame(history)

# --- 6. 技術分析 ---

def get_trend_analysis(symbol):
    clean = standardize_symbol(symbol)
    q_sym = f"{clean}.TW" if clean.isdigit() else clean
    try:
        stock = yf.Ticker(q_sym)
        df = stock.history(period="1y")
        if len(df) < 60: return None
        
        current = df['Close'].iloc[-1]
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        ma60 = df['Close'].rolling(60).mean().iloc[-1]
        
        # 簡單趨勢判斷
        t_short = "🔴 看多" if current > ma5 else "🟢 看空"
        t_mid = "🔴 看多" if current > ma20 else "🟢 看空"
        t_long = "🔴 看多" if current > ma60 else "🟢 看空"
        
        return pd.Series([t_short, t_mid, t_long], index=['短', '中', '長'])
    except: return None

# --- 7. 介面 ---

tab1, tab2, tab3, tab4 = st.tabs(["📝 交易錄入", "📥 批次匯入", "📊 趨勢戰情", "💰 資產透視"])

# --- Tab 1: 交易錄入 (Auto Name) ---
with tab1:
    st.subheader("📝 交易錄入")
    
    # 使用 Session State 來處理自動帶入
    if 'input_sym' not in st.session_state: st.session_state.input_sym = ""
    if 'auto_name' not in st.session_state: st.session_state.auto_name = ""

    def on_sym_change():
        sym = st.session_state.input_sym
        if sym:
            st.session_state.auto_name = fetch_name_only(sym)

    with st.form("entry"):
        c1, c2 = st.columns(2)
        ttype = c1.selectbox("交易類別", ["買入", "賣出", "現金股息", "配股"])
        tdate = c2.date_input("日期")
        
        c3, c4 = st.columns(2)
        # key 綁定 session_state，on_change 綁定 callback
        tsym = c3.text_input("股票代號 (Enter後自動帶入名稱)", key="input_sym", on_change=on_sym_change)
        tname = c4.text_input("股票名稱", key="auto_name")
        
        c5, c6 = st.columns(2)
        tqty = c5.number_input("股數", min_value=0.0)
        tprice = c6.number_input("價格/總金額", min_value=0.0)
        
        c7, c8 = st.columns(2)
        tfee = c7.number_input("手續費", 0.0)
        ttax = c8.number_input("交易稅", 0.0)
        
        if st.form_submit_button("💾 儲存交易"):
            if tsym:
                final_name = tname if tname else fetch_name_only(tsym)
                amt = 0
                if "買" in ttype: amt = -(tqty*tprice + tfee)
                elif "賣" in ttype: amt = (tqty*tprice - tfee - ttax)
                elif "現金" in ttype: amt = tprice
                
                row = [str(tdate), ttype, standardize_symbol(tsym), final_name, tprice, tqty, tfee, ttax, amt]
                if save_data(row):
                    st.success(f"已儲存 {final_name}")
            else:
                st.warning("請輸入代號")

# --- Tab 2: 批次匯入 (xlsx + 00919 fix) ---
with tab2:
    st.subheader("📥 批次匯入 Excel (.xlsx)")
    
    # 下載範本
    template_data = {
        "日期": ["2026-01-01", "2026-01-02"],
        "類別": ["買入", "買入"],
        "代號": ["00919", "2330"], # 範例為字串
        "名稱": ["群益台灣精選高息", "台積電"],
        "價格": [22.5, 600],
        "股數": [1000, 100],
        "手續費": [20, 20],
        "交易稅": [0, 0],
        "總金額": [-22520, -60020]
    }
    df_temp = pd.DataFrame(template_data)
    
    # 轉為 Excel Bytes
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_temp.to_excel(writer, index=False)
    processed_data = output.getvalue()
    
    st.download_button("📥 下載 Excel 範本", processed_data, "template.xlsx")
    
    uploaded = st.file_uploader("上傳 .xlsx 檔案", type=['xlsx'])
    if uploaded and st.button("確認匯入"):
        try:
            # 重要：dtype={'代號': str} 確保 00919 不會變成 919
            df_u = pd.read_excel(uploaded, dtype={'代號': str})
            count = batch_save_data_xlsx(df_u)
            if count > 0:
                st.success(f"成功匯入 {count} 筆資料！")
        except Exception as e:
            st.error(f"匯入錯誤: {e}")

# --- Tab 3: 趨勢戰情 ---
with tab3:
    st.subheader("📊 趨勢戰情室")
    
    raw_df = load_data()
    if not raw_df.empty:
        rate = get_exchange_rate()
        holdings, _ = calculate_portfolio(raw_df, rate)
        
        # 02. 提供目前持股資訊總覽 或是 個股檢視
        view_mode = st.radio("檢視模式", ["持股總覽", "個股深度分析"], horizontal=True)
        
        if view_mode == "持股總覽":
            if not holdings.empty:
                st.markdown("##### 🚦 持股趨勢紅綠燈")
                trend_data = []
                # 遍歷所有持股
                for sym in holdings['代號'].unique():
                    name = holdings[holdings['代號']==sym]['顯示名稱'].iloc[0]
                    trends = get_trend_analysis(sym)
                    if trends is not None:
                        trend_data.append({
                            "名稱": name,
                            "短": trends['短'],
                            "中": trends['中'],
                            "長": trends['長']
                        })
                if trend_data:
                    st.dataframe(pd.DataFrame(trend_data), use_container_width=True)
                else:
                    st.info("無法取得趨勢資料")
            else:
                st.info("無庫存")
                
        else: # 個股深度分析
            target_list = holdings['顯示名稱'].tolist() if not holdings.empty else []
            target_sel = st.selectbox("選擇股票", ["請選擇"] + target_list)
            
            if target_sel != "請選擇":
                sym = target_sel.split()[0]
                hist, ana, err = get_trend_analysis(sym) # 這裡需改寫或沿用上個版本的 analyze_full_signal
                # 這裡為了簡化，直接呼叫 yfinance 重繪圖表
                clean = standardize_symbol(sym)
                q_sym = f"{clean}.TW" if clean.isdigit() else clean
                stock = yf.Ticker(q_sym)
                hist_df = stock.history(period="1y")
                
                if not hist_df.empty:
                    # 01. 顯示短中長期
                    current = hist_df['Close'].iloc[-1]
                    ma5 = hist_df['Close'].rolling(5).mean().iloc[-1]
                    ma20 = hist_df['Close'].rolling(20).mean().iloc[-1]
                    ma60 = hist_df['Close'].rolling(60).mean().iloc[-1]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("短期 (MA5)", "看多" if current > ma5 else "看空")
                    c2.metric("中期 (MA20)", "看多" if current > ma20 else "看空")
                    c3.metric("長期 (MA60)", "看多" if current > ma60 else "看空")
                    
                    # 畫圖
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist_df.index, open=hist_df['Open'], high=hist_df['High'], 
                                                 low=hist_df['Low'], close=hist_df['Close'], name='K線'))
                    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'].rolling(20).mean(), name='MA20', line=dict(color='orange')))
                    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: 資產透視 (大幅更新) ---
with tab4:
    st.subheader("💰 資產透視")
    
    raw_df = load_data()
    if not raw_df.empty:
        rate = get_exchange_rate()
        holdings_df, history_df = calculate_portfolio(raw_df, rate)
        
        # 01. 可選擇項目 (全部 / 美股 / 台股)
        filter_mode = st.radio("資產篩選", ["全部資產", "台股 (TWD)", "美股 (USD)"], horizontal=True)
        
        display_df = pd.DataFrame()
        currency_symbol = ""
        
        if filter_mode == "全部資產":
            # 混合顯示，需統一匯率 (全部轉台幣)
            display_df = holdings_df.copy()
            # 將美股轉台幣顯示
            display_df['投入成本'] = display_df.apply(lambda x: x['投入成本'] * rate if x['IsUS'] else x['投入成本'], axis=1)
            display_df['目前市值'] = display_df.apply(lambda x: x['目前市值'] * rate if x['IsUS'] else x['目前市值'], axis=1)
            # 平均單價混合顯示比較怪，建議分開，這裡先不特別處理單價
            currency_symbol = "NT$"
            
        elif filter_mode == "台股 (TWD)":
            display_df = holdings_df[holdings_df['IsUS'] == False].copy()
            currency_symbol = "NT$"
            
        elif filter_mode == "美股 (USD)":
            display_df = holdings_df[holdings_df['IsUS'] == True].copy()
            currency_symbol = "$"
            
        if not display_df.empty:
            # 顯示總覽 Metrics
            total_cost = display_df['投入成本'].sum()
            total_mkt = display_df['目前市值'].sum()
            total_unreal = total_mkt - total_cost
            total_ret = (total_unreal / total_cost * 100) if total_cost > 0 else 0
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總投入成本", f"{currency_symbol}{total_cost:,.0f}")
            m2.metric("總目前市值", f"{currency_symbol}{total_mkt:,.0f}")
            m3.metric("未實現損益", f"{currency_symbol}{total_unreal:,.0f}", delta_color="normal")
            m4.metric("總報酬率", f"{total_ret:,.2f}%", delta_color="normal")
            
            st.divider()
            
            # 03. 目前持有中的單個股 (列表顯示)
            st.markdown("##### 📋 持股明細")
            # 格式化顯示
            show_df = display_df[['顯示名稱', '持有股數', '平均單價', '投入成本', '目前現價', '目前市值', '損益率%']].copy()
            st.dataframe(
                show_df.style.format({
                    '平均單價': '{:.2f}', '投入成本': '{:,.0f}', 
                    '目前現價': '{:.2f}', '目前市值': '{:,.0f}', '損益率%': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            st.divider()
            
            # 02. 可選單個個股 -> 歷史交易與損益
            st.markdown("##### 🔎 個股歷史交易查詢")
            # 篩選清單 (包含已出清的)
            all_history_syms = history_df['DisplayName'].unique()
            sel_history = st.selectbox("選擇查詢個股", ["請選擇"] + list(all_history_syms))
            
            if sel_history != "請選擇":
                sub_h = history_df[history_df['DisplayName'] == sel_history].copy()
                sub_h['日期'] = sub_h['日期'].dt.strftime('%Y-%m-%d')
                
                # 計算該股總已實現
                realized_sum = sub_h['單筆損益'].sum()
                is_us_stock = sub_h['IsUS'].iloc[0]
                curr = "$" if is_us_stock else "NT$"
                
                st.metric(f"{sel_history} 累計已實現損益", f"{curr}{realized_sum:,.0f}", delta_color="normal")
                
                st.dataframe(sub_h[['日期', '類別', '股數', '價格', '單筆損益']], use_container_width=True)
                
        else:
            st.info("該類別無持股資料")
    else:
        st.info("尚無資料")
