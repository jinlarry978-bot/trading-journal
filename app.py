import streamlit as st  
import pandas as pd  
import plotly.graph_objects as go  
from plotly.subplots import make_subplots  
import gspread  
from oauth2client.service_account import ServiceAccountCredentials  
import yfinance as yf  
import datetime  
import io  
import google.generativeai as genai  
  
# --- 1. 頁面配置與 CSS ---  
st.set_page_config(page_title="專業投資戰情室 Pro", layout="wide", page_icon="💎")  
  
st.markdown("""  
    <style>  
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');  
      
    /* 強制淺色主題與手機優化 */  
    [data-testid="stAppViewContainer"], html, body {  
        background-color: #F8F9FA !important;  
        color: #212529 !important;  
        font-family: 'Inter', sans-serif;  
    }  
    h1, h2, h3, h4, p, span, div, label { color: #212529 !important; }  
      
    /* 自定義 KPI 卡片 (防數字切斷) */  
    .custom-kpi-card {  
        background-color: #FFFFFF;  
        border: 1px solid #E9ECEF;  
        padding: 18px;  
        border-radius: 12px;  
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);  
        margin-bottom: 12px;  
        min-height: 130px;  
        display: flex;  
        flex-direction: column;  
        justify-content: center;  
        transition: transform 0.3s ease;  
    }  
    .custom-kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.05); }  
    .kpi-label { font-size: 14px; color: #6C757D; font-weight: 600; margin-bottom: 6px; }  
    .kpi-val-usd { font-size: 24px; font-weight: 800; color: #212529; line-height: 1.1; }  
    .kpi-val-twd { font-size: 15px; color: #888; font-weight: 500; margin-top: 5px; }  
    .delta-text { font-size: 13px; font-weight: 700; margin-top: 8px; padding: 2px 8px; border-radius: 4px; width: fit-content; }  
    .pos { color: #D32F2F; background-color: rgba(211, 47, 47, 0.1); }  
    .neg { color: #2E7D32; background-color: rgba(46, 125, 50, 0.1); }  
      
    /* AI 與 策略卡片 */  
    .ai-box { background-color: #F0F4F8; border-left: 5px solid #4285F4; padding: 15px; border-radius: 5px; margin-top: 20px; color: #212529 !important; }  
    .strategy-card { padding: 18px; border-radius: 12px; margin-bottom: 15px; border: 1px solid #E9ECEF; background-color: white; }  
    .strategy-title { margin: 0; color: #495057; font-weight: 700; font-size: 14px; }  
    .strategy-signal { margin: 5px 0; font-weight: 800; font-size: 20px; }  
      
    @media (max-width: 640px) {  
        .kpi-val-usd { font-size: 20px !important; }  
        .custom-kpi-card { padding: 15px !important; min-height: 110px; }  
    }  
    </style>  
    """, unsafe_allow_html=True)  
  
# --- 2. 核心工具函式 ---  
  
def safe_float(val):  
    try:  
        if pd.isna(val) or str(val).strip() == "":  
            return 0.0  
        return float(val)  
    except:  
        return 0.0  
  
def standardize_symbol(symbol):  
    """  
    統一股票代號格式：  
    - 純數字且長度 <= 4：補成 4 碼（例如 233 → 0233）  
    - 純數字且長度 > 4：維持原樣（例如 00940 等 5 碼 ETF）  
    - 其他字串：去除空白、轉大寫  
    """  
    s = str(symbol).replace("'", "").strip().upper()  
    if s.isdigit():  
        if len(s) <= 4:  
            return s.zfill(4)   # 1~4 碼 → 補到 4 碼  
        else:  
            return s            # 5 碼（如 00940）等 → 不動  
    return s  
  
def standardize_date(date_val):  
    try:  
        if pd.isna(date_val) or str(date_val).strip() == "":  
            return None  
        # 處理 Excel 日期序號  
        if isinstance(date_val, (int, float)):  
            dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=date_val)  
            return dt.strftime("%Y-%m-%d")  
        dt = pd.to_datetime(str(date_val).replace('.', '-').replace('/', '-'))  
        return dt.strftime("%Y-%m-%d")  
    except:  
        return None  
  
def is_tw_stock(symbol):  
    s = str(symbol).upper()  
    return s.isdigit() or ".TW" in s  
  
# --- 3. 連線與 AI ---  
  
@st.cache_resource  
def init_connection():  
    scope = [  
        "https://spreadsheets.google.com/feeds",  
        "https://www.googleapis.com/auth/drive"  
    ]  
    creds = ServiceAccountCredentials.from_json_keyfile_dict(  
        st.secrets["gcp_service_account"], scope  
    )  
    return gspread.authorize(creds)  
  
def init_gemini():  
    if "GEMINI_API_KEY" in st.secrets:  
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  
        return True  
    return False  
  
def ask_gemini_analyst(symbol, name, data_summary):  
    try:  
        prompt = (  
            f"你是一位資深投資顧問。請分析標的：{symbol} {name}。"  
            f"收盤價：{data_summary['close']:.2f}, RSI(14)：{data_summary['rsi']:.1f}, "  
            f"KD(K)：{data_summary['k']:.1f}。"  
            "請給出專業短評、目前趨勢判定與具體操作建議（買進/減碼/觀望），"  
            "約120字繁體中文。"  
        )  
        for m_name in ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-pro']:  
            try:  
                model = genai.GenerativeModel(model_name=m_name)  
                response = model.generate_content(prompt)  
                if response.text:  
                    return f"{response.text}\n\n(AI引擎: {m_name})"  
            except:  
                continue  
        return "AI 分析暫時不可用，請稍後重試。"  
    except Exception as e:  
        return f"AI 連線錯誤: {str(e)}"  
  
# 輕量級取得股票名稱，避免為了名稱跑完整技術分析  
def resolve_stock_name(symbol: str) -> str:  
    clean = standardize_symbol(symbol)  
    q_sym = f"{clean}.TW" if clean.isdigit() else clean  
    try:  
        stock = yf.Ticker(q_sym)  
        # 優先使用 fast_info（較輕量）  
        fast_info = getattr(stock, "fast_info", None)  
        if fast_info and isinstance(fast_info, dict):  
            if "shortName" in fast_info:  
                return fast_info["shortName"]  
        # 再退回較重的 info  
        info = getattr(stock, "info", {}) or {}  
        return info.get("shortName") or info.get("longName") or clean  
    except:  
        return clean  
  
# --- 4. 資料庫操作 ---  
  
@st.cache_data(ttl=60)  
def load_data():  
    """  
    從 Google Sheet 讀取全部交易紀錄，使用 cache 減少頻繁讀取。  
    """  
    try:  
        client = init_connection()  
        spreadsheet = client.open("TradeLog")  
        try:  
            tw = pd.DataFrame(spreadsheet.worksheet("TW_Trades").get_all_records())  
            if not tw.empty:  
                tw['Market'] = 'TW'  
        except:  
            tw = pd.DataFrame()  
        try:  
            us = pd.DataFrame(spreadsheet.worksheet("US_Trades").get_all_records())  
            if not us.empty:  
                us['Market'] = 'US'  
        except:  
            us = pd.DataFrame()  
        if tw.empty and us.empty:  
            return pd.DataFrame()  
        return pd.concat([tw, us], ignore_index=True)  
    except Exception as e:  
        st.error(f"資料讀取失敗: {e}")  
        return pd.DataFrame()  
  
def save_data(row_data):  
    try:  
        client = init_connection()  
        spreadsheet = client.open("TradeLog")  
        sheet = spreadsheet.worksheet("TW_Trades" if is_tw_stock(row_data[2]) else "US_Trades")  
        sheet.append_row(row_data)  
        st.cache_data.clear()  
        return True  
    except Exception as e:  
        st.error(f"單筆寫入失敗：{e}")  
        return False  
  
def batch_save_data(rows, market):  
    if not rows:  
        return True  
    try:  
        client = init_connection()  
        spreadsheet = client.open("TradeLog")  
        sheet = spreadsheet.worksheet("TW_Trades" if market == 'TW' else "US_Trades")  
        sheet.append_rows(rows)  
        st.cache_data.clear()  
        return True  
    except Exception as e:  
        st.error(f"批次寫入 {market} 資料失敗：{e}")  
        return False  
  
# --- 5. 核心運算 ---  
  
@st.cache_data(ttl=3600)  
def get_exchange_rate():  
    try:  
        # TWD=X：1 USD 可換多少 TWD  
        h = yf.Ticker("TWD=X").history(period="1d")  
        return h['Close'].iloc[-1] if not h.empty else 32.5  
    except:  
        return 32.5  
  
def calculate_full_portfolio(df, rate):  
    portfolio = {}  
    if df.empty:  
        return pd.DataFrame(), {"twd": {}, "usd": {}}, pd.DataFrame()  
  
    # 日期標準化  
    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))  
    df = df.sort_values('日期')  
  
    for _, row in df.iterrows():  
        sym = standardize_symbol(row['代號'])  
        if sym not in portfolio:  
            portfolio[sym] = {  
                'Name': row['名稱'],  
                'Qty': 0,  
                'Cost': 0,  
                'Realized': 0,  
                'IsUS': not is_tw_stock(sym)  
            }  
  
        p = portfolio[sym]  
        q = safe_float(row['股數'])  
        pr = safe_float(row['價格'])  
        f = safe_float(row['手續費'])  
        t = safe_float(row['交易稅'])  
        type_str = str(row['類別'])  
  
        if "買" in type_str:  
            p['Cost'] += (q * pr + f)  
            p['Qty'] += q  
        elif "賣" in type_str and p['Qty'] > 0:  
            avg = p['Cost'] / p['Qty']  
            cost_sold = avg * q  
            p['Realized'] += (q * pr - f - t) - cost_sold  
            p['Qty'] -= q  
            p['Cost'] -= cost_sold  
        elif "息" in type_str:  
            p['Realized'] += pr  
  
    # 取得現價  
    active_syms = [s for s, v in portfolio.items() if v['Qty'] > 0]  
    prices = {}  
    if active_syms:  
        qs = [  
            f"{s}.TW" if is_tw_stock(s) and s.isdigit() else s  
            for s in active_syms  
        ]  
        data = yf.Tickers(" ".join(qs))  
        for i, s in enumerate(active_syms):  
            try:  
                h = data.tickers[qs[i]].history(period="1d")  
                prices[s] = h['Close'].iloc[-1] if not h.empty else 0  
            except:  
                prices[s] = 0  
  
    res = []  
    t_twd = {'mkt': 0, 'unreal': 0, 'real': 0}  
    t_usd = {'mkt': 0, 'unreal': 0, 'real': 0}  
  
    for s, v in portfolio.items():  
        cp = prices.get(s, 0)  
        mkt = v['Qty'] * cp  
        unreal = mkt - v['Cost'] if v['Qty'] > 0 else 0  
  
        if v['IsUS']:  
            t_usd['mkt'] += mkt  
            t_usd['unreal'] += unreal  
            t_usd['real'] += v['Realized']  
  
            t_twd['mkt'] += mkt * rate  
            t_twd['unreal'] += unreal * rate  
            t_twd['real'] += v['Realized'] * rate  
        else:  
            t_twd['mkt'] += mkt  
            t_twd['unreal'] += unreal  
            t_twd['real'] += v['Realized']  
  
        if v['Qty'] > 0 or v['Realized'] != 0:  
            res.append({  
                "代號": s,  
                "名稱": v['Name'],  
                "庫存": v['Qty'],  
                "現價": cp,  
                "市值": mkt,  
                "未實現": unreal,  
                "已實現+息": v['Realized'],  
                "IsUS": v['IsUS']  
            })  
  
    return pd.DataFrame(res), {"twd": t_twd, "usd": t_usd}, df  
  
@st.cache_data(ttl=300)  
def analyze_full_signal(symbol):  
    """  
    回傳：  
    - hist: 含技術指標的 DataFrame  
    - ana: dict {st, lt, metrics, name, symbol}  
    - err: str 或 None  
    """  
    try:  
        clean = standardize_symbol(symbol)  
        q_sym = f"{clean}.TW" if clean.isdigit() else clean  
        stock = yf.Ticker(q_sym)  
        df = stock.history(period="1y")  
  
        if df is None or df.empty or len(df) < 60:  
            return None, None, "資料不足，無法進行技術分析（需至少 60 根 K 線）"  
  
        # 技術指標計算  
        df['MA5'] = df['Close'].rolling(5).mean()  
        df['MA20'] = df['Close'].rolling(20).mean()  
        df['MA60'] = df['Close'].rolling(60).mean()  
  
        # RSI(14)  
        delta = df['Close'].diff()  
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()  
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()  
        df['RSI'] = 100 - (100 / (1 + gain / loss))  
  
        # KD (9,3,3)  
        low_min = df['Low'].rolling(9).min()  
        high_max = df['High'].rolling(9).max()  
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100  
  
        k, d = 50, 50  
        k_l, d_l = [], []  
        for r in rsv:  
            if pd.isna(r):  
                k_l.append(k)  
                d_l.append(d)  
            else:  
                k = (2/3) * k + (1/3) * r  
                d = (2/3) * d + (1/3) * k  
                k_l.append(k)  
                d_l.append(d)  
        df['K'], df['D'] = k_l, d_l  
  
        last = df.iloc[-1]  
  
        # 策略判定  
        if last['Close'] > last['MA5'] and last['K'] > last['D']:  
            st_sig = {  
                "txt": "🔴 強勢偏多",  
                "col": "#D32F2F",  
                "desc": "站上5日線 + KD 金叉"  
            }  
        else:  
            st_sig = {  
                "txt": "🟠 震盪觀望",  
                "col": "#FF9800",  
                "desc": "指標尚不明確"  
            }  
  
        if last['Close'] > last['MA60']:  
            lt_sig = {  
                "txt": "🔴 多頭格局",  
                "col": "#D32F2F",  
                "desc": "守穩生命線 (MA60)"  
            }  
        else:  
            lt_sig = {  
                "txt": "🟢 弱勢空頭",  
                "col": "#2E7D32",  
                "desc": "季線反壓顯著"  
            }  
  
        metrics = {  
            "close": float(last['Close']),  
            "rsi": float(last['RSI']),  
            "k": float(last['K']),  
            "d": float(last['D'])  
        }  
  
        # 優先使用快取資訊  
        name = None  
        try:  
            fast_info = getattr(stock, "fast_info", None)  
            if fast_info and isinstance(fast_info, dict):  
                name = fast_info.get("shortName")  
        except:  
            pass  
  
        if not name:  
            try:  
                info = getattr(stock, "info", {}) or {}  
                name = info.get('longName') or info.get('shortName')  
            except:  
                name = None  
  
        if not name:  
            name = clean  
  
        ana = {  
            "st": st_sig,  
            "lt": lt_sig,  
            "metrics": metrics,  
            "name": name,  
            "symbol": q_sym  
        }  
        return df, ana, None  
  
    except Exception as e:  
        return None, None, str(e)  
  
# --- 6. 介面呈現 ---  
  
tab1, tab2, tab3, tab4 = st.tabs([  
    "📝 交易錄入",  
    "📥 批次匯入",  
    "📊 趨勢戰情",  
    "💰 資產透視"  
])  
  
# --- Tab1：單筆交易記錄 ---  
  
with tab1:  
    with st.form("trade_input"):  
        st.subheader("📝 單筆交易記錄")  
        c1, c2, c3 = st.columns(3)  
        ttype = c1.selectbox("交易類別", ["買入", "賣出", "股息/配息"])  
        tdate = c2.date_input("交易日期")  
        tsym = c3.text_input("股票代號 (如 2330 或 00940)")  
  
        c4, c5, c6, c7 = st.columns(4)  
        tqty = c4.number_input("股數", min_value=0.0, step=1.0)  
        tprice = c5.number_input("價格/配息金額", min_value=0.0)  
        tfee = c6.number_input("手續費", min_value=0.0)  
        ttax = c7.number_input("交易稅", min_value=0.0)  
  
        submitted = st.form_submit_button("確認送出")  
  
        if submitted:  
            if not tsym:  
                st.warning("請輸入股票代號")  
            else:  
                # 輕量取得名稱，避免每次新增交易都跑完整技術分析  
                tname = resolve_stock_name(tsym)  
  
                if "買" in ttype:  
                    amt = -(tqty * tprice + tfee)  
                elif "賣" in ttype:  
                    amt = (tqty * tprice - tfee - ttax)  
                else:  
                    # 股息 / 配息  
                    amt = tprice  
  
                ok = save_data([  
                    str(tdate),  
                    ttype,  
                    standardize_symbol(tsym),  
                    tname,  
                    tprice,  
                    tqty,  
                    tfee,  
                    ttax,  
                    amt  
                ])  
                if ok:  
                    st.success("✅ 記錄已成功儲存！")  
                else:  
                    st.error("❌ 記錄儲存失敗，請稍後再試或檢查連線設定。")  
  
# --- Tab2：批次匯入 ---  
  
with tab2:  
    st.subheader("📥 批次匯入交易")  
  
    template = pd.DataFrame({  
        "日期": ["2026-01-01"],  
        "類別": ["買入"],  
        "代號": ["2330"],  
        "名稱": ["台積電"],  
        "價格": [1000],  
        "股數": [100],  
        "手續費": [20],  
        "交易稅": [0]  
    })  
  
    st.download_button(  
        "📥 下載 CSV 範本",  
        io.BytesIO(template.to_csv(index=False).encode('utf-8-sig')),  
        "template.csv"  
    )  
  
    uploaded = st.file_uploader("上傳 CSV 檔案", type=["csv"])  
  
    if uploaded and st.button("開始匯入檔案"):  
        try:  
            df_u = pd.read_csv(uploaded)  
        except Exception as e:  
            st.error(f"CSV 解析失敗：{e}")  
        else:  
            tw_rows, us_rows = [], []  
            for _, r in df_u.iterrows():  
                sym = standardize_symbol(r['代號'])  
                row = [  
                    standardize_date(r['日期']),  
                    r['類別'],  
                    sym,  
                    r['名稱'],  
                    r['價格'],  
                    r['股數'],  
                    r['手續費'],  
                    r['交易稅'],  
                    0  # 金額暫不使用，維持欄位  
                ]  
                if is_tw_stock(sym):  
                    tw_rows.append(row)  
                else:  
                    us_rows.append(row)  
  
            ok_tw = batch_save_data(tw_rows, 'TW')  
            ok_us = batch_save_data(us_rows, 'US')  
  
            if ok_tw and ok_us:  
                st.success("✅ 批次匯入完成！")  
            else:  
                st.warning("部分資料匯入失敗，請檢查錯誤訊息。")  
  
# --- Tab3：趨勢戰情診斷 ---  
  
with tab3:  
    st.subheader("📊 趨勢戰情診斷")  
  
    raw_for_filter = load_data()  
  
    # 庫存快選（只顯示目前仍有持股的代號）  
    inv = {}  
    for _, r in raw_for_filter.iterrows():  
        s = standardize_symbol(r['代號'])  
        q = safe_float(r['股數'])  
        if "買" in str(r['類別']):  
            inv[s] = inv.get(s, 0) + q  
        elif "賣" in str(r['類別']):  
            inv[s] = inv.get(s, 0) - q  
    held_stocks = [s for s, q in inv.items() if q > 0]  
  
    sel_col, search_col = st.columns([1, 1])  
    with sel_col:  
        sel_sym = st.selectbox("🎯 庫存快速診斷", ["請選擇"] + held_stocks)  
    with search_col:  
        search_sym = st.text_input("🔍 搜尋代號 (如 AAPL、2330 或 00940)", "")  
  
    target = search_sym if search_sym else (sel_sym if sel_sym != "請選擇" else None)  
  
    if target:  
        with st.spinner("正在生成深度診斷報告..."):  
            hist, ana, err = analyze_full_signal(target)  
  
        if err:  
            st.warning(f"無法完成技術分析：{err}")  
        elif hist is not None and ana is not None:  
            # 指標區塊  
            m1, m2, m3, m4 = st.columns(4)  
            m1.metric("目前股價", f"{ana['metrics']['close']:.2f}")  
            m2.metric("RSI (14)", f"{ana['metrics']['rsi']:.1f}")  
            m3.metric("K 值", f"{ana['metrics']['k']:.1f}")  
            m4.metric(  
                "布林位置",  
                "中軌上方" if ana['metrics']['close'] > hist['MA20'].iloc[-1] else "中軌下方"  
            )  
  
            # AI 分析（可選）  
            if init_gemini():  
                if st.button("🤖 啟動 AI 深度投顧分析"):  
                    with st.spinner("AI 分析師正在閱讀 K 線圖..."):  
                        res = ask_gemini_analyst(  
                            ana['symbol'],  
                            ana['name'],  
                            ana['metrics']  
                        )  
                        st.markdown(  
                            f'<div class="ai-box"><b>🤖 AI 投顧觀點：</b><br>{res}</div>',  
                            unsafe_allow_html=True  
                        )  
  
            # 策略卡片  
            s1, s2 = st.columns(2)  
            with s1:  
                st.markdown(  
                    f'''  
                    <div class="strategy-card" style="border-left:5px solid {ana["st"]["col"]}">  
                        <div class="strategy-title">短期趨勢 (K/D)</div>  
                        <div class="strategy-signal" style="color:{ana["st"]["col"]}">  
                            {ana["st"]["txt"]}  
                        </div>  
                        <div>{ana["st"]["desc"]}</div>  
                    </div>  
                    ''',  
                    unsafe_allow_html=True  
                )  
            with s2:  
                st.markdown(  
                    f'''  
                    <div class="strategy-card" style="border-left:5px solid {ana["lt"]["col"]}">  
                        <div class="strategy-title">長期趨勢 (MA60)</div>  
                        <div class="strategy-signal" style="color:{ana["lt"]["col"]}">  
                            {ana["lt"]["txt"]}  
                        </div>  
                        <div>{ana["lt"]["desc"]}</div>  
                    </div>  
                    ''',  
                    unsafe_allow_html=True  
                )  
  
            # K 線 + 成交量圖（手機上高度略縮小）  
            fig_height = 550  
            fig = make_subplots(  
                rows=2,  
                cols=1,  
                shared_xaxes=True,  
                row_heights=[0.7, 0.3],  
                vertical_spacing=0.05  
            )  
            fig.add_trace(  
                go.Candlestick(  
                    x=hist.index,  
                    open=hist['Open'],  
                    high=hist['High'],  
                    low=hist['Low'],  
                    close=hist['Close'],  
                    name='K線'  
                ),  
                row=1, col=1  
            )  
            fig.add_trace(  
                go.Scatter(  
                    x=hist.index,  
                    y=hist['MA20'],  
                    name='月線',  
                    line=dict(color='#FF9800', width=1)  
                ),  
                row=1, col=1  
            )  
            fig.add_trace(  
                go.Scatter(  
                    x=hist.index,  
                    y=hist['MA60'],  
                    name='季線',  
                    line=dict(color='#9C27B0', width=1)  
                ),  
                row=1, col=1  
            )  
            fig.add_trace(  
                go.Bar(  
                    x=hist.index,  
                    y=hist['Volume'],  
                    name='成交量',  
                    marker_color='rgba(100,100,100,0.3)'  
                ),  
                row=2, col=1  
            )  
            fig.update_layout(  
                height=fig_height,  
                template="plotly_white",  
                xaxis_rangeslider_visible=False,  
                margin=dict(l=10, r=10, t=10, b=10)  
            )  
            st.plotly_chart(fig, use_container_width=True)  
        else:  
            st.warning("無法取得該股票的技術分析資料，請稍後再試。")  
  
# --- Tab4：資產透視與績效分析 ---  
  
with tab4:  
    st.subheader("💰 資產透視與績效分析")  
  
    rate = get_exchange_rate()  
    raw_df = load_data()  
  
    if not raw_df.empty:  
        p_df, totals, _ = calculate_full_portfolio(raw_df, rate)  
  
        # KPI 卡片渲染  
        def render_kpi(label, usd, twd, d=None):  
            if d is not None:  
                cls = "pos" if d > 0 else "neg"  
                arrow = "↑" if d > 0 else "↓"  
                dt = f'<div class="delta-text {cls}">{arrow} {abs(d):.1f}%</div>'  
            else:  
                dt = ""  
            st.markdown(  
                f'''  
                <div class="custom-kpi-card">  
                    <div class="kpi-label">{label}</div>  
                    <div class="kpi-val-usd">US$ {usd:,.0f}</div>  
                    <div class="kpi-val-twd">≈ NT$ {twd:,.0f}</div>  
                    {dt}  
                </div>  
                ''',  
                unsafe_allow_html=True  
            )  
  
        k1, k2, k3, k4 = st.columns(4)  
        with k1:  
            render_kpi("資產總市值", totals['usd'].get('mkt', 0), totals['twd'].get('mkt', 0))  
        with k2:  
            mkt_usd = totals['usd'].get('mkt', 0)  
            unreal_usd = totals['usd'].get('unreal', 0)  
            d_p = (unreal_usd / mkt_usd * 100) if mkt_usd > 0 else 0  
            render_kpi("未實現損益", unreal_usd, totals['twd'].get('unreal', 0), d=d_p)  
        with k3:  
            render_kpi(  
                "累計已實現+息",  
                totals['usd'].get('real', 0),  
                totals['twd'].get('real', 0)  
            )  
        with k4:  
            total_unreal = totals['usd'].get('unreal', 0)  
            total_real = totals['usd'].get('real', 0)  
            total_unreal_twd = totals['twd'].get('unreal', 0)  
            total_real_twd = totals['twd'].get('real', 0)  
            render_kpi(  
                "總累計淨損益",  
                total_unreal + total_real,  
                total_unreal_twd + total_real_twd  
            )  
  
        st.write("---")  
        st.subheader("📋 現存持倉明細")  
  
        if not p_df.empty:  
            display_df = p_df[p_df['庫存'] > 0].copy()  
  
            # 顯示同時含 USD 與 TWD 的數字  
            for col in ['市值', '未實現', '已實現+息']:  
                def fmt_row(r):  
                    val = r[col]  
                    if r['IsUS']:  
                        return f"${val:,.0f} (NT${val * rate:,.0f})"  
                    else:  
                        return f"{val:,.0f}"  
                display_df[col] = display_df.apply(fmt_row, axis=1)  
  
            st.dataframe(  
                display_df.drop(columns=['IsUS']),  
                use_container_width=True  
            )  
    else:  
        st.info("尚未有任何交易紀錄，請先在「交易錄入」或「批次匯入」新增資料。")
