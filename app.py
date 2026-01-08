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
            return s.zfill(4)  
        else:  
            return s  
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
    """  
    回傳 AI 分析文字。若全部模型都失敗，回傳帶有說明的錯誤訊息。  
    """  
    try:  
        prompt = (  
            f"你是一位資深投資顧問。請分析標的：{symbol} {name}。"  
            f"收盤價：{data_summary['close']:.2f}, RSI(14)：{data_summary['rsi']:.1f}, "  
            f"KD(K)：{data_summary['k']:.1f}。"  
            "請給出專業短評、目前趨勢判定與具體操作建議（買進/減碼/觀望），"  
            "約120字繁體中文。"  
        )  
        # 使用正式穩定模型  
        model_names = ['gemini-2.0-flash-001']  
        last_error = None  
        for m_name in model_names:  
            try:  
                model = genai.GenerativeModel(model_name=m_name)  
                response = model.generate_content(prompt)  
                if response and getattr(response, "text", None):  
                    return f"{response.text}\n\n(AI引擎: {m_name})"  
            except Exception as e:  
                last_error = str(e)  
                continue  
        if last_error:  
            return f"AI 分析暫時不可用，請稍後重試。（錯誤：{last_error}）"  
        return "AI 分析暫時不可用，請稍後重試。"  
    except Exception as e:  
        return f"AI 連線錯誤: {str(e)}"  
  
# 輕量級取得股票名稱，避免為了名稱跑完整技術分析  
def resolve_stock_name(symbol: str) -> str:  
    clean = standardize_symbol(symbol)  
    q_sym = f"{clean}.TW" if clean.isdigit() else clean  
    try:  
        stock = yf.Ticker(q_sym)  
        fast_info = getattr(stock, "fast_info", None)  
        if fast_info and isinstance(fast_info, dict):  
            if "shortName" in fast_info:  
                return fast_info["shortName"]  
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
        return True, 0  
    try:  
        client = init_connection()  
        spreadsheet = client.open("TradeLog")  
        sheet = spreadsheet.worksheet("TW_Trades" if market == 'TW' else "US_Trades")  
        sheet.append_rows(rows)  
        st.cache_data.clear()  
        return True, len(rows)  
    except Exception as e:  
        st.error(f"批次寫入 {market} 資料失敗：{e}")  
        return False, 0  
  
# --- 5. 核心運算 ---  
  
@st.cache_data(ttl=3600)  
def get_exchange_rate():  
    try:  
        h = yf.Ticker("TWD=X").history(period="1d")  
        return h['Close'].iloc[-1] if not h.empty else 32.5  
    except:  
        return 32.5  
  
def calculate_full_portfolio(df, rate):  
    """  
    回傳：  
    - 當前持股明細 DataFrame（含 IsUS）  
    - totals: {'twd': {...}, 'usd': {...}}  
    - df_sorted: 整個交易資料（日期排序後）  
    """  
    portfolio = {}  
    if df.empty:  
        return pd.DataFrame(), {"twd": {}, "usd": {}}, pd.DataFrame()  
  
    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))  
    df = df.sort_values('日期')  
  
    for _, row in df.iterrows():  
        sym = standardize_symbol(row['代號'])  
        if sym not in portfolio:  
            portfolio[sym] = {  
                'Name': row.get('名稱', sym),  
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
        elif "現金股息" in type_str or ("股息" in type_str and "現金" not in type_str and "配股" not in type_str):  
            p['Realized'] += pr  
        elif "配股" in type_str:  
            p['Qty'] += q  
  
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
                name = fast_info.get('shortName')  
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
  
@st.cache_data(ttl=1800)  
def build_nav_series(trades_df: pd.DataFrame, rate: float):  
    """  
    建立簡易資產淨值曲線（TWD）：  
    - 依日期展開  
    - 依每日持股數 * 當日收盤價 + 已實現損益（用當下匯率換算 TWD）  
    """  
    if trades_df.empty:  
        return pd.DataFrame()  
  
    df = trades_df.copy()  
    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))  
    df = df.sort_values('日期')  
  
    df['代號_std'] = df['代號'].apply(standardize_symbol)  
    symbols = df['代號_std'].unique().tolist()  
    if not symbols:  
        return pd.DataFrame()  
  
    min_date = df['日期'].min()  
    max_date = df['日期'].max()  
  
    # 取每檔標的價史  
    price_dict = {}  
    for s in symbols:  
        q_sym = f"{s}.TW" if is_tw_stock(s) and s.isdigit() else s  
        try:  
            stock = yf.Ticker(q_sym)  
            hist = stock.history(start=min_date, end=max_date + datetime.timedelta(days=1))  
            if not hist.empty:  
                price_dict[s] = hist['Close']  
        except:  
            continue  
  
    if not price_dict:  
        return pd.DataFrame()  
  
    # 日期索引：所有價史的 union  
    all_dates = sorted(set().union(*[ser.index for ser in price_dict.values()]))  
    if not all_dates:  
        return pd.DataFrame()  
  
    pos = {s: 0.0 for s in symbols}  
    realized_twd = 0.0  
  
    nav_records = []  
    grouped = df.groupby('日期')  
  
    for d in all_dates:  
        date_only = pd.to_datetime(d).normalize()  
  
        # 當日交易  
        if date_only in grouped.groups:  
            day_trades = grouped.get_group(date_only)  
            for _, row in day_trades.iterrows():  
                s = row['代號_std']  
                q = safe_float(row['股數'])  
                pr = safe_float(row['價格'])  
                f = safe_float(row['手續費'])  
                t = safe_float(row['交易稅'])  
                type_str = str(row['類別'])  
                is_us = not is_tw_stock(s)  
  
                if "買" in type_str:  
                    pos[s] += q  
                    cash_flow = -(q * pr + f)  
                elif "賣" in type_str:  
                    pos[s] -= q  
                    cash_flow = (q * pr - f - t)  
                elif "現金股息" in type_str or ("股息" in type_str and "現金" not in type_str and "配股" not in type_str):  
                    cash_flow = pr  
                elif "配股" in type_str:  
                    pos[s] += q  
                    cash_flow = 0  
                else:  
                    cash_flow = 0  
  
                if cash_flow != 0:  
                    realized_twd += cash_flow * (rate if is_us else 1.0)  
  
        # 市值  
        mkt_twd = 0.0  
        for s in symbols:  
            if s not in price_dict:  
                continue  
            ser = price_dict[s]  
            if d not in ser.index:  
                continue  
            price = ser.loc[d]  
            qty = pos.get(s, 0.0)  
            if qty == 0:  
                continue  
            is_us = not is_tw_stock(s)  
            val = qty * price * (rate if is_us else 1.0)  
            mkt_twd += val  
  
        nav = mkt_twd + realized_twd  
        nav_records.append({  
            "日期": date_only,  
            "市值_TWD": mkt_twd,  
            "已實現_TWD": realized_twd,  
            "淨值_TWD": nav  
        })  
  
    nav_df = pd.DataFrame(nav_records).drop_duplicates(subset=['日期'])  
    nav_df = nav_df.sort_values('日期')  
    return nav_df  
  
# --- 6. 介面呈現 ---  
  
tab1, tab2, tab3, tab4 = st.tabs([  
    "📝 交易錄入",  
    "📥 批次匯入",  
    "📊 趨勢戰情",  
    "💰 資產透視"  
])  
  
# --- Tab1：單筆交易記錄 ---  
  
with tab1:  
    st.subheader("📝 單筆交易記錄")  
  
    with st.form("trade_input"):  
        c1, c2 = st.columns(2)  
        ttype = c1.selectbox("交易類別", ["買入", "賣出", "現金股息", "配股"])  
        tdate = c2.date_input("交易日期")  
  
        c3, c4 = st.columns(2)  
        tsym = c3.text_input("股票代號 (如 2330 / 00940 / AAPL)")  
        tname_hint = c4.text_input("名稱（可留空自動查詢）", "")  
  
        c5, c6 = st.columns(2)  
        tqty = c5.number_input("股數", min_value=0.0, step=1.0)  
        tprice = c6.number_input("價格/股息金額", min_value=0.0)  
  
        with st.expander("進階費用設定（選填）"):  
            c7, c8 = st.columns(2)  
            tfee = c7.number_input("手續費", min_value=0.0)  
            ttax = c8.number_input("交易稅", min_value=0.0)  
  
        submitted = st.form_submit_button("確認送出")  
  
        if submitted:  
            if not tsym:  
                st.warning("請輸入股票代號")  
            else:  
                sym_std = standardize_symbol(tsym)  
                if tname_hint.strip():  
                    tname = tname_hint.strip()  
                else:  
                    tname = resolve_stock_name(tsym)  
  
                if "買" in ttype:  
                    amt = -(tqty * tprice + tfee)  
                elif "賣" in ttype:  
                    amt = (tqty * tprice - tfee - ttax)  
                elif "現金股息" in ttype:  
                    amt = tprice  
                elif "配股" in ttype:  
                    amt = 0  
                else:  
                    amt = 0  
  
                ok = save_data([  
                    str(tdate),  
                    ttype,  
                    sym_std,  
                    tname,  
                    tprice,  
                    tqty,  
                    tfee,  
                    ttax,  
                    amt  
                ])  
                if ok:  
                    st.success(f"✅ 記錄已成功儲存：{sym_std} {tname}")  
                else:  
                    st.error("❌ 記錄儲存失敗，請稍後再試或檢查連線設定。")  
  
# --- Tab2：批次匯入 ---  
  
with tab2:  
    st.subheader("📥 批次匯入交易")  
  
    template = pd.DataFrame({  
        "日期": ["2026-01-01", "2026-01-10", "2026-01-15", "2026-01-20"],  
        "類別": ["買入", "賣出", "現金股息", "配股"],  
        "代號": ["2330", "2330", "2330", "00940"],  
        "名稱": ["台積電", "台積電", "台積電", "群益台灣科技優息"],  
        "價格": [600, 650, 20, 0],  
        "股數": [100, 50, 0, 500],  
        "手續費": [20, 20, 0, 0],  
        "交易稅": [0, 100, 0, 0]  
    })  
  
    st.download_button(  
        "📥 下載 CSV 範本",  
        io.BytesIO(template.to_csv(index=False).encode('utf-8-sig')),  
        "trade_template.csv"  
    )  
  
    st.markdown("上傳欄位需包含：`日期, 類別, 代號, 名稱, 價格, 股數, 手續費, 交易稅`")  
  
    uploaded = st.file_uploader("上傳 CSV 檔案", type=["csv"])  
  
    if uploaded and st.button("開始匯入檔案"):  
        try:  
            df_u = pd.read_csv(uploaded)  
        except Exception as e:  
            st.error(f"CSV 解析失敗：{e}")  
        else:  
            tw_rows, us_rows = [], []  
            error_rows = []  
  
            for idx, r in df_u.iterrows():  
                try:  
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
                        0  
                    ]  
                    if is_tw_stock(sym):  
                        tw_rows.append(row)  
                    else:  
                        us_rows.append(row)  
                except Exception as e:  
                    error_rows.append((idx + 2, str(e)))  # +2：含標題列  
  
            ok_tw, n_tw = batch_save_data(tw_rows, 'TW')  
            ok_us, n_us = batch_save_data(us_rows, 'US')  
  
            st.write("---")  
            st.markdown("### 匯入結果總結")  
            st.write(f"- TW_Trades 成功筆數：{n_tw}（成功：{ok_tw}）")  
            st.write(f"- US_Trades 成功筆數：{n_us}（成功：{ok_us}）")  
  
            if error_rows:  
                st.warning(f"有 {len(error_rows)} 筆列解析失敗：")  
                for row_no, msg in error_rows[:20]:  
                    st.write(f"- 第 {row_no} 列：{msg}")  
                if len(error_rows) > 20:  
                    st.write(f"... 其餘 {len(error_rows) - 20} 筆省略顯示")  
            elif ok_tw and ok_us:  
                st.success("✅ 批次匯入完成！")  
  
# --- Tab3：趨勢戰情診斷 ---  
  
with tab3:  
    st.subheader("📊 趨勢戰情診斷")  
  
    raw_for_filter = load_data()  
  
    # 代號 → 名稱 映射  
    names_map = {}  
    for _, r in raw_for_filter.iterrows():  
        s = standardize_symbol(r['代號'])  
        names_map[s] = r.get('名稱', s)  
  
    inv = {}  
    for _, r in raw_for_filter.iterrows():  
        s = standardize_symbol(r['代號'])  
        q = safe_float(r['股數'])  
        if "買" in str(r['類別']):  
            inv[s] = inv.get(s, 0) + q  
        elif "賣" in str(r['類別']):  
            inv[s] = inv.get(s, 0) - q  
        elif "配股" in str(r['類別']):  
            inv[s] = inv.get(s, 0) + q  
    held_syms = sorted([s for s, q in inv.items() if q > 0])  
  
    st.markdown("#### 🔎 選擇診斷標的")  
    mode = st.radio("選擇方式", ["從目前持股選", "手動輸入代號"], horizontal=True)  
  
    target = None  
    if mode == "從目前持股選":  
        options = ["請選擇"] + [f"{s} {names_map.get(s, '')}" for s in held_syms]  
        sel_label = st.selectbox("🎯 庫存快速診斷", options)  
        if sel_label != "請選擇":  
            target = sel_label.split()[0]  # 前半段為代號  
    else:  
        search_sym = st.text_input("🔍 手動輸入代號 (如 AAPL、2330、00940)", "")  
        if search_sym.strip():  
            target = search_sym.strip()  
  
    if target:  
        with st.spinner("正在生成深度診斷報告..."):  
            hist, ana, err = analyze_full_signal(target)  
  
        if err:  
            st.warning(f"無法完成技術分析：{err}")  
        elif hist is not None and ana is not None:  
            st.markdown(f"### {ana['name']}（{ana['symbol']}）趨勢診斷")  
  
            # 指標區塊  
            m1, m2, m3, m4 = st.columns(4)  
            m1.metric("目前股價", f"{ana['metrics']['close']:.2f}")  
            m2.metric("RSI (14)", f"{ana['metrics']['rsi']:.1f}")  
            m3.metric("K 值", f"{ana['metrics']['k']:.1f}")  
            m4.metric(  
                "布林位置",  
                "中軌上方" if ana['metrics']['close'] > hist['MA20'].iloc[-1] else "中軌下方"  
            )  
  
            # AI 分析  
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
            else:  
                st.info("尚未設定 GEMINI_API_KEY，無法啟用 AI 投顧分析。")  
  
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
  
            # K 線 + 成交量圖  
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
                height=550,  
                template="plotly_white",  
                xaxis_rangeslider_visible=False,  
                hovermode="x unified",  
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
        p_df_all, totals_all, df_sorted = calculate_full_portfolio(raw_df, rate)  
  
        # 市場篩選：全部 / 台股 / 美股  
        st.markdown("#### 🔍 市場篩選")  
        market_view = st.radio(  
            "選擇要看的市場",  
            ["全部", "僅台股", "僅美股"],  
            horizontal=True  
        )  
  
        if p_df_all.empty:  
            st.info("目前沒有任何持股。")  
        else:  
            if market_view == "僅台股":  
                p_df = p_df_all[~p_df_all['IsUS']].copy()  
            elif market_view == "僅美股":  
                p_df = p_df_all[p_df_all['IsUS']].copy()  
            else:  
                p_df = p_df_all.copy()  
  
            # 重新計算 totals 依照篩選後持股  
            t_twd = {'mkt': 0, 'unreal': 0, 'real': 0}  
            t_usd = {'mkt': 0, 'unreal': 0, 'real': 0}  
            for _, r in p_df.iterrows():  
                mkt = r['市值']  
                unreal = r['未實現']  
                real = r['已實現+息']  
                if r['IsUS']:  
                    t_usd['mkt'] += mkt  
                    t_usd['unreal'] += unreal  
                    t_usd['real'] += real  
  
                    t_twd['mkt'] += mkt * rate  
                    t_twd['unreal'] += unreal * rate  
                    t_twd['real'] += real * rate  
                else:  
                    t_twd['mkt'] += mkt  
                    t_twd['unreal'] += unreal  
                    t_twd['real'] += real  
  
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
                render_kpi("資產總市值", t_usd.get('mkt', 0), t_twd.get('mkt', 0))  
            with k2:  
                mkt_usd = t_usd.get('mkt', 0)  
                unreal_usd = t_usd.get('unreal', 0)  
                d_p = (unreal_usd / mkt_usd * 100) if mkt_usd > 0 else 0  
                render_kpi("未實現損益", unreal_usd, t_twd.get('unreal', 0), d=d_p)  
            with k3:  
                render_kpi(  
                    "累計已實現+息",  
                    t_usd.get('real', 0),  
                    t_twd.get('real', 0)  
                )  
            with k4:  
                total_unreal = t_usd.get('unreal', 0)  
                total_real = t_usd.get('real', 0)  
                total_unreal_twd = t_twd.get('unreal', 0)  
                total_real_twd = t_twd.get('real', 0)  
                render_kpi(  
                    "總累計淨損益",  
                    total_unreal + total_real,  
                    total_unreal_twd + total_real_twd  
                )  
  
            st.write("---")  
  
            # 圓餅圖：持股市值分布  
            st.markdown("#### 🥧 持股市值分布")  
            pie_df = p_df[p_df['庫存'] > 0].copy()  
            if not pie_df.empty:  
                pie_df['市值_TWD'] = pie_df.apply(  
                    lambda r: r['市值'] * (rate if r['IsUS'] else 1.0),  
                    axis=1  
                )  
                fig_pie = go.Figure(  
                    data=[go.Pie(  
                        labels=pie_df['名稱'] + " (" + pie_df['代號'] + ")",  
                        values=pie_df['市值_TWD'],  
                        hole=0.3  
                    )]  
                )  
                fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))  
                st.plotly_chart(fig_pie, use_container_width=True)  
            else:  
                st.info("目前無持股，無法顯示資產分布。")  
  
            # 資產淨值曲線  
            st.markdown("#### 📈 資產淨值曲線（TWD）")  
            try:  
                nav_df = build_nav_series(raw_df, rate)  
                if not nav_df.empty:  
                    fig_nav = go.Figure()  
                    fig_nav.add_trace(  
                        go.Scatter(  
                            x=nav_df['日期'],  
                            y=nav_df['淨值_TWD'],  
                            mode='lines',  
                            name='淨值'  
                        )  
                    )  
                    fig_nav.update_layout(  
                        height=400,  
                        template="plotly_white",  
                        margin=dict(l=10, r=10, t=10, b=10)  
                    )  
                    st.plotly_chart(fig_nav, use_container_width=True)  
                else:  
                    st.info("目前淨值曲線資料不足。")  
            except Exception as e:  
                st.warning(f"資產淨值曲線生成失敗：{e}")  
  
            st.write("---")  
            st.subheader("📋 現存持倉明細")  
  
            if not p_df.empty:  
                display_df = p_df[p_df['庫存'] > 0].copy()  
  
                for col in ['市值', '未實現', '已實現+息']:  
                    def fmt_row(r):  
                        val = r[col]  
                        if r['IsUS']:  
                            return f"${val:,.0f} (NT${val * rate:,.0f})"  
                        else:  
                            return f"{val:,.0f}"  
                    display_df[col] = display_df.apply(fmt_row, axis=1)  
  
                # 名稱 + 代號都保留顯示  
                st.dataframe(  
                    display_df.drop(columns=['IsUS']),  
                    use_container_width=True  
                )  
  
            # 單檔個股損益明細  
            st.write("---")  
            st.markdown("#### 🎯 單檔個股損益明細")  
  
            # 代號→名稱 map  
            name_map_all = {}  
            for _, r in raw_df.iterrows():  
                s = standardize_symbol(r['代號'])  
                name_map_all[s] = r.get('名稱', s)  
  
            all_syms = sorted(set(standardize_symbol(x) for x in raw_df['代號'].tolist()))  
            options_single = ["請選擇"] + [f"{s} {name_map_all.get(s, '')}" for s in all_syms]  
            sel_label = st.selectbox("選擇標的查看詳細損益", options_single)  
  
            if sel_label != "請選擇":  
                sym_std = sel_label.split()[0]  
                sub = raw_df[raw_df['代號'].apply(standardize_symbol) == sym_std].copy()  
                if not sub.empty:  
                    sub['日期'] = pd.to_datetime(sub['日期'].apply(standardize_date))  
                    sub = sub.sort_values('日期')  
  
                    qty = 0.0  
                    cost = 0.0  
                    realized = 0.0  
  
                    for _, row in sub.iterrows():  
                        q = safe_float(row['股數'])  
                        pr = safe_float(row['價格'])  
                        f = safe_float(row['手續費'])  
                        t = safe_float(row['交易稅'])  
                        tp = str(row['類別'])  
  
                        if "買" in tp:  
                            cost += q * pr + f  
                            qty += q  
                        elif "賣" in tp and qty > 0:  
                            avg = cost / qty  
                            cost_sold = avg * q  
                            realized += (q * pr - f - t) - cost_sold  
                            qty -= q  
                            cost -= cost_sold  
                        elif "現金股息" in tp or ("股息" in tp and "現金" not in tp and "配股" not in tp):  
                            realized += pr  
                        elif "配股" in tp:  
                            qty += q  
  
                    q_sym = f"{sym_std}.TW" if is_tw_stock(sym_std) and sym_std.isdigit() else sym_std  
                    try:  
                        stock = yf.Ticker(q_sym)  
                        h = stock.history(period="1d")  
                        cur_price = h['Close'].iloc[-1] if not h.empty else 0.0  
                    except:  
                        cur_price = 0.0  
  
                    is_us = not is_tw_stock(sym_std)  
                    mkt_val = qty * cur_price  
                    mkt_val_twd = mkt_val * (rate if is_us else 1.0)  
                    cost_twd = cost * (rate if is_us else 1.0)  
                    realized_twd = realized * (rate if is_us else 1.0)  
                    total_pnl_twd = (mkt_val_twd - cost_twd) + realized_twd  
                    total_invest = cost_twd  
                    total_ret = (total_pnl_twd / total_invest * 100) if total_invest > 0 else 0  
  
                    show_name = name_map_all.get(sym_std, sym_std)  
                    st.markdown(f"##### {show_name}（{sym_std}）損益概覽")  
  
                    c1, c2, c3, c4 = st.columns(4)  
                    with c1:  
                        st.metric("目前股數", f"{qty:,.0f}")  
                    with c2:  
                        st.metric("現價", f"{cur_price:,.2f}")  
                    with c3:  
                        st.metric("市值 (TWD)", f"{mkt_val_twd:,.0f}")  
                    with c4:  
                        st.metric("總報酬率", f"{total_ret:,.1f}%")  
  
                    c5, c6 = st.columns(2)  
                    with c5:  
                        st.metric("累計投入成本 (TWD)", f"{cost_twd:,.0f}")  
                    with c6:  
                        st.metric("累計已實現+息 (TWD)", f"{realized_twd:,.0f}")  
  
                    st.markdown("##### 交易明細")  
                    st.dataframe(sub, use_container_width=True)  
                else:  
                    st.info("找不到該標的的交易紀錄。")  
    else:  
        st.info("尚未有任何交易紀錄，請先在「交易錄入」或「批次匯入」新增資料。")
