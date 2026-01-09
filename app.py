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
    
    /* 自定義 KPI 卡片 */
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
    .news-box { background-color: #FFF3E0; border-left: 5px solid #FF9800; padding: 15px; border-radius: 5px; margin-top: 10px; color: #212529 !important; }
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
        return float(str(val).replace(',', ''))
    except:
        return 0.0

def standardize_symbol(symbol):
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
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # 嘗試從 Secrets 讀取，若無則報錯
    if "gcp_service_account" not in st.secrets:
        st.error("❌ 未設定 GCP Service Account Secrets")
        st.stop()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    return gspread.authorize(creds)

def init_gemini():
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return True
    return False

# --- 3.1 AI 分析邏輯 (技術面 + 消息面) ---

def ask_gemini_tech_analyst(symbol, name, data_summary):
    """技術面 AI 分析"""
    try:
        prompt = (
            f"你是一位資深投資顧問。請分析標的：{symbol} {name}。"
            f"技術指標：收盤價 {data_summary['close']:.2f}, RSI(14) {data_summary['rsi']:.1f}, "
            f"KD(K) {data_summary['k']:.1f}, KD(D) {data_summary['d']:.1f}。"
            "請給出專業短評、目前趨勢判定與具體操作建議（買進/減碼/觀望），"
            "限制 100 字以內繁體中文。"
        )
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text if response else "AI 分析無回應"
    except Exception as e:
        return f"AI 連線錯誤: {str(e)}"

def ask_gemini_sentiment_analyst(symbol, news_list):
    """消息面 AI 情緒分析 (回傳 JSON)"""
    if not news_list:
        return {"error": "無新聞資料"}

    news_text = "\n".join([f"- {n.get('title', '')}" for n in news_list[:5]])
    
    prompt = f"""
    你是一位華爾街情緒分析師。請閱讀以下關於 {symbol} 的新聞標題：
    {news_text}
    
    請進行情緒分析並回傳嚴格的 JSON 格式 (不要 Markdown)：
    {{
        "sentiment_score": (整數 -100 到 100, 負為看空, 正為看多),
        "sentiment_label": ("看多"/"看空"/"中立"),
        "summary": ("50字以內的繁體中文新聞重點摘要"),
        "prediction": ("基於新聞的短期走勢預測")
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = model.generate_content(prompt)
        text = response.text
        # 清洗 JSON
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "")
        elif text.startswith("```"):
            text = text.replace("```", "")
        return json.loads(text)
    except Exception as e:
        return {"error": f"AI 解析失敗: {str(e)}"}

# --- 4. 資料獲取與計算 ---

def resolve_stock_name(symbol: str) -> str:
    clean = standardize_symbol(symbol)
    q_sym = f"{clean}.TW" if clean.isdigit() else clean
    try:
        stock = yf.Ticker(q_sym)
        return stock.info.get('shortName') or stock.info.get('longName') or clean
    except:
        return clean

def get_stock_news(symbol):
    """獲取 Yahoo Finance 新聞"""
    clean = standardize_symbol(symbol)
    q_sym = f"{clean}.TW" if clean.isdigit() else clean
    try:
        stock = yf.Ticker(q_sym)
        news = stock.news
        return news if news else []
    except:
        return []

@st.cache_data(ttl=60)
def load_data():
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        # 讀取 TW 與 US 分頁
        dfs = []
        for sheet_name, market in [("TW_Trades", "TW"), ("US_Trades", "US")]:
            try:
                ws = spreadsheet.worksheet(sheet_name)
                recs = ws.get_all_records()
                if recs:
                    d = pd.DataFrame(recs)
                    d['Market'] = market
                    dfs.append(d)
            except:
                pass
        
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"資料讀取失敗 (請確認 Google Sheet 'TradeLog' 存在): {e}")
        return pd.DataFrame()

def save_data(row_data):
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        sheet_name = "TW_Trades" if is_tw_stock(row_data[2]) else "US_Trades"
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except:
            # 若分頁不存在則建立
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=10)
            sheet.append_row(["日期", "類別", "代號", "名稱", "價格", "股數", "手續費", "交易稅", "總金額"])
            
        sheet.append_row(row_data)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"寫入失敗: {e}")
        return False

def batch_save_data(rows, market):
    if not rows: return True, 0
    try:
        client = init_connection()
        spreadsheet = client.open("TradeLog")
        sheet_name = "TW_Trades" if market == 'TW' else "US_Trades"
        sheet = spreadsheet.worksheet(sheet_name)
        sheet.append_rows(rows)
        st.cache_data.clear()
        return True, len(rows)
    except Exception as e:
        st.error(f"批次寫入錯誤: {e}")
        return False, 0

# --- 5. 投資組合運算 ---

@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        h = yf.Ticker("TWD=X").history(period="1d")
        return h['Close'].iloc[-1] if not h.empty else 32.5
    except:
        return 32.5

def calculate_full_portfolio(df, rate):
    portfolio = {}
    if df.empty: return pd.DataFrame(), {}, pd.DataFrame()

    df['日期'] = pd.to_datetime(df['日期'].apply(standardize_date))
    df = df.sort_values('日期')

    for _, row in df.iterrows():
        sym = standardize_symbol(row['代號'])
        if sym not in portfolio:
            portfolio[sym] = {'Name': row.get('名稱', sym), 'Qty': 0, 'Cost': 0, 'Realized': 0, 'IsUS': not is_tw_stock(sym)}
        
        p = portfolio[sym]
        q = safe_float(row['股數'])
        pr = safe_float(row['價格'])
        f = safe_float(row['手續費'])
        t = safe_float(row['交易稅'])
        act = str(row['類別'])

        if "買" in act:
            p['Cost'] += (q * pr + f)
            p['Qty'] += q
        elif "賣" in act and p['Qty'] > 0:
            avg = p['Cost'] / p['Qty']
            cost_sold = avg * q
            p['Realized'] += (q * pr - f - t) - cost_sold
            p['Qty'] -= q
            p['Cost'] -= cost_sold
        elif "現金股息" in act:
            p['Realized'] += pr # 假設輸入的是總金額
        elif "配股" in act:
            p['Qty'] += q

    # 批次抓取現價
    active_syms = [s for s, v in portfolio.items() if v['Qty'] > 0]
    prices = {}
    if active_syms:
        qs = [f"{s}.TW" if is_tw_stock(s) and s.isdigit() else s for s in active_syms]
        try:
            data = yf.Tickers(" ".join(qs))
            for i, s in enumerate(active_syms):
                try:
                    h = data.tickers[qs[i]].history(period="1d")
                    prices[s] = h['Close'].iloc[-1] if not h.empty else 0
                except:
                    prices[s] = 0
        except:
            pass

    res = []
    t_twd = {'mkt': 0, 'unreal': 0, 'real': 0}
    t_usd = {'mkt': 0, 'unreal': 0, 'real': 0}

    for s, v in portfolio.items():
        cp = prices.get(s, 0)
        mkt = v['Qty'] * cp
        unreal = mkt - v['Cost'] if v['Qty'] > 0 else 0
        
        # 匯總
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
                "代號": s, "名稱": v['Name'], "庫存": v['Qty'], "現價": cp,
                "市值": mkt, "未實現": unreal, "已實現+息": v['Realized'], "IsUS": v['IsUS']
            })
            
    return pd.DataFrame(res), {"twd": t_twd, "usd": t_usd}, df

@st.cache_data(ttl=300)
def analyze_full_signal(symbol):
    """技術分析主邏輯"""
    try:
        clean = standardize_symbol(symbol)
        q_sym = f"{clean}.TW" if clean.isdigit() else clean
        stock = yf.Ticker(q_sym)
        df = stock.history(period="1y")

        if df is None or df.empty or len(df) < 60:
            return None, None, "K線資料不足 (需 > 60 天)"

        # 指標計算
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))

        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        
        k_l, d_l = [], []
        k, d = 50, 50
        for r in rsv:
            if pd.isna(r):
                k_l.append(50); d_l.append(50)
            else:
                k = (2/3)*k + (1/3)*r
                d = (2/3)*d + (1/3)*k
                k_l.append(k); d_l.append(d)
        df['K'], df['D'] = k_l, d_l

        last = df.iloc[-1]
        
        # 簡單策略判定
        if last['Close'] > last['MA5'] and last['K'] > last['D']:
            st_sig = {"txt": "🔴 強勢偏多", "col": "#D32F2F", "desc": "站上5日線 + KD金叉"}
        else:
            st_sig = {"txt": "🟠 震盪觀望", "col": "#FF9800", "desc": "指標尚不明確"}

        if last['Close'] > last['MA60']:
            lt_sig = {"txt": "🔴 多頭格局", "col": "#D32F2F", "desc": "守穩生命線 (MA60)"}
        else:
            lt_sig = {"txt": "🟢 弱勢空頭", "col": "#2E7D32", "desc": "季線反壓顯著"}

        metrics = {
            "close": float(last['Close']), "rsi": float(last['RSI']),
            "k": float(last['K']), "d": float(last['D'])
        }
        
        name = stock.info.get('shortName') or clean
        
        ana = {
            "st": st_sig, "lt": lt_sig, "metrics": metrics,
            "name": name, "symbol": q_sym
        }
        return df, ana, None
    except Exception as e:
        return None, None, str(e)

# --- 6. 介面呈現 ---

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 交易錄入", "📥 批次匯入", "📊 趨勢戰情", "💰 資產透視"
])

# --- Tab 1: 單筆輸入 ---
with tab1:
    st.subheader("📝 單筆交易記錄")
    with st.form("trade_input"):
        c1, c2 = st.columns(2)
        ttype = c1.selectbox("類別", ["買入", "賣出", "現金股息", "配股"])
        tdate = c2.date_input("日期")
        c3, c4 = st.columns(2)
        tsym = c3.text_input("代號 (如 2330 / AAPL)")
        tname = c4.text_input("名稱 (選填)", "")
        c5, c6 = st.columns(2)
        tqty = c5.number_input("股數", min_value=0.0)
        tprice = c6.number_input("價格/總金額", min_value=0.0)
        with st.expander("進階費用"):
            c7, c8 = st.columns(2)
            tfee = c7.number_input("手續費", 0.0)
            ttax = c8.number_input("交易稅", 0.0)
        
        if st.form_submit_button("送出"):
            if tsym:
                std_sym = standardize_symbol(tsym)
                final_name = tname if tname else resolve_stock_name(std_sym)
                
                amt = 0
                if "買" in ttype: amt = -(tqty*tprice + tfee)
                elif "賣" in ttype: amt = (tqty*tprice - tfee - ttax)
                elif "現金" in ttype: amt = tprice # 股息直接填總額
                
                row = [str(tdate), ttype, std_sym, final_name, tprice, tqty, tfee, ttax, amt]
                if save_data(row):
                    st.success(f"已儲存 {std_sym} {ttype}")
            else:
                st.warning("請輸入代號")

# --- Tab 2: 批次匯入 ---
with tab2:
    st.subheader("📥 批次匯入")
    st.markdown("格式範本：`日期, 類別, 代號, 名稱, 價格, 股數, 手續費, 交易稅`")
    
    # 產生範本 CSV
    sample_df = pd.DataFrame([{
        "日期": "2026-01-01", "類別": "買入", "代號": "2330", "名稱": "台積電",
        "價格": 600, "股數": 1000, "手續費": 20, "交易稅": 0
    }])
    st.download_button("下載範本 CSV", sample_df.to_csv(index=False).encode('utf-8-sig'), "template.csv", "text/csv")
    
    uploaded = st.file_uploader("上傳 CSV", type=["csv"])
    if uploaded and st.button("開始匯入"):
        try:
            df_u = pd.read_csv(uploaded)
            tw_r, us_r = [], []
            for _, r in df_u.iterrows():
                try:
                    sym = standardize_symbol(r['代號'])
                    # 計算總額
                    act = r['類別']
                    q, p, f, t = r['股數'], r['價格'], r['手續費'], r['交易稅']
                    amt = 0
                    if "買" in act: amt = -(q*p+f)
                    elif "賣" in act: amt = (q*p-f-t)
                    elif "現金" in act: amt = p
                    
                    row = [standardize_date(r['日期']), act, sym, r['名稱'], p, q, f, t, amt]
                    if is_tw_stock(sym): tw_r.append(row)
                    else: us_r.append(row)
                except: continue
            
            ok1, n1 = batch_save_data(tw_r, 'TW')
            ok2, n2 = batch_save_data(us_r, 'US')
            st.success(f"完成！TW: {n1} 筆, US: {n2} 筆")
        except Exception as e:
            st.error(f"檔案解析失敗: {e}")

# --- Tab 3: 趨勢戰情 (優化版) ---
with tab3:
    st.subheader("📊 AI 趨勢戰情室")
    
    # 搜尋與選擇
    raw_for_filter = load_data()
    held_syms = []
    if not raw_for_filter.empty:
        inv = raw_for_filter.groupby('代號')['股數'].sum() # 簡易計算
        held_syms = inv.index.tolist() # 這裡簡化，實際應用 portfolio 函數算比較準
        
    c_s1, c_s2 = st.columns([1, 2])
    target = c_s2.text_input("輸入代號 (如 2330)", "").upper()
    if not target:
        sel = c_s1.selectbox("或從庫存選擇", ["請選擇"] + held_syms)
        if sel != "請選擇": target = sel

    if target:
        st.divider()
        hist, ana, err = analyze_full_signal(target)
        
        if err:
            st.error(err)
        elif hist is not None:
            st.markdown(f"### {ana['name']} ({ana['symbol']})")
            
            # --- 技術面區塊 ---
            t1, t2 = st.columns([2, 1])
            with t1:
                # K線圖
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K線'), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='月線', line=dict(color='#FF9800', width=1)), row=1, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='成交量', marker_color='rgba(100,100,100,0.3)'), row=2, col=1)
                fig.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
            with t2:
                # 策略訊號
                st.markdown(f'<div class="strategy-card" style="border-left:5px solid {ana["st"]["col"]}">'
                            f'<div class="strategy-title">短期趨勢</div>'
                            f'<div class="strategy-signal" style="color:{ana["st"]["col"]}">{ana["st"]["txt"]}</div>'
                            f'<div>{ana["st"]["desc"]}</div></div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="strategy-card" style="border-left:5px solid {ana["lt"]["col"]}">'
                            f'<div class="strategy-title">長期趨勢</div>'
                            f'<div class="strategy-signal" style="color:{ana["lt"]["col"]}">{ana["lt"]["txt"]}</div>'
                            f'<div>{ana["lt"]["desc"]}</div></div>', unsafe_allow_html=True)
                
                # 技術面 AI
                if init_gemini() and st.button("🤖 技術面 AI 診斷"):
                    with st.spinner("Gemini 正在看線圖..."):
                        res = ask_gemini_tech_analyst(ana['symbol'], ana['name'], ana['metrics'])
                        st.markdown(f'<div class="ai-box"><b>🤖 技術觀點：</b><br>{res}</div>', unsafe_allow_html=True)

            # --- 消息面區塊 (新增功能) ---
            st.markdown("#### 📰 新聞情緒與 AI 判讀")
            
            if st.button("🚀 啟動新聞情緒分析 (Gemini 3 Flash Preview)"):
                with st.spinner(f"正在搜尋 {target} 近期新聞並進行情緒推論..."):
                    # 1. 抓新聞
                    news_list = get_stock_news(target)
                    
                    if news_list:
                        # 2. AI 分析
                        sentiment_res = ask_gemini_sentiment_analyst(target, news_list)
                        
                        if "error" not in sentiment_res:
                            # 顯示儀表板與摘要
                            col_gauge, col_text = st.columns([1, 2])
                            
                            with col_gauge:
                                score = sentiment_res.get('sentiment_score', 0)
                                fig_g = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = score,
                                    title = {'text': "市場情緒分數"},
                                    gauge = {
                                        'axis': {'range': [-100, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [-100, -30], 'color': "#FFEBEE"},
                                            {'range': [-30, 30], 'color': "#F5F5F5"},
                                            {'range': [30, 100], 'color': "#E8F5E9"}]
                                    }
                                ))
                                fig_g.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
                                st.plotly_chart(fig_g, use_container_width=True)
                                
                            with col_text:
                                st.markdown(f'<div class="news-box"><b>📝 新聞摘要：</b><br>{sentiment_res.get("summary", "無摘要")}</div>', unsafe_allow_html=True)
                                st.markdown(f"**🔮 預測：** {sentiment_res.get('prediction', '無預測')}")
                                st.divider()
                                st.caption("新聞來源 (Yahoo Finance):")
                                for n in news_list[:3]:
                                    st.markdown(f"- [{n.get('title')}]({n.get('link')})")
                        else:
                            st.error(sentiment_res['error'])
                    else:
                        st.warning("查無近期新聞，無法進行情緒分析。")

# --- Tab 4: 資產透視 ---
with tab4:
    st.subheader("💰 資產透視")
    
    rate = get_exchange_rate()
    raw_df = load_data()
    
    if not raw_df.empty:
        p_df, totals, _ = calculate_full_portfolio(raw_df, rate)
        
        # 顯示 KPI
        k1, k2, k3, k4 = st.columns(4)
        total_mkt = totals['usd']['mkt'] + totals['twd']['mkt']/rate # 這裡簡單加總，邏輯可依需求調整
        total_pl = (totals['usd']['unreal'] + totals['usd']['real']) + (totals['twd']['unreal'] + totals['twd']['real'])/rate
        
        # 為了顯示一致，統一轉 TWD 顯示
        twd_all_mkt = totals['twd']['mkt'] + totals['usd']['mkt'] * rate
        twd_all_unreal = totals['twd']['unreal'] + totals['usd']['unreal'] * rate
        twd_all_real = totals['twd']['real'] + totals['usd']['real'] * rate
        
        k1.metric("總資產 (TWD)", f"${twd_all_mkt:,.0f}")
        k2.metric("未實現損益", f"${twd_all_unreal:,.0f}", delta_color="normal")
        k3.metric("已實現損益", f"${twd_all_real:,.0f}")
        k4.metric("總損益合計", f"${twd_all_unreal + twd_all_real:,.0f}")
        
        st.divider()
        
        if not p_df.empty:
            # 圓餅圖
            c_p1, c_p2 = st.columns([1, 2])
            with c_p1:
                p_df['市值_TWD'] = p_df.apply(lambda x: x['市值'] * (rate if x['IsUS'] else 1), axis=1)
                fig_pie = px.pie(p_df[p_df['庫存']>0], values='市值_TWD', names='名稱', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with c_p2:
                # 持股表
                disp_df = p_df[p_df['庫存']>0].copy()
                st.dataframe(disp_df[['代號', '名稱', '庫存', '現價', '市值', '未實現', '已實現+息']], use_container_width=True)
        else:
            st.info("目前無庫存")
    else:
        st.info("尚無交易資料")
