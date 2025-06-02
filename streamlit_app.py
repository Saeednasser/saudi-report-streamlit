
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import json
from datetime import date, timedelta

bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
chat_id   = os.getenv('TELEGRAM_CHAT_ID', '').strip()

all_symbols = [
    "3010", "3040", "6014", "4071", "4162", "6040",
    "8210", "2082", "4346", "2090", "2180", "4031"
]

PERSIST_FILE = 'selected_symbols.json'
def save_selection(selection):
    with open(PERSIST_FILE, 'w') as f:
        json.dump(selection, f)

def load_selection():
    if os.path.exists(PERSIST_FILE):
        with open(PERSIST_FILE, 'r') as f:
            return json.load(f)
    return []

def fetch_data(symbols, start, end, interval):
    return yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval=interval,
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=True
    )

def detect_sell_breakout(df, lose_body=0.55):
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    if df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    o, h, l, c = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    range_ = np.where((h - l) != 0, (h - l), np.nan)
    ratio  = np.where(~np.isnan(range_), np.abs(o - c) / range_, 0)
    valid  = (c < o) & (ratio >= lose_body)
    highs, breakout = np.full(len(df), np.nan), np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if not np.isnan(highs[i-1]) and c[i] > highs[i-1] and not valid[i]:
            breakout[i] = True
            highs[i] = np.nan
        else:
            highs[i] = h[i] if valid[i] else highs[i-1]
    df['breakout'] = breakout
    return df

st.set_page_config(page_title="تقرير السوق السعودي", page_icon="📊")
st.title("📊 واجهة اختراقات السوق السعودي")
previous_selection = load_selection()
selected_symbols = st.multiselect(
    "اختر الرموز للتحليل:",
    all_symbols,
    default=previous_selection
)

if st.button("💥 تشغيل التقرير"):
    if not selected_symbols:
        st.warning("⚠️ لم تختَر أي رموز!")
    else:
        save_selection(selected_symbols)
        symbols = [s + ".SR" for s in selected_symbols]
        start = '2023-01-01'
        end = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        data = fetch_data(symbols, start, end, '1d')
        report = []
        if data is not None:
            for code in symbols:
                try:
                    df = data[code].reset_index()
                    df = detect_sell_breakout(df)
                    if df.empty or df['Date'].iloc[-1].date() != date.today():
                        continue
                    if df['breakout'].iloc[-1]:
                        price = round(df['Close'].iloc[-1], 2)
                        report.append((code.replace('.SR', ''), price))
                except Exception as e:
                    st.error(f"⚠️ خطأ في الرمز {code}: {e}")
        if report:
            text = f"📊 تقرير اختراقات السوق السعودي ({date.today()}):\n"
            for sym, pr in report:
                text += f"🔹 {sym} – {pr} ريال\n"
            st.success("✅ تم تجهيز التقرير! انظر أدناه.")
            st.text(text)
        else:
            text = f"🔎 لا توجد اختراقات جديدة اليوم ({date.today()})."
            st.info(text)
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, params={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'})
        if resp.status_code == 200:
            st.success("✅ تم الإرسال إلى Telegram")
            st.audio("https://www.soundjay.com/buttons/sounds/button-3.mp3")
        else:
            st.error(f"❌ خطأ {resp.status_code}: {resp.text}")
