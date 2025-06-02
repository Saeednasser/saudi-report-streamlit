import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import date, timedelta

bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
chat_id   = os.getenv('TELEGRAM_CHAT_ID', '').strip()

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
    if df.empty or 'Date' not in df.columns:
        return None
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

st.set_page_config(page_title="تقرير الأسواق", page_icon="📊")
st.title("📊 واجهة اختراقات الأسواق")

market_option = st.selectbox("اختر السوق:", ["السوق السعودي", "السوق الأمريكي"])
timeframe_option = st.selectbox("اختر الفاصل الزمني:", ["1h (ساعة)", "1d (يوم)", "1wk (أسبوع)", "1mo (شهر)"])
timeframe_map = {"1h (ساعة)": "1h", "1d (يوم)": "1d", "1wk (أسبوع)": "1wk", "1mo (شهر)": "1mo"}
interval = timeframe_map[timeframe_option]

symbols_input = st.text_area("ألصق الرموز هنا (رمز في كل سطر، بدون مسافات إضافية):")
selected_symbols = [line.strip() for line in symbols_input.strip().splitlines() if line.strip()]

if st.button("💥 تشغيل التقرير"):
    if not selected_symbols:
        st.warning("⚠️ الرجاء لصق رموز السوق في المربع أعلاه!")
    else:
        if market_option == "السوق السعودي":
            symbols = [s + ".SR" for s in selected_symbols]
            currency = 'ريال'
        elif market_option == "السوق الأمريكي":
            symbols = [s.upper() for s in selected_symbols]
            currency = 'USD'
        else:
            st.error("⚠️ سوق غير معروف.")
            symbols = []

        if symbols:
            start = '2023-01-01'
            end = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
            data = fetch_data(symbols, start, end, interval)
            report = []
            if data is not None:
                for code in symbols:
                    try:
                        df = data[code].reset_index()
                        result_df = detect_sell_breakout(df)
                        if result_df is None or result_df.empty or 'Date' not in result_df.columns:
                            continue
                        if result_df['breakout'].iloc[-1]:
                            clean_code = code.replace('.SR', '')
                            price = round(result_df['Close'].iloc[-1], 2)
                            report.append((clean_code, price))
                    except Exception as e:
                        st.error(f"⚠️ خطأ في الرمز {code}: {e}")
            if report:
                text = f"📊 تقرير اختراقات {market_option} ({date.today()}) - الفاصل الزمني {interval}:\n"
                for sym, pr in report:
                    text += f"🔹 {sym} – {pr} {currency}\n"
                st.success("✅ تم تجهيز التقرير! انظر أدناه.")
                st.text(text)
            else:
                text = f"🔎 لا توجد اختراقات جديدة اليوم ({date.today()}) على الفاصل الزمني {interval}."
                st.info(text)

            if bot_token and chat_id:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                resp = requests.post(url, params={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'})
                if resp.status_code == 200:
                    st.success("✅ تم الإرسال إلى Telegram")
                    st.audio("https://www.soundjay.com/buttons/sounds/button-3.mp3")
                else:
                    st.error(f"❌ خطأ {resp.status_code}: {resp.text}")
            else:
                st.warning("⚠️ لم يتم ضبط متغيرات TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID.")
