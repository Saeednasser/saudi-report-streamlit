import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
from datetime import date, timedelta, datetime

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

def get_company_name(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('shortName', 'اسم غير متوفر')
    except:
        return 'اسم غير متوفر'

st.set_page_config(page_title="تقرير الأسواق", page_icon="📊")
st.title("📊 واجهة اختراقات الأسواق")

market_option = st.selectbox("اختر السوق:", ["السوق السعودي", "السوق الأمريكي"])
timeframe_option = st.selectbox("اختر الفاصل الزمني:", ["1d (يوم)", "1h (ساعة)", "1wk (أسبوع)", "1mo (شهر)"])
timeframe_map = {"1h (ساعة)": "1h", "1d (يوم)": "1d", "1wk (أسبوع)": "1wk", "1mo (شهر)": "1mo"}
interval = timeframe_map[timeframe_option]

selected_date = st.date_input("اختر التاريخ لاختبار الاختراقات:", value=date.today())

symbols_input = st.text_area("ألصق الرموز هنا (رمز في كل سطر، بدون مسافات إضافية):")
selected_symbols = [line.strip() for line in symbols_input.strip().splitlines() if line.strip()]

if st.button("💥 تشغيل التقرير"):
    if not selected_symbols:
        st.warning("⚠️ الرجاء لصق رموز السوق في المربع أعلاه!")
    else:
        if market_option == "السوق السعودي":
            symbols = [s + ".SR" for s in selected_symbols]
            currency = 'ريال'
            tv_prefix = "TADAWUL:"
        elif market_option == "السوق الأمريكي":
            symbols = [s.upper() for s in selected_symbols]
            currency = 'USD'
            tv_prefix = "NASDAQ:"
        else:
            st.error("⚠️ سوق غير معروف.")
            symbols = []

        if symbols:
            start = '2023-01-01'
            end = (selected_date + timedelta(days=1)).strftime('%Y-%m-%d')
            data = fetch_data(symbols, start, end, interval)
            report = []
            if data is not None:
                for code in symbols:
                    try:
                        df = data[code].reset_index()
                        result_df = detect_sell_breakout(df)
                        if result_df is None or result_df.empty or 'Date' not in result_df.columns:
                            continue
                        result_df['Date'] = pd.to_datetime(result_df['Date']).dt.date
                        if interval in ['1wk', '1mo']:
                            target_row = result_df[result_df['Date'] <= selected_date].iloc[-1:]
                        else:
                            target_row = result_df[result_df['Date'] == selected_date]
                        if not target_row.empty and target_row['breakout'].any():
                            clean_code = code.replace('.SR', '')
                            price = round(target_row['Close'].iloc[-1], 2)
                            company_name = get_company_name(code)
                            tv_link = f"https://www.tradingview.com/symbols/{tv_prefix}{clean_code}/"
                            report.append({"الرمز": clean_code, "الاسم": company_name, "السعر": price, "الرابط": tv_link})
                    except Exception as e:
                        st.error(f"⚠️ خطأ في الرمز {code}: {e}")
            if report:
                st.success(f"📊 تقرير اختراقات {market_option} ({selected_date}) - الفاصل الزمني {interval}:")
                df_report = pd.DataFrame(report)
                for idx, row in df_report.iterrows():
                    st.markdown(f"🔹 **[{row['الرمز']}]({row['الرابط']})**\n{row['الاسم']}\n{row['السعر']} {currency}")
                st.markdown("📌 منصة الإشعارات: [Triple Power](https://t.me/TriplePower1)")
            else:
                text = f"🔎 لا توجد اختراقات في التاريخ المحدد ({selected_date}) على الفاصل الزمني {interval}."
                st.info(text)

            if bot_token and chat_id:
                text_for_telegram = "\n".join([f"{row['الرمز']} – {row['الاسم']} – {row['السعر']} {currency} – {row['الرابط']}" for row in report])
                if text_for_telegram:
                    text_for_telegram = f"📊 تقرير اختراقات {market_option} ({selected_date}) - الفاصل الزمني {interval}:\n" + text_for_telegram
                else:
                    text_for_telegram = f"🔎 لا توجد اختراقات في التاريخ المحدد ({selected_date}) على الفاصل الزمني {interval}."

                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                resp = requests.post(url, params={'chat_id': chat_id, 'text': text_for_telegram, 'parse_mode': 'Markdown'})
                if resp.status_code == 200:
                    st.success("✅ تم الإرسال إلى Telegram")
                    st.audio("https://www.soundjay.com/buttons/sounds/button-3.mp3")
                else:
                    st.error(f"❌ خطأ {resp.status_code}: {resp.text}")
            else:
                st.warning("⚠️ لم يتم ضبط متغيرات TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID.")
