
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ✅ قراءة التوكن والمعرف من متغيرات البيئة (GitHub Secrets)
bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

def fetch_data(symbols, start, end, interval):
    return yf.download(tickers=symbols, start=start, end=end, interval=interval,
                       group_by='ticker', auto_adjust=True, progress=False, threads=True)

def detect_sell_breakout(df, lose_body=0.55):
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    if df.empty:
        return None
    o, h, l, c = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    range_ = np.where((h - l) != 0, (h - l), np.nan)
    ratio = np.where(~np.isnan(range_), np.abs(o - c) / range_, 0)
    valid = (c < o) & (ratio >= lose_body)
    highs = np.full(len(df), np.nan)
    breakout = np.zeros(len(df), dtype=bool)
    for i in range(1, len(df)):
        if not np.isnan(highs[i - 1]) and c[i] > highs[i - 1] and not valid[i]:
            breakout[i] = True
            highs[i] = np.nan
        else:
            highs[i] = h[i] if valid[i] else highs[i - 1]
    df['breakout'] = breakout
    return df

def generate_report(market, symbols, interval, date_today):
    if market == "السوق السعودي":
        symbols = [s + ".SR" for s in symbols]
        currency = 'ريال'
        tv_prefix = "TADAWUL:"
    else:
        symbols = [s.upper() for s in symbols]
        currency = 'USD'
        tv_prefix = "NASDAQ:"

    start = '2023-01-01'
    end = (date_today + timedelta(days=1)).strftime('%Y-%m-%d')
    data = fetch_data(symbols, start, end, interval)

    report = []
    for code in symbols:
        try:
            df = data[code].reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            result_df = detect_sell_breakout(df)
            if result_df is None:
                continue
            row = result_df[result_df['Date'] == date_today]
            if not row.empty and row['breakout'].any():
                clean_code = code.replace('.SR', '')
                price = round(row['Close'].iloc[-1], 2)
                tv_link = f"https://www.tradingview.com/symbols/{tv_prefix}{clean_code}/"
                report.append(f"{clean_code} – {price} {currency} – {tv_link}")
        except Exception as e:
            print(f"⚠️ خطأ في {code}: {e}")
    return report

def send_to_telegram(message):
    if not bot_token or not chat_id:
        print("❌ لم يتم العثور على مفاتيح Telegram في البيئة.")
        return None
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    return requests.post(url, data={'chat_id': chat_id, 'text': message})

def main():
    today = datetime.now().date()
    schedules = [
        ("السوق السعودي", "symbols_sa.txt"),
        ("السوق الأمريكي", "symbols_us.txt")
    ]

    for market, file_path in schedules:
        try:
            with open(file_path, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            interval = '1d'
            report_lines = generate_report(market, symbols, interval, today)
            if report_lines:
                message = f"📊 تقرير اختراقات {market} ({today}):\n" + "\n".join(report_lines)
            else:
                message = f"📊 تقرير {market} ({today}): لا توجد اختراقات اليوم."

            response = send_to_telegram(message)
            print(f"📤 {market} – {'✅ تم الإرسال' if response and response.ok else '❌ فشل الإرسال أو المفاتيح غير مضبوطة'}")
        except Exception as err:
            print(f"❌ خطأ أثناء تجهيز تقرير {market}: {err}")

if __name__ == "__main__":
    main()
