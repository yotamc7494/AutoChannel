import os
import urllib.request
from io import BytesIO
from PIL import Image
TICKERS_TO_CHECK = [
    "NVDA", "AVGO", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META",
    "AMD", "PLTR", "SMCI", "ARM", "NFLX", "COIN", "MSTR", "INTC",
    "BRK-B", "JPM", "LLY", "QQQ", "SPY",
]

LOGOS_DIR = "assets/logos"


def fetch_logo_from_sources(ticker):
    """מנסה להוריד לוגו עבור הטיקר ממספר מקורות שונים."""
    sources = [
        f"https://assets.parqet.com/logos/symbol/{ticker}",
        f"https://financialmodelingprep.com/image-stock/{ticker}.png",
        f"https://logo.clearbit.com/{ticker.lower()}.com",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for url in sources:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
                if len(data) < 500:
                    continue

                # אימות שהנתונים שהתקבלו הם אכן תמונה תקינה
                img = Image.open(BytesIO(data))
                img.verify()  # בדיקת תקינות התמונה

                # טעינה מחדש לאחר verify (המטופלת סוגרת את הקובץ)
                img = Image.open(BytesIO(data)).convert("RGBA")
                return img
        except Exception:
            continue

    return None


def download_all_logos(ticker_list=None, force_redownload=False):
    """עובר על רשימת הטיקרים ומוריד את הבלתי קיימים."""
    os.makedirs(LOGOS_DIR, exist_ok=True)
    tickers = ticker_list or TICKERS_TO_CHECK

    print(f"🚀 מתחיל בהורדת לוגואים עבור {len(tickers)} טיקרים...\n")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for ticker in tickers:
        file_path = os.path.join(LOGOS_DIR, f"{ticker}.png")

        if os.path.exists(file_path) and not force_redownload:
            print(f"⏭️  {ticker}: כבר קיים במערכת.")
            skipped_count += 1
            continue

        img = fetch_logo_from_sources(ticker)

        if img:
            img.save(file_path, "PNG")
            print(f"✅ {ticker}: הורד בהצלחה!")
            success_count += 1
        else:
            print(f"❌ {ticker}: לא נfound לוגו תקין במקורות.")
            failed_count += 1

    print("\n" + "=" * 40)
    print(f"📊 סיכום הורדה:")
    print(f"  - הורדו בהצלחה: {success_count}")
    print(f"  - דולגו (כבר קיימים): {skipped_count}")
    print(f"  - נכשלו: {failed_count}")
    print("=" * 40)


if __name__ == "__main__":
    download_all_logos()