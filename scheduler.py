import modal
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# הגדרת סביבת העבודה: התקנת Playwright וכרום ישירות על הקונטיינר של Modal
image = (
    modal.Image.debian_slim()
    .pip_install("playwright", "tzdata")
    .run_commands("playwright install chromium", "playwright install-deps chromium")
)

app = modal.App("stock-video-scheduler")


# פונקציית הלוגיקה המרכזית
def trigger_streamlit():
    from playwright.sync_api import sync_playwright

    now_et = datetime.now(ZoneInfo('America/New_York'))
    weekday = now_et.weekday()
    hour = now_et.hour
    is_trading_day = weekday < 5

    if hour >= 17:
        video_type = 'daily_summery' if is_trading_day else 'vs_video'
    elif hour < 11:
        video_type = 'update_video' if is_trading_day else 'vs_video'
    else:
        video_type = 'vs_video'

    # הטוקן שלך מוגדר כאן
    token = "XixSAdi234029fj28fn43890982"

    url = f"https://autochannel-stockmarketnews.streamlit.app/?WEBHOOK_TOKEN={token}&video_type={video_type}"
    print(f"🚀 Opening {url} (Type: {video_type})")

    # הפעלת הדפדפן הסמוי כדי לעורר את התהליך
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        time.sleep(15)  # המתנה קלה כדי ש-Streamlit תסיים לאתחל את התהליך
        browser.close()
        print(f"✅ Successfully triggered {video_type}!")


# ---------------------------------------------------------
# טריגרים המתוזמנים של Modal (בשעות ה-UTC המקוריות)
# ---------------------------------------------------------

# 1. בוקר לפני מסחר (12:14 UTC -> 08:14 EDT)
@app.function(image=image, schedule=modal.Cron("14 12 * * *"))
def morning_cron():
    trigger_streamlit()


# 2. שעה ויראלית באמצע היום (17:25 UTC -> 13:25 EDT)
@app.function(image=image, schedule=modal.Cron("25 17 * * *"))
def noon_cron():
    trigger_streamlit()


# 3. סיכום יומי (22:21 UTC -> 18:21 EDT)
@app.function(image=image, schedule=modal.Cron("21 22 * * 1-5"))
def evening_cron():
    trigger_streamlit()