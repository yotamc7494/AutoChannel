import modal

app = modal.App("streamlit-ping-bot")

# הגדרת סביבת עבודה עם Playwright מותקן
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("playwright")
    .run_commands("playwright install chromium", "playwright install-deps chromium")
)


# מופעל מדויק כל 10 דקות
@app.function(
    image=image,
    schedule=modal.Period(minutes=30),
    timeout=60
)
def ping_streamlit():
    from playwright.sync_api import sync_playwright
    import time

    token = "XixSAdi234029fj28fn43890982"
    url = f"https://autochannel-stockmarketnews.streamlit.app/?WEBHOOK_TOKEN={token}&video_type=ping"

    print(f"🚀 Opening Playwright session: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        time.sleep(12)  # מנהל חיבור WebSocket מלא מול Streamlit ומריץ את app.py
        browser.close()
    print("✅ Ping completed successfully!")