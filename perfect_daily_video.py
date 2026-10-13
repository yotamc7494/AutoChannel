import edge_tts
import os
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"
import sys
import logging
import asyncio
import concurrent.futures
import json
import requests
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from uploader import upload_video
from thumbnail import generate_daily_thumbnail
import numpy as np
import yfinance as yf
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from google import genai
from google.genai import types
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    VideoClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
import urllib.request
import moviepy.audio.fx as afx
from dotenv import load_dotenv
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
load_dotenv()
import aiohttp


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PerfectDailyVideo")
GEMINI_MODEL1 = "gemini-3.5-flash-lite"
GEMINI_MODEL2 = "gemini-flash-lite-latest"

# ---------------------------------------------------------
# Pydantic Schemas for Structured LLM Output
# ---------------------------------------------------------
class StockAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    company_spoken_name: str = Field(description="Full spoken company name for TTS, e.g. Nvidia")
    script: str = Field(description="Spoken narrative for TTS (~40 words).Starting with technical data and mostly predictions, level targets and some CTA.")
    timeframe: str = Field(description="Chart timeframe: '1d', '1mo', '3mo', '6mo', or '1y'")
    interval: str = Field(description="Chart interval: '5m' for 1d, '1d' for others")
    key_levels: List[float] = Field(default=[], description="Important price levels to highlight on chart")
    highlight_type: str = Field(default="none", description="Type of overlay: 'breakout', 'support', 'resistance', or 'none'")

class FullVideoScriptSchema(BaseModel):
    intro_script: str = Field(description="STRICTLY 15 to 25 words English intro script with high CTA.")
    news_ticker_text: str = Field(description="Single line ticker with top 3-5 market headlines + short 4-word summary each, separated by '  •  ' e.g. 'FED CUTS RATES: Policy shifted dovish  •  NVDA SURGES: AI chip demand peaks  •  BTC BREAKS $70K: Institutional inflow high'")
    analyzed_stocks: List[StockAnalysis]
    youtube_title: str
    description: str
    tags: str

# ---------------------------------------------------------
# Helper Functions & TTS Normalization
# ---------------------------------------------------------
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing!")
    return genai.Client(api_key=api_key)


def ask_llm(prompt, config=None):
    client = get_gemini_client()

    # ניסיון הרצה ראשון עם GEMINI_MODEL1
    try:
        if config:
            return client.models.generate_content(
                model=GEMINI_MODEL1,
                contents=prompt,
                config=config
            )
        else:
            return client.models.generate_content(model=GEMINI_MODEL1, contents=prompt)

    except Exception as e:
        logger.warning(f"⚠️ Failed to generate content with {GEMINI_MODEL1} ({e}). Falling back to {GEMINI_MODEL2}...")

        # ניסיון הרצה משני עם GEMINI_MODEL2
        if config:
            return client.models.generate_content(
                model=GEMINI_MODEL2,
                contents=prompt,
                config=config
            )
        else:
            return client.models.generate_content(model=GEMINI_MODEL2, contents=prompt)

def format_seconds_to_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def clean_script_for_tts(text: str) -> str:
    """Removes non-ASCII characters/emojis & converts symbols for clear TTS."""
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Strips emojis
    text = text.replace("$", "").replace("%", " percent")
    replacements = {
        r'\bNVDA\b': 'Nvidia', r'\bTSLA\b': 'Tesla', r'\bAAPL\b': 'Apple',
        r'\bAMZN\b': 'Amazon', r'\bMSFT\b': 'Microsoft', r'\bGOOGL\b': 'Google',
        r'\bMETA\b': 'Meta', r'\bBTC\b': 'Bitcoin', r'\bSPY\b': 'S and P 500', r'\bQQQ\b': 'Nasdaq'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def fetch_free_proxies():
    """שולף רשימת פרוקסים חינמיים מעודכנת בזמן אמת"""
    try:
        url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=2500&country=all&ssl=all&anonymity=all"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return [p.strip() for p in response.text.splitlines() if p.strip()]
    except Exception as e:
        logger.warning(f"Could not fetch proxy list: {e}")
    return []

def generate_voiceover_audio(script_text: str, output_path: str = "temp_speech.mp3") -> float:
    clean_text = clean_script_for_tts(script_text)
    logger.info(f"Generating TTS audio: {output_path}")

    async def _save():
        voice = "en-US-ChristopherNeural"
        rate = "+5%"

        # 1. ניסיון חיבור ישיר
        try:
            comm = edge_tts.Communicate(clean_text, voice, rate=rate)
            await comm.save(output_path)
            return
        except Exception as e:
            logger.warning(f"Direct TTS failed ({e}), switching to proxy fallback...")

        # 2. ניסיון דרך רשימת פרוקסים חינמיים
        proxies = fetch_free_proxies()
        for proxy_ip in proxies:
            proxy_url = f"http://{proxy_ip}"
            try:
                logger.info(f"Attempting TTS via proxy: {proxy_url}")
                comm = edge_tts.Communicate(clean_text, voice, rate=rate, proxy=proxy_url)
                await comm.save(output_path)
                logger.info(f"✅ TTS generated successfully via proxy: {proxy_url}")
                return
            except Exception:
                continue

        raise RuntimeError("All proxy attempts failed for TTS.")

    # הרצה בתוך ThreadPoolExecutor למניעת התנגשות ב-Event Loop של Streamlit
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, _save())
        future.result()

    audio_clip = AudioFileClip(output_path)
    duration = audio_clip.duration
    audio_clip.close()

    return duration

# ---------------------------------------------------------
# 4. Dynamic 30-Sec Intro Rendering
# ---------------------------------------------------------
def fetch_intro_index_data():
    logger.info("📊 Fetching intraday index data...")

    spy = yf.Ticker("SPY").history(period="1d", interval="5m", prepost=True)
    qqq = yf.Ticker("QQQ").history(period="1d", interval="5m", prepost=True)
    btc = yf.Ticker("BTC-USD").history(period="1d", interval="5m", prepost=True)

    def process_series(df, target_len=300):
        if df is None or df.empty or "Close" not in df:
            return np.zeros(target_len), 0.0

        vals = df["Close"].dropna().values.astype(float)
        N = len(vals)
        if N == 0:
            return np.zeros(target_len), 0.0

        first_val = vals[0] if vals[0] != 0 else 1.0
        pct = ((vals - first_val) / first_val) * 100.0

        x_smooth = np.linspace(0, 1, target_len)

        if N >= 4:
            x_raw = np.linspace(0, 1, N)
            spl = make_interp_spline(x_raw, pct, k=3)
            y_smooth = spl(x_smooth)
        else:
            x_raw = np.linspace(0, 1, N)
            y_smooth = np.interp(x_smooth, x_raw, pct)

        return y_smooth, float(vals[-1])

    spy_pct, spy_price = process_series(spy)
    qqq_pct, qqq_price = process_series(qqq)
    btc_pct, btc_price = process_series(btc)

    x_smooth = np.linspace(0, 1, 300)

    date_str = "TODAY"
    if not spy.empty and hasattr(spy.index[-1], 'strftime'):
        date_str = spy.index[-1].strftime("%b %d, %Y").upper()

    # שליפת רשימת המניות הצדדית
    watchlist = fetch_top_movers_and_watchlist()
    print(watchlist)
    return {
        "x": x_smooth,
        "QQQ": (qqq_pct, qqq_price),
        "SPY": (spy_pct, spy_price),
        "BTC": (btc_pct, btc_price),
        "date_str": date_str,
        "watchlist": watchlist
    }


# ---------------------------------------------------------
# 1. חיפוש כללי באינטרנט (24 שעות אחרונות) - במקום מיכה סטוקס
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. חיפוש כללי באינטרנט (24 שעות אחרונות) - במקום מיכה סטוקס
# ---------------------------------------------------------

def fetch_daily_market_news_from_rss() -> str:
    logger.info("📡 Fetching latest 24h market news via Google News RSS...")

    # רשימת מילות מפתח ומדדים מרכזיים לסריקה
    queries = ["SPY stock", "QQQ stock", "Bitcoin crypto market", "Nvidia Tesla stock news"]
    all_headlines = []

    for query in queries:
        try:
            # חיפוש ממוקד ל-24 השעות האחרונות (when:1d)
            encoded_query = query.replace(" ", "+")
            url = f"https://news.google.com/rss/search?q={encoded_query}+when:1d&hl=en-US&gl=US&ceid=US:en"

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)

                # שליפת 3 הכותרות המובילות לכל שאילתת חיפוש
                for item in root.findall(".//item")[:3]:
                    title = item.findtext("title") or ""
                    if " - " in title:
                        title = title.rsplit(" - ", 1)[0]  # הסרת שם מקור החדשות בסוף הכותרת
                    if title not in all_headlines:
                        all_headlines.append(title)
        except Exception as e:
            logger.warning(f"Failed pulling RSS for query '{query}': {e}")

    # גיבוש כל הכותרות למסמך טקסט קצר ונקי
    market_summary = "Latest 24h Market Headlines & Catalysts:\n" + "\n".join([f"- {h}" for h in all_headlines])
    logger.info(f"✅ Extracted {len(all_headlines)} clean headlines from RSS.")
    return market_summary


# ---------------------------------------------------------
# 2. חילוץ ואימות טיקרים מתוך סיכום הרשת
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2. Market Data Verification
# ---------------------------------------------------------
def extract_and_verify_ticker_data(market_summary: str) -> dict:
    logger.info("🔍 Extracting mentioned tickers from web summary and fetching ground-truth data...")


    prompt = f"Extract all stock tickers mentioned in this market summary as a JSON array of strings (e.g. ['NVDA', 'TSLA', 'AAPL']): {market_summary}"
    res = ask_llm(prompt)
    tickers = []
    try:
        tickers = json.loads(re.search(r'\[.*\]', res.text, re.DOTALL).group(0))
    except Exception:
        tickers = ["NVDA", "TSLA", "AAPL", "MSFT"]

    verified_data = {}
    for ticker in set(tickers):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                last_close = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else last_close
                pct_change = ((last_close - prev_close) / prev_close) * 100
                verified_data[ticker] = {
                    "price": round(last_close, 2),
                    "pct_change": round(pct_change, 2)
                }
        except Exception as e:
            logger.warning(f"Failed to fetch market data for {ticker}: {e}")

    return verified_data


def fetch_top_movers_and_watchlist() -> List[dict]:
    """שולפת רשימת מניות מוכרות ומניות עם תנועה חדה (Top Movers) כולל מחיר ואחוז שינוי יומי."""
    logger.info("📈 Fetching watchlist and top movers data...")

    # רשימת מניות מוכרות + מניות פופולריות נוספות לבדיקה
    candidate_tickers = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "AMD", "META", "GOOGL", "PLTR", "MARA"]
    stocks_data = []

    for ticker in candidate_tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                last_price = float(hist['Close'].iloc[-1])
                prev_price = float(hist['Close'].iloc[-2])
                pct_change = ((last_price - prev_price) / prev_price) * 100.0
                stocks_data.append({
                    "ticker": ticker,
                    "price": last_price,
                    "pct": pct_change,
                    "abs_pct": abs(pct_change)
                })
        except Exception as e:
            logger.warning(f"Failed fetching data for {ticker}: {e}")

    # מיון: שמירה על 3 מניות מוכרות ראשיות + המניות עם אחוז השינוי הגבוה ביותר היום
    stocks_data.sort(key=lambda x: x["abs_pct"], reverse=True)

    # החזרת עד 7 מניות מובילות לתצוגה נקייה
    return stocks_data[:7]


# ---------------------------------------------------------
# 3. יצירת תסריט מבוסס נתוני רשת ואימות שוק
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. AI Script Generation with Structured Schema
# ---------------------------------------------------------
def generate_verified_script(market_summary: str, verified_market_data: dict) -> FullVideoScriptSchema:
    logger.info("Generating structured video script with News Ticker from web insights...")

    prompt = f"""
    You are a top-tier Wall Street video producer and YouTube Shorts retention strategist.
    Synthesize this web market summary and VERIFIED market data into an English script.

    Web Market Summary:
    {market_summary}

    Verified Market Data:
    {json.dumps(verified_market_data, indent=2)}

    RULES:
    1. "intro_script": ULTRA-SHORT SCROLL-STOPPING HOOK (15 TO 25 WORDS).
    2. "news_ticker_text": Generate 3 high-impact breaking news headlines with ultra-short descriptions, separated by ' • '. Keep text uppercase and ultra punchy for top scrolling ticker.
    3. "analyzed_stocks": Max 4 key stocks with precise key levels.
    4. ABSOLUTELY NO EMOJIS in any field.
    """

    response = ask_llm(prompt, config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FullVideoScriptSchema
            ))
    return FullVideoScriptSchema.model_validate_json(response.text)

# ---------------------------------------------------------
# 4. עדכון ה-Pipeline המרכזי להרצה עצמאית
# ---------------------------------------------------------
def run_perfect_pipeline(
        output_filename="perfect_daily_recap.mp4",
        bg_music_path="assets/weekly_recup_music.mp3",
        upload=False
):
    logger.info("🚀 Launching Independent Daily Video Pipeline (Web Search Mode)...")
    temp_files = []
    timestamps = []
    current_time = 0.0

    try:
        # === חלק 1: חיפוש כללי באינטרנט על 24 השעות האחרונות ===
        market_summary = fetch_daily_market_news_from_rss()

        # === חלק 2: אימות נתוני שוק ===
        verified_data = extract_and_verify_ticker_data(market_summary)

        # === חלק 3: יצירת תסריט וכותרות ב-LLM מתוך הסיכום הכללי ===
        script_schema = generate_verified_script(market_summary, verified_data)

        # === חלק 4: רינדור אינטרו (גרף מדדים קפוא לאחר 7 שניות) ===
        intro_audio_file = "temp_intro.mp3"
        intro_duration = generate_voiceover_audio(script_schema.intro_script, intro_audio_file)
        temp_files.append(intro_audio_file)

        market_data = fetch_intro_index_data()
        market_data["news_ticker_text"] = script_schema.news_ticker_text

        # העברת global_start_time = current_time (0.0)
        intro_clip = render_dynamic_intro_clip(market_data, intro_audio_file, intro_duration,
                                               global_start_time=current_time)
        intro_clip = add_subtitles(intro_clip, script_schema.intro_script)

        timestamps.append(f"{format_seconds_to_timestamp(current_time)} Market Overview")
        current_time += intro_duration

        # === חלק 5: רינדור סקציות המניות ===
        stock_clips = []
        for stock_info in script_schema.analyzed_stocks:
            stock_audio_file = f"temp_{stock_info.ticker}.mp3"
            stock_duration = generate_voiceover_audio(stock_info.script, stock_audio_file)
            temp_files.append(stock_audio_file)

            verified_price = verified_data.get(stock_info.ticker, {}).get("price", 0.0)

            # 👈 העברת news_ticker_text ו-global_start_time בלולאה
            s_clip = render_annotated_stock_clip(
                stock_info,
                verified_price,
                stock_audio_file,
                stock_duration,
                news_ticker_text=script_schema.news_ticker_text,
                global_start_time=current_time
            )
            s_clip = add_subtitles(s_clip, stock_info.script)
            stock_clips.append(s_clip)

            timestamps.append(f"{format_seconds_to_timestamp(current_time)} {stock_info.ticker}")
            current_time += stock_duration

        # === חלק 6: רינדור Outro וchapters ===
        outro_clip = render_outro_clip(duration=5.0)
        timestamps.append(f"{format_seconds_to_timestamp(current_time)} Outro")

        chapters_text = "\n\nCHAPTERS:\n" + "\n".join(timestamps)
        final_description = (script_schema.description + chapters_text).strip()

        # === חלק 7: איחוד והפקת סרטון סופי ===
        final_video = concatenate_videoclips([intro_clip] + stock_clips + [outro_clip])

        if bg_music_path and os.path.exists(bg_music_path):
            logger.info("🎵 Mixing background music...")
            bg_music = AudioFileClip(bg_music_path)
            if bg_music.duration < final_video.duration:
                bg_music = bg_music.with_effects([afx.AudioLoop(duration=final_video.duration)])
            else:
                bg_music = bg_music.subclipped(0, final_video.duration)
            bg_music = bg_music.with_effects([
                afx.MultiplyVolume(0.12),
                afx.AudioFadeOut(2.0)
            ])
            final_video = final_video.with_audio(CompositeAudioClip([final_video.audio, bg_music]))

        logger.info(f"💾 Rendering final video file: {output_filename}...")
        final_video.write_videofile(
            output_filename,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k",
            threads=8,
            preset="ultrafast"
        )

        # === חלק 8: Thumbnail והעלאה ליוטיוב ===
        img_path = None
        if callable(generate_daily_thumbnail):
            try:
                img_path = generate_daily_thumbnail(
                    sp500_val=float(market_data["SPY"][1]),
                    sp500_pct=float(market_data["SPY"][0][-1]),
                    qqq_val=float(market_data["QQQ"][1]),
                    qqq_pct=float(market_data["QQQ"][0][-1]),
                    btc_val=float(market_data["BTC"][1]),
                    btc_pct=float(market_data["BTC"][0][-1]),
                    date_str=market_data.get("date_str"),
                    template_path="assets/Thumbnail_Daily.jpg",
                    output_path="d_thumbnail.png",
                )
            except Exception as e:
                logger.error(f"⚠️ Thumbnail generation failed: {e}")

        if upload and callable(upload_video):
            logger.info("🚀 Uploading video to YouTube...")
            upload_video(
                file_path=output_filename,
                title=script_schema.youtube_title,
                description=final_description,
                tags=script_schema.tags,
                thumbnail_path=img_path
            )

        logger.info("✅ Pipeline executed successfully (Independent Mode)!")
        return output_filename

    finally:
        logger.info("🧹 Cleaning up temporary files...")
        for tmp in temp_files:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {tmp}: {e}")

def render_outro_clip(duration: float = 5.0) -> VideoClip:
    logger.info("Rendering Outro Screen with CTA...")
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor('#0B0E14')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0B0E14')
    ax.axis('off')

    fig.text(0.5, 0.65, "THANKS FOR WATCHING!", fontsize=38, fontweight='bold', color='#00FFA3', ha='center')
    fig.text(0.5, 0.50, "LIKE & SUBSCRIBE FOR DAILY MARKET RECAPS", fontsize=24, fontweight='bold', color='#FFFFFF', ha='center')
    fig.text(0.5, 0.38, "What is your target? Drop your thoughts in the comments below!", fontsize=18, color='#8B949E', ha='center')

    # 🚀 רינדור פעם אחת מחוץ ל-make_frame למניעת חישובים חוזרים
    canvas.draw()
    frame_img = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    def make_frame(t):
        return frame_img

    return VideoClip(make_frame, duration=duration)


def render_dynamic_intro_clip(market_data: dict, audio_path: str, duration: float,
                              global_start_time: float = 0.0) -> VideoClip:
    logger.info(f"Rendering Dynamic Intro with News Ticker Bar (Duration: {duration:.2f}s)...")
    voice_clip = AudioFileClip(audio_path)

    x_smooth = market_data["x"]
    assets = ["QQQ", "SPY", "BTC"]
    colors = {"QQQ": "#00E5FF", "SPY": "#00FFA3", "BTC": "#FFB800"}
    watchlist = market_data.get("watchlist", [])
    raw_ticker = market_data.get("news_ticker_text", "MARKET NEWS TICKER • QQQ & SPY IN FOCUS")

    seg_dur = duration / 3.0

    sidebar_w, sidebar_h = int(1920 * 0.18), int(1080 * 0.68)
    sidebar_img = render_static_sidebar_image(watchlist, width_px=sidebar_w, height_px=sidebar_h)

    x1, x2 = int(1920 * 0.77), int(1920 * 0.77) + sidebar_w
    y1, y2 = int(1080 * 0.22), int(1080 * 0.22) + sidebar_h

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    def make_frame(t):
        global_t = global_start_time + t
        asset_idx = min(int(t // seg_dur), 2)
        local_t = t % seg_dur
        progress = min(local_t / seg_dur, 1.0)
        curr_step = max(1, int(progress * len(x_smooth)))

        fig.clf()
        fig.patch.set_facecolor('#0B0E14')

        # 📰 1. ציור פס החדשות לפי global_t
        draw_news_ticker_bar(fig, raw_ticker, global_t)

        # 📈 2. אזור הגרף הראשי
        ax = fig.add_axes([0.05, 0.10, 0.70, 0.68])
        ax.set_facecolor('#0B0E14')

        fig.text(0.05, 0.84, f"MARKET OVERVIEW | {market_data['date_str']}", fontsize=20, fontweight='bold',
                 color='#FFFFFF')

        all_visible_y = []
        for idx, asset in enumerate(assets):
            y_vals, current_price = market_data[asset]
            c_color = colors[asset]
            asset_start_t = idx * seg_dur

            if t < asset_start_t:
                continue
            elif t >= asset_start_t + seg_dur:
                p_step = len(x_smooth)
            else:
                p_step = curr_step

            ax.plot(x_smooth[:p_step], y_vals[:p_step], color=c_color, linewidth=3.5, label=asset)
            ax.scatter(x_smooth[p_step - 1], y_vals[p_step - 1], color=c_color, s=120, zorder=10)

            ax.text(
                x_smooth[p_step - 1], y_vals[p_step - 1],
                f" {asset}: ${current_price:,.2f}",
                color=c_color, fontsize=12, fontweight='bold', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='#0B0E14', edgecolor=c_color, alpha=0.85)
            )
            all_visible_y.extend(y_vals[:p_step])

        ax.set_xlim(-0.02, 1.15)
        if all_visible_y:
            ax.set_ylim(min(all_visible_y) - 0.5, max(all_visible_y) + 0.8)

        ax.grid(True, linestyle='--', alpha=0.15, color='#8B949E')
        ax.tick_params(colors='#FFFFFF', labelsize=11)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#30363D')
            spine.set_linewidth(2)

        canvas.draw()
        frame_img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

        # הדבקת הסיידבר
        frame_img[y1:y2, x1:x2] = sidebar_img

        return frame_img

    clip = VideoClip(make_frame, duration=duration).with_audio(voice_clip)
    return clip


def render_static_sidebar_image(watchlist: List[dict], width_px: int = 345, height_px: int = 810) -> np.ndarray:
    """מייצרת תמונה סטטית של סיידבר המניות המובילות פעם אחת בלבד."""
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor('#161B22')

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#161B22')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # מסגרת היקפית
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#30363D')
        spine.set_linewidth(2)

    # כותרת וקו מפריד
    ax.text(0.5, 0.93, "TOP MOVERS", fontsize=15, fontweight='bold', color='#FFFFFF', ha='center')
    ax.plot([0.1, 0.9], [0.88, 0.88], color='#30363D', linewidth=1.5)

    # טקסט המניות
    start_y = 0.80
    y_step = 0.11
    for i, item in enumerate(watchlist[:7]):
        curr_y = start_y - (i * y_step)
        pct_val = item.get("pct", 0.0)
        pct_color = "#00FFA3" if pct_val >= 0 else "#FF3366"
        sign = "+" if pct_val >= 0 else ""

        # טיקר
        ax.text(0.08, curr_y, item.get("ticker", ""), fontsize=13, fontweight='bold', color='#FFFFFF', va='center')
        # מחיר
        ax.text(0.08, curr_y - 0.04, f"${item.get('price', 0.0):.2f}", fontsize=10, color='#8B949E', va='center')
        # אחוז שינוי (ירוק / אדום)
        ax.text(0.92, curr_y - 0.015, f"{sign}{pct_val:.2f}%", fontsize=11, fontweight='bold', color=pct_color, ha='right', va='center')

    canvas.draw()
    sidebar_img = np.asarray(canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return sidebar_img


def draw_news_ticker_bar(fig, raw_ticker_text: str, global_t: float, scroll_speed: float = 0.06):
    """מציירת את פס החדשות הרץ בראש ה-Figure לפי זמן גלובלי (global_t) לרצף מושלם בכל הסרטון."""
    full_ticker_str = f"BREAKING NEWS:  {raw_ticker_text}   •   " * 6

    ax_ticker = fig.add_axes([0.05, 0.91, 0.90, 0.055])
    ax_ticker.set_facecolor('#161B22')
    ax_ticker.set_xlim(0, 1)
    ax_ticker.set_ylim(0, 1)
    ax_ticker.set_xticks([])
    ax_ticker.set_yticks([])

    for spine in ax_ticker.spines.values():
        spine.set_visible(True)
        spine.set_color('#30363D')
        spine.set_linewidth(2)

    # חישוב המיקום לפי הזמן המצטבר של כל הסרטון (global_t)
    x_pos = 0.02 - (global_t * scroll_speed)

    ax_ticker.text(
        x_pos, 0.5, full_ticker_str,
        fontsize=12, fontweight='bold', color='#00FFA3', va='center',
        clip_on=True, transform=ax_ticker.transData
    )

# ---------------------------------------------------------
# 5. Dynamic Stock Chart with Annotations
# ---------------------------------------------------------


def render_annotated_stock_clip(stock_info: StockAnalysis, verified_price: float, audio_path: str,
                                duration: float, news_ticker_text: str = "", global_start_time: float = 0.0) -> VideoClip:
    ticker = stock_info.ticker
    logger.info(f"Rendering animated chart for {ticker} with news ticker bar...")
    voice_clip = AudioFileClip(audio_path)

    df = yf.Ticker(ticker).history(period=stock_info.timeframe, interval=stock_info.interval, prepost=True).dropna()
    if df.empty:
        df = yf.Ticker(ticker).history(period="1mo", interval="1d").dropna()

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    canvas = FigureCanvasAgg(fig)

    opens = np.asarray(df['Open'].values, dtype=float).flatten()
    closes = np.asarray(df['Close'].values, dtype=float).flatten()
    highs = np.asarray(df['High'].values, dtype=float).flatten()
    lows = np.asarray(df['Low'].values, dtype=float).flatten()

    x_idxs = np.arange(len(df))
    candle_colors = np.where(closes >= opens, '#00FFA3', '#FF3366')

    y_min, y_max = lows.min(), highs.max()
    y_range = max(1.0, y_max - y_min)

    def make_frame(t):
        global_t = global_start_time + t
        fig.clf()
        fig.patch.set_facecolor('#0B0E14')

        # 📰 1. ציור פס החדשות ברצף מושלם
        if news_ticker_text:
            draw_news_ticker_bar(fig, news_ticker_text, global_t)

        # 📈 2. הגרף הראשי (מותאם גובה לפס החדשות)
        ax = fig.add_axes([0.08, 0.10, 0.88, 0.68])
        ax.set_facecolor('#0B0E14')

        # Draw Candlesticks
        ax.vlines(x_idxs, lows, highs, color=candle_colors, linewidth=1.2)
        ax.bar(x_idxs, closes - opens, bottom=opens, color=candle_colors, width=0.6)

        progress = min(t / duration, 1.0)
        start_x = int(len(df) * 0.15 * progress)
        end_x = len(df) - 1
        ax.set_xlim(start_x, end_x + max(2, int(len(df) * 0.05)))
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.12)

        blink_alpha = 0.45 + 0.45 * np.abs(np.sin(np.pi * t * 1.5))

        for level in stock_info.key_levels:
            ax.axhline(y=level, color='#FFD700', linestyle='--', linewidth=2.0, alpha=blink_alpha)
            label_x_idx = int(start_x + (end_x - start_x) * 0.82)
            ax.text(label_x_idx, level + y_range * 0.02, f"Key Level: ${level:.2f}",
                    color='#FFD700', fontsize=12, fontweight='bold', ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#0B0E14', edgecolor='#FFD700', alpha=0.85))

        if stock_info.key_levels:
            target_level = stock_info.key_levels[0]
        else:
            target_level = closes[-1]

        arrow_x = int(start_x + (end_x - start_x) * 0.70)
        text_x = int(start_x + (end_x - start_x) * 0.40)

        h_type = stock_info.highlight_type.lower()
        if h_type == "breakout":
            ax.annotate('BREAKOUT ZONE',
                        xy=(arrow_x, target_level),
                        xytext=(text_x, target_level + y_range * 0.08),
                        arrowprops=dict(facecolor='#00FFA3', edgecolor='#00FFA3', shrink=0.08, width=2, headwidth=7),
                        fontsize=13, fontweight='bold', color='#00FFA3',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#0B0E14', edgecolor='#00FFA3', alpha=0.85))
        elif h_type == "support":
            ax.annotate('SUPPORT LEVEL',
                        xy=(arrow_x, target_level),
                        xytext=(text_x, target_level - y_range * 0.08),
                        arrowprops=dict(facecolor='#00E5FF', edgecolor='#00E5FF', shrink=0.08, width=2, headwidth=7),
                        fontsize=13, fontweight='bold', color='#00E5FF',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#0B0E14', edgecolor='#00E5FF', alpha=0.85))

        fig.text(0.08, 0.84, f"{ticker} - {stock_info.company_spoken_name}", fontsize=22, fontweight='bold', color='#FFFFFF')
        fig.text(0.08, 0.80, f"Verified Price: ${verified_price:.2f} | Timeframe: {stock_info.timeframe}", fontsize=13, color='#8B949E')

        ax.grid(True, linestyle='--', alpha=0.15, color='#8B949E')
        ax.tick_params(colors='#FFFFFF', labelsize=11)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#30363D')
            spine.set_linewidth(2)

        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3]

    clip = VideoClip(make_frame, duration=duration).with_audio(voice_clip)
    return clip

# ---------------------------------------------------------
# Subtitles & Final Pipeline Assembly
# ---------------------------------------------------------
def get_system_font_path() -> str:
    possible_fonts = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/Library/Fonts/Arial.ttf"
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return font_path
    return "DejaVu-Sans"


def add_subtitles(video_clip: VideoClip, text: str) -> CompositeVideoClip:
    clean_text = re.sub(r'[^\x00-\x7F]+', '', text).strip()

    # Split text into sentences for synchronized line-by-line display
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if s.strip()]
    if not sentences:
        sentences = [clean_text]

    words_per_sentence = [max(1, len(s.split())) for s in sentences]
    total_words = sum(words_per_sentence)
    total_duration = video_clip.duration

    box_width = int(video_clip.w * 0.85)
    box_height = 140
    font_path = get_system_font_path()

    subtitle_clips = [video_clip]
    start_time = 0.0

    for idx, (sent, w_count) in enumerate(zip(sentences, words_per_sentence)):
        sent_duration = (w_count / total_words) * total_duration
        if idx == len(sentences) - 1:
            sent_duration = max(sent_duration, total_duration - start_time)

        try:
            txt = TextClip(
                text=sent,
                font=font_path,
                font_size=34,
                color='white',
                method='caption',
                size=(box_width, box_height)
            ).with_start(start_time).with_duration(sent_duration).with_position(('center', 0.9), relative=True)

            subtitle_clips.append(txt)
        except Exception as e:
            logger.warning(f"Failed to render subtitle for line '{sent}': {e}")

        start_time += sent_duration

    return CompositeVideoClip(subtitle_clips)

if __name__ == "__main__":
    #print(fetch_top_movers_and_watchlist())
    run_perfect_pipeline()