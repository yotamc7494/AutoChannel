import math
import os
import time
import tempfile
import random
import json
import urllib.request
import shutil
import subprocess
from functools import lru_cache
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.interpolate import PchipInterpolator
import requests
from PIL import Image, ImageDraw, ImageEnhance

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from google import genai
from google.genai import types
from dotenv import load_dotenv
import yfinance as yf

from uploader import upload_video

SECTORS = {
    "Semiconductors & Chips": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "MU", "AMAT", "ASML"],
    "Big Tech & Cloud": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRM", "ADBE"],
    "Electric Vehicles & Auto": ["TSLA", "RIVN", "LCID", "F", "GM", "TM"],
    "Fintech & Banking": ["V", "MA", "PYPL", "SQ", "JPM", "BAC", "GS", "MS", "COIN"],
    "Streaming & Media": ["NFLX", "DIS", "WBD", "SPOT", "ROKU"],
    "Retail & E-commerce": ["WMT", "TGT", "COST", "BABA", "SHOP", "MELI"],
    "Pharma & Healthcare": ["LLY", "NVO", "PFE", "JNJ", "UNH", "ABBV"],
    "Energy & Defense": ["XOM", "CVX", "LMT", "RTX", "NOC"]
}

# --- פלטות צבעים כהות ויוקרתיות לרקע ---
BACKGROUND_PALETTES = [
    {
        "bg1": (24, 32, 47, 255),
        "bg2": (15, 19, 28, 255),
        "line": (42, 52, 71, 255),
        "card_bg": "#141A26",
        "grid": "#2A3447",
    },
    {
        "bg1": (32, 22, 48, 255),
        "bg2": (18, 12, 28, 255),
        "line": (60, 42, 85, 255),
        "card_bg": "#1B1226",
        "grid": "#3B2854",
    },
    {
        "bg1": (18, 38, 32, 255),
        "bg2": (10, 22, 18, 255),
        "line": (35, 70, 58, 255),
        "card_bg": "#0F211C",
        "grid": "#22453A",
    },
    {
        "bg1": (42, 20, 28, 255),
        "bg2": (25, 11, 16, 255),
        "line": (75, 35, 50, 255),
        "card_bg": "#210F16",
        "grid": "#4A2231",
    },
    {
        "bg1": (35, 35, 38, 255),
        "bg2": (18, 18, 20, 255),
        "line": (60, 60, 65, 255),
        "card_bg": "#1C1C1E",
        "grid": "#3A3A3C",
    },
    {
        "bg1": (20, 28, 55, 255),
        "bg2": (12, 16, 32, 255),
        "line": (45, 60, 105, 255),
        "card_bg": "#11182E",
        "grid": "#2C3A66",
    },
]

load_dotenv()

LOGOS_DIR = "assets/logos"
TRADING_FLOOR_DIR = "assets/trading_floor_videos"

plt.rcParams['font.family'] = ['Segoe UI Emoji', 'DejaVu Sans']

try:
    cache_dir = os.path.join(tempfile.gettempdir(), "yf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    yf.set_tz_cache_location(cache_dir)
except Exception:
    pass


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.")


def get_ticker_logo(ticker):
    """מורידה לוגו שקוף אוטומטית מ-CDN עם אימות תקינות הקובץ."""
    os.makedirs(LOGOS_DIR, exist_ok=True)
    logo_path = os.path.join(LOGOS_DIR, f"{ticker}.png")

    if os.path.exists(logo_path):
        try:
            with Image.open(logo_path) as img:
                img.verify()
            return logo_path
        except Exception:
            try:
                os.remove(logo_path)
            except Exception:
                pass

    urls = [
        f"https://assets.parqet.com/logos/symbol/{ticker}",
        f"https://financialmodelingprep.com/image-stock/{ticker}.png",
        f"https://logo.clearbit.com/{ticker.lower()}.com",
    ]

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = resp.read()
                if len(data) < 500:
                    continue
                img = Image.open(BytesIO(data))
                img.verify()
                img = Image.open(BytesIO(data)).convert("RGBA")
                img.save(logo_path, "PNG")
                return logo_path
        except Exception:
            continue

    return None


def prepare_image_background_and_thumbnail(ticker1, ticker2, palette, width=1080, height=1920):
    bg1 = palette["bg1"]
    bg2 = palette["bg2"]
    line_color = palette["line"]

    bg = Image.new("RGBA", (width, height), bg2)
    draw = ImageDraw.Draw(bg)

    comp_type = random.choice(["diagonal", "horizontal", "vertical"])

    if comp_type == "diagonal":
        poly_tl = [(0, 0), (width, 0), (0, height)]
        poly_br = [(width, 0), (width, height), (0, height)]
        draw.polygon(poly_tl, fill=bg1)
        draw.polygon(poly_br, fill=bg2)
        draw.line([(width, 0), (0, height)], fill=line_color, width=12)
    elif comp_type == "horizontal":
        draw.rectangle([(0, 0), (width, height // 2)], fill=bg1)
        draw.rectangle([(0, height // 2), (width, height)], fill=bg2)
        draw.line([(0, height // 2), (width, height // 2)], fill=line_color, width=12)
    else:  # vertical
        draw.rectangle([(0, 0), (width // 2, height)], fill=bg1)
        draw.rectangle([(width // 2, 0), (width, height)], fill=bg2)
        draw.line([(width // 2, 0), (width // 2, height)], fill=line_color, width=12)

    logo1_path = get_ticker_logo(ticker1)
    logo2_path = get_ticker_logo(ticker2)
    logo_size = 380

    def paste_logo(bg_img, logo_path, center_pos):
        if logo_path and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path).convert("RGBA")
                img.thumbnail((logo_size, logo_size), Image.Resampling.LANCZOS)
                pos = (int(center_pos[0] - img.width / 2), int(center_pos[1] - img.height / 2))
                bg_img.paste(img, pos, img)
            except Exception as e:
                print(f"⚠️ Failed to paste logo {logo_path}: {e}")

    if comp_type == "diagonal":
        paste_logo(bg, logo2_path, (width * 0.25, height * 0.30))
        paste_logo(bg, logo1_path, (width * 0.75, height * 0.70))
    elif comp_type == "horizontal":
        paste_logo(bg, logo2_path, (width * 0.5, height * 0.25))
        paste_logo(bg, logo1_path, (width * 0.5, height * 0.75))
    else:  # vertical
        paste_logo(bg, logo2_path, (width * 0.25, height * 0.5))
        paste_logo(bg, logo1_path, (width * 0.75, height * 0.5))

    temp_dir = tempfile.gettempdir()
    unique_id = f"{ticker1}_vs_{ticker2}_{int(time.time())}"
    thumb_path = os.path.join(temp_dir, f"thumb_{unique_id}.png")
    dark_bg_path = os.path.join(temp_dir, f"dark_bg_{unique_id}.png")

    bg.convert("RGB").save(thumb_path, quality=95)

    enhancer = ImageEnhance.Brightness(bg)
    dark_bg = enhancer.enhance(0.35)
    dark_bg.convert("RGB").save(dark_bg_path)

    return thumb_path, dark_bg_path


def fetch_yahoo_chart_direct(ticker, period="max", interval="1wk"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={period}&interval={interval}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
    }
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = data["chart"]["result"][0]
                timestamps = result["timestamp"]
                closes = result["indicators"]["quote"][0]["close"]

                df = pd.DataFrame(
                    {"Close": closes},
                    index=pd.to_datetime(timestamps, unit="s", utc=True),
                )
                df.index = df.index.tz_localize(None)
                df = df.dropna()
                if not df.empty:
                    return df["Close"]
            elif response.status_code == 429:
                time.sleep(3)
        except Exception:
            time.sleep(2)
    return pd.Series(dtype=float)


def get_comparison_data(ticker1, ticker2, initial_investment=100):
    print(f"📥 Fetching historical data for {ticker1} and {ticker2}...", flush=True)

    h1 = fetch_yahoo_chart_direct(ticker1, period="max", interval="1wk")
    time.sleep(1)
    h2 = fetch_yahoo_chart_direct(ticker2, period="max", interval="1wk")

    if h1.empty or h2.empty:
        return None, None, None

    start_date = max(h1.index[0], h2.index[0])
    h1_aligned = h1[h1.index >= start_date]
    h2_aligned = h2[h2.index >= start_date]

    common_index = h1_aligned.index.intersection(h2_aligned.index)
    h1_final = h1_aligned.loc[common_index]
    h2_final = h2_aligned.loc[common_index]

    if h1_final.empty or h2_final.empty:
        return None, None, None

    v1 = (h1_final / h1_final.iloc[0]) * initial_investment
    v2 = (h2_final / h2_final.iloc[0]) * initial_investment

    return v1, v2, start_date.year


def generate_fomo_metadata_and_header_with_ai(ticker1, ticker2, sector_name, investment, start_year):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise ValueError("🚨 GEMINI_API_KEY is missing!")

    client = genai.Client(api_key=api_key)

    CREATIVE_ANGLES = [
        f"REGRET & FOMO: Focus on the pain of choosing the wrong stock in {start_year}",
        f"BRUTAL WAR: Frame it as an aggressive battle between {sector_name} giants",
        f"WEALTH / ROI: Focus purely on the massive stack of money made from ${investment}",
        f"MYSTERY / SHOCK: Frame it as a surprising twist that nobody expected",
        f"INVESTOR COMPARISON: Ask a direct, provocative question to investors",
    ]
    chosen_angle = random.choice(CREATIVE_ANGLES)

    EMOJI_SETS = ["⚔️, 💥, 🥊, 💣", "👑, 🏆, 🥇, 💰", "🤯, 🚨, 🔥, 😱", "📊, 📉, 📈, 💵"]
    chosen_emojis = random.choice(EMOJI_SETS)

    prompt = f"""
        You are a top-tier viral financial content creator for TikTok, Shorts, and Reels.
        Create the ON-SCREEN HEADER and YouTube metadata for an animated stock race video.

        Video Details:
        - Stocks: {ticker1} vs {ticker2}
        - Sector: {sector_name}
        - Investment: ${investment} starting in {start_year}

        CREATIVE ANGLE FOR THIS VIDEO:
        {chosen_angle}

        STRICT RULES FOR DIVERSITY:
        1. "video_header": 3 to 5 words MAX in ALL CAPS for the overlay on top of the video.
        2. "youtube_title": NO CLICHÉS! DO NOT use boring formats. Use diverse hooks. Emojis: {chosen_emojis}
        3. "description": Punchy summary ending with an engaging question.
        4. "tags": 5-7 comma-separated tags.

        STRICT OUTPUT FORMAT:
        Return ONLY a valid JSON object:
        {{
          "video_header": "SHORT ON-SCREEN HEADER",
          "youtube_title": "Creative title",
          "description": "Engaging description",
          "tags": "tag1,tag2,tag3"
        }}
    """
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=1.0)
    )
    return json.loads(response.text)


# =====================================================================
# FFmpeg Render Engine (Replaces moviepy)
# =====================================================================
def get_bg_frames(video_path, width, height, fps, total_frames):
    """שואב פריימים גולמיים מווידאו הרקע באמצעות FFmpeg כדי לשלב אותם בזיכרון."""
    cmd = [
        ffmpeg_exe(), '-loglevel', 'error',
        '-stream_loop', '-1', '-i', video_path,
        # חותך למרכז ומכהה משמעותית (מדמה את vfx.colorx(0.25))
        '-vf', f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
               f"colorlevels=romax=0.25:gomax=0.25:bomax=0.25",
        '-r', str(fps),
        '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3 * 5)
    frame_size = width * height * 3

    try:
        for _ in range(total_frames):
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)
    finally:
        proc.kill()


def render_chart_video_to_disk(
        v1, v2, ticker1, ticker2, start_year, duration, color1, color2,
        header_text, palette, dark_bg_path, bg_video_path, show_fill,
        width, height, fps, output_path
):
    """מצייר את הגרף ב-Matplotlib ומזרים את הפיקסלים ישירות ל-FFmpeg כדי לקודד קובץ וידאו חסר קול."""
    total_frames = int(duration * fps)
    freeze_frames = int(1.0 * fps)  # מסגרת קפואה בסוף של שנייה אחת

    n_original = len(v1)
    n_dense = 1000
    x_orig = np.arange(n_original)
    x_dense = np.linspace(0, n_original - 1, n_dense)

    y1_dense = PchipInterpolator(x_orig, v1.values)(x_dense)
    y2_dense = PchipInterpolator(x_orig, v2.values)(x_dense)

    # הכנת מטמון לרקע (וידאו או תמונה)
    bg_generator = None
    static_bg = None
    if bg_video_path and os.path.exists(bg_video_path):
        bg_generator = get_bg_frames(bg_video_path, width, height, fps, total_frames + freeze_frames)
    elif dark_bg_path and os.path.exists(dark_bg_path):
        bg_pil = Image.open(dark_bg_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        static_bg = np.array(bg_pil, dtype=np.float32)
    else:
        static_bg = np.full((height, width, 3), palette["bg2"][:3], dtype=np.float32)

    # הכנת משטח הציור של Matplotlib (בלי חלון אמיתי)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)

    fig.text(0.5, 0.90, header_text, color="white", fontsize=26, fontweight="bold",
             ha="center", va="top", bbox=dict(boxstyle="square,pad=0.5", facecolor=palette["card_bg"],
                                              edgecolor=palette["grid"], alpha=0.9))
    date_text_obj = fig.text(0.5, 0.82, "", color="white", fontsize=24, fontweight="bold", ha="center", va="top")

    try:
        dates = v1.index.strftime("%b %Y")
    except AttributeError:
        dates = [str(start_year)] * n_original

    # פתיחת תהליך FFmpeg כדי לקלוט פריימים וליצור קובץ MP4
    out_cmd = [
        ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-', '-an',  # אין שמע עדיין
        '-c:v', 'libx264', '-preset', 'ultrafast', '-b:v', '8000k', '-pix_fmt', 'yuv420p',
        output_path
    ]
    out_proc = subprocess.Popen(out_cmd, stdin=subprocess.PIPE)

    last_frame = None

    # לולאת רינדור פריימים
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        ax.clear()
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.grid(True, color=palette["grid"], linestyle="--", linewidth=1, alpha=0.5)

        h_p = 1 - math.pow(1 - min(t / duration, 1.0), 3)
        s_p = 1 - math.pow(1 - min(t / duration, 1.0), 2)
        progress = 0.8*s_p + 0.2*h_p
        idx = max(1, int(progress * (n_dense - 1)))
        idx_orig = min(max(0, int(progress * (n_original - 1))), n_original - 1)

        date_text_obj.set_text(dates[idx_orig])

        sub_x, sub_y1, sub_y2 = x_dense[:idx + 1], y1_dense[:idx + 1], y2_dense[:idx + 1]

        ax.plot(sub_x, sub_y1, color=color1, linewidth=6, label=ticker1)
        ax.plot(sub_x, sub_y2, color=color2, linewidth=6, label=ticker2)

        if show_fill:
            ax.fill_between(sub_x, sub_y1, min(sub_y1), color=color1, alpha=0.15)
            ax.fill_between(sub_x, sub_y2, min(sub_y2), color=color2, alpha=0.15)

        ax.plot(sub_x[-1], sub_y1[-1], marker="o", markersize=12, color=color1)
        ax.plot(sub_x[-1], sub_y2[-1], marker="o", markersize=12, color=color2)

        v1_curr, v2_curr = sub_y1[-1], sub_y2[-1]
        crown_icon = "\U0001F451"

        if v1_curr >= v2_curr:
            l1_str, l2_str = f"{ticker1} {crown_icon}\n${v1_curr:,.1f}", f"{ticker2}\n${v2_curr:,.1f}"
        else:
            l1_str, l2_str = f"{ticker1}\n${v1_curr:,.1f}", f"{ticker2} {crown_icon}\n${v2_curr:,.1f}"

        current_y_max = max(np.max(sub_y1), np.max(sub_y2))
        current_y_min = min(np.min(sub_y1), np.min(sub_y2))
        ax.set_ylim(current_y_min * 0.85, current_y_max * 1.35)

        x_max = max(10, sub_x[-1])
        ax.set_xlim(-(x_max * 0.05), x_max + (x_max * 0.15))

        ax.text(sub_x[-1], v1_curr + (current_y_max * 0.03), l1_str, color=color1, fontsize=24, fontweight="bold",
                ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.3", facecolor=palette["card_bg"],
                                                    edgecolor=color1, alpha=0.85))
        ax.text(sub_x[-1], v2_curr + (current_y_max * 0.03), l2_str, color=color2, fontsize=24, fontweight="bold",
                ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.3", facecolor=palette["card_bg"],
                                                    edgecolor=color2, alpha=0.85))

        ax.tick_params(axis="both", colors="white", labelsize=18)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # המרת הגרף למערך נתונים RGB ו-Alpha
        canvas.draw()
        chart_rgba = np.asarray(canvas.buffer_rgba())
        alpha = (chart_rgba[:, :, 3:4].astype(np.float32)) / 255.0
        chart_rgb = chart_rgba[:, :, :3].astype(np.float32)

        # טעינת פריים רקע
        current_bg = static_bg
        if bg_generator is not None:
            try:
                current_bg = next(bg_generator)
            except StopIteration:
                pass

        # ערבוב השכבות ושליחה ל-FFmpeg
        blended = (chart_rgb * alpha + current_bg * (1.0 - alpha)).astype(np.uint8)
        last_frame = blended
        out_proc.stdin.write(blended.tobytes())

    # פריים קפוא (השהיה בסוף)
    if last_frame is not None:
        for _ in range(freeze_frames):
            if bg_generator is not None:
                try:
                    current_bg = next(bg_generator)
                    blended = (chart_rgb * alpha + current_bg * (1.0 - alpha)).astype(np.uint8)
                    out_proc.stdin.write(blended.tobytes())
                except StopIteration:
                    out_proc.stdin.write(last_frame.tobytes())
            else:
                out_proc.stdin.write(last_frame.tobytes())

    out_proc.stdin.close()
    out_proc.wait()
    plt.close(fig)


def mix_audio_and_video(silent_video_path, music_path, output_path, total_duration):
    """מוסיפה מוזיקת רקע לווידאו המרונדר עם Fade Out בסיום (Stream Copy מהיר מאוד)."""
    if not music_path or not os.path.exists(music_path):
        shutil.copy(silent_video_path, output_path)
        return

    fade_start = max(0.0, total_duration - 2.0)
    cmd = [
        ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-i', silent_video_path,
        '-stream_loop', '-1', '-i', music_path,
        # הגדרת עוצמה ודהייה בסוף
        '-filter_complex', f"[1:a]volume=0.15,afade=t=out:st={fade_start}:d=2[aout]",
        '-map', '0:v:0', '-map', '[aout]',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
        '-t', str(total_duration),
        output_path
    ]
    subprocess.run(cmd, check=True)


def create_comparison_fomo_video(ticker1, ticker2, sector_name, music_path, investment,
                                 output_filename="fomo_comparison.mp4"):
    duration = random.randint(15, 20)
    fps = 15
    width, height = 1080, 1920

    v1, v2, start_year = get_comparison_data(ticker1, ticker2, investment)
    if v1 is None or v2 is None or len(v1) < 2 or len(v2) < 2:
        print(f"❌ Insufficient price data for {ticker1} vs {ticker2}. Skipping...", flush=True)
        return None, None

    selected_palette = random.choice(BACKGROUND_PALETTES)
    print(f"🎬 Generating {duration}s Video For {ticker1} vs {ticker2} ({sector_name})", flush=True)

    thumb_path, dark_image_bg_path = prepare_image_background_and_thumbnail(
        ticker1, ticker2, palette=selected_palette, width=width, height=height
    )

    palette = ["#00E676", "#FF1744", "#29B6F6", "#FFA000", "#E040FB", "#1DE9B6", "#FFEA00"]
    c1, c2 = random.sample(palette, 2)

    ai_data = generate_fomo_metadata_and_header_with_ai(ticker1, ticker2, sector_name, investment, start_year)
    show_fill = random.random() > 0.30

    background_type = random.choice(["video", "logos"])
    bg_video_path = None
    dark_bg_for_chart = None

    if background_type == "video":
        if os.path.exists(TRADING_FLOOR_DIR) and (
        video_files := [f for f in os.listdir(TRADING_FLOOR_DIR) if f.endswith(".mp4")]):
            bg_video_path = os.path.join(TRADING_FLOOR_DIR, random.choice(video_files))
            print("🟢 Using darkened trading floor video background.", flush=True)
        else:
            background_type = "logos"

    if background_type == "logos":
        print("🟢 Using static darkened logo composition background.", flush=True)
        dark_bg_for_chart = dark_image_bg_path

    # רינדור הווידאו האילם ישירות לכונן באמצעות FFmpeg
    temp_silent_video = os.path.join(tempfile.gettempdir(), f"silent_{output_filename}")

    render_chart_video_to_disk(
        v1, v2, ticker1, ticker2, start_year, duration, c1, c2,
        ai_data["video_header"], selected_palette, dark_bg_for_chart, bg_video_path,
        show_fill, width, height, fps, temp_silent_video
    )

    print("🎵 Mixing background music...", flush=True)
    total_duration = duration + 1.0  # + שנייה אחת של פריים קפוא
    mix_audio_and_video(temp_silent_video, music_path, output_filename, total_duration)

    # מחיקת הקובץ הזמני (הסרטון האילם)
    if os.path.exists(temp_silent_video):
        os.remove(temp_silent_video)

    print(f"✅ Done! Saved to {output_filename}", flush=True)
    return ai_data, thumb_path


def run_generator(test_mode=False):
    MUSIC_PATH = f"assets/suspense_music/{random.choice(range(1, 16))}.mp3"
    if not os.path.exists(MUSIC_PATH):
        MUSIC_PATH = ""

    sector_name, tickers = random.choice(list(SECTORS.items()))

    investment = random.choice([100, 200, 500, 1000])
    upload_data, thumb_path = None, None
    attempts = 0

    while upload_data is None and thumb_path is None:
        ticker1, ticker2 = random.sample(tickers, 2)
        output_filename = f"{ticker1}_vs_{ticker2}.mp4"
        attempts += 1
        upload_data, thumb_path = create_comparison_fomo_video(
            ticker1, ticker2, sector_name, MUSIC_PATH, investment, output_filename
        )
        if attempts > 10:
            break

    if test_mode:
        return output_filename
    else:
        if upload_data:
            upload_video(
                output_filename,
                upload_data["youtube_title"],
                upload_data["description"],
                upload_data["tags"],
                thumbnail_path=thumb_path,
            )
            return output_filename


if __name__ == "__main__":
    run_generator(test_mode=True)