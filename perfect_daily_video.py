import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from scipy.interpolate import make_interp_spline

from thumbnail import generate_daily_thumbnail
from uploader import upload_video

load_dotenv()
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PerfectDailyVideo")


# =====================================================================
# 1. CONFIG & SCHEMAS
# =====================================================================
# --- LLM ---
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]  # tried in order

# --- Video ---
VIDEO_W, VIDEO_H = 1920, 1080          # drop to 1280x720 for ~2x faster renders
DPI = VIDEO_W / 19.2                   # keeps every font/layout proportion identical at any size
OUTPUT_FPS = 30
RENDER_FPS = 15                        # heavy matplotlib layer is redrawn at this rate, frames are repeated
VIDEO_BITRATE = "8000k"
X264_PRESET = "ultrafast"
OUTRO_SECONDS = 5.0
BG_MUSIC_VOLUME = 0.12
BG_MUSIC_FADE_OUT = 2.0

# --- Parallelism ---
IO_WORKERS = 8                         # threads for network calls
PARALLEL_RENDER = True                 # render each scene in its own process
MAX_RENDER_WORKERS = os.cpu_count() or 1

# --- TTS ---
PIPER_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
PIPER_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
PIPER_MODEL_FILE = "en_US-ryan-high.onnx"
PIPER_CONFIG_FILE = "en_US-ryan-high.onnx.json"

# --- Data ---
RSS_QUERIES = [
    "SPY QQQ stock market rally crash Fed rates",
    "S&P 500 Nasdaq stock market today close",
    "Nvidia AMD TSMC AI chip stock news",
    "Tesla Apple Microsoft Amazon Meta stock news",
    "top stock gainers losers earnings surprise market news",
    "Bitcoin crypto market price surge MicroStrategy Coinbase"
]
WATCHLIST_CANDIDATES = [
    # Mega-Cap Tech (ענקיות הטכנולוגיה)
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",

    # Semiconductors & AI (שבבים ובינה מלאכותית)
    "AMD", "AVGO", "TSM", "ARM", "MU", "SMCI",

    # High-Beta & Retail Favorites (מניות מומנטום ומסחר ער)
    "PLTR", "COIN", "MSTR", "MARA", "HOOD", "NFLX", "ORCL",

    # Market Leaders & Heavyweights (ענקיות סקטוריאליות)
    "LLY", "CRM", "DIS", "JPM", "BABA"
]
TTS_SYMBOL_NAMES = {
    r"\bNVDA\b": "Nvidia", r"\bTSLA\b": "Tesla", r"\bAAPL\b": "Apple", r"\bAMZN\b": "Amazon",
    r"\bMSFT\b": "Microsoft", r"\bGOOGL\b": "Google", r"\bMETA\b": "Meta", r"\bBTC\b": "Bitcoin",
    r"\bSPY\b": "S and P 500", r"\bQQQ\b": "Nasdaq",
}

# --- Palette ---
C_BG, C_PANEL, C_BORDER = "#0B0E14", "#161B22", "#30363D"
C_WHITE, C_MUTED = "#FFFFFF", "#8B949E"
C_GREEN, C_RED, C_CYAN, C_GOLD, C_AMBER = "#00FFA3", "#FF3366", "#00E5FF", "#FFD700", "#FFB800"

AVATAR_OPEN_PATH = "assets/avatar/open.png"
AVATAR_CLOSED_PATH = "assets/avatar/closed.png"


class StockAnalysis(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    company_spoken_name: str = Field(description="Full spoken company name for TTS, e.g. Nvidia")
    script: str = Field(description="Spoken narrative for TTS (~40 words).Starting with technical data and mostly predictions and level targets")
    timeframe: str = Field(description="Chart timeframe: '1d', '1mo', '3mo', '6mo', or '1y'")
    interval: str = Field(description="Chart interval: '5m' for 1d, '1d' for others")
    key_levels: List[float] = Field(default=[], description="Important price levels to highlight on chart")
    highlight_type: str = Field(default="none", description="Type of overlay: 'breakout', 'support', 'resistance', or 'none'")


class FullVideoScriptSchema(BaseModel):
    intro_script: str = Field(description="STRICTLY 25 to 40 words English intro script with high CTA.")
    news_ticker_text: str = Field(description="Single line ticker with top 3-5 market headlines + short 4-word summary each, separated by '  •  ' e.g. 'FED CUTS RATES: Policy shifted dovish  •  NVDA SURGES: AI chip demand peaks  •  BTC BREAKS $70K: Institutional inflow high'")
    analyzed_stocks: List[StockAnalysis]
    youtube_title: str
    description: str
    tags: str


# =====================================================================
# 2. GENERIC HELPERS
# =====================================================================
def parallel_map(fn: Callable, items: Sequence, max_workers: int = IO_WORKERS) -> list:
    """ThreadPool map that preserves order (use for network-bound work)."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(fn, items))


def run_async(coro):
    """Run a coroutine in a fresh thread/event loop (safe inside Streamlit)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def format_seconds_to_timestamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


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


AUDIO_ANALYSIS_SR = 22050


def decode_audio(path: str, sr: int = AUDIO_ANALYSIS_SR) -> np.ndarray:
    """Decode any audio file to mono float32 samples via ffmpeg."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='ignore')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def audio_duration(path: str) -> float:
    return len(decode_audio(path)) / AUDIO_ANALYSIS_SR


def get_audio_peaks(audio_path: str, fps: int = OUTPUT_FPS) -> np.ndarray:
    """Normalised RMS volume per video frame (vectorised; used for lip sync)."""
    samples = decode_audio(audio_path)
    spf = int(AUDIO_ANALYSIS_SR / fps)
    n_frames = max(1, int(len(samples) / AUDIO_ANALYSIS_SR * fps))
    needed = n_frames * spf
    if len(samples) < needed:
        samples = np.pad(samples, (0, needed - len(samples)))
    chunks = samples[:needed].reshape(n_frames, spf).astype(np.float64)
    vol = np.sqrt((chunks ** 2).mean(axis=1))
    peak = vol.max()
    return vol / peak if peak > 0 else vol


# =====================================================================
# 3. LLM
# =====================================================================
@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing!")
    return genai.Client(api_key=api_key)


def ask_llm(prompt, config=None):
    """Try each model in GEMINI_MODELS in order; raise the last error if all fail."""
    client = get_gemini_client()
    kwargs = {"config": config} if config else {}
    last_exc = None
    for model in GEMINI_MODELS:
        try:
            return client.models.generate_content(model=model, contents=prompt, **kwargs)
        except Exception as e:
            last_exc = e
            logger.warning(f"⚠️ Failed to generate content with {model} ({e}).")
    raise last_exc


# =====================================================================
# 4. MARKET DATA
# =====================================================================
def _fetch_rss_headlines(query: str) -> List[str]:
    try:
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+when:1d&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            root = ET.fromstring(resp.read())
        headlines = []
        for item in root.findall(".//item")[:3]:
            title = item.findtext("title") or ""
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]  # strip the publisher name
            headlines.append(title)
        return headlines
    except Exception as e:
        logger.warning(f"Failed pulling RSS for query '{query}': {e}")
        return []


def fetch_daily_market_news_from_rss() -> str:
    logger.info("📡 Fetching latest 24h market news via Google News RSS...")
    all_headlines: List[str] = []
    for headlines in parallel_map(_fetch_rss_headlines, RSS_QUERIES):  # queries run concurrently
        for h in headlines:
            if h not in all_headlines:
                all_headlines.append(h)
    logger.info(f"✅ Extracted {len(all_headlines)} clean headlines from RSS.")
    return "Latest 24h Market Headlines & Catalysts:\n" + "\n".join(f"- {h}" for h in all_headlines)


def _fetch_close_change(ticker: str, period: str, min_rows: int = 1) -> Optional[dict]:
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty or len(hist) < min_rows:
            return None
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last
        return {"ticker": ticker, "price": last, "pct": (last - prev) / prev * 100.0}
    except Exception as e:
        logger.warning(f"Failed to fetch market data for {ticker}: {e}")
        return None


def extract_and_verify_ticker_data(market_summary: str) -> dict:
    logger.info("🔍 Extracting mentioned tickers and fetching ground-truth data...")
    prompt = f"Extract all stock tickers mentioned in this market summary as a JSON array of strings (e.g. ['NVDA', 'TSLA', 'AAPL']): {market_summary}"
    res = ask_llm(prompt)
    try:
        raw = json.loads(re.search(r"\[.*\]", res.text, re.DOTALL).group(0))
        tickers = {str(t).strip().lstrip("$").upper() for t in raw if str(t).strip()}
    except Exception:
        tickers = {"NVDA", "TSLA", "AAPL", "MSFT"}

    rows = parallel_map(lambda t: _fetch_close_change(t, "5d"), sorted(tickers))
    return {
        r["ticker"]: {"price": round(r["price"], 2), "pct_change": round(r["pct"], 2)}
        for r in rows if r
    }


def fetch_top_movers_and_watchlist(max_items: int = 7) -> List[dict]:
    """Candidate stocks sorted by the size of today's move (largest first)."""
    logger.info("📈 Fetching watchlist and top movers data...")
    rows = [r for r in parallel_map(lambda t: _fetch_close_change(t, "2d", min_rows=2), WATCHLIST_CANDIDATES) if r]
    for r in rows:
        r["abs_pct"] = abs(r["pct"])
    rows.sort(key=lambda r: r["abs_pct"], reverse=True)
    return rows[:max_items]


def _fetch_intraday(symbol: str):
    try:
        return yf.Ticker(symbol).history(period="1d", interval="5m", prepost=True)
    except Exception as e:
        logger.warning(f"Failed intraday fetch for {symbol}: {e}")
        return None


def _process_series(df, target_len: int = 300) -> Tuple[np.ndarray, float]:
    """Percent-change curve smoothed to target_len points + last price."""
    if df is None or df.empty or "Close" not in df:
        return np.zeros(target_len), 0.0
    vals = df["Close"].dropna().values.astype(float)
    n = len(vals)
    if n == 0:
        return np.zeros(target_len), 0.0
    first = vals[0] if vals[0] != 0 else 1.0
    pct = (vals - first) / first * 100.0
    x_smooth = np.linspace(0, 1, target_len)
    x_raw = np.linspace(0, 1, n)
    y = make_interp_spline(x_raw, pct, k=3)(x_smooth) if n >= 4 else np.interp(x_smooth, x_raw, pct)
    return y, float(vals[-1])


def fetch_intro_index_data() -> dict:
    logger.info("📊 Fetching intraday index data...")
    symbols = {"SPY": "SPY", "QQQ": "QQQ", "BTC": "BTC-USD"}
    with ThreadPoolExecutor(max_workers=4) as ex:
        watch_f = ex.submit(fetch_top_movers_and_watchlist)
        hist_f = {k: ex.submit(_fetch_intraday, s) for k, s in symbols.items()}
        hist = {k: f.result() for k, f in hist_f.items()}
        watchlist = watch_f.result()

    date_str = "TODAY"
    spy = hist["SPY"]
    if spy is not None and not spy.empty and hasattr(spy.index[-1], "strftime"):
        date_str = spy.index[-1].strftime("%b %d, %Y").upper()

    return {
        "x": np.linspace(0, 1, 300),
        "QQQ": _process_series(hist["QQQ"]),
        "SPY": _process_series(hist["SPY"]),
        "BTC": _process_series(hist["BTC"]),
        "date_str": date_str,
        "watchlist": watchlist,
    }


def fetch_stock_history(stock: StockAnalysis):
    """Price history for one stock; falls back to 1 month of daily bars. None if nothing is available."""
    attempts = [
        dict(period=stock.timeframe, interval=stock.interval, prepost=True),
        dict(period="1mo", interval="1d"),
    ]
    for kwargs in attempts:
        try:
            df = yf.Ticker(stock.ticker).history(**kwargs).dropna()
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"History fetch failed for {stock.ticker} ({kwargs}): {e}")
    return None


def build_thumbnail(market_data: dict) -> Optional[str]:
    if not callable(generate_daily_thumbnail):
        return None
    try:
        return generate_daily_thumbnail(
            sp500_val=float(market_data["SPY"][1]), sp500_pct=float(market_data["SPY"][0][-1]),
            qqq_val=float(market_data["QQQ"][1]), qqq_pct=float(market_data["QQQ"][0][-1]),
            btc_val=float(market_data["BTC"][1]), btc_pct=float(market_data["BTC"][0][-1]),
            date_str=market_data.get("date_str"),
            template_path="assets/Thumbnail_Daily.jpg",
            output_path="d_thumbnail.png",
        )
    except Exception as e:
        logger.error(f"⚠️ Thumbnail generation failed: {e}")
        return None


def fetch_market_data_and_thumbnail() -> Tuple[dict, Optional[str]]:
    """Independent of news/LLM, so the pipeline runs it in the background from second zero."""
    market_data = fetch_intro_index_data()
    return market_data, build_thumbnail(market_data)


# =====================================================================
# 5. SCRIPT GENERATION
# =====================================================================
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
    1. "intro_script": ULTRA-SHORT SCROLL-STOPPING HOOK.
    2. "news_ticker_text": Generate 3 high-impact breaking news headlines with ultra-short descriptions, separated by ' • '. Keep text uppercase and ultra punchy for top scrolling ticker.
    3. "analyzed_stocks": Max 4 key stocks with precise key levels.
    4. ABSOLUTELY NO EMOJIS or Dollar Symbols in any field.
    """
    response = ask_llm(prompt, config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FullVideoScriptSchema,
    ))
    return FullVideoScriptSchema.model_validate_json(response.text)


# =====================================================================
# 6. TEXT-TO-SPEECH  (Sequential fallback logic from the old script)
# =====================================================================
def ensure_piper_model():
    """מוריד את מודל הקול פעם אחת בלבד אם הוא לא קיים בתיקייה המקומית"""
    if not os.path.exists(PIPER_MODEL_FILE) or not os.path.exists(PIPER_CONFIG_FILE):
        logger.info("Downloading Piper voice model (one-time setup)...")
        urllib.request.urlretrieve(PIPER_MODEL_URL, PIPER_MODEL_FILE)
        urllib.request.urlretrieve(PIPER_CONFIG_URL, PIPER_CONFIG_FILE)
        logger.info("✅ Piper voice model ready!")


def clean_script_for_tts(text: str) -> str:
    """Removes non-ASCII characters/emojis & converts symbols for clear TTS."""
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.replace("$", "").replace("%", " percent")
    for pattern, replacement in TTS_SYMBOL_NAMES.items():
        text = re.sub(pattern, replacement, text)

    # ניקוי נוסף עבור Piper: הסרת ירידות שורה מיותרות
    text = text.replace('\n', ' ').strip()
    return text


def generate_voiceovers(items: List[Tuple[str, str]]) -> List[float]:
    """items = [(script_text, output_path), ...]  ->  durations in seconds (same order)."""
    logger.info(f"Generating {len(items)} TTS clips locally with Piper...")
    durations = []

    # מוודא שהמודל קיים לפני שמתחילים
    ensure_piper_model()

    for text, path in items:
        clean_text = clean_script_for_tts(text)
        path_str = str(path)
        logger.info(f"Generating TTS audio: {Path(path_str).name}")

        # הרצת פקודת Piper עם הזרקת הטקסט ישירות למנוע, ללא צורך בהתעסקות במרכאות
        cmd = ["piper", "--model", PIPER_MODEL_FILE, "--output_file", path_str]
        proc = subprocess.run(cmd, input=clean_text.encode('utf-8'), capture_output=True)

        if proc.returncode != 0:
            raise RuntimeError(f"Piper TTS failed for {path_str}: {proc.stderr.decode(errors='ignore')}")

        # חישוב האורך באמצעות הפונקציה הקיימת בקוד
        dur = audio_duration(path_str)
        durations.append(dur)
        logger.info(f"✅ Created: {Path(path_str).name} ({dur:.2f}s)")

    return durations


# =====================================================================
# 7. DRAWING PRIMITIVES
# =====================================================================
def new_figure(width_px: int = VIDEO_W, height_px: int = VIDEO_H, facecolor: str = C_BG):
    """Pyplot-free figure (no global state, safe in threads/processes).
    The +1e-3 guards against int() truncation giving a 1919/1079 px canvas."""
    fig = Figure(figsize=((width_px + 1e-3) / DPI, (height_px + 1e-3) / DPI), dpi=DPI, facecolor=facecolor)
    return fig, FigureCanvasAgg(fig)


def grab(canvas: FigureCanvasAgg, width_px: int = VIDEO_W, height_px: int = VIDEO_H) -> np.ndarray:
    """Draw the figure and return an independent RGB copy of the pixels."""
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    if img.shape[:2] != (height_px, width_px):
        img = np.asarray(Image.fromarray(img).resize((width_px, height_px)))
    return img


def style_axes(ax, grid: bool = True) -> None:
    ax.set_facecolor(C_BG)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.15, color=C_MUTED)
    ax.tick_params(colors=C_WHITE, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_BORDER)
        spine.set_linewidth(2)


class FrameCache:
    """Re-renders only when t crosses a RENDER_FPS boundary; otherwise returns the last frame."""

    def __init__(self, render_fps: int = RENDER_FPS):
        self.fps = render_fps
        self._key = None
        self._frame = None

    def get(self, t: float, render_fn: Callable[[float], np.ndarray]) -> np.ndarray:
        key = int(t * self.fps + 1e-6)
        if key != self._key:
            self._frame = render_fn(key / self.fps)
            self._key = key
        return self._frame


# ---------------------------------------------------------------------
# News ticker (one text artist; scrolling = moving its x position)
# ---------------------------------------------------------------------
class NewsTickerBar:
    def __init__(self, fig, raw_text: str, scroll_speed: float = 0.06):
        self.speed = scroll_speed
        ax = fig.add_axes([0.05, 0.91, 0.90, 0.055])
        ax.set_facecolor(C_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(C_BORDER)
            spine.set_linewidth(2)

        safe_text = raw_text.replace("$", r"\$")  # a bare "$" pair would switch matplotlib into math mode
        unit = f"BREAKING NEWS:  {safe_text}   •   "
        self._text = ax.text(0.02, 0.5, unit, fontsize=12, fontweight="bold", color=C_GREEN,
                             va="center", clip_on=True, transform=ax.transData)

        # Measure one repeat of the text so the scroll can wrap seamlessly (never runs out of text).
        renderer = fig.canvas.get_renderer()
        w1 = self._text.get_window_extent(renderer).width
        self._text.set_text(unit * 2)
        w2 = self._text.get_window_extent(renderer).width
        self.period = max((w2 - w1) / ax.bbox.width, 1e-3)  # axes-width units
        reps = int(math.ceil(1.0 / self.period)) + 2
        self._text.set_text(unit * reps)

    def update(self, global_t: float) -> None:
        self._text.set_x(0.02 - (global_t * self.speed) % self.period)


# ---------------------------------------------------------------------
# Avatar (every open/closed x bob combination is composited once, then cached)
# ---------------------------------------------------------------------
class AvatarPanel:
    def __init__(self, size: int):
        self.size = size
        self._open = self._load(AVATAR_OPEN_PATH)
        self._closed = self._load(AVATAR_CLOSED_PATH)
        bg = np.full((size, size, 3), [22, 27, 34], dtype=np.uint8)
        bg[:4, :] = bg[-4:, :] = bg[:, :4] = bg[:, -4:] = [48, 54, 61]
        self._bg = bg
        self._cache: Dict[Tuple[bool, int], np.ndarray] = {}

    def _load(self, path: str) -> np.ndarray:
        try:
            return np.array(Image.open(path).convert("RGBA").resize((self.size, self.size)))
        except Exception as e:
            logger.warning(f"Avatar image missing ({path}): {e}")
            return np.zeros((self.size, self.size, 4), dtype=np.uint8)

    def frame(self, is_open: bool, bob_px: int) -> np.ndarray:
        key = (is_open, bob_px)
        if key not in self._cache:
            sprite = self._open if is_open else self._closed
            box = self._bg.copy()
            alpha = sprite[:, :, 3:4] / 255.0          # float64 -> bit-identical to the original compositing
            rgb = sprite[:, :, :3].astype(np.float64)
            if bob_px > 0:  # character shifts up by bob_px
                a, c, dst = alpha[bob_px:], rgb[bob_px:], box[:-bob_px]
                box[:-bob_px] = (a * c + (1 - a) * dst).astype(np.uint8)
            else:
                box[:] = (alpha * rgb + (1 - alpha) * box).astype(np.uint8)
            self._cache[key] = box
        return self._cache[key]


# ---------------------------------------------------------------------
# Subtitles (pre-rendered once with Pillow, alpha-blended into the frame)
# ---------------------------------------------------------------------
def find_system_font() -> Optional[str]:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf", "/Library/Fonts/Arial.ttf",
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


@lru_cache(maxsize=32)
def load_font(size: int):
    path = find_system_font()
    try:
        if path:
            return ImageFont.truetype(path, size)
    except OSError:
        pass
    try:
        return ImageFont.load_default(size)
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_w: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if not cur or font.getlength(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


@dataclass
class SubtitleCue:
    start: float
    end: float
    x: int
    y: int
    alpha: np.ndarray  # (h, w, 1) float32, cropped to the text's bounding box


class SubtitleTrack:
    BASE_FONT_SIZE = 34   # at 1080p; scaled with the video height
    MIN_FONT_SIZE = 14

    def __init__(self, text: str, duration: float):
        clean = re.sub(r"[^\x00-\x7F]+", "", text).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()] or [clean]
        counts = [max(1, len(s.split())) for s in sentences]
        total_words = sum(counts)

        # Subtitle strip = the band below the chart (bottom 10% of the frame), horizontally centred.
        self.box_w = int(VIDEO_W * 0.85)
        self.box_x = (VIDEO_W - self.box_w) // 2
        self.box_y = int(VIDEO_H * 0.9)
        self.box_h = VIDEO_H - self.box_y

        self.cues: List[SubtitleCue] = []
        start = 0.0
        for i, (sent, n_words) in enumerate(zip(sentences, counts)):
            dur = n_words / total_words * duration
            if i == len(sentences) - 1:
                dur = max(dur, duration - start)
            cue = self._render_cue(sent, start, start + dur)
            if cue:
                self.cues.append(cue)
            start += dur

    def _render_cue(self, sentence: str, start: float, end: float) -> Optional[SubtitleCue]:
        try:
            size = max(self.MIN_FONT_SIZE, int(self.BASE_FONT_SIZE * VIDEO_H / 1080))
            while True:  # shrink the font until the wrapped text fits in the strip
                font = load_font(size)
                lines = _wrap_text(sentence, font, self.box_w)
                ascent, descent = font.getmetrics()
                line_h = int((ascent + descent) * 1.15)
                if line_h * len(lines) <= self.box_h or size <= self.MIN_FONT_SIZE:
                    break
                size -= 2

            mask = Image.new("L", (self.box_w, self.box_h), 0)
            draw = ImageDraw.Draw(mask)
            y = max(0, (self.box_h - line_h * len(lines)) // 2)
            for line in lines:
                draw.text(((self.box_w - draw.textlength(line, font=font)) / 2, y), line, font=font, fill=255)
                y += line_h
            arr = np.asarray(mask)
            rows, cols = np.where(arr > 0)
            if len(rows) == 0:
                return None
            y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            alpha = (arr[y0:y1, x0:x1].astype(np.float32) / 255.0)[:, :, None]
            return SubtitleCue(start, end, self.box_x + int(x0), self.box_y + int(y0), alpha)
        except Exception as e:
            logger.warning(f"Failed to render subtitle for line '{sentence}': {e}")
            return None

    def overlay(self, frame: np.ndarray, t: float, inplace: bool = False) -> np.ndarray:
        cue = next((c for c in self.cues if c.start <= t < c.end), None)
        if cue is None and self.cues and t >= self.cues[-1].end:
            cue = self.cues[-1]
        if cue is None:
            return frame
        if not inplace:
            frame = frame.copy()
        h, w = cue.alpha.shape[:2]
        region = frame[cue.y:cue.y + h, cue.x:cue.x + w].astype(np.float32)
        region += (255.0 - region) * cue.alpha  # white text over the frame
        frame[cue.y:cue.y + h, cue.x:cue.x + w] = (region + 0.5).astype(np.uint8)
        return frame


# ---------------------------------------------------------------------
# Static sidebar (top movers) - rendered once
# ---------------------------------------------------------------------
def render_static_sidebar_image(watchlist: List[dict], width_px: int, height_px: int) -> np.ndarray:
    fig, canvas = new_figure(width_px, height_px, facecolor=C_PANEL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "TOP MOVERS", fontsize=15, fontweight="bold", color=C_WHITE, ha="center")
    ax.plot([0.1, 0.9], [0.88, 0.88], color=C_BORDER, linewidth=1.5)

    start_y, y_step = 0.80, 0.11
    for i, item in enumerate(watchlist[:7]):
        y = start_y - i * y_step
        pct = item.get("pct", 0.0)
        ax.text(0.08, y, item.get("ticker", ""), fontsize=13, fontweight="bold", color=C_WHITE, va="center")
        ax.text(0.08, y - 0.04, f"${item.get('price', 0.0):.2f}", fontsize=10, color=C_MUTED, va="center")
        ax.text(0.92, y - 0.015, f"{'+' if pct >= 0 else ''}{pct:.2f}%", fontsize=11, fontweight="bold",
                color=C_GREEN if pct >= 0 else C_RED, ha="right", va="center")
    return grab(canvas, width_px, height_px)


# =====================================================================
# 8. SCENES   (each exposes .frame_at(t) -> RGB uint8 array of shape (H, W, 3); t is scene-local)
#    Callers must treat the returned array as read-only (it may be shared between frames).
# =====================================================================
class IntroScene:
    ASSETS = ("QQQ", "SPY", "BTC")
    COLORS = {"QQQ": C_CYAN, "SPY": C_GREEN, "BTC": C_AMBER}
    SPEAKING_THRESHOLD = 0.08  # lower = mouth moves more easily (useful range 0.02 - 0.15)

    def __init__(self, market_data: dict, audio_path: str, duration: float, global_start_time: float,
                 ticker_text: str, subtitle_text: str):
        self.duration = duration
        self.global_start = global_start_time
        self.x = market_data["x"]
        self.seg_dur = duration / 3.0
        self.cache = FrameCache()
        self.subtitles = SubtitleTrack(subtitle_text, duration)

        # --- layout (pixels), anchored to the chart axes below ---
        graph_y_top = int(VIDEO_H * (1 - 0.78))
        graph_y_bottom = int(VIDEO_H * (1 - 0.10))
        self.side_w = int(VIDEO_W * 0.18)
        self.x1 = int(VIDEO_W * 0.77)
        self.x2 = self.x1 + self.side_w
        self.y2_av = graph_y_bottom
        self.y1_av = self.y2_av - self.side_w
        self.y1_side = graph_y_top
        self.y2_side = self.y1_av - 10
        side_h = self.y2_side - self.y1_side

        self.sidebar = render_static_sidebar_image(market_data.get("watchlist", []), self.side_w, side_h)
        self.avatar = AvatarPanel(self.side_w)
        self.peaks = get_audio_peaks(audio_path)

        # --- figure: built ONCE, artists are updated per frame ---
        self.fig, self.canvas = new_figure()
        self.ticker = NewsTickerBar(self.fig, ticker_text or "MARKET NEWS TICKER • QQQ & SPY IN FOCUS")
        self.ax = self.fig.add_axes([0.05, 0.10, 0.70, 0.68])
        style_axes(self.ax)
        self.ax.set_xlim(-0.02, 1.15)
        self.fig.text(0.05, 0.84, f"MARKET OVERVIEW | {market_data['date_str']}", fontsize=20,
                      fontweight="bold", color=C_WHITE)

        self.series = {}
        for asset in self.ASSETS:
            y_vals, price = market_data[asset]
            color = self.COLORS[asset]
            line, = self.ax.plot([], [], color=color, linewidth=3.5)
            dot, = self.ax.plot([], [], marker="o", markersize=math.sqrt(120), linestyle="none",
                                color=color, markeredgewidth=0, zorder=10)
            label = self.ax.text(0, 0, f" {asset}: ${price:,.2f}", color=color, fontsize=12, fontweight="bold",
                                 va="center", visible=False,
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor=C_BG, edgecolor=color, alpha=0.85))
            self.series[asset] = dict(
                y=y_vals, line=line, dot=dot, label=label,
                run_min=np.minimum.accumulate(y_vals), run_max=np.maximum.accumulate(y_vals),
            )

    def _render_chart(self, t: float) -> np.ndarray:
        n = len(self.x)
        curr_step = max(1, int(min((t % self.seg_dur) / self.seg_dur, 1.0) * n))
        lo, hi = math.inf, -math.inf

        for idx, asset in enumerate(self.ASSETS):
            s = self.series[asset]
            start = idx * self.seg_dur
            visible = t >= start
            s["line"].set_visible(visible)
            s["dot"].set_visible(visible)
            s["label"].set_visible(visible)
            if not visible:
                continue
            p = n if t >= start + self.seg_dur else curr_step
            x_end, y_end = self.x[p - 1], s["y"][p - 1]
            s["line"].set_data(self.x[:p], s["y"][:p])
            s["dot"].set_data([x_end], [y_end])
            s["label"].set_position((x_end, y_end))
            lo, hi = min(lo, s["run_min"][p - 1]), max(hi, s["run_max"][p - 1])

        if hi >= lo:
            self.ax.set_ylim(lo - 0.5, hi + 0.8)
        self.ticker.update(self.global_start + t)

        frame = grab(self.canvas)
        frame[self.y1_side:self.y2_side, self.x1:self.x2] = self.sidebar
        return frame

    def frame_at(self, t: float) -> np.ndarray:
        frame = self.cache.get(t, self._render_chart).copy()  # we paint on it, so own a copy

        vol = self.peaks[min(int(t * OUTPUT_FPS), len(self.peaks) - 1)] if len(self.peaks) else 0.0
        speaking = vol > self.SPEAKING_THRESHOLD
        bob_px = int(min(max(vol * 50, 4), 20)) if speaking else 0
        frame[self.y1_av:self.y2_av, self.x1:self.x2] = self.avatar.frame(speaking, bob_px)

        return self.subtitles.overlay(frame, t, inplace=True)


class StockScene:
    def __init__(self, stock: StockAnalysis, df, verified_price: float, duration: float,
                 global_start_time: float, ticker_text: str, subtitle_text: str):
        self.duration = duration
        self.global_start = global_start_time
        self.cache = FrameCache()
        self.subtitles = SubtitleTrack(subtitle_text, duration)

        opens = np.asarray(df["Open"].values, dtype=float).flatten()
        closes = np.asarray(df["Close"].values, dtype=float).flatten()
        highs = np.asarray(df["High"].values, dtype=float).flatten()
        lows = np.asarray(df["Low"].values, dtype=float).flatten()
        self.n = len(df)
        x_idxs = np.arange(self.n)
        colors = np.where(closes >= opens, C_GREEN, C_RED)
        y_min, y_max = lows.min(), highs.max()
        y_range = max(1.0, y_max - y_min)

        # --- figure: candles and labels are created ONCE ---
        self.fig, self.canvas = new_figure()
        self.ticker = NewsTickerBar(self.fig, ticker_text) if ticker_text else None
        self.ax = self.fig.add_axes([0.08, 0.10, 0.88, 0.68])
        style_axes(self.ax)
        self.ax.vlines(x_idxs, lows, highs, color=colors, linewidth=1.2)
        self.ax.bar(x_idxs, closes - opens, bottom=opens, color=colors, width=0.6)
        self.ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.12)

        self.level_lines, self.level_labels = [], []
        for level in stock.key_levels:
            self.level_lines.append(self.ax.axhline(y=level, color=C_GOLD, linestyle="--", linewidth=2.0))
            self.level_labels.append(self.ax.text(
                0, level + y_range * 0.02, f"Key Level: ${level:.2f}", color=C_GOLD, fontsize=12,
                fontweight="bold", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C_BG, edgecolor=C_GOLD, alpha=0.85)))

        target = stock.key_levels[0] if stock.key_levels else closes[-1]
        h_type = stock.highlight_type.lower()
        self.annotation, self._ann_dy = None, 0.0
        if h_type in ("breakout", "support"):
            text, color, self._ann_dy = (("BREAKOUT ZONE", C_GREEN, y_range * 0.08) if h_type == "breakout"
                                         else ("SUPPORT LEVEL", C_CYAN, -y_range * 0.08))
            self._target = target
            self.annotation = self.ax.annotate(
                text, xy=(0, target), xytext=(0, target + self._ann_dy),
                arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.08, width=2, headwidth=7),
                fontsize=13, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BG, edgecolor=color, alpha=0.85))

        self.fig.text(0.08, 0.84, f"{stock.ticker} - {stock.company_spoken_name}", fontsize=22,
                      fontweight="bold", color=C_WHITE)
        self.fig.text(0.08, 0.80, f"Verified Price: ${verified_price:.2f} | Timeframe: {stock.timeframe}",
                      fontsize=13, color=C_MUTED)

    def _render_chart(self, t: float) -> np.ndarray:
        progress = min(t / self.duration, 1.0)
        start_x = int(self.n * 0.15 * progress)
        end_x = self.n - 1
        self.ax.set_xlim(start_x, end_x + max(2, int(self.n * 0.05)))

        blink = 0.45 + 0.45 * abs(math.sin(math.pi * t * 1.5))
        label_x = int(start_x + (end_x - start_x) * 0.82)
        for line, label in zip(self.level_lines, self.level_labels):
            line.set_alpha(blink)
            label.set_x(label_x)

        if self.annotation is not None:
            self.annotation.xy = (int(start_x + (end_x - start_x) * 0.70), self._target)
            self.annotation.set_position((int(start_x + (end_x - start_x) * 0.40), self._target + self._ann_dy))

        if self.ticker:
            self.ticker.update(self.global_start + t)
        return grab(self.canvas)

    def frame_at(self, t: float) -> np.ndarray:
        return self.subtitles.overlay(self.cache.get(t, self._render_chart), t)


class OutroScene:
    def __init__(self, duration: float = OUTRO_SECONDS):
        self.duration = duration
        fig, canvas = new_figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(C_BG)
        ax.axis("off")
        fig.text(0.5, 0.65, "THANKS FOR WATCHING!", fontsize=38, fontweight="bold", color=C_GREEN, ha="center")
        fig.text(0.5, 0.50, "LIKE & SUBSCRIBE FOR DAILY MARKET RECAPS", fontsize=24, fontweight="bold",
                 color=C_WHITE, ha="center")
        fig.text(0.5, 0.38, "What is your target? Drop your thoughts in the comments below!", fontsize=18,
                 color=C_MUTED, ha="center")
        self._frame = grab(canvas)

    def frame_at(self, t: float) -> np.ndarray:
        return self._frame


# =====================================================================
# 9. VIDEO WRITING  (frames are piped straight to ffmpeg; scenes render in parallel processes)
# =====================================================================
@dataclass
class SceneJob:
    kind: str                      # "intro" | "stock" | "outro"
    out_path: str
    start_time: float              # position on the global timeline (seconds)
    duration: float
    ffmpeg_threads: int = 2
    args: Dict[str, Any] = field(default_factory=dict)


def build_scene(job: SceneJob):
    a = job.args
    if job.kind == "intro":
        return IntroScene(a["market_data"], a["audio_path"], job.duration, job.start_time,
                          a["ticker_text"], a["subtitle_text"])
    if job.kind == "stock":
        return StockScene(a["stock"], a["df"], a["verified_price"], job.duration, job.start_time,
                          a["ticker_text"], a["subtitle_text"])
    if job.kind == "outro":
        return OutroScene(job.duration)
    raise ValueError(f"Unknown scene kind: {job.kind}")


def frame_range(start_time: float, duration: float) -> Tuple[int, int]:
    """Global frame indices [first, last). Using cumulative rounding avoids A/V drift between scenes."""
    return round(start_time * OUTPUT_FPS), round((start_time + duration) * OUTPUT_FPS)


def render_scene_job(job: SceneJob) -> str:
    """Top-level (picklable) worker: build one scene and encode it to a silent mp4."""
    scene = build_scene(job)
    first, last = frame_range(job.start_time, job.duration)
    n_frames = max(1, last - first)

    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(OUTPUT_FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", X264_PRESET, "-b:v", VIDEO_BITRATE,
        "-pix_fmt", "yuv420p", "-threads", str(job.ffmpeg_threads), job.out_path,
    ]
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                g = first + i
                t = max(0.0, g / OUTPUT_FPS - job.start_time)
                proc.stdin.write(np.ascontiguousarray(scene.frame_at(t)))
            proc.stdin.close()
            proc.wait()
        except BrokenPipeError:
            proc.wait()
        except Exception:
            proc.kill()
            raise
        if proc.returncode != 0:
            err_file.seek(0)
            raise RuntimeError(f"ffmpeg failed for {job.kind} scene: {err_file.read().decode(errors='ignore')}")
    return job.out_path


def render_scenes(jobs: List[SceneJob]) -> List[str]:
    workers = min(len(jobs), MAX_RENDER_WORKERS)
    for j in jobs:
        j.ffmpeg_threads = max(1, (os.cpu_count() or 1) // max(1, workers))

    if PARALLEL_RENDER and workers > 1:
        logger.info(f"🎞️ Rendering {len(jobs)} scenes in parallel on {workers} processes...")
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(render_scene_job, jobs))
        except Exception as e:
            logger.warning(f"Parallel rendering failed ({e}); falling back to sequential rendering.")

    logger.info(f"🎞️ Rendering {len(jobs)} scenes sequentially...")
    return [render_scene_job(j) for j in jobs]


# =====================================================================
# 10. FINAL ASSEMBLY  (concat with stream copy + a single audio mix; no video re-encode)
# =====================================================================
def assemble_final_video(scene_paths: List[str], voice_tracks: List[Tuple[str, float]],
                         total_frames: int, music_path: Optional[str], output_path: str, work_dir: Path) -> None:
    total = total_frames / OUTPUT_FPS

    list_file = work_dir / "scenes.txt"
    list_file.write_text("".join(
        "file '{}'\n".format(Path(p).resolve().as_posix().replace("'", "'\\''")) for p in scene_paths))

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    for path, _ in voice_tracks:
        cmd += ["-i", str(path)]

    have_music = bool(music_path) and os.path.exists(music_path)
    if have_music:
        logger.info("🎵 Mixing background music...")
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    stereo = "aformat=sample_rates=44100:channel_layouts=stereo"
    # voice is mono: duplicate it to both channels (ffmpeg's default upmix would lose 3 dB)
    voice_fmt = "aformat=sample_rates=44100:channel_layouts=mono,pan=stereo|c0=c0|c1=c0"
    graph, labels = [], []
    for i, (_, dur) in enumerate(voice_tracks, start=1):  # each clip is forced to exactly its video length
        graph.append(f"[{i}:a]{voice_fmt},apad=whole_dur={dur:.6f},atrim=duration={dur:.6f},asetpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
    graph.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1,"
                 f"apad=whole_dur={total:.6f},atrim=duration={total:.6f},asetpts=PTS-STARTPTS[voice]")

    if have_music:
        m = len(voice_tracks) + 1
        fade_start = max(0.0, total - BG_MUSIC_FADE_OUT)
        graph.append(f"[{m}:a]{stereo},atrim=duration={total:.6f},asetpts=PTS-STARTPTS,"
                     f"volume={BG_MUSIC_VOLUME},afade=t=out:st={fade_start:.6f}:d={BG_MUSIC_FADE_OUT}[bg]")
        # amix scales each input by 1/2, so volume=2 restores the original voice level
        graph.append("[voice][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=2[aout]")
    else:
        graph.append("[voice]anull[aout]")

    cmd += ["-filter_complex", ";".join(graph), "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-t", f"{total:.6f}", str(output_path)]

    logger.info(f"💾 Assembling final video: {output_path}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg assembly failed: {proc.stderr.decode(errors='ignore')}")


# =====================================================================
# 11. PIPELINE
# =====================================================================
def run_perfect_pipeline(
        output_filename: str = "perfect_daily_recap.mp4",
        bg_music_path: Optional[str] = "assets/weekly_recup_music.mp3",
        upload: bool = False,
) -> str:
    logger.info("🚀 Launching Independent Daily Video Pipeline (Web Search Mode)...")
    work_dir = Path(tempfile.mkdtemp(prefix="daily_recap_"))

    try:
        # ---- Phase A: all network work, as concurrent as the dependencies allow ----
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            market_future = pool.submit(fetch_market_data_and_thumbnail)   # + thumbnail, in the background

            market_summary = fetch_daily_market_news_from_rss()
            verified_data = extract_and_verify_ticker_data(market_summary)
            script = generate_verified_script(market_summary, verified_data)

            stocks = list(script.analyzed_stocks)
            history_futures = [pool.submit(fetch_stock_history, s) for s in stocks]   # overlap with TTS

            speech_items = [(script.intro_script, work_dir / "intro.mp3")] + [
                (s.script, work_dir / f"stock_{i}_{s.ticker}.mp3") for i, s in enumerate(stocks)]
            durations = generate_voiceovers(speech_items)

            histories = [f.result() for f in history_futures]
            market_data, thumbnail_path = market_future.result()

        # ---- Phase B: build the timeline ----
        jobs: List[SceneJob] = []
        voice_tracks: List[Tuple[str, float]] = []
        timestamps: List[str] = []
        cursor = 0.0

        intro_path, intro_dur = str(speech_items[0][1]), durations[0]
        jobs.append(SceneJob("intro", str(work_dir / "scene_intro.mp4"), cursor, intro_dur, args=dict(
            market_data=market_data, audio_path=intro_path, ticker_text=script.news_ticker_text,
            subtitle_text=script.intro_script)))
        voice_tracks.append((intro_path, intro_dur))
        timestamps.append(f"{format_seconds_to_timestamp(cursor)} Market Overview")
        cursor += intro_dur

        for i, stock in enumerate(stocks):
            df = histories[i]
            if df is None:
                logger.warning(f"No price data for {stock.ticker}; skipping its segment.")
                continue
            audio_path, dur = str(speech_items[i + 1][1]), durations[i + 1]
            jobs.append(SceneJob("stock", str(work_dir / f"scene_{i}_{stock.ticker}.mp4"), cursor, dur, args=dict(
                stock=stock, df=df, verified_price=verified_data.get(stock.ticker, {}).get("price", 0.0),
                ticker_text=script.news_ticker_text, subtitle_text=stock.script)))
            voice_tracks.append((audio_path, dur))
            timestamps.append(f"{format_seconds_to_timestamp(cursor)} {stock.ticker}")
            cursor += dur

        jobs.append(SceneJob("outro", str(work_dir / "scene_outro.mp4"), cursor, OUTRO_SECONDS))
        timestamps.append(f"{format_seconds_to_timestamp(cursor)} Outro")
        total_frames = frame_range(0.0, cursor + OUTRO_SECONDS)[1]

        final_description = (script.description + "\n\nCHAPTERS:\n" + "\n".join(timestamps)).strip()

        # ---- Phase C: render + assemble ----
        scene_paths = render_scenes(jobs)
        assemble_final_video(scene_paths, voice_tracks, total_frames, bg_music_path, output_filename, work_dir)

        # ---- Phase D: upload ----
        if upload and callable(upload_video):
            logger.info("🚀 Uploading video to YouTube...")
            upload_video(
                file_path=output_filename,
                title=script.youtube_title,
                description=final_description,
                tags=script.tags,
                thumbnail_path=thumbnail_path,
            )

        logger.info("✅ Pipeline executed successfully (Independent Mode)!")
        return output_filename

    finally:
        logger.info("🧹 Cleaning up temporary files...")
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    run_perfect_pipeline()