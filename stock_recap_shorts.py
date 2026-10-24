"""
Single-stock vertical Short (9:16, ~30 seconds)  -  same look & feel as PerfectDailyShorts

Layout (identical to the daily recap)
    * top      : scrolling news ticker bar
    * header   : TICKER / company | timeframe on the left, price + % change on the right
    * chart    : candlesticks that draw in, live price tag, pulsing previous-close line
    * middle   : sentence subtitles (white, centred)
    * bottom L : talking / bobbing avatar
    * bottom R : KEY STATS panel for the stock (market cap, day range, 52w range, volume, P/E ...)
                 a news headline card slides over it for a few seconds when the script talks about that headline

Pipeline
    1. pick the biggest mover (Redis "used today" set)   2. news (Google News RSS) + stats (yfinance)
    3. Gemini writes 5 narration "beats" (hook / catalyst / context / bulls vs bears / CTA)
    4. Kokoro TTS per beat -> exact beat timings (no Whisper needed)   5. render frames -> ffmpeg (+ music)

Dependencies: numpy, scipy, matplotlib, Pillow, yfinance, google-genai, pydantic, python-dotenv,
              soundfile, kokoro, upstash-redis (+ an ffmpeg binary). No Whisper, no worker processes.
"""
from __future__ import annotations

import gc
import html
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

try:
    import httpx  # installed with google-genai; used to recognise network errors
except ImportError:  # pragma: no cover
    httpx = None

load_dotenv()
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StockShort")


# =====================================================================
# 1. CONFIG
# =====================================================================
# --- LLM ---
GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]   # tried in order, per round
LLM_MAX_ROUNDS = 5
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# --- Video (9:16 vertical) - same numbers as the daily recap ---
VIDEO_W = int(os.environ.get("PERFECT_VIDEO_WIDTH", "720"))      # 540 / 720 / 1080
if VIDEO_W not in (540, 720, 1080):
    raise ValueError("PERFECT_VIDEO_WIDTH must be 540, 720 or 1080")
VIDEO_H = VIDEO_W * 16 // 9
UI_SCALE = VIDEO_W / 1080            # every layout pixel below is authored for 1080 wide and scaled by this
DPI = VIDEO_W / 10.8                 # keeps every font/layout proportion identical at any size
OUTPUT_FPS = 30
RENDER_FPS = 15                      # matplotlib layer is redrawn at this rate, frames are repeated
VIDEO_BITRATE = "4500k" if VIDEO_W <= 720 else "8000k"
X264_PRESET = "ultrafast"
BG_MUSIC_VOLUME = 0.12
BG_MUSIC_FADE_OUT = 2.0
SHORTS_MAX_SECONDS = 59.0

# --- Pacing ---
TARGET_SECONDS = 30                  # what the prompt aims for (~80 spoken words)
SCRIPT_WORDS_MIN, SCRIPT_WORDS_MAX = 60, 105
TTS_VOICE, TTS_SPEED, TTS_RATE = "am_liam", 1.1, 24000
BEAT_GAP = 0.18                      # silence between narration beats (s)
END_PAD = 0.9                        # video keeps going a moment after the last word (s)
REVEAL_FRACTION = 0.45               # candles finish drawing at this fraction of the video, then hold
CAPTION_MAX_WORDS = 12               # long sentences are split into several subtitle cues

# --- Headline pop-up (slides over the stats panel) ---
NEWS_CARD_LABEL = "LATEST NEWS"
POPUP_SECONDS = 3.2                  # "a quick moment"
POPUP_SLIDE = 0.30
POPUP_DELAY = 0.25                   # into the beat that talks about the headline
MAX_POPUPS = 3

# --- Data ---
TICKERS_TO_CHECK = [
    "NVDA", "AVGO", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META",
    "AMD", "PLTR", "SMCI", "ARM", "NFLX", "COIN", "MSTR", "INTC",
    "BRK-B", "JPM", "LLY", "QQQ", "SPY",
]
MARKET_REFS = ("SPY", "QQQ")         # always downloaded: used for "vs the market" context
ETF_TICKERS = {"QQQ", "SPY"}         # ETFs have no market cap; "net assets" is shown instead
COMPANY_NAMES = {
    "NVDA": "Nvidia", "AVGO": "Broadcom", "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft",
    "AMZN": "Amazon", "GOOGL": "Alphabet Google", "META": "Meta Platforms", "AMD": "Advanced Micro Devices",
    "PLTR": "Palantir", "SMCI": "Super Micro Computer", "ARM": "Arm Holdings", "NFLX": "Netflix",
    "COIN": "Coinbase", "MSTR": "Strategy MicroStrategy", "INTC": "Intel", "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase", "LLY": "Eli Lilly", "QQQ": "Invesco QQQ Nasdaq 100 ETF", "SPY": "SPDR S&P 500 ETF",
}
TTS_SYMBOL_NAMES = {
    r"\bNVDA\b": "Nvidia", r"\bTSLA\b": "Tesla", r"\bAAPL\b": "Apple", r"\bAMZN\b": "Amazon",
    r"\bMSFT\b": "Microsoft", r"\bGOOGL\b": "Google", r"\bMETA\b": "Meta", r"\bAVGO\b": "Broadcom",
    r"\bPLTR\b": "Palantir", r"\bSMCI\b": "Super Micro", r"\bMSTR\b": "MicroStrategy", r"\bCOIN\b": "Coinbase",
    r"\bNFLX\b": "Netflix", r"\bINTC\b": "Intel", r"\bJPM\b": "JP Morgan", r"\bLLY\b": "Eli Lilly",
    r"\bBRK-B\b": "Berkshire Hathaway", r"\bBTC\b": "Bitcoin", r"\bSPY\b": "S and P 500", r"\bQQQ\b": "Nasdaq",
}

# --- Palette (same as the daily recap) ---
C_BG, C_PANEL, C_BORDER = "#0B0E14", "#161B22", "#30363D"
C_WHITE, C_MUTED = "#FFFFFF", "#8B949E"
C_GREEN, C_RED, C_CYAN, C_GOLD, C_AMBER = "#00FFA3", "#FF3366", "#00E5FF", "#FFD700", "#FFB800"
C_PILL = "#FF1744"

AVATAR_OPEN_PATH = "assets/avatar/open.png"
AVATAR_CLOSED_PATH = "assets/avatar/closed.png"


def px(v: float) -> int:
    return int(round(v * UI_SCALE))


# --- Layout boxes (authored for 1080x1920, identical to the daily recap) ---
TICKER_BOX = tuple(px(v) for v in (40, 100, 1000, 65))     # x, y_top, w, h
HEADER_TOP = px(190)
CHART_BOX = tuple(px(v) for v in (140, 335, 900, 605))
SUB_BOX = tuple(px(v) for v in (40, 1000, 1000, 170))
PANEL_Y = px(1190)
PANEL_SIZE = px(480)
AVATAR_X = px(40)
SIDEBAR_X = px(540)
SIDEBAR_W = px(500)


def axes_rect(box: Tuple[int, int, int, int]) -> List[float]:
    """Pixel box (x, y_top, w, h) -> matplotlib figure-fraction rect [left, bottom, w, h]."""
    x, y, w, h = box
    return [x / VIDEO_W, 1 - (y + h) / VIDEO_H, w / VIDEO_W, h / VIDEO_H]


def fig_x(x_px: float) -> float:
    return x_px / VIDEO_W


def fig_y(y_px_from_top: float) -> float:
    return 1 - y_px_from_top / VIDEO_H


# =====================================================================
# 2. HELPERS
# =====================================================================
def with_retries(fn: Callable, attempts: int = 3, base_delay: float = 1.0, what: str = "operation"):
    """Call fn(); on failure wait base_delay * 2^n (+jitter) and try again. Raises the last error."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                delay = base_delay * 2 ** i * random.uniform(0.7, 1.3)
                logger.warning(f"{what} failed ({e}); retrying in {delay:.1f}s ({i + 2}/{attempts})")
                time.sleep(delay)
    raise last_exc


def get_secret(name: str) -> Optional[str]:
    """Environment / .env first, then Streamlit secrets (only if Streamlit is available)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


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
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='ignore')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


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


def mpl_safe(text: str) -> str:
    """A pair of bare '$' makes matplotlib switch to math mode ('$6,800 | BTC: $70,000' breaks)."""
    return text.replace("$", r"\$")


def ascii_clean(text: str) -> str:
    """Normalises typographic characters and drops anything the bold fonts may not render (emoji etc.)."""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", "", text)).strip()


def _positive_number(value) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_attr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


def format_money(value: Optional[float]) -> Optional[str]:
    if not value:
        return None
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= limit:
            return f"${value / limit:.2f}{suffix}"
    return None


def format_count(value: Optional[float]) -> str:
    if not value:
        return "-"
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:.0f}"


def fmt_price(v: float) -> str:
    return f"${v:,.0f}" if v >= 1000 else f"${v:,.2f}"


# =====================================================================
# 3. LLM  (model rotation + exponential backoff)
# =====================================================================
class LLMUnavailableError(RuntimeError):
    """Every model kept failing after all retry rounds."""


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing!")
    return genai.Client(api_key=api_key)


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _is_transient(exc: Exception) -> bool:
    code = _status_code(exc)
    if code is not None:
        return code in LLM_RETRY_STATUS
    if httpx is not None and isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))


def ask_llm(prompt, config=None, parse: Optional[Callable] = None, max_rounds: int = LLM_MAX_ROUNDS):
    """Each round tries every model once; transient errors move on to the next model, then back off and
    retry. Permanent errors drop the model. `parse` failures (bad JSON, wrong length ...) are retried too."""
    client = get_gemini_client()
    kwargs = {"config": config} if config else {}
    models = list(GEMINI_MODELS)
    last_exc: Optional[Exception] = None

    for round_no in range(1, max_rounds + 1):
        for model in list(models):
            try:
                response = client.models.generate_content(model=model, contents=prompt, **kwargs)
            except Exception as e:
                last_exc = e
                if _is_transient(e):
                    logger.warning(f"{model} temporarily unavailable (round {round_no}/{max_rounds}): {e}")
                    continue
                logger.error(f"{model} failed with a non-retryable error, removing it from rotation: {e}")
                models.remove(model)
                if not models:
                    raise
                continue
            if parse is None:
                return response
            try:
                return parse(response)
            except Exception as e:
                last_exc = e
                logger.warning(f"{model} returned unusable output (round {round_no}/{max_rounds}): {e}")

        if round_no < max_rounds:
            delay = min(LLM_MAX_DELAY, LLM_BASE_DELAY * 2 ** (round_no - 1)) * random.uniform(0.5, 1.0)
            logger.info(f"All models busy; retrying in {delay:.1f}s (round {round_no + 1}/{max_rounds})...")
            time.sleep(delay)

    raise LLMUnavailableError(
        f"Gemini still failing after {max_rounds} rounds across {GEMINI_MODELS}. Last error: {last_exc}") from last_exc


# =====================================================================
# 4. STOCK SELECTION  (one batch download)
# =====================================================================


def _download_batch(tickers: List[str]):
    def _go():
        df = yf.download(tickers=tickers, period="2d", interval="5m", group_by="ticker",
                         progress=False, threads=True)
        if df is None or df.empty:
            raise RuntimeError("yfinance returned no data")
        return df
    return with_retries(_go, attempts=3, base_delay=2.0, what="Batch download")


def _ticker_frame(batch_df, ticker: str):
    """Slice one ticker out of a batch download (works for one or many tickers)."""
    if hasattr(batch_df.columns, "levels"):
        if ticker not in batch_df.columns.get_level_values(0):
            return None
        return batch_df[ticker].dropna(how="all")
    return batch_df.dropna(how="all")


def _session_stats(df) -> Optional[dict]:
    """Latest session's candles + price / previous close / % change out of a 2-day 5-minute frame."""
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        return None
    dates = df.index.date
    latest = dates[-1]
    today = df[dates == latest]
    if today.empty:
        return None
    price = float(today["Close"].iloc[-1])
    prev_day = df[dates < latest]
    prev_close = float(prev_day["Close"].iloc[-1]) if not prev_day.empty else float(today["Open"].iloc[0])
    if not prev_close:
        return None
    return {"df": today, "price": price, "prev_close": prev_close,
            "pct": (price - prev_close) / prev_close * 100.0}


def get_top_moving_stock(test: bool = False) -> dict:
    """The ticker with the biggest move (in %) in the latest session, skipping ones already used today."""
    candidates = list(TICKERS_TO_CHECK)

    download_list = sorted(set(candidates) | set(MARKET_REFS))
    logger.info(f"Batch fetching {len(download_list)} tickers in one request...")
    batch_df = _download_batch(download_list)

    changes: Dict[str, float] = {}
    best = None
    for symbol in download_list:
        try:
            s = _session_stats(_ticker_frame(batch_df, symbol))
            if not s:
                continue
            changes[symbol] = s["pct"]
            if symbol in candidates and (best is None or abs(s["pct"]) > abs(best["change_pct"])):
                best = {"symbol": symbol, "current_price": round(s["price"], 2),
                        "change_pct": round(s["pct"], 2), "prev_close": s["prev_close"], "df": s["df"]}
        except Exception as e:
            logger.warning(f"Error analyzing batch data for {symbol}: {e}")

    if best is None:
        raise RuntimeError("No valid market data found for any ticker.")

    best["market"] = {k: changes[k] for k in MARKET_REFS if k in changes}
    logger.info(f"Top Moving Stock: {best['symbol']} ({best['change_pct']}%)")
    return best


def get_company_name(symbol: str) -> str:
    if symbol in COMPANY_NAMES:
        return COMPANY_NAMES[symbol]
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol


def pick_chart_history(stock: dict) -> Tuple[object, str]:
    """Today's 5-minute candles; if the session is very young, fall back to 5 days of 15-minute candles."""
    cols = ["Open", "High", "Low", "Close"]
    df = stock["df"].dropna(subset=cols)
    if len(df) >= 20:
        return df, "1D"
    try:
        hist = yf.Ticker(stock["symbol"]).history(period="5d", interval="15m").dropna(subset=cols)
        if len(hist) >= 20:
            return hist, "5D"
    except Exception as e:
        logger.warning(f"5-day fallback history failed for {stock['symbol']}: {e}")
    if len(df) < 2:
        raise RuntimeError(f"Not enough candles to draw a chart for {stock['symbol']}")
    return df, "1D"


# =====================================================================
# 4b. NEWS  (Google News RSS, company-specific)
# =====================================================================
COMPANY_NEWS_ALIASES = {
    "GOOGL": ("Alphabet", "Google"),
    "MSTR": ("MicroStrategy", "Strategy"),
    "BRK-B": ("Berkshire Hathaway",),
    "QQQ": ("Invesco QQQ", "Nasdaq 100"),
    "SPY": ("S&P 500", "SPDR"),
    "AMD": ("AMD", "Advanced Micro Devices"),
    "ARM": ("Arm Holdings",),
    "META": ("Meta", "Meta Platforms"),
    "JPM": ("JPMorgan", "JPMorgan Chase"),
    "LLY": ("Eli Lilly",),
}
RSS_NEWS_MAX_ARTICLES = 5
RSS_NEWS_TIMEOUT = 7


def _rss_company_terms(symbol: str) -> List[str]:
    company = COMPANY_NAMES.get(symbol, symbol).strip()
    choices = list(COMPANY_NEWS_ALIASES.get(symbol, ()))
    if company and company != symbol:
        choices.append(company)
    if company and company != symbol and len(company.split()[0]) >= 4:
        choices.append(company.split()[0])
    return list(dict.fromkeys(c for c in choices if c))


def _rss_queries(symbol: str) -> List[str]:
    terms = _rss_company_terms(symbol)
    queries = [f'"{symbol}" stock']
    if terms:
        queries.append(f'"{terms[0]}" stock')
        queries.append(f'"{terms[0]}" shares OR earnings')
    return list(dict.fromkeys(queries))


def _fetch_rss_headlines(query: str, days: int = 1) -> List[dict]:
    url = ('https://news.google.com/rss/search?q=' + urllib.parse.quote_plus(f'{query} when:{days}d')
           + '&hl=en-US&gl=US&ceid=US:en')
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        })
        with urllib.request.urlopen(req, timeout=RSS_NEWS_TIMEOUT) as resp:
            root = ET.fromstring(resp.read())

        headlines = []
        for item in root.findall('.//item')[:10]:
            title = html.unescape((item.findtext('title') or '').strip())
            publisher = html.unescape((item.findtext('source') or '').strip())
            if publisher and title.endswith(' - ' + publisher):
                title = title[:-(len(publisher) + 3)].strip()
            elif not publisher and ' - ' in title:
                title, publisher = title.rsplit(' - ', 1)
            if not title:
                continue
            pub_dt = None
            pub_text = (item.findtext('pubDate') or '').strip()
            if pub_text:
                try:
                    pub_dt = parsedate_to_datetime(pub_text)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    pub_dt = pub_dt.astimezone(timezone.utc)
                except (TypeError, ValueError, IndexError):
                    pass
            headlines.append({
                'provider': publisher or 'Market News',
                'title': title,
                'summary': '',
                'published': pub_dt.isoformat() if pub_dt else None,
                '_published_ts': pub_dt.timestamp() if pub_dt else 0.0,
            })
        return headlines
    except Exception as e:
        logger.warning('Google News RSS query %r (%sd) failed: %s', query, days, e)
        return []


def _rss_title_matches_company(title: str, symbol: str) -> bool:
    for term in [symbol, *_rss_company_terms(symbol)]:
        if re.search(r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])', title, flags=re.IGNORECASE):
            return True
    return False


def _fetch_stock_rss_items(symbol: str, days: int) -> List[dict]:
    queries = _rss_queries(symbol)
    with ThreadPoolExecutor(max_workers=min(3, len(queries))) as pool:
        lists = list(pool.map(lambda q: _fetch_rss_headlines(q, days), queries))

    now_ts = datetime.now(timezone.utc).timestamp()
    max_age = (days * 24 + 8) * 3600
    seen, articles = set(), []
    for results in lists:
        for item in results:
            title = item['title']
            if not _rss_title_matches_company(title, symbol):
                continue
            ts = item.get('_published_ts') or 0.0
            if ts and (now_ts - ts > max_age or ts > now_ts + 7200):
                continue
            key = re.sub(r'[^a-z0-9]+', '', title.casefold())
            if key in seen:
                continue
            seen.add(key)
            articles.append(item)
    articles.sort(key=lambda item: item.get('_published_ts') or 0.0, reverse=True)
    return articles


def format_news_context(items: List[dict]) -> str:
    """Numbered headlines: the script refers to them by index to trigger the on-screen pop-up."""
    if not items:
        return "No recent company-specific news found. Do not invent a cause for today's move."
    blocks = []
    for i, it in enumerate(items):
        published = it.get('published')
        date_note = f", published {published[:10]} UTC" if published else ""
        blocks.append(f"[{i}] ({it['provider']}{date_note}) {it['title']}")
    return ("Numbered company news headlines from Google News RSS (headlines are NOT proof that an event "
            "caused today's stock move):\n" + "\n".join(blocks))


def fetch_news_context(symbol: str) -> Tuple[str, List[dict]]:
    symbol = symbol.strip().upper()
    logger.info('Fetching %s news via Google News RSS (past 24h)...', symbol)
    fresh = _fetch_stock_rss_items(symbol, days=1)
    if len(fresh) >= 3:
        articles = fresh[:RSS_NEWS_MAX_ARTICLES]
    else:  # weekend / slow news cycle fallback; the dates are shown to Gemini
        logger.info('Only %s recent RSS headlines for %s, checking past 7 days.', len(fresh), symbol)
        older = _fetch_stock_rss_items(symbol, days=7)
        seen = {re.sub(r'[^a-z0-9]+', '', x['title'].casefold()) for x in fresh}
        articles = list(fresh)
        for item in older:
            key = re.sub(r'[^a-z0-9]+', '', item['title'].casefold())
            if key not in seen:
                articles.append(item)
                seen.add(key)
            if len(articles) >= RSS_NEWS_MAX_ARTICLES:
                break

    items = [{k: v for k, v in item.items() if not k.startswith('_')} for item in articles[:RSS_NEWS_MAX_ARTICLES]]
    logger.info('%s: received %d relevant RSS headlines.', symbol, len(items))
    return format_news_context(items), items


# =====================================================================
# 5. STOCK STATS  (everything the right-hand panel shows - every field is optional)
# =====================================================================
def _fetch_market_cap(symbol: str) -> Tuple[float, str]:
    """(value, label). Several sources, because Yahoo's `.info` is often rate-limited. ETFs -> net assets."""
    t = yf.Ticker(symbol)
    fi = t.fast_info

    def from_fast_info():
        return _positive_number(_safe_attr(fi, "market_cap"))

    def from_info():
        info = t.info or {}
        cap = _positive_number(info.get("marketCap"))
        if cap:
            return cap
        shares, price = _positive_number(info.get("sharesOutstanding")), _positive_number(info.get("currentPrice"))
        return shares * price if shares and price else None

    def from_etf_assets():
        info = t.info or {}
        return _positive_number(info.get("totalAssets") or info.get("netAssets"))

    quote_type = str(_safe_attr(fi, "quote_type") or "").upper()
    if symbol in ETF_TICKERS or quote_type == "ETF":
        strategies = [("info.totalAssets", from_etf_assets, "NET ASSETS")]
    else:
        strategies = [("fast_info.market_cap", from_fast_info, "MARKET CAP"),
                      ("info.marketCap", from_info, "MARKET CAP")]
    for source, getter, label in strategies:
        try:
            value = getter()
        except Exception as e:
            logger.info(f"   market cap source '{source}' failed for {symbol}: {e}")
            continue
        if value:
            return value, label
    raise LookupError(f"no market cap source worked for {symbol}")


def fetch_stock_stats(stock: dict) -> dict:
    symbol, df, price = stock["symbol"], stock["df"], stock["current_price"]
    stats: dict = {
        "symbol": symbol, "price": price, "pct": stock["change_pct"], "prev_close": stock.get("prev_close"),
        "open": float(df["Open"].iloc[0]),
        "day_high": float(np.nanmax(df["High"].values)), "day_low": float(np.nanmin(df["Low"].values)),
        "volume": float(np.nansum(df["Volume"].values)) if "Volume" in df else None,
        "mcap": None, "mcap_label": "MARKET CAP", "year_high": None, "year_low": None,
        "avg_volume": None, "rel_volume": None, "pe": None, "pe_label": "P/E RATIO", "target": None,
    }
    market = stock.get("market", {})
    stats["spy_pct"], stats["qqq_pct"] = market.get("SPY"), market.get("QQQ")
    stats["vs_spy"] = (stock["change_pct"] - market["SPY"]) if "SPY" in market and symbol != "SPY" else None

    t = yf.Ticker(symbol)
    try:
        fi = t.fast_info
    except Exception:
        fi = None

    def fi_get(name):
        return _positive_number(_safe_attr(fi, name)) if fi is not None else None

    stats["year_high"], stats["year_low"] = fi_get("year_high"), fi_get("year_low")
    stats["avg_volume"] = fi_get("three_month_average_volume") or fi_get("ten_day_average_volume")

    try:
        stats["mcap"], stats["mcap_label"] = with_retries(lambda: _fetch_market_cap(symbol), attempts=2,
                                                          base_delay=1.0, what=f"{symbol} market cap")
    except Exception as e:
        logger.warning(f"Market cap unavailable for {symbol}: {e}")

    try:
        info = with_retries(lambda: t.info or {}, attempts=2, base_delay=1.0, what=f"{symbol} info")
    except Exception as e:
        logger.warning(f"yfinance info unavailable for {symbol}: {e}")
        info = {}
    stats["pe"] = _positive_number(info.get("trailingPE"))
    if not stats["pe"]:
        stats["pe"] = _positive_number(info.get("forwardPE"))
        stats["pe_label"] = "FORWARD P/E"
    stats["target"] = _positive_number(info.get("targetMeanPrice"))
    stats["year_high"] = stats["year_high"] or _positive_number(info.get("fiftyTwoWeekHigh"))
    stats["year_low"] = stats["year_low"] or _positive_number(info.get("fiftyTwoWeekLow"))
    stats["avg_volume"] = stats["avg_volume"] or _positive_number(info.get("averageVolume"))

    # Relative volume is only meaningful once the session is (almost) complete.
    try:
        ts = df.index[-1]
        if (ts.hour, ts.minute) >= (15, 55) and stats["volume"] and stats["avg_volume"]:
            stats["rel_volume"] = stats["volume"] / stats["avg_volume"]
    except Exception:
        pass
    logger.info("Stats: " + ", ".join(f"{k}={v}" for k, v in stats.items() if v is not None))
    return stats


@dataclass
class StatRow:
    label: str
    value: str
    color: str = C_WHITE
    bar: Optional[Tuple[float, float, float]] = None     # (low, high, current) -> position bar under the row


def build_stat_rows(stats: dict, max_rows: int = 6) -> List[StatRow]:
    """Priority-ordered rows; anything we could not get is simply left out (never 'N/A')."""
    rows: List[StatRow] = []
    if stats.get("mcap"):
        rows.append(StatRow(stats["mcap_label"], format_money(stats["mcap"]) or "-"))
    lo, hi, price = stats.get("day_low"), stats.get("day_high"), stats["price"]
    if lo and hi and hi > lo:
        rows.append(StatRow("DAY RANGE", f"{fmt_price(lo)} - {fmt_price(hi)}", bar=(lo, hi, price)))
    if stats.get("year_low") and stats.get("year_high") and stats["year_high"] > stats["year_low"]:
        rows.append(StatRow("52W RANGE", f"{fmt_price(stats['year_low'])} - {fmt_price(stats['year_high'])}",
                            bar=(stats["year_low"], stats["year_high"], price)))
    if stats.get("volume"):
        text, color = format_count(stats["volume"]), C_WHITE
        if stats.get("rel_volume"):
            text += f"  ({stats['rel_volume']:.1f}x avg)"
            color = C_GOLD if stats["rel_volume"] >= 1.5 else C_WHITE
        rows.append(StatRow("VOLUME", text, color))
    if stats.get("pe"):
        rows.append(StatRow(stats["pe_label"], f"{stats['pe']:.1f}"))
    if stats.get("vs_spy") is not None:
        d = stats["vs_spy"]
        rows.append(StatRow("VS S&P 500", f"{'+' if d >= 0 else ''}{d:.2f}%", C_GREEN if d >= 0 else C_RED))
    if stats.get("target"):
        up = (stats["target"] - price) / price * 100
        rows.append(StatRow("ANALYST TARGET", f"{fmt_price(stats['target'])} ({up:+.0f}%)",
                            C_GREEN if up >= 0 else C_RED))
    if stats.get("prev_close"):
        rows.append(StatRow("PREV CLOSE", fmt_price(stats["prev_close"])))
    return rows[:max_rows]


def stats_for_prompt(stats: dict) -> str:
    p = stats["price"]
    lines = [f"Price ${p:.2f}, {stats['pct']:+.2f}% today" +
             (f" (previous close ${stats['prev_close']:.2f})" if stats.get("prev_close") else "")]
    if stats.get("mcap"):
        lines.append(f"{stats['mcap_label'].title()}: {format_money(stats['mcap'])}")
    lines.append(f"Today's range: ${stats['day_low']:.2f} to ${stats['day_high']:.2f}")
    if stats.get("year_high") and stats.get("year_low"):
        from_high = (p - stats["year_high"]) / stats["year_high"] * 100
        lines.append(f"52 week range: ${stats['year_low']:.2f} to ${stats['year_high']:.2f} "
                     f"(price is {abs(from_high):.0f}% {'below' if from_high < 0 else 'above'} the 52 week high)")
    if stats.get("volume"):
        v = f"Volume today: {format_count(stats['volume'])} shares"
        if stats.get("rel_volume"):
            v += f", {stats['rel_volume']:.1f} times the 3 month average"
        lines.append(v)
    if stats.get("pe"):
        lines.append(f"{stats['pe_label'].title()}: {stats['pe']:.1f}")
    if stats.get("target"):
        lines.append(f"Average analyst price target: ${stats['target']:.2f}")
    if stats.get("spy_pct") is not None:
        lines.append(f"S&P 500 (SPY) today: {stats['spy_pct']:+.2f}%"
                     + (f", Nasdaq 100 (QQQ): {stats['qqq_pct']:+.2f}%" if stats.get("qqq_pct") is not None else ""))
    if stats.get("vs_spy") is not None:
        lines.append(f"Stock vs S&P 500 today: {stats['vs_spy']:+.2f} percentage points")
    return "\n".join(f"- {l}" for l in lines)


# =====================================================================
# 6. SCRIPT + TITLES  (5 narration "beats" -> exact timing without Whisper)
# =====================================================================
class Beat(BaseModel):
    role: str = Field(description="One of: hook, catalyst, context, bulls_bears, cta")
    text: str = Field(description="Spoken narration for this beat: 1 or 2 short sentences")
    headline_index: int = Field(default=-1, description=(
        "Index of the numbered news headline to pop up on screen while this beat is spoken, or -1 for none"))


class StockShortScript(BaseModel):
    youtube_title: str
    news_ticker_text: str = Field(description="Single line, 3 or 4 punchy UPPERCASE items separated by '  •  '")
    beats: List[Beat]
    description: str
    tags: str


def _make_script_parser(n_headlines: int) -> Callable:
    def parse(response) -> StockShortScript:
        script = StockShortScript.model_validate_json(response.text)
        beats = [b for b in script.beats if b.text and b.text.strip()]
        if not 3 <= len(beats) <= 7:
            raise ValueError(f"expected about 5 beats, got {len(beats)}")
        words = sum(len(b.text.split()) for b in beats)
        if not SCRIPT_WORDS_MIN <= words <= SCRIPT_WORDS_MAX:
            raise ValueError(f"narration has {words} words, need {SCRIPT_WORDS_MIN}-{SCRIPT_WORDS_MAX}")
        used = set()
        for b in beats:                                   # one pop-up per headline, valid indices only
            if not (0 <= b.headline_index < n_headlines) or b.headline_index in used:
                b.headline_index = -1
            else:
                used.add(b.headline_index)
        if not script.youtube_title.strip():
            raise ValueError("empty youtube_title")
        script.beats = beats
        return script
    return parse


def generate_script_and_titles_with_ai(stock: dict, stats: dict, news_text: str, n_headlines: int) -> StockShortScript:
    direction = "UP" if stock["change_pct"] >= 0 else "DOWN"
    symbol, company = stock["symbol"], stock["company_name"]
    prompt = f"""
        You are a high-energy financial content creator and YouTube SEO expert for YouTube Shorts, TikTok and Reels.
        Write a fast-paced vertical video script (about {TARGET_SECONDS} seconds when spoken) and SEO metadata about this stock move.

        STOCK: {symbol} ({company}) is {direction} {abs(stock['change_pct']):.1f}% today.

        VERIFIED STATS (ground truth, never contradict them):
        {stats_for_prompt(stats)}

        {news_text}

        RULES:
        1. "beats": exactly 5 beats, in this order, each 1 or 2 SHORT sentences. All beats together: 75 to 90 words.
           - hook: the FIRST sentence says WHAT the stock did and WHY, using the company's plain name (not the ticker).
             Real substance, no vague teasers like "you won't believe this". About 15 words.
           - catalyst: the real reason, grounded in the headlines above. NEVER invent a catalyst. If no headline clearly
             explains the move, say honestly that no single headline explains it yet and name the most relevant
             market-wide factor instead. Set "headline_index" to the headline that supports this beat, else -1.
        - context: one or two facts from the VERIFIED STATS that make the move meaningful (volume, how it compares with
             the S&P 500, where it sits versus its 52 week range, valuation, market cap). Set "headline_index" to a SECOND
             headline only if it adds a different fact, otherwise -1.
           - bulls_bears: one sentence for what the bulls argue and one for what the bears argue.
           - cta: ask viewers for their PRICE TARGET or AVERAGE BUY PRICE for {symbol} in the comments.
             Never use the generic "type BUY or SELL".
        2. "headline_index" must be one of the numbers in the numbered list (or -1). Use each number at most once.
        3. TTS FRIENDLY: natural spoken English. ABSOLUTELY NO dashes, ellipses, semicolons, parentheses or emojis.
           Only periods, commas and question marks. No decimal points: round to speakable numbers ("up more than 5
           percent", "about 12 percent below its high"). No dollar signs. Avoid abbreviations with periods (write "US").
        4. "news_ticker_text": 3 or 4 items in UPPERCASE, separated by '  •  ', each "TOPIC: ultra short summary",
           based ONLY on the headlines and stats above, e.g. "NVDA JUMPS 5%: AI DEMAND SURGES  •  CHIPS LEAD NASDAQ".
        5. SEO: "youtube_title" MUST contain BOTH the ticker symbol and the full company name (emojis allowed here only).
           "description" must naturally include the ticker and company name plus key financial terms, end with a call
           to share your price target in the comments, and include the line "Not financial advice."
           "tags": 5 to 7 comma separated tags including the ticker and company name.
        """
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=StockShortScript)
    return ask_llm(prompt, config=config, parse=_make_script_parser(n_headlines))


def fallback_ticker_text(stock: dict, items: List[dict]) -> str:
    sign = "+" if stock["change_pct"] >= 0 else ""
    parts = [f"{stock['symbol']} {sign}{stock['change_pct']:.1f}%: BIGGEST MOVER TODAY"]
    for it in items[:3]:
        parts.append(ascii_clean(it["title"]).upper()[:70])
    return "  \u2022  ".join(parts)


def clean_ticker_text(text: str) -> str:
    return re.sub(r"[^\x20-\x7E\u2022]+", "", text or "").strip()


# =====================================================================
# 7. TEXT-TO-SPEECH  (Kokoro-82M, local, in-process, one clip per beat)
# =====================================================================
import requests   # put with the other top-level imports

KOKORO_DIR = Path(__file__).resolve().parent / "kokoro_assets"
_KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_NAME = "kokoro-v1.0.int8.onnx"   # ~90 MB. Use "kokoro-v1.0.onnx" (~310 MB) for best quality.
KOKORO_VOICES_NAME = "voices-v1.0.bin"


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading %s ...", dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)  # only becomes the real file once the download completed


def _get_or_download_kokoro_assets() -> Tuple[str, str]:
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    model_path = KOKORO_DIR / KOKORO_MODEL_NAME
    voices_path = KOKORO_DIR / KOKORO_VOICES_NAME
    for path in (model_path, voices_path):
        if not path.exists():
            _download(f"{_KOKORO_BASE}/{path.name}", path)
    return str(model_path), str(voices_path)


@lru_cache(maxsize=1)
def get_kokoro_onnx():
    from kokoro_onnx import Kokoro  # lazy: render workers never load onnxruntime
    model_path, voices_path = _get_or_download_kokoro_assets()
    return Kokoro(model_path, voices_path)


def clean_script_for_tts(text: str) -> str:
    """Removes non-ASCII characters/emojis & converts symbols for clear TTS."""
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.replace("$", "").replace("%", " percent")
    text = re.sub(r"\s+[-\u2013\u2014]+\s+|[\u2013\u2014]", ", ", text).replace(";", ",").replace("...", ".")
    for pattern, replacement in TTS_SYMBOL_NAMES.items():
        text = re.sub(pattern, replacement, text)
    return text.replace("\n", " ").strip()


def generate_voiceover(beat_texts: List[str], out_path: Path) -> Tuple[List[Tuple[float, float]], float]:
    """One Kokoro clip per beat, joined with a short gap into a single WAV.
    Returns ([(start, end) of each beat in seconds], end of the last word). Beat timings are exact,
    which is how the captions and the headline pop-ups stay in sync without Whisper."""
    import soundfile as sf
    kokoro = get_kokoro_onnx()
    pieces: List[np.ndarray] = []
    spans: List[Tuple[float, float]] = []
    gap, sample_rate, cursor = None, TTS_RATE, 0
    for i, text in enumerate(beat_texts):
        clip, sample_rate = kokoro.create(clean_script_for_tts(text), voice=TTS_VOICE,
                                          speed=TTS_SPEED, lang="en-us")
        clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        if clip.size == 0:
            raise RuntimeError(f"Kokoro returned empty audio for beat {i}: {text!r}")
        if sample_rate != TTS_RATE:
            logger.warning("Kokoro sample rate is %d, TTS_RATE is %d", sample_rate, TTS_RATE)
        if gap is None:
            gap = np.zeros(int(sample_rate * BEAT_GAP), dtype=np.float32)
        spans.append((cursor / sample_rate, (cursor + len(clip)) / sample_rate))
        pieces.append(clip)
        cursor += len(clip)
        if i < len(beat_texts) - 1:
            pieces.append(gap)
            cursor += len(gap)
    sf.write(str(out_path), np.concatenate(pieces), sample_rate, subtype="PCM_16")
    logger.info("Voiceover: %d beats, %.1fs", len(spans), spans[-1][1])
    return spans, spans[-1][1]


# =====================================================================
# 8. DRAWING PRIMITIVES  (identical to the daily recap)
# =====================================================================
def new_figure(width_px: int = VIDEO_W, height_px: int = VIDEO_H, facecolor: str = C_BG):
    """Pyplot-free figure. The +1e-3 guards against int() truncation giving a 1079/1919 px canvas."""
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
    ax.tick_params(colors=C_WHITE, labelsize=14)
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


class NewsTickerBar:
    """Scrolling news bar (one text artist; scrolling = moving its x position)."""

    def __init__(self, fig, raw_text: str, scroll_speed: float = 0.12, fontsize: int = 18):
        self.speed = scroll_speed
        ax = fig.add_axes(axes_rect(TICKER_BOX))
        ax.set_facecolor(C_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(C_BORDER)
            spine.set_linewidth(2)

        safe_text = raw_text.replace("$", r"\$")
        unit = f"BREAKING NEWS:  {safe_text}   \u2022   "
        self._text = ax.text(0.02, 0.5, unit, fontsize=fontsize, fontweight="bold", color=C_GREEN,
                             va="center", clip_on=True, transform=ax.transData)
        renderer = fig.canvas.get_renderer()
        w1 = self._text.get_window_extent(renderer).width
        self._text.set_text(unit * 2)
        w2 = self._text.get_window_extent(renderer).width
        self.period = max((w2 - w1) / ax.bbox.width, 1e-3)
        reps = int(math.ceil(1.0 / self.period)) + 2
        self._text.set_text(unit * reps)

    def update(self, t: float) -> None:
        self._text.set_x(0.02 - (t * self.speed) % self.period)


class AvatarPanel:
    """Every open/closed x bob combination is composited once, then cached."""

    def __init__(self, size: int):
        self.size = size
        self._open = self._load(AVATAR_OPEN_PATH)
        self._closed = self._load(AVATAR_CLOSED_PATH)
        self._bg = Image.new("RGB", (size, size), (22, 27, 34))
        draw = ImageDraw.Draw(self._bg)
        draw.rectangle((0, 0, size - 1, size - 1), outline=(48, 54, 61), width=max(1, px(4)))
        self._cache: Dict[Tuple[bool, int], np.ndarray] = {}

    def _load(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as im:
                return im.convert("RGBA").resize((self.size, self.size))
        except Exception as e:
            logger.warning("Avatar image missing (%s): %s", path, e)
            return Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))

    def frame(self, is_open: bool, bob_px: int) -> np.ndarray:
        step = max(1, px(4))
        bob_px = max(0, (bob_px + step // 2) // step * step)
        key = (is_open, bob_px)
        if key not in self._cache:
            sprite = self._open if is_open else self._closed
            out = self._bg.copy()
            out.paste(sprite, (0, -bob_px), sprite.getchannel("A"))
            self._cache[key] = np.asarray(out, dtype=np.uint8)
        return self._cache[key]


# ---------------------------------------------------------------------
# Fonts / text wrapping (Pillow)
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


def _font_height(font) -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except Exception:
        return int(getattr(font, "size", 16) * 1.2)


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


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while len(text) > 1 and draw.textlength(text + "...", font=font) > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


# ---------------------------------------------------------------------
# Subtitles (pre-rendered once with Pillow, alpha-blended into the frame)
# ---------------------------------------------------------------------
@dataclass
class SubtitleCue:
    start: float
    end: float
    x: int
    y: int
    alpha: np.ndarray  # (h, w, 1) float32, cropped to the text's bounding box


def _split_for_captions(sentence: str, max_words: int = CAPTION_MAX_WORDS) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    n_chunks = math.ceil(len(words) / max_words)
    size = math.ceil(len(words) / n_chunks)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


class SubtitleTrack:
    BASE_FONT_SIZE = 54   # at 1080 wide; scaled with UI_SCALE
    MIN_FONT_SIZE = 14

    def __init__(self, segments: List[Tuple[str, float, float]], total: float):
        """segments = [(text, start, end)] per spoken beat. Cues are contiguous (no flicker in the pauses)."""
        self.box_x, self.box_y, self.box_w, self.box_h = SUB_BOX
        specs: List[List] = []
        for text, start, end in segments:
            clean = re.sub(r"[^\x00-\x7F]+", "", text).strip()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()] or [clean]
            pieces = [p for s in sentences for p in _split_for_captions(s)]
            counts = [max(1, len(p.split())) for p in pieces]
            cursor = start
            for piece, n_words in zip(pieces, counts):
                dur = n_words / sum(counts) * (end - start)
                specs.append([piece, cursor, cursor + dur])
                cursor += dur
        for i in range(len(specs) - 1):
            specs[i][2] = specs[i + 1][1]
        if specs:
            specs[-1][2] = max(total, specs[-1][2])
        self.cues: List[SubtitleCue] = [c for c in (self._render_cue(*s) for s in specs) if c]

    def _render_cue(self, sentence: str, start: float, end: float) -> Optional[SubtitleCue]:
        try:
            size = max(self.MIN_FONT_SIZE, int(self.BASE_FONT_SIZE * UI_SCALE))
            while True:  # shrink the font until the wrapped text fits in the strip
                font = load_font(size)
                lines = _wrap_text(sentence, font, self.box_w)
                line_h = int(_font_height(font) * 1.15)
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
# Bottom-right: KEY STATS panel (static image) + headline pop-up cards
# ---------------------------------------------------------------------
def render_stats_panel(rows: List[StatRow], symbol: str, change_pct: float,
                       width_px: int = SIDEBAR_W, height_px: int = PANEL_SIZE) -> np.ndarray:
    accent = C_GREEN if change_pct >= 0 else C_RED
    fig, canvas = new_figure(width_px, height_px, facecolor=C_PANEL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=C_BORDER, linewidth=6, transform=ax.transAxes))

    ax.text(0.5, 0.925, f"{symbol} KEY STATS", fontsize=22, fontweight="bold", color=C_WHITE, ha="center", va="center")
    ax.plot([0.08, 0.92], [0.865, 0.865], color=C_BORDER, linewidth=1.5)

    n = max(1, len(rows))
    row_h = min(0.16, 0.80 / n)
    x0, x1 = 0.07, 0.93
    for i, row in enumerate(rows):
        yc = 0.865 - row_h * (i + 0.5)
        ty = yc + row_h * 0.17 if row.bar else yc
        ax.text(x0, ty, row.label, fontsize=12.5, color=C_MUTED, va="center", ha="left")
        ax.text(x1, ty, mpl_safe(row.value), fontsize=15, fontweight="bold", color=row.color, va="center", ha="right")
        if row.bar:
            lo, hi, cur = row.bar
            pos = min(max((cur - lo) / (hi - lo), 0.0), 1.0)
            xp, yb = x0 + pos * (x1 - x0), yc - row_h * 0.22
            ax.plot([x0, x1], [yb, yb], color=C_BORDER, linewidth=6, solid_capstyle="round")
            ax.plot([x0, xp], [yb, yb], color=accent, linewidth=6, solid_capstyle="round")
            ax.plot([xp], [yb], marker="o", markersize=8, color=C_WHITE, markeredgewidth=0, zorder=5)
    return grab(canvas, width_px, height_px)


def render_headline_card(item: dict, change_pct: float,
                         width: int = SIDEBAR_W, height: int = PANEL_SIZE) -> Optional[np.ndarray]:
    """Opaque RGB card the same size as the stats panel; it slides over the panel for a moment."""
    title = ascii_clean(item.get("title", ""))
    if len(title) < 10:
        return None
    provider = ascii_clean(item.get("provider", "")) or "Market News"
    accent = C_GREEN if change_pct >= 0 else C_RED

    img = Image.new("RGB", (width, height), C_PANEL)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width - 1, height - 1), outline=accent, width=max(2, px(6)))
    pad = px(36)

    label_font = load_font(max(12, px(26)))
    pill_h = px(46)
    label_w = int(d.textlength(NEWS_CARD_LABEL, font=label_font)) + px(40)
    d.rounded_rectangle((pad, pad, pad + label_w, pad + pill_h), radius=pill_h // 2, fill=C_PILL)
    d.text((pad + label_w / 2, pad + pill_h / 2), NEWS_CARD_LABEL, font=label_font, fill="white", anchor="mm")

    prov_font = load_font(max(11, px(24)))
    prov = _fit_text(d, provider, prov_font, width - 2 * pad)
    d.text((pad, height - pad), prov, font=prov_font, fill=C_MUTED, anchor="ls")

    top = pad + pill_h + px(28)
    bottom = height - pad - px(34)
    max_w = width - 2 * pad
    size = max(14, px(44))
    while True:
        font = load_font(size)
        lines = _wrap_text(title, font, max_w)
        line_h = int(_font_height(font) * 1.2)
        if line_h * len(lines) <= bottom - top or size <= 14:
            break
        size -= 2
    max_lines = max(1, (bottom - top) // line_h)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and d.textlength(last + "...", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last.rstrip() + "..."
    y = top
    for line in lines:
        d.text((pad, y), line, font=font, fill="white", anchor="lt")
        y += line_h
    return np.asarray(img, dtype=np.uint8).copy()


@dataclass
class Popup:
    start: float
    end: float
    card: np.ndarray   # (PANEL_SIZE, SIDEBAR_W, 3)


def plan_popups(beats: List[Beat], spans: List[Tuple[float, float]], items: List[dict],
                change_pct: float, total: float) -> List[Popup]:
    """A headline card appears for a quick moment at the start of the beat that talks about it."""
    if not items:
        return []
    if not any(0 <= b.headline_index < len(items) for b in beats):      # the model gave no pairing
        target = next((b for b in beats if b.role == "catalyst"), beats[min(1, len(beats) - 1)])
        target.headline_index = 0
    popups: List[Popup] = []
    for beat, (s, _) in zip(beats, spans):
        if not (0 <= beat.headline_index < len(items)) or len(popups) >= MAX_POPUPS:
            continue
        card = render_headline_card(items[beat.headline_index], change_pct)
        if card is None:
            continue
        start = s + POPUP_DELAY
        end = min(start + POPUP_SECONDS, total - 0.1)
        if end - start >= 1.5:
            popups.append(Popup(start, end, card))
    for a, b in zip(popups, popups[1:]):                                # never overlap
        a.end = min(a.end, b.start - 0.15)
    return [p for p in popups if p.end - p.start >= 1.0]


class BottomPanel:
    """Avatar bottom-left (follows the voice volume); stats panel bottom-right, with headline pop-ups on top."""
    SPEAKING_THRESHOLD = 0.08  # lower = mouth moves more easily (useful range 0.02 - 0.15)

    def __init__(self, stats_img: np.ndarray, popups: List[Popup], audio_path: str):
        self.sidebar = stats_img
        self.popups = popups
        self.avatar = AvatarPanel(PANEL_SIZE)
        self.peaks = get_audio_peaks(audio_path)

    def paint_sidebar(self, frame: np.ndarray) -> None:
        frame[PANEL_Y:PANEL_Y + PANEL_SIZE, SIDEBAR_X:SIDEBAR_X + SIDEBAR_W] = self.sidebar

    def paint_popup(self, frame: np.ndarray, t: float) -> None:
        for pop in self.popups:
            if not pop.start <= t < pop.end:
                continue
            p_in = min(max((t - pop.start) / POPUP_SLIDE, 0.0), 1.0)
            p_out = min(max((t - (pop.end - POPUP_SLIDE)) / POPUP_SLIDE, 0.0), 1.0)
            dx = int(round(min(1.0, (1 - p_in) ** 2 + p_out ** 2) * SIDEBAR_W))   # slides in/out from the right
            if dx < SIDEBAR_W:
                frame[PANEL_Y:PANEL_Y + PANEL_SIZE, SIDEBAR_X + dx:SIDEBAR_X + SIDEBAR_W] = pop.card[:, :SIDEBAR_W - dx]
            return

    def paint_avatar(self, frame: np.ndarray, t: float) -> None:
        idx = int(t * OUTPUT_FPS)
        vol = float(self.peaks[idx]) if idx < len(self.peaks) else 0.0
        speaking = vol > self.SPEAKING_THRESHOLD
        bob_px = px(min(max(vol * 50, 4), 20)) if speaking else 0
        frame[PANEL_Y:PANEL_Y + PANEL_SIZE, AVATAR_X:AVATAR_X + PANEL_SIZE] = self.avatar.frame(speaking, bob_px)


# =====================================================================
# 9. THE SCENE  (candles draw in, live price tag, pulsing previous-close line)
# =====================================================================
def _x_tick_labels(df, count: int = 5) -> Tuple[np.ndarray, List[str]]:
    n = len(df)
    idxs = np.unique(np.linspace(0, n - 1, min(count, n)).astype(int))
    try:
        multi_day = len(set(df.index.date)) > 1
        fmt = "%b %d" if multi_day else "%H:%M"
        labels = [df.index[i].strftime(fmt) for i in idxs]
    except Exception:
        labels = [str(i) for i in idxs]
    return idxs, labels


class StockShortScene:
    def __init__(self, stock: dict, df, tf_label: str, ticker_text: str, subtitles: SubtitleTrack,
                 panel: BottomPanel, duration: float):
        self.duration = duration
        self.cache = FrameCache()
        self.subtitles = subtitles
        self.panel = panel

        opens = np.asarray(df["Open"].values, dtype=float).flatten()
        closes = np.asarray(df["Close"].values, dtype=float).flatten()
        highs = np.asarray(df["High"].values, dtype=float).flatten()
        lows = np.asarray(df["Low"].values, dtype=float).flatten()
        self.closes = closes
        self.n = len(df)
        x_idxs = np.arange(self.n)
        self.accent = C_GREEN if stock["change_pct"] >= 0 else C_RED
        colors = list(np.where(closes >= opens, C_GREEN, C_RED))
        y_min, y_max = float(lows.min()), float(highs.max())
        y_range = max(y_max - y_min, y_max * 0.002, 0.01)

        # previous close is the key level; if it is far away from the candles it would squash the chart
        prev = stock.get("prev_close")
        self.level = prev if prev and (y_min - 0.5 * y_range <= prev <= y_max + 0.5 * y_range) else None
        view_lo = min(y_min, self.level) if self.level else y_min
        view_hi = max(y_max, self.level) if self.level else y_max
        self.view_range = max(view_hi - view_lo, 0.01)

        # --- figure: everything is created ONCE; candles are revealed by changing visibility ---
        self.fig, self.canvas = new_figure()
        self.ticker = NewsTickerBar(self.fig, ticker_text)
        self.ax = self.fig.add_axes(axes_rect(CHART_BOX))
        style_axes(self.ax)
        self.segs = [[(float(i), lows[i]), (float(i), highs[i])] for i in range(self.n)]
        self.wicks = self.ax.vlines(x_idxs, lows, highs, color=colors, linewidth=1.4)
        body = closes - opens
        tiny = y_range * 0.002
        body = np.where(np.abs(body) < tiny, tiny, body)          # doji candles stay visible
        self.bodies = self.ax.bar(x_idxs, body, bottom=opens, color=colors, width=0.7)
        self.x_min = -1.0
        self.x_max = self.n - 1 + max(2, int(self.n * 0.05))
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(view_lo - self.view_range * 0.06, view_hi + self.view_range * 0.14)
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
        tick_idx, tick_labels = _x_tick_labels(df)
        self.ax.set_xticks(tick_idx)
        self.ax.set_xticklabels(tick_labels)

        label_x = self.x_max - (self.x_max - self.x_min) * 0.02
        self.level_line = self.level_label = None
        if self.level:
            self.level_line = self.ax.axhline(y=self.level, color=C_GOLD, linestyle="--", linewidth=2.2)
            self.level_label = self.ax.text(
                label_x, self.level, f"Prev Close: ${self.level:.2f}", color=C_GOLD, fontsize=14, fontweight="bold",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C_BG, edgecolor=C_GOLD, alpha=0.85))

        self.live_line = self.ax.axhline(y=closes[0], color=self.accent, linestyle=":", linewidth=1.6, alpha=0.6)
        self.halo, = self.ax.plot([], [], marker="o", markersize=20, linestyle="none", color=self.accent,
                                  markeredgewidth=0, alpha=0.3, zorder=9)
        self.dot, = self.ax.plot([], [], marker="o", markersize=8, linestyle="none", color=self.accent,
                                 markeredgewidth=0, zorder=10)
        self.tag = self.ax.text(label_x, closes[0], "", color=C_BG, fontsize=15, fontweight="bold", ha="right",
                                va="bottom", zorder=11,
                                bbox=dict(boxstyle="round,pad=0.25", facecolor=self.accent, edgecolor="none", alpha=0.95))
        self._label_x = label_x
        self._shown = None

        # --- header: ticker + company on the left, price + change on the right ---
        price = stock["current_price"] if stock["current_price"] > 0 else float(closes[-1])
        pct = stock["change_pct"]
        self.fig.text(fig_x(px(40)), fig_y(HEADER_TOP), stock["symbol"], fontsize=42, fontweight="bold",
                      color=C_WHITE, va="top")
        self.fig.text(fig_x(px(40)), fig_y(HEADER_TOP + px(76)), mpl_safe(f"{stock['company_name']}  |  {tf_label}"),
                      fontsize=20, color=C_MUTED, va="top")
        self.fig.text(fig_x(px(1040)), fig_y(HEADER_TOP), f"${price:,.2f}", fontsize=38, fontweight="bold",
                      color=C_WHITE, ha="right", va="top")
        self.fig.text(fig_x(px(1040)), fig_y(HEADER_TOP + px(70)), f"{'+' if pct >= 0 else ''}{pct:.2f}%",
                      fontsize=26, fontweight="bold", color=self.accent, ha="right", va="top")

    def _visible_candles(self, t: float) -> int:
        p = min(t / max(self.duration * REVEAL_FRACTION, 1e-6), 1.0)
        p = 1 - (1 - p) ** 2                                   # ease-out
        return min(self.n, max(2, int(round(p * self.n))))

    def _render_chart(self, t: float) -> np.ndarray:
        k = self._visible_candles(t)
        if k != self._shown:
            self.wicks.set_segments(self.segs[:k])
            for i, rect in enumerate(self.bodies):
                rect.set_visible(i < k)
            self._shown = k
        last = float(self.closes[k - 1])
        above = self.level is None or last >= self.level
        off = self.view_range * 0.015

        self.live_line.set_ydata([last, last])
        self.dot.set_data([k - 1], [last])
        self.halo.set_data([k - 1], [last])
        self.halo.set_alpha(0.12 + 0.28 * abs(math.sin(math.pi * t * 1.2)))
        self.tag.set_text(mpl_safe(f" ${last:,.2f} "))
        self.tag.set_position((self._label_x, last + (off if above else -off)))
        self.tag.set_va("bottom" if above else "top")
        if self.level_line is not None:
            self.level_line.set_alpha(0.45 + 0.45 * abs(math.sin(math.pi * t * 1.5)))
            self.level_label.set_position((self._label_x, self.level + (-off if above else off)))
            self.level_label.set_va("top" if above else "bottom")

        self.ticker.update(t)
        frame = grab(self.canvas)
        self.panel.paint_sidebar(frame)
        return frame

    def frame_at(self, t: float) -> np.ndarray:
        frame = self.cache.get(t, self._render_chart).copy()   # we paint on it, so own a copy
        self.panel.paint_popup(frame, t)
        self.panel.paint_avatar(frame, t)
        return self.subtitles.overlay(frame, t, inplace=True)


# =====================================================================
# 10. VIDEO RENDER  (frames piped straight to ffmpeg, voice + music mixed in the same pass)
# =====================================================================
def render_video(scene: StockShortScene, voice_path: str, total: float, out_path: str) -> None:
    n_frames = max(1, round(total * OUTPUT_FPS))
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(OUTPUT_FPS), "-i", "-",
           "-i", str(voice_path)]
    music_path = f"assets/music/{random.choice([1,2,3,4,5,6])}.mp3"
    have_music = bool(music_path) and os.path.exists(music_path)
    if have_music:
        logger.info("Mixing background music...")
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    # voice is mono: duplicate it to both channels (ffmpeg's default upmix would lose 3 dB)
    voice = (f"[1:a]aformat=sample_rates=44100:channel_layouts=mono,pan=stereo|c0=c0|c1=c0,"
             f"apad=whole_dur={total:.6f},atrim=duration={total:.6f},asetpts=PTS-STARTPTS")
    if have_music:
        fade_start = max(0.0, total - BG_MUSIC_FADE_OUT)
        graph = [voice + "[voice]",
                 f"[2:a]aformat=sample_rates=44100:channel_layouts=stereo,atrim=duration={total:.6f},"
                 f"asetpts=PTS-STARTPTS,volume={BG_MUSIC_VOLUME},afade=t=out:st={fade_start:.6f}:d={BG_MUSIC_FADE_OUT}[bg]",
                 # amix scales each input by 1/2, so volume=2 restores the original voice level
                 "[voice][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=2[aout]"]
    else:
        graph = [voice + "[aout]"]

    cmd += ["-filter_complex", ";".join(graph), "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "libx264", "-preset", X264_PRESET, "-b:v", VIDEO_BITRATE, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.6f}", str(out_path)]

    logger.info(f"Rendering {n_frames} frames ({total:.1f}s) -> {out_path}")
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                if i and i % (OUTPUT_FPS * 5) == 0:
                    logger.info(f"   rendered {i / OUTPUT_FPS:.0f}s / {total:.0f}s")
                proc.stdin.write(np.ascontiguousarray(scene.frame_at(i / OUTPUT_FPS)))
            proc.stdin.close()
            proc.wait()
        except BrokenPipeError:
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if proc.returncode != 0:
            err_file.seek(0)
            raise RuntimeError(f"ffmpeg failed: {err_file.read().decode(errors='ignore')}")


def extract_frame(video_path: str, t: float, out_png: str) -> Optional[str]:
    proc = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video_path,
                           "-frames:v", "1", out_png], capture_output=True)
    return out_png if proc.returncode == 0 and os.path.exists(out_png) else None


# =====================================================================
# 11. PIPELINE + ENTRY POINTS
# =====================================================================
def create_stock_short(upload: bool = False, test: bool = False,
                       on_preview: Optional[Callable[[str], None]] = None) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="stock_short_"))
    try:
        # ---- Phase A: data ----
        stock = get_top_moving_stock(test=test)
        symbol = stock["symbol"]
        with ThreadPoolExecutor(max_workers=3) as pool:
            news_f = pool.submit(fetch_news_context, symbol)
            name_f = pool.submit(get_company_name, symbol)
            stats_f = pool.submit(fetch_stock_stats, stock)
            news_text, news_items = news_f.result()
            stock["company_name"] = name_f.result()
            stats = stats_f.result()
        chart_df, tf_label = pick_chart_history(stock)

        # ---- Phase B: script + voice ----
        logger.info("Generating script...")
        script = generate_script_and_titles_with_ai(stock, stats, news_text, len(news_items))
        beats = script.beats
        logger.info("=" * 60)
        logger.info(f"YOUTUBE TITLE: {script.youtube_title}")
        for b in beats:
            logger.info(f"  [{b.role}] (headline {b.headline_index}) {b.text}")
        logger.info("=" * 60)

        audio_file = work_dir / "voiceover.wav"
        spans, speech_end = generate_voiceover([b.text for b in beats], audio_file)
        total = speech_end + END_PAD
        if total > SHORTS_MAX_SECONDS:
            logger.warning(f"Video is {total:.1f}s, longer than the {SHORTS_MAX_SECONDS:.0f}s Shorts limit!")

        # ---- Phase C: graphics + render ----
        popups = plan_popups(beats, spans, news_items, stock["change_pct"], total)
        subtitles = SubtitleTrack([(b.text, s, e) for b, (s, e) in zip(beats, spans)], total)
        stats_img = render_stats_panel(build_stat_rows(stats), symbol, stock["change_pct"])
        panel = BottomPanel(stats_img, popups, str(audio_file))
        ticker_text = clean_ticker_text(script.news_ticker_text) or fallback_ticker_text(stock, news_items)
        scene = StockShortScene(stock, chart_df, tf_label, ticker_text, subtitles, panel, total)

        output_filename = f"STOCK_{symbol}_{int(stock['change_pct'])}pct.mp4"
        render_video(scene, str(audio_file), total, output_filename)
        scene.fig.clear()
        del scene, panel, subtitles
        gc.collect()
        logger.info(f"SUCCESS! Video saved as: {output_filename} ({total:.1f}s)")

        if on_preview:
            t_prev = (popups[0].start + popups[0].end) / 2 if popups else total / 2
            png = extract_frame(output_filename, t_prev, str(work_dir / "preview.png"))
            if png:
                on_preview(png)

        # ---- Phase D: upload ----
        if upload:
            from uploader import upload_video  # deferred until actually needed
            description = script.description.strip()
            if "#shorts" not in description.lower():
                description += "\n\n#Shorts"
            tags = [t.strip() for t in script.tags.split(",") if t.strip()]
            upload_video(file_path=output_filename, title=script.youtube_title, description=description, tags=tags, category_id="25")
            logger.info(f"Uploaded To YouTube: {script.youtube_title}")
        return output_filename
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    create_stock_short(upload=False)