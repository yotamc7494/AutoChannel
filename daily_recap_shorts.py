"""
daily_recap_shorts.py - retention-first VERTICAL (9:16) daily market recap Short, driven by the same
"director" logic as weekly_recap.py.

Pipeline
  A. Data + news  ->  Director LLM (story angle, 2-4 chapters with a screen plan + time budget, thumbnail idea,
     YouTube metadata)  ->  Writer LLM (one call PER chapter)  ->  master timeline of screens ("segments")
  B. Pre-fetch of everything the timeline shows (yfinance) BEFORE rendering. Every number spoken or shown for a
     ticker comes from verified data via [[TICKER:day]] style placeholders, never from the LLM's memory.
  C. TTS (Kokoro ONNX), synthesised sentence by sentence so every screen knows when each sentence is spoken
     (used to flash the right card / sector / ticker / headline and to time the captions).
     HARD TIME LIMIT: the Short never exceeds SHORTS_MAX_SECONDS (50 s). If the voice runs long, the least valuable
     screens are cut; if that is still not enough the voice is sped up slightly.
  D. Render: one isolated OS process per chapter; every scene is an MP4 chunk that already contains its audio
     (TTS + SFX at t=0).
  E. Assemble: ffmpeg concat demuxer (stream copy) + optional ducked background music.
  F. Thumbnail: drawn by thumbnail.py (Pillow, 1080x1920) once the video is saved.

Screens the Director/Writers can cut between (all 9:16, all driven by verified data)
  market_pulse, pro_candlestick_chart, stock_dashboard, sector_heatmap, top_movers, news_flash, plain_text, outro

Usage
  python daily_recap_shorts.py [--upload] [--output out.mp4] [--news-file notes.txt] [--work-dir DIR]
                               [--save-script s.json] [--load-script s.json] [--no-music]
  Streamlit / other callers keep using run_perfect_pipeline(output_filename, upload) exactly as before.
Env
  GEMINI_API_KEY, PERFECT_VIDEO_WIDTH=540|720|1080 (default 720), RECAP_RENDER_WORKERS (default 1),
  RECAP_RENDER_FPS (default 15), RECAP_IO_WORKERS (network threads, default 1; 3-4 makes the data phase faster)
Optional assets (never fatal if missing)
  assets/avatar/open.png, assets/avatar/closed.png, assets/music/*.mp3, assets/sfx/<name>.wav|mp3,
  assets/logos/<TICKER>.png (company logos are also downloaded on a best-effort basis)
Needs thumbnail.py next to this file (only for the thumbnail step; the video does not depend on it).
"""
from __future__ import annotations

import argparse
import datetime
import gc
import inspect
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_k):
        return False

try:
    import httpx  # installed with google-genai; used to recognise network errors
except ImportError:  # pragma: no cover
    httpx = None

load_dotenv()
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("PerfectDailyShorts")

# =====================================================================
# 1. CONFIG
# =====================================================================
# --- LLM ---
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]  # tried in order, every round
LLM_MAX_ROUNDS = 4
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
CONTEXT_MAX_CHARS = 9000

# --- Video (9:16 vertical, YouTube Shorts) ---
# Low-RAM default (2.25x fewer pixels than 1080x1920). PERFECT_VIDEO_WIDTH=1080 restores full resolution.
VIDEO_W = int(os.environ.get("PERFECT_VIDEO_WIDTH", "720"))
if VIDEO_W not in (540, 720, 1080):
    raise ValueError("PERFECT_VIDEO_WIDTH must be 540, 720 or 1080")
VIDEO_H = VIDEO_W * 16 // 9
UI_SCALE = VIDEO_W / 1080  # every layout pixel is authored for a 1080x1920 grid and scaled by this
DPI = VIDEO_W / 10.8
REF_PT = 0.72  # 1 reference pixel (1080 grid) in matplotlib points, at ANY size
OUTPUT_FPS = 30
RENDER_FPS = int(os.environ.get("RECAP_RENDER_FPS", "15"))  # heavy matplotlib layer redraw rate (frames repeat)
VIDEO_BITRATE = "4500k" if VIDEO_W <= 720 else "8000k"
X264_PRESET = "ultrafast"
BG_MUSIC_VOLUME = 0.12
BG_MUSIC_FADE_OUT = 2.0
USE_MUSIC = True

# --- HARD time limit + pacing. The story decides the screens; the clock decides how many survive. ---
SHORTS_MAX_SECONDS = 50.0  # the finished video is never longer than this
TIME_SAFETY = 0.4  # planning margin under the hard limit (frame rounding, container padding)
TARGET_SPEECH_SECONDS = 41.0  # what the Director plans for (narration only)
WORDS_PER_SECOND = 3.0  # Kokoro am_liam at 1.1x, used for word budgets
CHAPTERS_MIN, CHAPTERS_MAX = 2, 4
SEG_PER_CHAPTER_MIN, SEG_PER_CHAPTER_MAX = 1, 3
SEG_WORDS_MIN, SEG_WORDS_MAX = 4, 42  # safety bounds for broken output, NOT a target
CHAPTER_MIN_SECONDS, CHAPTER_MAX_SECONDS = 6, 20
CHAPTER_MAX_ROUNDS = 3
MIN_SCREENS = 3  # the time-fit editor never cuts below this many screens
SEGMENT_PAD = 0.12  # silence after every screen
END_PAD = 0.7  # hold on the closing screen after the last word
SENTENCE_GAP = 0.10  # silence between sentences inside one segment
MAX_VOICE_SPEED = 1.4
SPEEDUP_BEFORE_CUT = 1.12  # a small overshoot is fixed by a slightly faster voice; only bigger ones cut screens

# --- Parallelism / memory ---
IO_WORKERS = int(os.environ.get("RECAP_IO_WORKERS", "1"))
RENDER_IN_SUBPROCESS = True
RENDER_WORKERS = int(os.environ.get("RECAP_RENDER_WORKERS", "1"))
FFMPEG_THREADS = 1
LOG_EVERY_SECONDS = 8

# --- Animation speeds (seconds). Everything builds fast; the voice, not the animation, sets the pace. ---
CHART_BUILD_SECONDS = 0.45
PULSE_BUILD_SECONDS = 1.8

# --- Voice (Kokoro) ---
HOST_VOICE, HOST_SPEED = "am_liam", 1.1
SFX_GAIN = 0.2

# --- Screens ---
LAYOUTS = ("market_pulse", "pro_candlestick_chart", "stock_dashboard", "sector_heatmap", "top_movers",
           "news_flash", "plain_text", "outro")
DEFAULT_HEADLINES = {
    "market_pulse": "MARKET PULSE", "pro_candlestick_chart": "THE CHART", "stock_dashboard": "BY THE NUMBERS",
    "sector_heatmap": "SECTOR MAP", "top_movers": "BIGGEST MOVERS", "news_flash": "BREAKING NEWS",
    "plain_text": "THE TAKEAWAY", "outro": "SEE YOU TOMORROW",
}
# Keep-score used when the clock forces screens to be cut (higher = more valuable, cut last)
LAYOUT_KEEP_SCORE = {"pro_candlestick_chart": 7, "top_movers": 7, "market_pulse": 6, "sector_heatmap": 6,
                     "stock_dashboard": 5, "news_flash": 4, "plain_text": 3}
CANDLE_SPECS = {"1d": ("1d", "5m"), "5d": ("5d", "30m"), "1mo": ("1mo", "1d"), "3mo": ("3mo", "1d")}
CANDLE_FALLBACK_TF = "1mo"
INDEX_SYMBOLS = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
                 "^VIX": "VIX volatility index", "BTC-USD": "Bitcoin"}
# market_pulse benchmarks: key -> (yfinance symbol, short label, colour)
PULSE_ASSETS = {"SPY": ("SPY", "S&P 500", "#00FFA3"), "QQQ": ("QQQ", "NASDAQ", "#00E5FF"),
                "DIA": ("DIA", "DOW", "#B388FF"), "IWM": ("IWM", "SMALL CAPS", "#FF9100"),
                "BTC": ("BTC-USD", "BITCOIN", "#FFB800")}
PULSE_DEFAULT = ["SPY", "QQQ", "BTC"]
SECTOR_ETFS = {"XLK": "TECH", "XLF": "FINANCIALS", "XLE": "ENERGY", "XLV": "HEALTH CARE", "XLY": "DISCRETIONARY",
               "XLP": "STAPLES", "XLI": "INDUSTRIALS", "XLU": "UTILITIES", "XLRE": "REAL ESTATE",
               "XLB": "MATERIALS", "XLC": "COMM SERVICES"}
ETF_ALIASES = {
    "XLK": ["technology", "tech"], "XLF": ["financials", "financial", "banks", "banking"], "XLE": ["energy", "oil"],
    "XLV": ["health care", "healthcare", "health"], "XLY": ["consumer discretionary", "discretionary"],
    "XLP": ["consumer staples", "staples"], "XLI": ["industrials", "industrial"], "XLU": ["utilities", "utility"],
    "XLRE": ["real estate", "property"], "XLB": ["materials"],
    "XLC": ["communication services", "communications", "communication"],
}
SECTOR_WEIGHTS = {"XLK": 32, "XLF": 13, "XLV": 10, "XLY": 10, "XLC": 9, "XLI": 8, "XLP": 6, "XLE": 3.5,
                  "XLU": 2.5, "XLRE": 2, "XLB": 2}

# --- Data ---
RSS_QUERIES = [
    "SPY QQQ stock market rally crash Fed rates",
    "S&P 500 Nasdaq stock market today close",
    "Nvidia AMD TSMC AI chip stock news",
    "Tesla Apple Microsoft Amazon Meta stock news",
    "top stock gainers losers earnings surprise market news",
    "Bitcoin crypto market price surge MicroStrategy Coinbase",
]
WATCHLIST_CANDIDATES = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",  # mega-cap tech
    "AMD", "AVGO", "TSM", "ARM", "MU", "SMCI",  # semiconductors & AI
    "PLTR", "COIN", "MSTR", "MARA", "HOOD", "NFLX", "ORCL",  # high-beta & retail favourites
    "LLY", "CRM", "DIS", "JPM", "BABA",  # market leaders
]
# ticker -> spoken company name: topic checks, headline->ticker matching, TTS
COMPANY_NAMES: Dict[str, str] = {
    "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta",
    "TSLA": "Tesla", "AMD": "AMD", "AVGO": "Broadcom", "TSM": "TSMC", "ARM": "Arm Holdings", "MU": "Micron",
    "SMCI": "Super Micro", "PLTR": "Palantir", "COIN": "Coinbase", "MSTR": "MicroStrategy",
    "MARA": "Marathon Digital", "HOOD": "Robinhood", "NFLX": "Netflix", "ORCL": "Oracle", "LLY": "Eli Lilly",
    "CRM": "Salesforce", "DIS": "Disney", "JPM": "JPMorgan", "BABA": "Alibaba",
    "INTC": "Intel", "BA": "Boeing", "WMT": "Walmart", "COST": "Costco", "XOM": "Exxon Mobil", "UBER": "Uber",
    "NKE": "Nike", "PFE": "Pfizer", "KO": "Coca Cola", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "JNJ": "Johnson and Johnson", "UNH": "UnitedHealth", "QCOM": "Qualcomm", "SHOP": "Shopify",
    "SNOW": "Snowflake", "PYPL": "PayPal",
}
MAX_EXTRA_TICKERS = 6  # tickers found in the headlines that are not on the watchlist
TTS_OVERRIDES = {"GOOGL": "Google", "GOOG": "Google", "BTC": "Bitcoin", "SPY": "S and P 500", "QQQ": "Nasdaq",
                 "TSM": "T S M C", "AMD": "A M D", "MSTR": "MicroStrategy", "VIX": "the VIX", "ETF": "E T F",
                 "CPI": "C P I", "FOMC": "the Fed meeting", "GDP": "G D P", "IPO": "I P O", "AI": "A I"}


def _build_tts_names() -> Dict[str, str]:
    out = {rf"\b{k}\b": v for k, v in TTS_OVERRIDES.items()}
    for t, name in COMPANY_NAMES.items():
        if len(t) >= 3 and rf"\b{t}\b" not in out:
            out[rf"\b{t}\b"] = name
    return out


TTS_SYMBOL_NAMES = _build_tts_names()

# --- Palette ---
C_BG, C_PANEL, C_BORDER = "#0B0E14", "#161B22", "#30363D"
C_WHITE, C_MUTED = "#FFFFFF", "#8B949E"
C_GREEN, C_RED, C_CYAN, C_GOLD, C_AMBER = "#00FFA3", "#FF3366", "#00E5FF", "#FFD700", "#FFB800"
C_UP, C_DOWN = "#26A69A", "#EF5350"
COLOR_THEMES = {"red": C_RED, "green": C_GREEN, "gold": C_GOLD, "cyan": C_CYAN, "amber": C_AMBER, "white": C_WHITE,
                "purple": "#B388FF", "blue": "#448AFF", "orange": "#FF9100"}
STAGE_FACE = (11 / 255, 14 / 255, 20 / 255, 0.62)  # translucent so the animated background shows through
PANEL_FACE = (22 / 255, 27 / 255, 34 / 255, 0.90)

AVATAR_OPEN_PATH = "assets/avatar/open.png"
AVATAR_CLOSED_PATH = "assets/avatar/closed.png"
LOGO_DIR = Path("assets/logos")
THUMB_CANDLES_KEPT = 60  # candles kept for the thumbnail (the engine merges them down to ~18 chunky ones)


def px(v: float) -> int:
    return int(round(v * UI_SCALE))


# --- Layout boxes, authored for 1080x1920 (x, y_top, w, h) ---
TICKER_BOX = tuple(px(v) for v in (40, 100, 1000, 65))
HEADER_TOP = px(188)
HEADER_SUB_TOP = px(262)
STAGE_BOX = tuple(px(v) for v in (40, 335, 1000, 800))
STAGE_REF = (1000, 800)  # stage size in reference pixels
SUB_BOX = tuple(px(v) for v in (40, 1160, 1000, 170))  # captions
OUTRO_SUB_BOX = tuple(px(v) for v in (40, 1160, 1000, 170))
INFO_BOX = tuple(px(v) for v in (400, 1360, 640, 330))
INFO_REF = (640, 330)
AVATAR_X, AVATAR_Y, AVATAR_SIZE = px(40), px(1360), px(330)
FULL_REF = (1080, 1920)


def axes_rect(box: Tuple[int, int, int, int]) -> List[float]:
    """Pixel box (x, y_top, w, h) -> matplotlib figure-fraction rect [left, bottom, w, h]."""
    x, y, w, h = box
    return [x / VIDEO_W, 1 - (y + h) / VIDEO_H, w / VIDEO_W, h / VIDEO_H]


def fig_x(x_px: float) -> float:
    return x_px / VIDEO_W


def fig_y(y_px_from_top: float) -> float:
    return 1 - y_px_from_top / VIDEO_H


def px_box(ref_box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    return tuple(px(v) for v in ref_box)


# =====================================================================
# 1b. SCHEMAS  (Director -> Outline, Writer -> ChapterScript, master timeline)
# =====================================================================
class _Model(BaseModel):
    model_config = ConfigDict(use_enum_values=True, validate_default=True)


class LayoutType(str, Enum):
    market_pulse = "market_pulse"
    pro_candlestick_chart = "pro_candlestick_chart"
    stock_dashboard = "stock_dashboard"
    sector_heatmap = "sector_heatmap"
    top_movers = "top_movers"
    news_flash = "news_flash"
    plain_text = "plain_text"
    outro = "outro"


class SfxType(str, Enum):
    none = "none"
    swoosh = "swoosh"
    cash_register = "cash_register"
    heartbeat = "heartbeat"


class AccentColor(str, Enum):
    red = "red"
    green = "green"
    gold = "gold"
    cyan = "cyan"
    amber = "amber"
    white = "white"
    purple = "purple"


class CandleTimeframe(str, Enum):
    d1 = "1d"  # today's session, 5-minute candles
    d5 = "5d"  # five sessions, 30-minute candles
    m1 = "1mo"  # one month, daily candles
    m3 = "3mo"  # three months, daily candles


class NoteAnchor(str, Enum):
    high = "high"
    low = "low"
    start = "start"
    end = "end"
    biggest_drop = "biggest_drop"


class HighlightMetric(str, Enum):
    none = "none"
    price = "price"
    change = "change"
    week_change = "week_change"
    market_cap = "market_cap"
    pe_ratio = "pe_ratio"
    dividend_yield = "dividend_yield"


class MoversSide(str, Enum):
    both = "both"
    gainers = "gainers"
    losers = "losers"


class PulseAsset(str, Enum):
    SPY = "SPY"
    QQQ = "QQQ"
    DIA = "DIA"
    IWM = "IWM"
    BTC = "BTC"


class ChartNote(_Model):
    anchor: NoteAnchor = Field(default=NoteAnchor.high)
    text: str = Field(description="Annotation text, max 3 words")


class Segment(_Model):
    layout: LayoutType = Field(description="Which screen to show for this segment")
    headline: str = Field(description="Short on-screen title, UPPERCASE, max 22 characters")
    script: str = Field(description="Spoken narration: 1-3 short punchy sentences, only what the screen needs")
    accent: AccentColor = Field(default=AccentColor.gold)
    sfx: SfxType = Field(default=SfxType.none, description="Special sound effect at the start (a swoosh is added automatically)")
    caption: str = Field(default="", description="LEAVE EMPTY. Filled by code (on-screen caption)")

    # --- market_pulse ---
    pulse_assets: List[PulseAsset] = Field(default_factory=list, description="1 to 3 benchmarks, e.g. SPY, QQQ, BTC")
    # --- pro_candlestick_chart ---
    chart_ticker: str = Field(default="", description="Real liquid ticker or index, e.g. NVDA, SPY, ^GSPC")
    chart_timeframe: CandleTimeframe = Field(default=CandleTimeframe.d1)
    chart_notes: List[ChartNote] = Field(default_factory=list, description="0 to 2 annotations")
    # --- stock_dashboard ---
    dash_ticker: str = Field(default="", description="Real INDIVIDUAL company ticker (never an index)")
    highlight_metric: HighlightMetric = Field(default=HighlightMetric.none,
                                              description="Fallback card to glow if the script names none")
    # --- top_movers ---
    movers_side: MoversSide = Field(default=MoversSide.both, description="both = two boards, or one side only")
    movers_gainers: List[str] = Field(default_factory=list, description="0-3 gainer tickers from the supplied data")
    movers_losers: List[str] = Field(default_factory=list, description="0-3 loser tickers from the supplied data")
    # --- news_flash ---
    news_headlines: List[str] = Field(default_factory=list,
                                      description="1 to 3 headlines COPIED EXACTLY from the supplied headlines")
    # --- plain_text ---
    plain_text: str = Field(default="", description="Strong statement shown centred (max 18 words)")
    plain_highlight: List[str] = Field(default_factory=list, description="Up to 3 words from plain_text")


class ChapterPlan(_Model):
    title: str = Field(description="Chapter title, max 5 words")
    summary: str = Field(description="What this chapter covers and which story it tells, 1-2 sentences")
    key_tickers: List[str] = Field(default_factory=list, description="Tickers this chapter is about")
    talking_points: List[str] = Field(description="2 to 4 facts or angles taken from the supplied data/news")
    screens: List[LayoutType] = Field(default_factory=list,
                                      description="The 1-3 screens that tell this chapter best, in order")
    seconds: int = Field(default=12, description="Spoken seconds for this chapter (6-20); all chapters together ~41")


class ThumbnailIdea(BaseModel):
    """The Director's idea for the YouTube thumbnail. The featured stock, the colours, the % badge and the candles are
    decided / verified in code (see plan_thumbnail + thumbnail.py); the Director only picks the angle and the text."""
    template: str = Field(description="'candle_hero' (real candlestick chart + logo + % badge + surprised face) or "
                                      "'massive_number' (giant headline only)")
    ticker: str = Field(description="Ticker of the ONE stock to feature, taken from the supplied data")
    main_text: str = Field(description="MAX 3 WORDS, UPPERCASE clickbait about that stock, e.g. 'NVIDIA EXPLODES?'. "
                                       "Must match the direction of its verified move. No emojis, dollar signs or numbers.")
    avatar_emotion: str = Field(description="'shocked', 'pointing', 'thinking' or 'neutral'")


THUMBNAIL_FALLBACK = {"template": "candle_hero", "ticker": "", "main_text": "MARKET MOVERS",
                      "avatar_emotion": "shocked"}


class DailyOutline(_Model):
    date_label: str = Field(description="e.g. 'FRI OCT 30'")
    story_angle: str = Field(description="The ONE tension or question that defines today's market, 1 sentence")
    hook: str = Field(description="Specific, irresistible cold-open idea for the first 3 seconds")
    chapters: List[ChapterPlan]
    news_ticker_text: str = Field(
        description="3 UPPERCASE items 'TOPIC: ultra short summary' joined by ' \u2022 '")
    youtube_title: str
    description: str
    tags: str
    thumbnail: ThumbnailIdea = Field(description="Idea for the YouTube thumbnail (see the THUMBNAIL rules)")

    @model_validator(mode="before")
    @classmethod
    def fill_missing_thumbnail(cls, data: Any) -> Any:
        """A response without a thumbnail idea must never cost the whole script: use a neutral default."""
        if isinstance(data, dict) and not data.get("thumbnail"):
            logger.warning("Outline has no thumbnail idea; using the default")
            data = {**data, "thumbnail": dict(THUMBNAIL_FALLBACK)}
        return data


class ChapterScript(_Model):
    segments: List[Segment]


class TimelineItem(_Model):
    chapter_index: int
    chapter_title: str
    segment: Segment


class MasterTimeline(_Model):
    outline: DailyOutline
    items: List[TimelineItem]
    headlines: List[str] = Field(default_factory=list)  # the only quotable sources (news_flash is checked against them)


# Heavy dependencies are imported only where used.
@lru_cache(maxsize=1)
def _yf():
    import yfinance as yf
    return yf


def load_render_dependencies() -> None:
    global Figure, FigureCanvasAgg, Rectangle, FancyBboxPatch, FuncFormatter, Circle
    if "Figure" not in globals():
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
        from matplotlib.ticker import FuncFormatter


# =====================================================================
# 2. MEMORY CONTROL / PROCESS ISOLATION / GENERIC HELPERS
# =====================================================================
def log_memory(stage: str) -> None:
    """Linux RSS + total cgroup usage (including ffmpeg and other processes)."""
    try:
        rss_mib = None
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_mib = int(line.split()[1]) / 1024
                    break
        cur, lim = Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")
        total = int(cur.read_text()) / (1024 ** 2) if cur.exists() else None
        raw_lim = lim.read_text().strip() if lim.exists() else "max"
        max_mib = int(raw_lim) / (1024 ** 2) if raw_lim.isdecimal() else None
        fmt = lambda n: f"{n:.0f} MiB" if n is not None else "N/A"
        logger.info("RAM [%s] process=%s container=%s limit=%s", stage, fmt(rss_mib), fmt(total), fmt(max_mib))
    except Exception as e:
        logger.debug("RAM logging unavailable: %s", e)


def run_memory_worker(kind: str, *args: str) -> None:
    """Start one isolated worker, wait for exit, reclaim its RAM before the next stage."""
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [sys.executable, str(Path(__file__).resolve()), f"--{kind}-worker", *map(str, args)]
    log_memory(f"before {kind} worker")
    logger.info("Starting isolated %s worker", kind)
    completed = subprocess.run(cmd, env=env, check=False)
    gc.collect()
    log_memory(f"after {kind} worker")
    if completed.returncode:
        msg = " (OS killed it; likely OOM)" if completed.returncode in (-9, 137) else ""
        raise RuntimeError(f"{kind} worker exit code {completed.returncode}{msg}; check the logs")


def with_retries(fn: Callable, attempts: int = 3, base_delay: float = 1.0, what: str = "operation"):
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


def parallel_map(fn: Callable, items: Sequence, max_workers: int = IO_WORKERS) -> list:
    """ThreadPool map that preserves order (use for network-bound work)."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as ex:
        return list(ex.map(fn, items))


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


def write_wav_mono(path: str, samples: np.ndarray, sr: int) -> None:
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def ev(x: Any) -> str:
    """Enum member or plain value -> plain string."""
    return str(x.value if isinstance(x, Enum) else x)


def mpl_safe(text: str) -> str:
    return text.replace("$", r"\$")


def ascii_clean(text: str) -> str:
    text = text or ""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        text = text.replace(src, dst)
    text = text.replace("\\", "")
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", "", text)).strip()


def shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:max(1, n - 3)].rstrip() + "..."


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def first_sentence(text: str) -> str:
    parts = split_sentences(text)
    return parts[0] if parts else (text or "").strip()


def theme_color(name: Any, default: str = C_GOLD) -> str:
    return COLOR_THEMES.get(ev(name or "").strip().lower(), default)


def norm_color_name(name: Any, default: str = "gold") -> str:
    n = ev(name or "").strip().lower()
    return n if n in COLOR_THEMES else default


def _rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def blend_hex(fg: str, bg: str, k: float) -> str:
    """Mix two hex colours; k is the share of `bg` (0 = pure fg, 1 = pure bg)."""
    a, b = _rgb(fg), _rgb(bg)
    return "#%02x%02x%02x" % tuple(int(round(x * (1 - k) + y * k)) for x, y in zip(a, b))


def clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def ramp(t: float, start: float, length: float) -> float:
    return clamp01((t - start) / max(length, 1e-6))


def ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def ease_in_out(p: float) -> float:
    return 3 * p * p - 2 * p ** 3


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def fmt_big(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    for lim, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(v) >= lim:
            return f"${v / lim:,.2f}{suf}"
    return f"${v:,.0f}"


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# =====================================================================
# 3. LLM  (model rotation + exponential backoff + retry on unusable output)
# =====================================================================
class LLMUnavailableError(RuntimeError):
    """Every model kept failing after all retry rounds."""


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing!")
    from google import genai
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
    """Each round tries every model once; transient errors move on, permanent ones drop the model.
    `parse` failures (bad JSON, wrong length, ...) are retried too."""
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
            logger.info(f"Retrying in {delay:.1f}s (round {round_no + 1}/{max_rounds})...")
            time.sleep(delay)
    raise LLMUnavailableError(
        f"Gemini still failing after {max_rounds} rounds across {GEMINI_MODELS}. Last error: {last_exc}") from last_exc


# =====================================================================
# 4. MARKET DATA  (pre-fetch, memoised, strict fallbacks)
# =====================================================================
def clean_ticker(t: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.\-^=]", "", str(t or "")).upper()[:12]


def normalize_timeframe(tf: Any) -> str:
    t = re.sub(r"\s+", "", ev(tf or "").lower())
    t = {"1w": "5d", "1wk": "5d", "1week": "5d", "week": "5d", "1m": "1mo", "1month": "1mo", "month": "1mo",
         "3m": "3mo", "3month": "3mo", "6mo": "3mo", "1y": "3mo", "1yr": "3mo", "year": "3mo", "1day": "1d",
         "day": "1d", "today": "1d"}.get(t, t)
    return t if t in CANDLE_SPECS else "1mo"


def _clean_df(df, need_volume: bool = True):
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if need_volume and "Volume" in df.columns:
        df = df.assign(Volume=df["Volume"].fillna(0))
    return df


_DAY_MEMO: Dict[str, dict] = {}


def fetch_day_perf(ticker: str) -> Optional[dict]:
    """Memoised: one daily-history call gives today's move, the 5-session move and the previous close.
    Only successful downloads are cached; failures are retried on the next call."""
    if ticker not in _DAY_MEMO:
        data = _fetch_day_perf_uncached(ticker)
        if data:
            _DAY_MEMO[ticker] = data
        return data
    return _DAY_MEMO[ticker]


def _fetch_day_perf_uncached(ticker: str) -> Optional[dict]:
    def _go():
        df = _clean_df(_yf().Ticker(ticker).history(period="1mo", interval="1d"), need_volume=False)
        if len(df) < 3:
            raise RuntimeError(f"only {len(df)} rows")
        return df

    try:
        df = with_retries(_go, attempts=3, base_delay=2.0, what=f"{ticker} daily history")
    except Exception as e:
        logger.warning(f"No daily data for {ticker}: {e}")
        return None
    closes = df["Close"].astype(float).tolist()
    win = min(5, len(closes) - 1)
    base, last, prev = closes[-win - 1], closes[-1], closes[-2]
    if base <= 0 or prev <= 0:
        return None
    return {"ticker": ticker, "last": last, "prev_close": prev, "base": base,
            "day_pct": (last / prev - 1) * 100, "pct": (last / base - 1) * 100,
            "end_date": str(df.index[-1].date())}


def _fetch_candles_once(ticker: str, tf: str) -> Optional[dict]:
    period, interval = CANDLE_SPECS[tf]

    def _go():
        df = _clean_df(_yf().Ticker(ticker).history(period=period, interval=interval))
        if tf == "1d" and len(df):  # keep only the most recent session
            last_day = df.index[-1].date()
            df = df[[d.date() == last_day for d in df.index]]
        if len(df) < 8:
            raise RuntimeError(f"only {len(df)} rows")
        return df

    try:
        logger.info(f"Fetching candles: {ticker} {tf} ({interval})")
        df = with_retries(_go, attempts=2, base_delay=1.5, what=f"{ticker} {tf} candles")
    except Exception as e:
        logger.warning(f"No {tf} candles for {ticker}: {e}")
        return None
    n = len(df)
    idxs = np.unique(np.linspace(0, n - 1, min(5, n)).astype(int))
    fmt = {"1d": "%H:%M", "5d": "%a %H:%M", "1mo": "%b %d", "3mo": "%b %d"}[tf]
    try:
        labels = [df.index[i].strftime(fmt) for i in idxs]
    except Exception:
        labels = [str(i) for i in idxs]
    vol = df["Volume"].astype(float).tolist() if "Volume" in df.columns else [0.0] * n
    return {"ticker": ticker, "timeframe": tf, "interval": interval,
            "open": df["Open"].astype(float).tolist(), "high": df["High"].astype(float).tolist(),
            "low": df["Low"].astype(float).tolist(), "close": df["Close"].astype(float).tolist(),
            "volume": vol, "tick_idx": [int(i) for i in idxs], "tick_labels": labels}


def fetch_candles(ticker: str, requested_tf: str) -> Optional[dict]:
    """Anything empty falls back to 1 month of daily candles. The reference close for the on-screen % change is the
    same one the narration placeholders use (previous close for 1d, the close before the window for 5d)."""
    chain = [requested_tf] + ([CANDLE_FALLBACK_TF] if requested_tf != CANDLE_FALLBACK_TF else [])
    for tf in chain:
        data = _fetch_candles_once(ticker, tf)
        if not data:
            continue
        data["requested_timeframe"] = requested_tf
        data["fallback"] = tf != requested_tf
        perf = fetch_day_perf(ticker)
        if tf == "1d" and perf:
            data["ref_close"], data["ref_label"] = perf["prev_close"], "vs previous close"
        elif tf == "5d" and perf:
            data["ref_close"], data["ref_label"] = perf["base"], "vs last week's close"
        else:
            data["ref_close"], data["ref_label"] = data["open"][0], "over the period"
        if data["fallback"]:
            logger.warning(f"{ticker}: {requested_tf} data unavailable, fell back to {tf} daily candles")
        return data
    return None


def fetch_pulse_series(key: str, points: int = 160) -> Optional[dict]:
    """Intraday % curve of one benchmark versus the PREVIOUS CLOSE, ending exactly on the verified day move."""
    symbol, label, color = PULSE_ASSETS[key]
    perf = fetch_day_perf(symbol)
    if not perf:
        return None
    try:
        df = with_retries(lambda: _yf().Ticker(symbol).history(period="1d", interval="5m"), attempts=2,
                          base_delay=1.5, what=f"{symbol} intraday")
        vals = df["Close"].dropna().values.astype(float)
    except Exception as e:
        logger.warning(f"No intraday data for {symbol}: {e}")
        return None
    if len(vals) < 6:
        return None
    y = (vals / perf["prev_close"] - 1) * 100.0
    y = np.interp(np.linspace(0, len(y) - 1, points), np.arange(len(y)), y)
    k = np.ones(3) / 3
    y = np.convolve(np.pad(y, (1, 1), mode="edge"), k, mode="valid")
    y = y + (perf["day_pct"] - y[-1]) * np.linspace(0, 1, points) ** 3  # land exactly on the verified number
    return {"key": key, "symbol": symbol, "label": label, "color": color, "y": [float(v) for v in y],
            "pct": float(perf["day_pct"]), "last": float(perf["last"])}


def fetch_stock_info(ticker: str) -> Optional[dict]:
    """Price / changes from daily history (reliable); market cap, P/E, dividend yield from Yahoo (best effort).
    Anything Yahoo does not provide stays None, and the dashboard simply leaves that card out."""
    perf = fetch_day_perf(ticker)
    if not perf:
        return None
    info: dict = {}
    try:
        info = with_retries(lambda: dict(_yf().Ticker(ticker).info or {}), attempts=2, base_delay=1.5,
                            what=f"{ticker} info")
    except Exception as e:
        logger.warning(f"No fundamentals for {ticker}: {e}")
    price = perf["last"]
    mcap = _num(info.get("marketCap"))
    if not mcap:
        try:
            mcap = _num(_yf().Ticker(ticker).fast_info["market_cap"])
        except Exception:
            mcap = None
    pe = _num(info.get("trailingPE")) or _num(info.get("forwardPE"))
    rate = _num(info.get("dividendRate")) or _num(info.get("trailingAnnualDividendRate"))
    dy = rate / price * 100 if rate and price else None
    if dy is None and _num(info.get("trailingAnnualDividendYield")):
        dy = _num(info.get("trailingAnnualDividendYield")) * 100
    name = COMPANY_NAMES.get(ticker) or ascii_clean(str(info.get("shortName") or info.get("longName") or ticker))
    return {"ticker": ticker, "name": name, "price": price, "change_pct": perf["day_pct"],
            "week_pct": perf["pct"], "market_cap": mcap, "pe": pe, "div_yield": dy}


_LOGO_MISS: set = set()


def ensure_logo(ticker: str) -> Optional[str]:
    """Best effort; a coloured placeholder is drawn when nothing is found."""
    ticker = clean_ticker(ticker)
    for name in (f"{ticker}.png", f"{ticker.lower()}.png"):
        if (LOGO_DIR / name).exists():
            return str(LOGO_DIR / name)
    if not ticker or ticker.startswith("^") or ticker in _LOGO_MISS:
        return None
    try:
        import requests
        r = requests.get(f"https://financialmodelingprep.com/image-stock/{ticker}.png", timeout=6)
        if r.ok and len(r.content) > 600 and "image" in r.headers.get("content-type", "image"):
            LOGO_DIR.mkdir(parents=True, exist_ok=True)
            (LOGO_DIR / f"{ticker}.png").write_bytes(r.content)
            Image.open(LOGO_DIR / f"{ticker}.png").verify()
            return str(LOGO_DIR / f"{ticker}.png")
    except Exception:
        (LOGO_DIR / f"{ticker}.png").unlink(missing_ok=True)
    _LOGO_MISS.add(ticker)
    return None


# ---------------------------------------------------------------------
# News + context for the Director / Writers
# ---------------------------------------------------------------------
@dataclass
class DailyContext:
    text: str
    headlines: List[str] = field(default_factory=list)
    date_label: str = "TODAY"


def _fetch_rss_headlines(query: str) -> List[str]:
    try:
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+when:1d&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            root = ET.fromstring(resp.read())
        out = []
        for item in root.findall(".//item")[:3]:
            title = item.findtext("title") or ""
            if " - " in title:
                title = title.rsplit(" - ", 1)[0]  # strip the publisher name
            title = ascii_clean(title)
            if title:
                out.append(title)
        return out
    except Exception as e:
        logger.warning(f"Failed pulling RSS for query '{query}': {e}")
        return []


def fetch_headlines() -> List[str]:
    logger.info("Fetching latest 24h market news via Google News RSS...")
    seen, out = set(), []
    for headlines in parallel_map(_fetch_rss_headlines, RSS_QUERIES):
        for h in headlines:
            key = h.lower()[:60]
            if key not in seen:
                seen.add(key)
                out.append(h)
    logger.info(f"Extracted {len(out)} clean headlines from RSS.")
    return out


def mentioned_tickers(headlines: Sequence[str]) -> List[str]:
    """Companies named in the headlines that are not on the watchlist (verified later, never guessed by the LLM)."""
    found = []
    blob = " | ".join(headlines).lower()
    for t, name in COMPANY_NAMES.items():
        if t in WATCHLIST_CANDIDATES or len(name) < 4:
            continue
        if re.search(r"(?<![a-z])" + re.escape(name.lower()) + r"(?![a-z])", blob):
            found.append(t)
    return found[:MAX_EXTRA_TICKERS]


def _pct(p: float) -> str:
    return f"{p:+.2f}%"


def gather_daily_context(news_file: Optional[str] = None) -> DailyContext:
    """Everything the Director/Writers may use: real index, sector and watchlist moves plus the 24h headlines."""
    logger.info("Gathering daily market data and news...")
    headlines = fetch_headlines()
    extras = mentioned_tickers(headlines)
    symbols = list(INDEX_SYMBOLS) + list(SECTOR_ETFS) + WATCHLIST_CANDIDATES + extras
    days = {s: d for s, d in zip(symbols, parallel_map(fetch_day_perf, symbols)) if d}
    if not days and not headlines and not news_file:
        raise RuntimeError("Could not download any market data or news and no --news-file was given.")

    lines: List[str] = []
    ref = days.get("^GSPC") or next(iter(days.values()), None)
    date_label = "TODAY"
    if ref:
        end = datetime.date.fromisoformat(ref["end_date"])
        date_label = end.strftime("%a %b %d").upper()
        lines.append(f"SESSION: {end.strftime('%A %B %d %Y')} (last close). Use placeholders like [[^GSPC:day]] in scripts.")
    lines.append("INDEXES (day % change, 5-session % change):")
    for s, name in INDEX_SYMBOLS.items():
        if s in days:
            lines.append(f"- {name} ({s}): {_pct(days[s]['day_pct'])} day, {_pct(days[s]['pct'])} week")
    lines.append("SECTORS via sector ETFs (day % change):")
    for s, name in sorted(SECTOR_ETFS.items(), key=lambda kv: -(days.get(kv[0]) or {"day_pct": 0})["day_pct"]):
        if s in days:
            lines.append(f"- {name.title()} ({s}): {_pct(days[s]['day_pct'])}")
    stocks = sorted(((s, days[s]["day_pct"]) for s in WATCHLIST_CANDIDATES + extras if s in days),
                    key=lambda x: x[1], reverse=True)
    if stocks:
        lines.append("WATCHLIST MOVES today (day %), biggest gainers first:")
        lines.append("- " + ", ".join(f"{COMPANY_NAMES.get(s, s)} ({s}) {p:+.1f}%" for s, p in stocks))
        big = sorted(stocks, key=lambda x: -abs(x[1]))[:3]
        lines.append("- Biggest absolute moves: " + ", ".join(f"{s} {p:+.1f}%" for s, p in big))
    if headlines:
        lines.append("HEADLINES, last 24h (the only quotable sources; news_flash must copy them EXACTLY):")
        lines += [f"- {h}" for h in headlines[:22]]
    if news_file:
        try:
            lines.append("ADDITIONAL NOTES FROM THE PRODUCER:")
            lines.append(Path(news_file).read_text(encoding="utf-8", errors="ignore")[:3000])
        except Exception as e:
            logger.warning(f"Could not read {news_file}: {e}")
    return DailyContext("\n".join(lines)[:CONTEXT_MAX_CHARS], headlines, date_label)


@dataclass
class MarketCache:
    candles: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    info: Dict[str, dict] = field(default_factory=dict)
    days: Dict[str, dict] = field(default_factory=dict)
    pulse: Dict[str, dict] = field(default_factory=dict)
    sectors: List[dict] = field(default_factory=list)
    logos: Dict[str, Optional[str]] = field(default_factory=dict)


# =====================================================================
# 5. SCRIPT NORMALISATION / VALIDATION / LLM PROMPTS
# =====================================================================
PLACEHOLDER_RE = re.compile(r"\[\[\s*([A-Za-z0-9.\-^=]{1,12})\s*:\s*(week|day|price)\s*\]\]", re.I)
DISCLAIMER_RE = re.compile(r"not\s+(a\s+)?(financial|investment)\s+advice", re.I)
GREETING_RE = re.compile(r"^\s*(hey|hi|hello|welcome|what'?s up|good (morning|evening|afternoon)|today on)\b", re.I)
DRY_JARGON = re.compile(
    r"\b(?:market participants|sustained liquidation|institutional sponsorship|risk assets|amidst|"
    r"impending trouble|broader index uncertainty|reassessed their risk exposure|downward pressure|"
    r"on the other side of the ledger|let'?s break it down|watch till the end|stay tuned)\b", re.I)


def spoken_word_estimate(script: str) -> int:
    """Word count with every [[TICKER:day]] counted as the ~5 words it will be spoken as."""
    return len(PLACEHOLDER_RE.sub("up two point one percent", script or "").split())


def chapter_word_budget(plan: ChapterPlan) -> int:
    return max(14, int(round(plan.seconds * WORDS_PER_SECOND)))


def downgrade_to_plain(seg: Segment, reason: str, apology: Optional[str] = None) -> None:
    """A screen that lacks its data becomes a centred statement taken from the narration."""
    logger.warning(f"Segment '{seg.headline}': {ev(seg.layout)} -> plain_text ({reason})")
    summary = shorten(first_sentence(seg.script), 120)
    if apology:
        seg.script = f"{apology} {seg.script}".strip()
        summary = shorten(f"{apology} {summary}", 130)
    seg.layout = "plain_text"
    seg.plain_text = summary
    seg.plain_highlight = []


def _payload_problem(seg: Segment) -> Optional[str]:
    L = seg.layout
    if L == "pro_candlestick_chart" and not seg.chart_ticker:
        return "no ticker"
    if L == "stock_dashboard" and not seg.dash_ticker:
        return "no ticker"
    if L == "news_flash" and not seg.news_headlines:
        return "no headlines"
    if L == "plain_text" and len(seg.plain_text) < 3:
        seg.plain_text = shorten(first_sentence(seg.script), 120)
    return None


def normalize_segment(seg: Segment) -> None:
    """Cleans every field and repairs / rejects what the renderer cannot draw. Raising triggers an LLM retry."""
    seg.layout = ev(seg.layout) if ev(seg.layout) in LAYOUTS else "plain_text"
    L = seg.layout
    seg.script = ascii_clean(seg.script)
    words = spoken_word_estimate(seg.script)
    if words < SEG_WORDS_MIN or words > SEG_WORDS_MAX:
        raise ValueError(f"segment narration has {words} words: {seg.script[:60]!r}")
    seg.headline = shorten(ascii_clean(seg.headline).upper(), 26) or DEFAULT_HEADLINES[L]
    seg.accent = norm_color_name(seg.accent)
    seg.sfx = ev(seg.sfx) if ev(seg.sfx) in {s.value for s in SfxType} else "none"
    seg.caption = ""

    seg.pulse_assets = list(dict.fromkeys(ev(a) for a in seg.pulse_assets if ev(a) in PULSE_ASSETS))[:3]
    seg.chart_ticker = clean_ticker(seg.chart_ticker)
    seg.chart_timeframe = normalize_timeframe(seg.chart_timeframe)
    for n in seg.chart_notes:
        n.text, n.anchor = shorten(ascii_clean(n.text), 28), ev(n.anchor)
    seg.chart_notes = [n for n in seg.chart_notes if n.text][:2]
    seg.dash_ticker = clean_ticker(seg.dash_ticker)
    seg.highlight_metric = ev(seg.highlight_metric)
    seg.movers_side = ev(seg.movers_side) if ev(seg.movers_side) in ("both", "gainers", "losers") else "both"
    seg.movers_gainers = [t for t in dict.fromkeys(clean_ticker(t) for t in seg.movers_gainers) if t][:3]
    seg.movers_losers = [t for t in dict.fromkeys(clean_ticker(t) for t in seg.movers_losers) if t][:3]
    seg.news_headlines = [h for h in (ascii_clean(h) for h in seg.news_headlines) if h][:3]
    seg.plain_text = shorten(ascii_clean(seg.plain_text), 130)
    seg.plain_highlight = [ascii_clean(w) for w in seg.plain_highlight if ascii_clean(w)][:3]

    problem = _payload_problem(seg)
    if problem:
        downgrade_to_plain(seg, problem)


def _norm_tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def match_headline(h: str, sources: Sequence[str]) -> Optional[str]:
    """The source headline that `h` copies (the writer is asked to copy it, but may drop a word), else None."""
    th = set(_norm_tokens(h))
    if not th:
        return None
    best, best_score = None, 0.0
    for src in sources:
        ts = set(_norm_tokens(src))
        if not ts:
            continue
        score = len(th & ts) / len(th | ts)
        if score > best_score:
            best, best_score = src, score
    return best if best_score >= 0.8 else None


def clean_ticker_text(text: str) -> str:
    return re.sub(r"[^\x20-\x7E\u2022]+", "", text or "").strip()


BEAR_TITLE_WORDS = ("CRASH", "PLUNGE", "TANK", "SLIDE", "SLIDES", "DUMP", "DROP", "FALL", "FALLS", "SINK", "SINKS",
                    "COLLAPSE", "TUMBLE", "BLEED", "BLEEDS", "SLUMP", "DIVE", "DIVES")
BULL_TITLE_WORDS = ("SURGE", "SURGES", "SOAR", "SOARS", "EXPLODE", "EXPLODES", "JUMP", "JUMPS", "RALLY", "POP",
                    "POPS", "ROCKET", "ROCKETS", "BOOM", "SPIKE", "SPIKES", "RIP", "RIPS", "BREAKOUT", "MOON")


def _title_word_forms(text: str) -> set:
    """Upper-case words of a thumbnail headline plus their singular forms (CRASHES -> CRASH, SLIDES -> SLIDE)."""
    out = set()
    for w in text.split():
        w = re.sub(r"[^A-Z]", "", w.upper())
        out.update({w, w.rstrip("S"), w[:-2] if w.endswith("ES") else w})
    return out - {""}


def normalize_thumbnail_idea(t: ThumbnailIdea) -> ThumbnailIdea:
    t.template = "massive_number" if str(t.template).strip().lower() == "massive_number" else "candle_hero"
    t.ticker = clean_ticker(t.ticker)
    words = re.sub(r"[^A-Z?!' ]+", "", ascii_clean(t.main_text).upper()).split()[:3]
    t.main_text = " ".join(words) or "MARKET MOVERS"
    t.avatar_emotion = (t.avatar_emotion or "").strip().lower()
    if t.avatar_emotion not in ("shocked", "pointing", "thinking", "neutral"):
        t.avatar_emotion = "shocked"
    return t


def normalize_outline(o: DailyOutline) -> DailyOutline:
    if not CHAPTERS_MIN <= len(o.chapters) <= CHAPTERS_MAX:
        raise ValueError(f"expected {CHAPTERS_MIN}-{CHAPTERS_MAX} chapters, got {len(o.chapters)}")
    if not o.youtube_title.strip():
        raise ValueError("empty youtube_title")
    o.date_label = shorten(ascii_clean(o.date_label).upper(), 24) or "DAILY RECAP"
    o.story_angle = ascii_clean(o.story_angle)
    o.hook = ascii_clean(o.hook)
    for ch in o.chapters:
        ch.title = shorten(ascii_clean(ch.title), 34)
        if not ch.title or not ch.talking_points:
            raise ValueError("chapter without title or talking points")
        ch.summary = ascii_clean(ch.summary)
        ch.key_tickers = [t for t in (clean_ticker(t) for t in ch.key_tickers) if t][:5]
        ch.talking_points = [ascii_clean(p) for p in ch.talking_points if ascii_clean(p)][:4]
        ch.screens = [s for s in dict.fromkeys(ev(s) for s in ch.screens) if s in LAYOUTS][:3]
    # time budget: clamp, then scale so that all chapters together match the narration target
    secs = [min(CHAPTER_MAX_SECONDS, max(CHAPTER_MIN_SECONDS, int(c.seconds or 10))) for c in o.chapters]
    scale = TARGET_SPEECH_SECONDS / max(1, sum(secs))
    secs = [max(CHAPTER_MIN_SECONDS, int(round(s * scale))) for s in secs]
    while sum(secs) > TARGET_SPEECH_SECONDS + 2:
        secs[secs.index(max(secs))] -= 1
    for ch, s in zip(o.chapters, secs):
        ch.seconds = s
    o.news_ticker_text = clean_ticker_text(o.news_ticker_text) or "DAILY RECAP  \u2022  MARKETS  \u2022  NOT FINANCIAL ADVICE"
    o.youtube_title = shorten(re.sub(r"\s+", " ", re.sub(r"\[\[[^\]]*\]\]", "", ascii_clean(o.youtube_title))).strip(), 95)
    o.tags = ascii_clean(o.tags)
    o.thumbnail = normalize_thumbnail_idea(o.thumbnail)
    return o


def parse_outline(response) -> DailyOutline:
    return normalize_outline(DailyOutline.model_validate_json(response.text))


def _normalize_segments(segments: List[Segment]) -> List[Segment]:
    """Normalise every segment; one that cannot be repaired is dropped instead of failing the chapter."""
    good = []
    for seg in segments[:SEG_PER_CHAPTER_MAX]:
        try:
            normalize_segment(seg)
            good.append(seg)
        except ValueError as e:
            logger.warning(f"Dropping unusable segment: {e}")
    return good


def _narration_signature(script: str) -> set:
    return set(re.findall(r"[a-z0-9]+", PLACEHOLDER_RE.sub("", script.lower())))


def _nearly_same_story(a: str, b: str) -> bool:
    aa, bb = _narration_signature(a), _narration_signature(b)
    return bool(aa and bb and len(aa & bb) / len(aa | bb) >= 0.80)


def _screen_subjects(seg: Segment) -> frozenset:
    """What this screen covers; different stocks do NOT constitute duplicate sections."""
    layout = ev(seg.layout)
    if layout == "stock_dashboard":
        return frozenset([seg.dash_ticker]) if seg.dash_ticker else frozenset()
    if layout == "pro_candlestick_chart":
        return frozenset([seg.chart_ticker]) if seg.chart_ticker else frozenset()
    if layout == "top_movers":
        return frozenset(seg.movers_gainers + seg.movers_losers)
    return frozenset()


def _duplicated_story(a: Segment, b: Segment) -> bool:
    aa, bb = _screen_subjects(a), _screen_subjects(b)
    if aa and bb and aa != bb:
        return False
    return _nearly_same_story(a.script, b.script)


def parse_chapter(response, plan: ChapterPlan, headlines: Sequence[str], near_misses: Optional[list] = None,
                  first_chapter: bool = False, last_chapter: bool = False) -> ChapterScript:
    """Judge story quality, screen rhythm and the clock budget."""
    cs = ChapterScript.model_validate_json(response.text)
    cs.segments = _normalize_segments(cs.segments)
    if len(cs.segments) < SEG_PER_CHAPTER_MIN:
        raise ValueError("no usable chapter segments")
    issues, seen = [], []
    for s in cs.segments:
        if DRY_JARGON.search(s.script):
            issues.append("corporate jargon / banned phrase")
        if s.layout == "news_flash" and any(not match_headline(h, headlines) for h in s.news_headlines):
            issues.append("news_flash headline not copied from the supplied headlines")
        if any(_duplicated_story(s, old) for old in seen):
            issues.append("repeated narration about the same subject")
        if s.layout == "outro" and not last_chapter:
            issues.append("outro before the last chapter")
        seen.append(s)
    total_words = sum(spoken_word_estimate(s.script) for s in cs.segments)
    budget = chapter_word_budget(plan)
    if total_words > budget * 1.3 + 4:
        issues.append(f"{total_words} words, budget is {budget}")
    if first_chapter:
        hook = first_sentence(cs.segments[0].script)
        if GREETING_RE.search(hook):
            issues.append("greeting instead of a hook")
        if spoken_word_estimate(hook) > 18:
            issues.append("hook sentence is too long")
        if cs.segments[0].layout == "outro":
            issues.append("outro as the first screen")
    if last_chapter and cs.segments[-1].layout != "outro":
        issues.append("last screen is not the outro")
    if issues:
        if near_misses is not None:
            near_misses.append((len(issues), cs))
        raise ValueError("; ".join(dict.fromkeys(issues)))
    return cs


def _gen_config(types, schema, temperature: float):
    kwargs = dict(response_mime_type="application/json", response_schema=schema, temperature=temperature)
    try:  # the SDK's automatic function calling is not needed here and only produces warnings
        return types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **kwargs)
    except Exception:
        return types.GenerateContentConfig(**kwargs)


LAYOUT_GUIDE = """
        SCREENS. Fill ONLY the fields of the screen you pick. Every number on a screen is verified by code.
        - market_pulse: intraday lines of 1-3 benchmarks versus the previous close (pulse_assets from SPY, QQQ, DIA, IWM,
          BTC). Best for the hook or for "how did the market close".
        - pro_candlestick_chart: ONE ticker or index (chart_ticker). chart_timeframe: 1d (today's 5-minute candles,
          the default), 5d, 1mo or 3mo. chart_notes: 0-2 notes (anchor high/low/start/end/biggest_drop, max 3 words).
        - stock_dashboard: metric cards for ONE company (dash_ticker), never an index or ETF. Name the metric you talk
          about (price, today, this week, market cap, P E, dividend) and its card glows.
        - sector_heatmap: the real 11-sector map of the day. No payload. NAME the sector you talk about and its block
          flashes.
        - top_movers: today's biggest verified gainers and losers of the watchlist. movers_side both / gainers /
          losers. movers_gainers / movers_losers are optional ticker hints taken from the supplied data (code fills the
          rest). Name the tickers or companies and their rows flash.
        - news_flash: 1-3 headlines COPIED EXACTLY from the supplied headlines (news_headlines), one sentence of
          narration per headline. Use it only when the news itself is the story.
        - plain_text: one strong statement (plain_text, max 18 words, plain_highlight up to 3 words from it).
        - outro: ONLY as the very last screen: a punchy closing line, a question for the comments, a nudge to follow.

        EDITING RULES
        1. Cut between DIFFERENT screens. Never the same screen type back to back unless it shows a different ticker.
        2. market_pulse and sector_heatmap: at most once per video. top_movers 'both': at most once. news_flash: max 2.
        3. One chart OR one dashboard per ticker, never both.
        4. Every screen earns its time: 1-3 short sentences, 3 to 8 seconds. No filler screens, no recaps.
        5. Follow the chapter's planned screens, but swap one if the data cannot support it.
        """

STYLE_GUIDE = """
        VOICE: a sharp, high-energy finance YouTuber on a vertical Short, talking to ONE viewer. Short punchy sentences
        (6-12 words), contractions, 'you'. Lead with the twist, the stakes or a contrast, then the evidence, then why
        the viewer should care. No greeting, no channel intro, no 'let's break it down', no 'watch till the end'.
        NEVER write like a press release ('market participants', 'risk assets', 'amid', 'downward pressure').
        No unsupported causes, fake quotes, invented catalysts or guaranteed outcomes. Explain the REASON behind a move
        only if the supplied headlines support it.
        NUMBERS: never invent prices, percentages or levels. Use verified placeholders: [[NVDA:day]] is a COMPLETE spoken
        move INCLUDING direction, so write 'Nvidia finished [[NVDA:day]].' and NOT 'Nvidia rose [[NVDA:day]].'
        Also [[NVDA:week]] and [[NVDA:price]]. Indexes work too: [[^GSPC:day]], [[^IXIC:day]], [[BTC-USD:day]].
        At most two placeholders per sentence; never read a list of numbers the screen already shows.
        TTS: natural spoken English, company names instead of ticker codes, simple punctuation (periods, commas,
        question marks). No emojis, markdown, dollar signs, percent signs or digits. Headlines are ASCII, UPPERCASE.
        A single-ticker screen (chart, dashboard) talks ONLY about that company.
        Do not speak financial-advice disclaimers.
        """

THUMBNAIL_GUIDE = """
        THUMBNAIL. template: 'candle_hero' (that stock's real candlestick chart, logo and verified percent badge next to
        a surprised face) or 'massive_number' (giant headline only; use it only if no single stock stands out).
        ticker: the ONE stock from the supplied data with the biggest or most surprising move today.
        main_text: MAX 3 WORDS, UPPERCASE clickbait question or exclamation about it ('NVIDIA EXPLODES?', 'TESLA CRASH?').
        It MUST match the DIRECTION of that stock's verified move: never crash for a riser, never surge for a faller.
        No emojis, dollar signs, numbers or placeholders. avatar_emotion: shocked, pointing, thinking or neutral.
        """


def generate_outline(ctx: DailyContext) -> DailyOutline:
    from google.genai import types
    logger.info("Director: planning a %.0f-second story...", TARGET_SPEECH_SECONDS)
    prompt = f"""
            You are the DIRECTOR of a fast, high-retention vertical YouTube Short that recaps today's stock market.
            HARD LIMIT: the finished video must stay UNDER {SHORTS_MAX_SECONDS:.0f} seconds, so plan about
            {TARGET_SPEECH_SECONDS:.0f} seconds of narration (about {int(TARGET_SPEECH_SECONDS * WORDS_PER_SECOND)} words
            in total). Information density beats coverage: it is fine to ignore most of the data.
            Use ONLY the verified information below. Find the ONE story that explains today (a sharp tension, a
            surprising contrast or a big question) and build a tight narrative around it, NOT a list of market facts.

            Return {CHAPTERS_MIN} to {CHAPTERS_MAX} chapters (story beats). A typical shape: 1) hook + how the market
            closed, 2) the story behind the biggest move, 3) the takeaway, ending on a question for the comments.
            For each chapter give: title, summary, key_tickers, 2-4 talking_points from the supplied data, screens
            (1-3 screens in order, from the SCREENS list below) and seconds (6-20; all chapters together about
            {TARGET_SPEECH_SECONDS:.0f}). Plan 6 to 8 screens in total, varied, cutting between different screen types.
            The very FIRST spoken sentence (max 15 words) must stop the scroll: a verified twist, a sharp question or
            a surprising contrast, spoken to the viewer. The LAST chapter ends with the outro screen.
            Also return: date_label, story_angle, hook, news_ticker_text (3 short UPPERCASE items 'TOPIC: summary' joined
            by ' \u2022 '), youtube_title (max 70 characters, truthful, curiosity-driven, no placeholders),
            description (60-100 words, conversational, concrete stories from the data, no disclaimer, no placeholders),
            tags (6-10 comma-separated), thumbnail (see THUMBNAIL below).

            {LAYOUT_GUIDE}
            {STYLE_GUIDE}
            {THUMBNAIL_GUIDE}

            VERIFIED MARKET DATA AND NEWS
            {ctx.text}
            """
    return ask_llm(prompt, config=_gen_config(types, DailyOutline, 0.9), parse=parse_outline)


def write_chapter(outline: DailyOutline, index: int, ctx: DailyContext, previous_tail: str) -> ChapterScript:
    from google.genai import types
    plan = outline.chapters[index]
    last = index == len(outline.chapters) - 1
    budget = chapter_word_budget(plan)
    logger.info("Writer: chapter %d/%d '%s' (%d s, about %d words)", index + 1, len(outline.chapters), plan.title,
                plan.seconds, budget)
    roles = []
    if index == 0:
        roles.append("FIRST CHAPTER: the first spoken sentence is the cold open (max 15 words, no greeting). "
                     f"Use this idea, reworded to sound natural and direct: {outline.hook}")
    if last:
        roles.append("LAST CHAPTER: your FINAL screen MUST be the outro: one punchy closing line, a question that "
                     "makes viewers comment, and a nudge to follow for tomorrow's recap. No spoken disclaimer.")
    overview = "\n".join(f"  {i + 1}. {c.title} ({c.seconds}s): {c.summary}" for i, c in enumerate(outline.chapters))
    prompt = f"""
            You are a charismatic finance YouTuber writing ONE chapter of a vertical Short that must stay under
            {SHORTS_MAX_SECONDS:.0f} seconds in total. Return {SEG_PER_CHAPTER_MIN} to {SEG_PER_CHAPTER_MAX} screens.
            WORD BUDGET FOR THIS CHAPTER: about {budget} spoken words in total (a [[TICKER:day]] placeholder is spoken
            as about 5 words). Going over it makes the video too long, so write TIGHT: every sentence must add a new
            fact, a concrete takeaway or a hook. Never repeat the chapter title, never pad, never summarise.
            {' '.join(roles)}
            {('Prior chapter ended: ' + previous_tail) if previous_tail else ''}

            STORY OF THE DAY: {outline.story_angle}
            EPISODE OVERVIEW ({outline.date_label})
        {overview}
            THIS CHAPTER: {plan.title}. {plan.summary}
            Key tickers: {', '.join(plan.key_tickers) or 'none'}
            Talking points: {' | '.join(plan.talking_points)}
            Planned screens, in order: {', '.join(plan.screens) or 'your choice'}

            {LAYOUT_GUIDE}
            {STYLE_GUIDE}

            MARKET DATA AND NEWS (the only source of facts)
            {ctx.text}
            """
    near_misses: list = []
    try:
        return ask_llm(prompt, config=_gen_config(types, ChapterScript, 0.85),
                       parse=lambda r: parse_chapter(r, plan, ctx.headlines, near_misses, index == 0, last),
                       max_rounds=CHAPTER_MAX_ROUNDS)
    except LLMUnavailableError:
        if not near_misses:
            raise
        score, cs = min(near_misses, key=lambda x: x[0])
        logger.warning("Chapter '%s': accepting best available draft (%d editorial issues); the editor and the "
                       "time-fit step will repair it", plan.title, score)
        return cs


def generate_master_timeline(ctx: DailyContext) -> MasterTimeline:
    outline = generate_outline(ctx)
    items: List[TimelineItem] = []
    tail = ""
    for ci, plan in enumerate(outline.chapters):
        script = write_chapter(outline, ci, ctx, tail)
        for seg in script.segments:
            items.append(TimelineItem(chapter_index=ci, chapter_title=plan.title, segment=seg))
        tail = shorten(script.segments[-1].script, 140) if script.segments else tail
    return MasterTimeline(outline=outline, items=items, headlines=list(ctx.headlines))


# =====================================================================
# 6. DATA PRE-FETCH  (before any video processing; zero-crash fallbacks)
#    + verified-number placeholders, topic checks, editorial rules and the clock
# =====================================================================
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n: int) -> str:
    n = int(n)
    if n < 0:
        return "minus " + int_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_words(n % 100))
    if n < 1_000_000:
        return int_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + int_words(n % 1000))
    return int_words(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else " " + int_words(n % 1_000_000))


def decimal_words(x: float, decimals: int = 1) -> str:
    x = round(abs(x), decimals)
    whole = int(x)
    frac = int(round((x - whole) * 10 ** decimals))
    if decimals == 0 or frac == 0:
        return int_words(whole)
    return f"{int_words(whole)} point " + " ".join(_ONES[int(d)] for d in str(frac).zfill(decimals))


def spoken_pct(p: float) -> str:
    a = abs(p)
    if a < 0.05:
        return "flat"
    return f"{'up' if p > 0 else 'down'} {decimal_words(a, 1 if a < 20 else 0)} percent"


def spoken_price(v: float, index: bool = False) -> str:
    if index:
        return f"about {int_words(int(round(v)))} points"
    if v >= 100:
        return f"{int_words(int(round(v)))} dollars"
    d = int(v)
    c = int(round((v - d) * 100))
    if c == 100:
        d, c = d + 1, 0
    unit = "dollar" if d == 1 else "dollars"
    return f"{int_words(d)} {unit}" + (f" and {int_words(c)} cents" if c else "")


def screen_pct(p: float) -> str:
    return f"{p:+.1f}%"


def screen_price(v: float, index: bool = False) -> str:
    return f"{v:,.0f}" if index else f"${v:,.2f}"


def _text_fields(seg: Segment) -> List[str]:
    return [seg.script, seg.plain_text]


def placeholder_tickers(seg: Segment) -> set:
    out = set()
    for text in _text_fields(seg):
        out.update(clean_ticker(m.group(1)) for m in PLACEHOLDER_RE.finditer(text))
    return out


def resolve_text(text: str, lookup: Callable[[str, str], Optional[float]], spoken: bool) -> str:
    def repl(m):
        t, kind = clean_ticker(m.group(1)), m.group(2).lower()
        v = lookup(t, kind)
        index = t.startswith("^")
        if v is None:
            return ("by a notable amount" if kind in ("week", "day") else "its latest level") if spoken else "n/a"
        if kind in ("week", "day"):
            return spoken_pct(v) if spoken else screen_pct(v)
        return spoken_price(v, index) if spoken else screen_price(v, index)

    out = PLACEHOLDER_RE.sub(repl, text)
    return re.sub(r"\[\[[^\]]*\]\]", "", out).replace("  ", " ").strip()


def resolve_segment(seg: Segment, lookup: Callable[[str, str], Optional[float]]) -> None:
    """Spoken script (numbers as words) + on-screen caption (numbers as digits) from the same verified data."""
    raw = seg.script
    seg.script = resolve_text(raw, lookup, True)
    caption = resolve_text(raw, lookup, False)
    seg.caption = caption if len(split_sentences(caption)) == len(split_sentences(seg.script)) else seg.script
    seg.plain_text = resolve_text(seg.plain_text, lookup, False)


def resolve_outline_metadata(outline: DailyOutline, lookup: Callable[[str, str], Optional[float]]) -> None:
    """YouTube text must never expose [[...]]. An unverified claim is dropped, not replaced by an invented number."""

    def verified(text: str) -> str:
        missing = [m.group(0) for m in PLACEHOLDER_RE.finditer(text)
                   if lookup(clean_ticker(m.group(1)), m.group(2).lower()) is None]
        return "" if missing else resolve_text(text, lookup, spoken=False)

    paragraphs = []
    for paragraph in outline.description.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        paragraphs.append(" ".join(s for s in (verified(x) for x in sentences if x.strip()) if s))
    outline.description = "\n".join(paragraphs).strip()
    if not outline.description:
        outline.description = "Daily market recap: " + "; ".join(c.title for c in outline.chapters) + "."
    outline.youtube_title = re.sub(r"\s+", " ", re.sub(r"\[\[[^\]]*\]\]", "", outline.youtube_title)).strip()


def scrub_offtopic(seg: Segment) -> None:
    """Chart / dashboard segments are about ONE company: sentences that talk about another company are removed."""
    focus = seg.chart_ticker if seg.layout == "pro_candlestick_chart" else seg.dash_ticker
    if not focus or focus.startswith("^"):
        return
    focus_name = COMPANY_NAMES.get(focus, "").lower()
    others = [n.lower() for t, n in COMPANY_NAMES.items()
              if t != focus and len(n) >= 4 and n.lower() != focus_name
              and n.lower() not in focus_name and focus_name not in n.lower()]
    sentences = split_sentences(seg.script)
    caps = split_sentences(seg.caption) if seg.caption else []
    keep_idx = [i for i, s in enumerate(sentences)
                if not any(re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", s.lower()) for n in others)]
    kept = [sentences[i] for i in keep_idx]
    if kept and len(kept) < len(sentences) and len(" ".join(kept).split()) >= 8:
        logger.warning(f"'{seg.headline}': removed {len(sentences) - len(kept)} off-topic sentence(s) about other companies")
        seg.script = " ".join(kept)
        seg.caption = " ".join(caps[i] for i in keep_idx) if len(caps) == len(sentences) else seg.script


def _strip_spoken_disclaimer(text: str) -> str:
    return " ".join(s for s in split_sentences(text) if not DISCLAIMER_RE.search(s)).strip()


def default_outro() -> Segment:
    seg = Segment(layout="outro", headline="SEE YOU TOMORROW", accent="gold",
                  script="Which stock are you watching tomorrow? Tell me in the comments, and follow for the next recap.")
    normalize_segment(seg)
    return seg


LAYOUT_CAPS = {"market_pulse": 1, "sector_heatmap": 1, "top_movers": 2, "news_flash": 2, "plain_text": 2}


def enforce_editorial_rules(master: MasterTimeline) -> None:
    """Deterministic editor for new OR --load-script JSON: removes disclaimers, repeated screens and repeated stories,
    keeps ONE screen per ticker and guarantees the closing outro. Idempotent."""
    counts: Dict[str, int] = {}
    seen_tickers, seen_movers = set(), set()
    outro_item: Optional[TimelineItem] = None
    staged: List[TimelineItem] = []
    for it in master.items:
        s = it.segment
        s.script = _strip_spoken_disclaimer(s.script)
        L = ev(s.layout)
        if not s.script:
            logger.info("Editor removed disclaimer-only segment: %s", s.headline)
            continue
        if L == "outro":
            if outro_item is None:  # only the first outro survives, and it is moved to the very end
                outro_item = it
            continue
        if counts.get(L, 0) >= LAYOUT_CAPS.get(L, 99):
            logger.info("Editor removed extra %s screen: %s", L, s.headline)
            continue
        if L in ("pro_candlestick_chart", "stock_dashboard"):
            ticker = s.chart_ticker if L == "pro_candlestick_chart" else s.dash_ticker
            if ticker in seen_tickers:
                logger.info("Editor removed a second analysis screen for %s", ticker)
                continue
            seen_tickers.add(ticker)
        if L == "top_movers":
            key = (ev(s.movers_side), tuple(sorted(s.movers_gainers)), tuple(sorted(s.movers_losers)))
            if key in seen_movers or (ev(s.movers_side) == "both" and ("both",) in seen_movers):
                logger.info("Editor removed repeated movers screen")
                continue
            seen_movers.add(key)
            if ev(s.movers_side) == "both":
                seen_movers.add(("both",))
        if any(_duplicated_story(s, old.segment) for old in staged):
            logger.info("Editor removed duplicated narration screen: %s", s.headline)
            continue
        counts[L] = counts.get(L, 0) + 1
        staged.append(it)
    if not staged:
        raise ValueError("No content left after editorial cleanup")
    last = staged[-1]
    if outro_item is None:
        outro_item = TimelineItem(chapter_index=last.chapter_index, chapter_title=last.chapter_title,
                                  segment=default_outro())
    else:
        outro_item.chapter_index, outro_item.chapter_title = last.chapter_index, last.chapter_title
    staged.append(outro_item)
    master.items = staged
    logger.info("Editorial cleanup retained %d screen(s)", len(master.items))


# ---------------------------------------------------------------------
# Grounding of the data-driven screens
# ---------------------------------------------------------------------
def ground_movers(seg: Segment, days: Dict[str, dict]) -> bool:
    """Replace the writer's hints by verified boards (auto-filled from the watchlist). False = cannot be drawn."""
    pool = {t: d["day_pct"] for t, d in days.items()
            if t in WATCHLIST_CANDIDATES or t in seg.movers_gainers or t in seg.movers_losers}
    gains = sorted((t for t, p in pool.items() if p > 0), key=lambda t: -pool[t])
    losses = sorted((t for t, p in pool.items() if p < 0), key=lambda t: pool[t])
    side = ev(seg.movers_side)
    n = 3 if side == "both" else 5

    def pick(wanted: List[str], ranked: List[str]) -> List[str]:
        out = [t for t in wanted if t in ranked][:n]
        for t in ranked:
            if len(out) >= n:
                break
            if t not in out:
                out.append(t)
        return out

    g, l = pick(seg.movers_gainers, gains), pick(seg.movers_losers, losses)
    if side == "both" and (len(g) < 2 or len(l) < 2):
        if len(g) >= 2:
            side, l = "gainers", []
        elif len(l) >= 2:
            side, g = "losers", []
    if side == "gainers":
        g, l = pick(seg.movers_gainers, gains), []
    elif side == "losers":
        g, l = [], pick(seg.movers_losers, losses)
    need = {"both": (len(g) >= 2 and len(l) >= 2), "gainers": len(g) >= 2, "losers": len(l) >= 2}[side]
    if not need:
        return False
    seg.movers_side, seg.movers_gainers, seg.movers_losers = side, g, l
    return True


def prefetch_market_data(master: MasterTimeline) -> MarketCache:
    """Collect every ticker / timeframe the timeline asks for, fetch once, ground the numbers, repair the segments."""
    cache = MarketCache()
    enforce_editorial_rules(master)
    segs = [it.segment for it in master.items]
    for seg in segs:  # an index/ETF has no P/E or market cap: show a chart instead of a grid of "N/A"
        if seg.layout == "stock_dashboard" and seg.dash_ticker.startswith("^"):
            logger.info(f"Dashboard for index {seg.dash_ticker} -> pro_candlestick_chart")
            seg.layout, seg.chart_ticker, seg.chart_timeframe = "pro_candlestick_chart", seg.dash_ticker, "1d"
        if seg.layout == "market_pulse" and not seg.pulse_assets:
            seg.pulse_assets = list(PULSE_DEFAULT)

    candle_req = sorted({(s.chart_ticker, s.chart_timeframe) for s in segs if s.layout == "pro_candlestick_chart"})
    info_req = sorted({s.dash_ticker for s in segs if s.layout == "stock_dashboard"})
    pulse_req = sorted({k for s in segs if s.layout == "market_pulse" for k in s.pulse_assets})
    day_req = set(info_req)
    for s in segs:
        day_req.update(placeholder_tickers(s))
        if s.layout == "top_movers":
            day_req.update(WATCHLIST_CANDIDATES)
            day_req.update(s.movers_gainers + s.movers_losers)
        if s.layout == "sector_heatmap":
            day_req.update(SECTOR_ETFS)
    for metadata_text in [master.outline.description]:
        for m in PLACEHOLDER_RE.finditer(metadata_text):
            day_req.add(clean_ticker(m.group(1)))
    logger.info(f"Prefetching {len(candle_req)} chart(s), {len(info_req)} dashboard(s), {len(pulse_req)} pulse "
                f"series, {len(day_req)} daily series")

    for t, w in zip(sorted(day_req), parallel_map(fetch_day_perf, sorted(day_req))):
        if w:
            cache.days[t] = w
    for key, data in zip(candle_req, parallel_map(lambda k: fetch_candles(*k), candle_req)):
        if data:
            cache.candles[key] = data
    for t, info in zip(info_req, parallel_map(fetch_stock_info, info_req)):
        if info:
            cache.info[t] = info
    for k, series in zip(pulse_req, parallel_map(fetch_pulse_series, pulse_req)):
        if series:
            cache.pulse[k] = series
    cache.sectors = [dict(etf=etf, name=name, pct=cache.days[etf]["day_pct"], weight=SECTOR_WEIGHTS[etf])
                     for etf, name in SECTOR_ETFS.items() if etf in cache.days]
    for t, path in zip(info_req, parallel_map(ensure_logo, info_req)):
        cache.logos[t] = path

    # --- ground every data-driven screen on the verified numbers ---
    for seg in segs:
        L = seg.layout
        if L == "pro_candlestick_chart":
            if (seg.chart_ticker, seg.chart_timeframe) not in cache.candles:
                downgrade_to_plain(seg, f"no price data for {seg.chart_ticker}")
        elif L == "stock_dashboard":
            if seg.dash_ticker not in cache.info:
                downgrade_to_plain(seg, f"no data for {seg.dash_ticker}")
        elif L == "market_pulse":
            seg.pulse_assets = [k for k in seg.pulse_assets if k in cache.pulse] or \
                               [k for k in PULSE_DEFAULT if k in cache.pulse]
            if not seg.pulse_assets:
                downgrade_to_plain(seg, "no intraday data")
        elif L == "sector_heatmap":
            if len(cache.sectors) < 6:
                downgrade_to_plain(seg, "sector data unavailable")
        elif L == "top_movers":
            if not ground_movers(seg, cache.days):
                downgrade_to_plain(seg, "could not verify the movers")
        elif L == "news_flash":
            matched = [m for m in (match_headline(h, master.headlines) for h in seg.news_headlines) if m]
            seg.news_headlines = list(dict.fromkeys(matched))[:3]
            if not seg.news_headlines:
                downgrade_to_plain(seg, "headlines could not be verified")

    def lookup(t: str, kind: str) -> Optional[float]:
        w = cache.days.get(t)
        return None if not w else {"week": w["pct"], "day": w["day_pct"], "price": w["last"]}[kind]

    for seg in segs:
        resolve_segment(seg, lookup)
        if seg.layout in ("pro_candlestick_chart", "stock_dashboard"):
            scrub_offtopic(seg)
    resolve_outline_metadata(master.outline, lookup)
    enforce_editorial_rules(master)  # downgrades may have created new repeats
    return cache


def data_for_segment(seg: Segment, cache: MarketCache) -> Optional[dict]:
    L = seg.layout
    if L == "pro_candlestick_chart":
        return cache.candles.get((seg.chart_ticker, seg.chart_timeframe))
    if L == "stock_dashboard":
        info = dict(cache.info.get(seg.dash_ticker) or {})
        info["logo"] = cache.logos.get(seg.dash_ticker)
        return info
    if L == "market_pulse":
        return {"series": [cache.pulse[k] for k in seg.pulse_assets if k in cache.pulse]}
    if L == "sector_heatmap":
        return {"sectors": cache.sectors}
    if L == "top_movers":
        row = lambda t: dict(ticker=t, pct=cache.days[t]["day_pct"], name=COMPANY_NAMES.get(t, ""))
        return {"gainers": [row(t) for t in seg.movers_gainers if t in cache.days],
                "losers": [row(t) for t in seg.movers_losers if t in cache.days]}
    return None


# ---------------------------------------------------------------------
# The clock: the Short never exceeds SHORTS_MAX_SECONDS
# ---------------------------------------------------------------------
def timeline_length(durations: Sequence[float], keep: Sequence[int]) -> float:
    if not keep:
        return 0.0
    return sum(durations[k] + SEGMENT_PAD for k in keep[:-1]) + durations[keep[-1]] + END_PAD


def choose_segments_to_keep(segs: Sequence[Segment], durations: Sequence[float],
                            max_seconds: float = SHORTS_MAX_SECONDS - TIME_SAFETY) -> List[int]:
    """Cut the least valuable middle screens (never the opening hook, never the outro) until the video fits."""
    keep = list(range(len(segs)))
    while timeline_length(durations, keep) > max_seconds and len(keep) > MIN_SCREENS:
        cands = keep[1:-1]
        if not cands:
            break
        victim = min(cands, key=lambda k: (LAYOUT_KEEP_SCORE.get(ev(segs[k].layout), 4), -durations[k]))
        logger.warning("Over the %.0fs limit (%.1fs): cutting screen %d '%s' (%s)", SHORTS_MAX_SECONDS,
                       timeline_length(durations, keep), victim, segs[victim].headline, ev(segs[victim].layout))
        keep.remove(victim)
    return keep


# =====================================================================
# 7. TEXT-TO-SPEECH (Kokoro-82M ONNX, sentence-level timings; no PyTorch, so no separate worker is needed)
# =====================================================================
KOKORO_DIR = Path(__file__).resolve().parent / "kokoro_assets"
_KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_NAME = "kokoro-v1.0.int8.onnx"  # ~90 MB. Use "kokoro-v1.0.onnx" (~310 MB) for best quality.
KOKORO_VOICES_NAME = "voices-v1.0.bin"


def _download(url: str, dest: Path) -> None:
    import requests
    logger.info("Downloading %s ...", dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)  # only becomes the real file if the download completed


def _get_or_download_kokoro_assets() -> Tuple[str, str]:
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    model_path, voices_path = KOKORO_DIR / KOKORO_MODEL_NAME, KOKORO_DIR / KOKORO_VOICES_NAME
    for path in (model_path, voices_path):
        if not path.exists():
            _download(f"{_KOKORO_BASE}/{path.name}", path)
    return str(model_path), str(voices_path)


@lru_cache(maxsize=1)
def get_kokoro_onnx():
    from kokoro_onnx import Kokoro
    model_path, voices_path = _get_or_download_kokoro_assets()
    return Kokoro(model_path, voices_path)


def clean_script_for_tts(text: str) -> str:
    """Removes non-ASCII characters/emojis & converts symbols for clear TTS."""
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.replace("$", "").replace("%", " percent")
    text = re.sub(r"\s+[-\u2013\u2014]+\s+|[\u2013\u2014]", ", ", text).replace(";", ",").replace("...", ".")
    text = re.sub(r"\^([A-Z]+)", r"\1", text)
    for pattern, replacement in TTS_SYMBOL_NAMES.items():
        text = re.sub(pattern, replacement, text)
    return text.replace("\n", " ").strip()


def generate_voiceovers(items: List[dict]) -> List[dict]:
    """items = [{text, voice, speed, path}] -> [{duration, sentences: [[start, end], ...]}]. Every sentence is
    synthesised on its own and joined with a short pause, so each screen knows when each sentence is spoken."""
    import soundfile as sf
    kokoro = get_kokoro_onnx()
    try:
        available = set(kokoro.get_voices())
    except Exception:
        available = set()
    results = []
    logger.info("Generating %d Kokoro clips", len(items))
    for it in items:
        voice = it["voice"] if (not available or it["voice"] in available) else HOST_VOICE
        if voice != it["voice"]:
            logger.warning("Voice %s missing, using %s", it["voice"], voice)
        sentences = split_sentences(it["text"]) or [it["text"]]
        parts, times, cursor, sr = [], [], 0.0, 24000
        for i, sent in enumerate(sentences):
            clean = clean_script_for_tts(sent)
            samples = np.zeros(0, dtype=np.float32)
            if re.search(r"[A-Za-z0-9]", clean):
                samples, sr = kokoro.create(clean, voice=voice, speed=float(it["speed"]), lang="en-us")
                samples = np.asarray(samples, dtype=np.float32)
            dur = len(samples) / sr
            times.append([cursor, cursor + dur])
            if len(samples):
                parts.append(samples)
                cursor += dur
            if i < len(sentences) - 1:
                parts.append(np.zeros(int(SENTENCE_GAP * sr), dtype=np.float32))
                cursor += SENTENCE_GAP
        if not parts:
            raise RuntimeError(f"Kokoro returned empty audio for {it['path']}")
        audio = np.concatenate(parts)
        sf.write(str(it["path"]), audio, sr, subtype="PCM_16")
        results.append({"duration": len(audio) / sr, "sentences": times})
        logger.info("Created %s (%.2fs, %s, %d sentences)", Path(it["path"]).name, results[-1]["duration"], voice,
                    len(sentences))
    return results


def run_tts(items: List[dict], work_dir: Path) -> List[dict]:
    """Reuses the voiceover of a previous run in the same work dir when the text and speed are unchanged."""
    in_file, out_file = work_dir / "tts_in.json", work_dir / "tts_out.json"
    payload = json.dumps(items)
    if (in_file.exists() and out_file.exists() and in_file.read_text(encoding="utf-8") == payload
            and all(Path(i["path"]).exists() for i in items)):
        logger.info("Reusing the voiceover from the previous run")
        return json.loads(out_file.read_text(encoding="utf-8"))
    out_file.unlink(missing_ok=True)
    results = generate_voiceovers(items)
    in_file.write_text(payload, encoding="utf-8")
    out_file.write_text(json.dumps(results), encoding="utf-8")
    return results


# =====================================================================
# 8. SOUND EFFECTS  (user files in assets/sfx win; otherwise synthesised, so nothing is ever missing)
# =====================================================================
SFX_SR = 44100


def synth_sfx(name: str, sr: int = SFX_SR) -> np.ndarray:
    rng = np.random.default_rng(5)

    def tt(d):
        return np.arange(int(d * sr)) / sr

    def norm(x, peak=0.8):
        return x / (np.abs(x).max() + 1e-9) * peak

    def bell(f, d=0.9):
        t = tt(d)
        return np.sin(2 * np.pi * f * t) * np.exp(-t * 6) + 0.5 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 9)

    def place(buf, clip, at):
        i = int(at * sr)
        n = max(0, min(len(clip), len(buf) - i))
        buf[i:i + n] += clip[:n]

    if name == "swoosh":
        d = 0.45
        t = tt(d)
        a = 0.03 + 0.55 * np.sin(np.pi * t / d) ** 2  # cut-off sweeps up then down
        noise, y, acc = rng.standard_normal(len(t)), np.zeros(len(t)), 0.0
        for i in range(len(t)):
            acc += a[i] * (noise[i] - acc)
            y[i] = acc
        return norm(y * np.sin(np.pi * t / d) ** 2)
    if name == "cash_register":
        buf = np.zeros(int(1.3 * sr))
        click = np.convolve(rng.standard_normal(int(0.03 * sr)), np.ones(6) / 6, mode="same")
        place(buf, click * np.linspace(1, 0, len(click)), 0.0)
        place(buf, bell(1568.0) * 0.8, 0.10)
        place(buf, bell(2093.0), 0.22)
        return norm(buf)
    if name == "heartbeat":
        buf = np.zeros(int(1.7 * sr))
        t = tt(0.35)
        thump = np.sin(2 * np.pi * 58 * t) * np.exp(-t * 22)
        for at, g in ((0.0, 1.0), (0.26, 0.75), (0.95, 1.0), (1.21, 0.75)):
            place(buf, thump * g, at)
        return norm(buf, 0.9)
    return np.zeros(1, dtype=np.float32)


def get_sfx_path(name: str, work_dir: Path) -> Optional[str]:
    if not name or name == "none":
        return None
    for ext in ("wav", "mp3", "ogg"):
        p = Path("assets/sfx") / f"{name}.{ext}"
        if p.exists():
            return str(p)
    out = work_dir / f"sfx_{name}.wav"
    if not out.exists():
        try:
            write_wav_mono(str(out), synth_sfx(name), SFX_SR)
        except Exception as e:
            logger.warning(f"Could not create SFX {name}: {e}")
            return None
    return str(out)


# =====================================================================
# 9. DRAWING PRIMITIVES
# =====================================================================
def new_figure(width_px: int = VIDEO_W, height_px: int = VIDEO_H, facecolor: str = C_BG):
    """Pyplot-free figure (no global state). The +1e-3 guards against int() truncation giving a 1079/1919 canvas."""
    load_render_dependencies()
    fig = Figure(figsize=((width_px + 1e-3) / DPI, (height_px + 1e-3) / DPI), dpi=DPI, facecolor=facecolor)
    return fig, FigureCanvasAgg(fig)


def grab(canvas, width_px: int = VIDEO_W, height_px: int = VIDEO_H) -> np.ndarray:
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    if img.shape[:2] != (height_px, width_px):
        img = np.asarray(Image.fromarray(img).resize((width_px, height_px)))
    return img


def style_axes(ax, grid: bool = True) -> None:
    ax.set_facecolor(STAGE_FACE)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.15, color=C_MUTED)
    ax.tick_params(colors=C_WHITE, labelsize=13)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_BORDER)
        spine.set_linewidth(2)


class FrameCache:
    """Re-renders only when t crosses a RENDER_FPS boundary; otherwise returns the last frame."""

    def __init__(self, render_fps: int = RENDER_FPS):
        self.fps, self._key, self._frame = render_fps, None, None

    def get(self, t: float, render_fn: Callable[[float], np.ndarray]) -> np.ndarray:
        key = int(t * self.fps + 1e-6)
        if key != self._key:
            self._frame = render_fn(key / self.fps)
            self._key = key
        return self._frame


def blend_rgba(frame: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    h, w = rgba.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[:, :, 3:4].astype(np.float32) / 255.0) * alpha
    region = frame[y0:y1, x0:x1].astype(np.float32)
    region += (sub[:, :, :3].astype(np.float32) - region) * a
    frame[y0:y1, x0:x1] = (region + 0.5).astype(np.uint8)


def blend_mask(frame: np.ndarray, mask: np.ndarray, x: int, y: int, color: Tuple[int, int, int],
               alpha: float = 1.0) -> None:
    h, w = mask.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    a = mask[y0 - y:y1 - y, x0 - x:x1 - x, None] * alpha
    region = frame[y0:y1, x0:x1].astype(np.float32)
    region += (np.asarray(color, dtype=np.float32) - region) * a
    frame[y0:y1, x0:x1] = (region + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------
# Animated background ("visual noise"): drifting gradient + faint scrolling grid + floating particles.
# Pure NumPy, drawn BEFORE the matplotlib axes (as a figimage under translucent axes).
# ---------------------------------------------------------------------
class BackgroundFX:
    def __init__(self, seed: int = 11):
        W, H = VIDEO_W, VIDEO_H
        rng = np.random.default_rng(seed)
        self.cell = max(24, px(96))
        xs = (np.arange(2 * W, dtype=np.float32) / W)[None, :]  # two widths wide => seamless horizontal drift
        ys = (np.arange(H, dtype=np.float32) / H)[:, None]
        field_ = 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (2 * xs + 0.9 * ys))) \
                 + 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * (xs - 1.3 * ys + 0.3)))
        vignette = 0.35 + 0.65 * np.clip(1 - np.abs(ys - 0.5) * 1.3, 0, 1)
        base = np.array(_rgb(C_BG), np.float32)
        tint = np.array((2, 11, 18), np.float32)
        strip = base + (field_ * vignette)[..., None] * tint
        self.strip = np.clip(strip + 0.5, 0, 255).astype(np.uint8)
        grid = np.zeros((H + self.cell, W), np.int16)
        grid[::self.cell, :] = 5
        grid[:, ::self.cell] = 5
        self.grid = grid
        n = 40
        self.x0, self.y0 = rng.uniform(0, W, n), rng.uniform(0, H, n)
        self.vy = rng.uniform(8, 26, n) * UI_SCALE
        self.sway = rng.uniform(6, 22, n) * UI_SCALE
        self.phase = rng.uniform(0, 2 * math.pi, n)
        self.rad = np.maximum(2, (rng.integers(2, 6, n) * UI_SCALE).astype(int))
        self.amp = rng.uniform(18, 50, n)
        self.stamps: Dict[int, np.ndarray] = {}
        for r in set(int(v) for v in self.rad):
            yy, xx = np.mgrid[-r * 2:r * 2 + 1, -r * 2:r * 2 + 1]
            self.stamps[r] = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.8) ** 2)).astype(np.float32)
        self.tint = np.array((0.35, 0.9, 1.0), np.float32)
        self.speed = 14 * UI_SCALE

    def frame(self, t: float) -> np.ndarray:
        W, H = VIDEO_W, VIDEO_H
        off = int(t * self.speed) % W
        out = self.strip[:, off:off + W].astype(np.int16)
        gy = int(t * 5 * UI_SCALE) % self.cell
        out += self.grid[gy:gy + H, :, None]
        for i in range(len(self.x0)):
            cx = int(self.x0[i] + self.sway[i] * math.sin(0.4 * t + self.phase[i])) % W
            cy = int(self.y0[i] - self.vy[i] * t) % H
            r = int(self.rad[i])
            st = self.stamps[r]
            k = st.shape[0] // 2
            twinkle = 0.55 + 0.45 * math.sin(1.3 * t + self.phase[i] * 2)
            x0, y0, x1, y1 = max(0, cx - k), max(0, cy - k), min(W, cx + k + 1), min(H, cy + k + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = st[y0 - (cy - k):y1 - (cy - k), x0 - (cx - k):x1 - (cx - k)]
            out[y0:y1, x0:x1] += (sub[..., None] * self.tint * (self.amp[i] * twinkle)).astype(np.int16)
        return np.clip(out, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def get_background() -> BackgroundFX:
    return BackgroundFX()


def attach_background(fig, t: float):
    """Pillow/NumPy background under every axes (axes faces are translucent, figure image has zorder -1)."""
    return fig.figimage(get_background().frame(t), xo=0, yo=0, origin="upper", zorder=-1)


# ---------------------------------------------------------------------
# Scrolling ticker
# ---------------------------------------------------------------------
class NewsTickerBar:
    def __init__(self, fig, raw_text: str, label: str = "DAILY RECAP", scroll_speed: float = 0.12,
                 fontsize: int = 18):
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
        unit = f"{label}:  {mpl_safe(raw_text)}   \u2022   "
        self._text = ax.text(0.02, 0.5, unit, fontsize=fontsize, fontweight="bold", color=C_GREEN,
                             va="center", clip_on=True, transform=ax.transData)
        renderer = fig.canvas.get_renderer()
        w1 = self._text.get_window_extent(renderer).width
        self._text.set_text(unit * 2)
        w2 = self._text.get_window_extent(renderer).width
        self.period = max((w2 - w1) / ax.bbox.width, 1e-3)  # axes-width units: the scroll wraps seamlessly
        self._text.set_text(unit * (int(math.ceil(1.0 / self.period)) + 2))

    def update(self, global_t: float) -> None:
        self._text.set_x(0.02 - (global_t * self.speed) % self.period)


# ---------------------------------------------------------------------
# Avatar. The mouth-state cache is CLASS level so it can be cleared (and RAM reclaimed) between chapters.
# ---------------------------------------------------------------------
class AvatarPanel:
    _cache: Dict[Tuple[int, bool, int], np.ndarray] = {}

    def __init__(self, size: int, framed: bool = True):
        self.size, self.framed = size, framed
        self._open = self._load(AVATAR_OPEN_PATH, (0, 229, 255, 255))
        self._closed = self._load(AVATAR_CLOSED_PATH, (0, 160, 190, 255))
        self._bg = Image.new("RGB", (size, size), (22, 27, 34))
        ImageDraw.Draw(self._bg).rectangle((0, 0, size - 1, size - 1), outline=(48, 54, 61), width=max(1, px(4)))

    def _load(self, path: str, fallback_rgba: Tuple[int, int, int, int]) -> Image.Image:
        try:
            with Image.open(path) as im:
                return im.convert("RGBA").resize((self.size, self.size))
        except Exception as e:  # missing asset: coloured placeholder, never fatal
            logger.warning("Avatar image missing (%s): %s - using a placeholder", path, e)
            ph = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
            d = ImageDraw.Draw(ph)
            m = self.size // 5
            d.ellipse((m, m, self.size - m, self.size - m), fill=fallback_rgba)
            if fallback_rgba[1] > 200:  # "open" placeholder gets an open mouth
                d.ellipse((self.size * 0.42, self.size * 0.55, self.size * 0.58, self.size * 0.70),
                          fill=(11, 14, 20, 255))
            else:
                d.line((self.size * 0.40, self.size * 0.62, self.size * 0.60, self.size * 0.62),
                       fill=(11, 14, 20, 255), width=max(2, self.size // 40))
            return ph

    def frame(self, is_open: bool, bob_px: int) -> np.ndarray:
        step = max(1, px(4))
        bob_px = max(0, (bob_px + step // 2) // step * step)
        key = (self.size, is_open, bob_px)
        if key not in AvatarPanel._cache:
            sprite = self._open if is_open else self._closed
            out = self._bg.copy()
            out.paste(sprite, (0, -bob_px), sprite.getchannel("A"))
            AvatarPanel._cache[key] = np.asarray(out, dtype=np.uint8)
        return AvatarPanel._cache[key]


def release_chapter_memory() -> None:
    """Called between chapters: drop the avatar frame cache and force a full garbage collection."""
    AvatarPanel._cache.clear()
    gc.collect()
    log_memory("chapter memory released")


def lip_sync(peaks: np.ndarray, t: float, amp: float = 50.0) -> Tuple[bool, int]:
    idx = int(t * OUTPUT_FPS)
    vol = float(peaks[idx]) if idx < len(peaks) else 0.0
    speaking = vol > 0.08
    return speaking, (px(min(max(vol * amp, 4), 20)) if speaking else 0)


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


@lru_cache(maxsize=64)
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


@lru_cache(maxsize=1)
def _mpl_bold_font_path() -> Optional[str]:
    try:
        from matplotlib import font_manager
        return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    except Exception:
        return find_system_font()


@lru_cache(maxsize=128)
def ref_font(size: int):
    """Measuring font on the 1080 reference grid (1 reference pixel == 1 font pixel)."""
    path = _mpl_bold_font_path()
    try:
        if path:
            return ImageFont.truetype(path, int(size))
    except OSError:
        pass
    return load_font(int(size))


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


def _ellipsize(text: str, font, max_w: float) -> str:
    while len(text) > 1 and font.getlength(text + "...") > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


def fit_wrap(text: str, max_w: float, max_h: float, max_size: int, min_size: int = 12,
             pitch: float = 1.25) -> Tuple[List[str], int]:
    """Largest font size (reference px) at which `text` wraps into max_w x max_h, with NO line wider than max_w.
    At the minimum size anything still too wide is cut with '...' so text can never leave its box."""
    size = int(max_size)
    while True:
        font = ref_font(size)
        lines = _wrap_text(text, font, max_w) or [""]
        widest = max(font.getlength(l) for l in lines)
        if (len(lines) * size * pitch <= max_h and widest <= max_w) or size <= min_size:
            if widest > max_w:
                lines = [_ellipsize(l, font, max_w) if font.getlength(l) > max_w else l for l in lines]
            return lines, size
        size -= 2


def fit_one(text: str, max_w: float, max_h: float, max_size: int, min_size: int = 12) -> int:
    """Largest font size (reference px) at which a SINGLE line fits max_w x max_h."""
    size = int(max_size)
    while size > min_size and (ref_font(size).getlength(text) > max_w or size * 1.15 > max_h):
        size -= 2
    return max(size, min_size)


def draw_lines(ax, lines: List[str], cx: float, cy: float, size: float, color: str, pitch: float = 1.25,
               ha: str = "center", **kw) -> List:
    n = len(lines)
    return [ax.text(cx, cy + (i - (n - 1) / 2) * size * pitch, mpl_safe(line), fontsize=size * REF_PT,
                    fontweight="bold", color=color, ha=ha, va="center", **kw)
            for i, line in enumerate(lines)]


# ---------------------------------------------------------------------
# Captions: pre-rendered once with Pillow (white text, black outline), timed from the real sentence timings
# ---------------------------------------------------------------------
@dataclass
class CaptionCue:
    start: float
    end: float
    x: int
    y: int
    rgba: np.ndarray


class CaptionTrack:
    BASE_SIZE = 58  # reference px
    MIN_SIZE = 26
    CHUNK_WORDS = 6  # a caption shows at most this many words at once (Shorts-style pacing)

    def __init__(self, text: str, sentence_times: Optional[List[Tuple[float, float]]], speech_dur: float,
                 box: Tuple[int, int, int, int] = SUB_BOX):
        self.box_x, self.box_y, self.box_w, self.box_h = box
        clean = ascii_clean(text)
        sentences = split_sentences(clean) or ([clean] if clean else [])
        if sentence_times and len(sentence_times) == len(sentences):
            windows = [(float(a), float(b)) for a, b in sentence_times]
        else:  # proportional fallback
            counts = [max(1, len(s.split())) for s in sentences]
            total, cur, windows = max(1, sum(counts)), 0.0, []
            for c in counts:
                d = speech_dur * c / total
                windows.append((cur, cur + d))
                cur += d
        self.cues: List[CaptionCue] = []
        for sent, (t0, t1) in zip(sentences, windows):
            words = sent.split()
            n_chunks = max(1, math.ceil(len(words) / self.CHUNK_WORDS))
            per = math.ceil(len(words) / n_chunks)
            chunks = [" ".join(words[i:i + per]) for i in range(0, len(words), per)]
            weights = [max(1, len(c)) for c in chunks]
            cur = t0
            for chunk, w in zip(chunks, weights):
                d = (t1 - t0) * w / sum(weights)
                cue = self._render_cue(chunk, cur, cur + d)
                if cue:
                    self.cues.append(cue)
                cur += d

    def _render_cue(self, text: str, start: float, end: float) -> Optional[CaptionCue]:
        try:
            stroke = max(2, px(5))
            size = max(self.MIN_SIZE, int(self.BASE_SIZE * UI_SCALE))
            min_size = max(12, int(self.MIN_SIZE * UI_SCALE))
            while True:
                font = load_font(size)
                lines = _wrap_text(text, font, self.box_w - 2 * stroke)
                line_h = int(_font_height(font) * 1.12)
                if (line_h * len(lines) <= self.box_h and max(font.getlength(l) for l in lines) <= self.box_w) \
                        or size <= min_size:
                    break
                size -= 2
            img = Image.new("RGBA", (self.box_w, self.box_h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            y = max(0, (self.box_h - line_h * len(lines)) // 2)
            for line in lines:
                x = (self.box_w - d.textlength(line, font=font)) / 2
                d.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=stroke,
                       stroke_fill=(0, 0, 0, 255))
                y += line_h
            bbox = img.getbbox()
            if not bbox:
                return None
            return CaptionCue(start, end, self.box_x + bbox[0], self.box_y + bbox[1],
                              np.asarray(img.crop(bbox)))
        except Exception as e:
            logger.warning(f"Failed to render caption '{text}': {e}")
            return None

    def overlay(self, frame: np.ndarray, t: float) -> None:
        cue = None
        for c in self.cues:  # the last cue that has started stays up until the next one
            if c.start <= t + 1e-6:
                cue = c
        if cue is not None:
            blend_rgba(frame, cue.rgba, cue.x, cue.y)


# =====================================================================
# 11. SCREENS ("layouts"), authored for the 1000x800 reference stage of a 1080x1920 frame
#     Each layout builds its artists ONCE; update(t) tweaks them and paint(frame, t) may add Pillow overlays.
#     `t` is segment-local. Everything builds in well under a second; the narration timing (sentence_times) only
#     decides WHAT is emphasised (flash / dim / light up), never how fast things appear.
# =====================================================================
def make_stage_axes(fig, box: Tuple[int, int, int, int], ref_w: float, ref_h: float, border: bool = True,
                    face=STAGE_FACE):
    """Axes whose data units are reference pixels, y pointing DOWN (author it like a web page)."""
    ax = fig.add_axes(axes_rect(box))
    ax.set_facecolor(face)
    ax.set_xlim(0, ref_w)
    ax.set_ylim(ref_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border)
        spine.set_color(C_BORDER)
        spine.set_linewidth(2)
    return ax


class Reveal:
    """A group of artists that fades in together."""

    def __init__(self, start: float, length: float = 0.25):
        self.start, self.length, self.artists = start, length, []

    def add(self, *arts):
        for a in arts:
            if isinstance(a, (list, tuple)):
                self.artists.extend(a)
            elif a is not None:
                self.artists.append(a)
        return self

    def apply(self, t: float, mult: float = 1.0) -> float:
        a = ease_out(ramp(t, self.start, self.length)) * mult
        for art in self.artists:
            art.set_alpha(a)
        return a


class MentionTrack:
    """Finds WHEN each keyword is spoken (sentence_times + word position inside the sentence) so the right card /
    sector / ticker can flash exactly while the voice talks about it."""

    def __init__(self, sentences: List[str], times: List[Tuple[float, float]], patterns: List[Tuple[str, Any]],
                 default_key: Any = None, hold: float = 0.35, lead: float = 0.12):
        regs = [(re.compile(r"(?<![a-z0-9])" + r"\s+".join(re.escape(w) for w in p.split()) + r"(?![a-z0-9])"), key)
                for p, key in sorted(patterns, key=lambda x: -len(x[0])) if p.strip()]
        events: List[list] = []
        for sent, (t0, t1) in zip(sentences, times):
            low, taken, found = sent.lower(), [], []
            for rg, key in regs:  # longest patterns first, no overlapping matches
                for m in rg.finditer(low):
                    if not any(m.start() < b and a < m.end() for a, b in taken):
                        taken.append((m.start(), m.end()))
                        found.append((m.start(), key))
            n_words = max(1, len(sent.split()))
            for pos, key in sorted(found, key=lambda x: x[0]):
                tt = t0 + (t1 - t0) * len(sent[:pos].split()) / n_words - lead
                events.append([max(0.0, tt), 0.0, key, t1])
        for i, e in enumerate(events):
            nxt = events[i + 1][0] if i + 1 < len(events) else None
            e[1] = min(nxt, e[3] + hold) if nxt is not None else e[3] + hold
        self.events = [(a, b, k) for a, b, k, _ in events]
        self.default_key = default_key

    def active(self, t: float) -> Tuple[Any, float]:
        best = (None, 0.0)
        for t0, t1, key in self.events:
            if t0 <= t < t1:
                s = ease_out(ramp(t, t0, 0.15)) * (1 - ease_out(ramp(t, t1 - 0.15, 0.15)))
                best = (key, s)
        if best[0] is None and not self.events and self.default_key is not None and t >= 0.8:
            return self.default_key, ease_out(ramp(t, 0.8, 0.3))
        return best


class LayoutBase:
    header_sub: Optional[str] = None
    header_right: Optional[Tuple[str, str, str]] = None  # (line 1, line 2, line-2 colour) shown in the info card
    info_label: Optional[str] = None

    def __init__(self, fig, seg: dict, duration: float, speech: float, data: Optional[dict] = None,
                 sentence_times: Optional[List[Tuple[float, float]]] = None):
        self.fig, self.seg, self.data = fig, seg, data
        self.duration = duration
        self.dur = max(speech, 0.5)
        self.accent = theme_color(seg.get("accent"), C_GOLD)
        self.sentences = split_sentences(seg.get("script", "")) or [seg.get("script", "")]
        self.sentence_times = self._fix_times(sentence_times)

    def _fix_times(self, st) -> List[Tuple[float, float]]:
        n = len(self.sentences)
        if st and len(st) == n:
            return [(float(a), float(b)) for a, b in st]
        words = [max(1, len(s.split())) for s in self.sentences]
        tot, cur, out = sum(words), 0.0, []
        for w in words:
            d = self.dur * w / tot
            out.append((cur, cur + d))
            cur += d
        return out

    def track(self, patterns, default_key=None) -> MentionTrack:
        return MentionTrack(self.sentences, self.sentence_times, patterns, default_key)

    def sentence_at(self, t: float) -> int:
        idx = 0
        for i, (a, _b) in enumerate(self.sentence_times):
            if t >= a:
                idx = i
        return idx

    def update(self, t: float) -> None:
        pass

    def paint(self, frame: np.ndarray, t: float) -> None:
        pass


def _anchor_point(anchor: str, highs, lows, closes) -> Tuple[int, float]:
    n = len(closes)
    if anchor == "high":
        i = int(np.argmax(highs))
        return i, float(highs[i])
    if anchor == "low":
        i = int(np.argmin(lows))
        return i, float(lows[i])
    if anchor == "start":
        return 0, float(closes[0])
    if anchor in ("biggest_drop", "drop"):
        i = int(np.argmin(closes / np.maximum.accumulate(closes) - 1.0))
        return i, float(closes[i])
    return n - 1, float(closes[-1])


# ---------------------------------------------------------------------
# Candle pane (candles + volume + moving averages + live price tag)
# ---------------------------------------------------------------------
class CandlePane:
    """Everything is drawn once; update(k) reveals k candles."""

    def __init__(self, fig, d: dict, price_ref, vol_ref, ma_lines: List[Tuple[str, str, Any]], title: str,
                 cur: str = "$"):
        o, h, lo, c, v = (np.asarray(d[k], dtype=float) for k in ("open", "high", "low", "close", "volume"))
        n = self.n = len(c)
        self.x, self.closes, self.cur = np.arange(n), c, cur
        ax = self.ax = fig.add_axes(axes_rect(px_box(price_ref)))
        vax = self.vax = fig.add_axes(axes_rect(px_box(vol_ref)), sharex=ax)
        style_axes(ax)
        style_axes(vax)
        for a in (ax, vax):
            a.yaxis.tick_right()
            a.tick_params(labelsize=15)
        ax.tick_params(labelbottom=False)
        colors = [C_UP if cl >= op else C_DOWN for op, cl in zip(o, c)]
        rng = max(float(h.max() - lo.min()), float(h.max()) * 0.002, 0.01)
        self.segs = [[(float(i), lo[i]), (float(i), h[i])] for i in range(n)]
        self.wicks = ax.vlines(self.x, lo, h, colors=colors, linewidth=2.0)
        body = c - o
        body = np.where(np.abs(body) < rng * 0.002, rng * 0.002, body)  # dojis stay visible
        self.bodies = ax.bar(self.x, body, bottom=o, color=colors, edgecolor=colors, width=0.68)
        self.has_vol = float(v.max()) > 0
        self.vbars = vax.bar(self.x, v if self.has_vol else np.zeros(n), color=colors, width=0.68, alpha=0.6)
        self.mas = []
        for i, (name, color, arr) in enumerate(ma_lines):
            line, = ax.plot([], [], color=color, linewidth=2.8, zorder=6)
            vals = np.asarray([np.nan if x_ is None else x_ for x_ in arr], dtype=float)
            self.mas.append((line, vals))
            ax.text(1.0 - 0.12 * (len(ma_lines) - 1 - i), 1.03, name, transform=ax.transAxes,
                    fontsize=22 * REF_PT, fontweight="bold", color=color, ha="right", va="bottom", clip_on=False)
        ax.text(0.0, 1.03, mpl_safe(title), transform=ax.transAxes, fontsize=28 * REF_PT, fontweight="bold",
                color=C_WHITE, ha="left", va="bottom", clip_on=False)
        pad = max(2.0, n * 0.07)
        ax.set_xlim(-1, n - 1 + pad)
        self.y_lo, self.y_hi = float(lo.min()) - rng * 0.08, float(h.max()) + rng * 0.20  # headroom, no overlap
        ax.set_ylim(self.y_lo, self.y_hi)
        vax.set_ylim(0, (float(v.max()) * 1.2) if self.has_vol else 1)
        vax.set_xticks(d["tick_idx"])
        vax.set_xticklabels(d["tick_labels"])
        vax.set_yticks([])
        vax.text(0.008, 0.9, "VOLUME" if self.has_vol else "NO VOLUME DATA", transform=vax.transAxes,
                 fontsize=20 * REF_PT, fontweight="bold", color=C_MUTED, ha="left", va="top")
        big = float(h.max()) >= 1000
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}" if big else f"{val:,.2f}"))
        ax.text(0.5, 0.5, mpl_safe(str(d["ticker"]).lstrip("^")), transform=ax.transAxes, fontsize=150 * REF_PT,
                fontweight="bold", color=C_WHITE, alpha=0.05, ha="center", va="center", zorder=1)
        self.trend = C_UP if c[-1] >= c[0] else C_DOWN
        self.live_line = ax.axhline(y=c[0], color=self.trend, linestyle=":", linewidth=1.8, alpha=0.7)
        self.dot, = ax.plot([], [], marker="o", markersize=9, linestyle="none", color=self.trend, zorder=10)
        self.halo, = ax.plot([], [], marker="o", markersize=22, linestyle="none", color=self.trend,
                             markeredgewidth=0, alpha=0.3, zorder=9)
        self.tag = ax.text(1.005, c[0], "", transform=ax.get_yaxis_transform(), color=C_BG, fontsize=18 * REF_PT,
                           fontweight="bold", ha="left", va="center", zorder=12, clip_on=False, visible=False,
                           bbox=dict(boxstyle="round,pad=0.25", facecolor=self.trend, edgecolor="none"))
        self._shown = None

    def include_levels(self, levels: Sequence[float]) -> None:
        rng = self.y_hi - self.y_lo
        self.y_lo = min(self.y_lo, min(levels) - rng * 0.05)
        self.y_hi = max(self.y_hi, max(levels) + rng * 0.08)
        self.ax.set_ylim(self.y_lo, self.y_hi)

    def update(self, k: int, t: float) -> None:
        n = self.n
        if k != self._shown:
            self.wicks.set_segments(self.segs[:k])
            for i, (rect, vb) in enumerate(zip(self.bodies, self.vbars)):
                rect.set_visible(i < k)
                vb.set_visible(i < k)
            for line, vals in self.mas:
                line.set_data(self.x[:k], vals[:k])
            self._shown = k
        last = float(self.closes[k - 1])
        self.live_line.set_ydata([last, last])
        self.dot.set_data([k - 1], [last])
        self.halo.set_data([k - 1], [last])
        self.halo.set_alpha(0.12 + 0.28 * abs(math.sin(math.pi * t * 1.2)))
        self.tag.set_visible(k >= n)  # the price label box only exists once the chart is built
        self.tag.set_text(mpl_safe(f" {self.cur}{last:,.2f} "))
        self.tag.set_y(min(max(last, self.y_lo), self.y_hi))
        self.tag.get_bbox_patch().set_facecolor(C_UP if last >= self.closes[0] else C_DOWN)


# --- 1. PRO CANDLESTICK CHART ------------------------------------------
class ProCandlestickLayout(LayoutBase):
    """The whole chart builds in under half a second; annotations and the reference line pop up right after."""
    NOTE_COLORS = {"high": C_GREEN, "low": C_RED, "biggest_drop": C_RED, "start": C_CYAN, "end": C_GOLD}
    BUILD_START = 0.08

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        make_stage_axes(fig, STAGE_BOX, *STAGE_REF)
        c = np.asarray(data["close"], dtype=float)
        o, h, lo = (np.asarray(data[k], dtype=float) for k in ("open", "high", "low"))
        n = len(c)

        def sma(a, w):
            out = np.full(len(a), np.nan)
            if len(a) >= w:
                out[w - 1:] = np.convolve(a, np.ones(w) / w, mode="valid")
            return out

        fast = max(3, min(9, n // 5))
        slow = max(fast + 2, min(21, n // 2))
        label = {"1d": "1D", "5d": "5D", "1mo": "1M", "3mo": "3M"}.get(data["timeframe"], data["timeframe"].upper())
        cur = "" if data["ticker"].startswith("^") else "$"
        self.pane = CandlePane(fig, data, (60, 400, 810, 490), (60, 900, 810, 170),
                               [(f"MA{fast}", C_AMBER, sma(c, fast)), (f"MA{slow}", C_CYAN, sma(c, slow))],
                               f"{data['ticker']} \u00b7 {data['interval']}", cur)
        ax, pane = self.pane.ax, self.pane
        ref = float(data.get("ref_close") or o[0])
        pane.include_levels([ref])
        self.ref_line = ax.axhline(ref, color=C_GOLD, linestyle="--", linewidth=2.2, alpha=0.0, zorder=5)
        ref_name = {"vs previous close": "PREV CLOSE", "vs last week's close": "WEEK START"}.get(
            data.get("ref_label", ""), "PERIOD START")
        self.ref_txt = ax.text(0.01, ref, f" {ref_name} ", transform=ax.get_yaxis_transform(), color=C_BG,
                               fontsize=17 * REF_PT, fontweight="bold", ha="left", va="bottom", zorder=11,
                               visible=False, bbox=dict(boxstyle="round,pad=0.2", facecolor=C_GOLD, edgecolor="none"))
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        self.notes, used = [], set()
        for note in seg.get("chart_notes", []):
            anchor = str(note["anchor"]).strip().lower()
            idx, y = _anchor_point(anchor, h, lo, c)
            if idx in used:
                continue
            used.add(idx)
            col = self.NOTE_COLORS.get(anchor, C_GOLD)
            side = -1 if idx > n * 0.5 else 1
            tx = min(max(idx + side * 0.2 * span, -1 + 0.18 * span), ax.get_xlim()[1] - 0.18 * span)
            ty = y + (0.16 if anchor in ("high", "start", "end") else -0.16) * (pane.y_hi - pane.y_lo)
            ty = min(max(ty, pane.y_lo + 0.07 * (pane.y_hi - pane.y_lo)), pane.y_hi - 0.10 * (pane.y_hi - pane.y_lo))
            ann = ax.annotate(mpl_safe("\n".join(_wrap_text(note["text"], ref_font(22), 260))), xy=(idx, y),
                              xytext=(tx, ty), arrowprops=dict(arrowstyle="-|>", color=col, lw=3), fontsize=22 * REF_PT,
                              fontweight="bold", color=col, ha="center", va="center", zorder=12, annotation_clip=False,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BG, edgecolor=col, alpha=0.9))
            ann.set_visible(False)  # a hidden annotation also hides its box and arrow
            self.notes.append(ann)
        pct = (c[-1] / ref - 1) * 100
        self.header_sub = f"{data['ticker']}  |  {label} {data.get('ref_label', '')}"
        self.info_label = f"{data['ticker']} \u00b7 {label}" + (" (daily fallback)" if data.get("fallback") else "")
        self.header_right = (f"{cur}{c[-1]:,.2f}", f"{pct:+.2f}%", C_GREEN if pct >= 0 else C_RED)

    def update(self, t: float) -> None:
        n = self.pane.n
        k = min(n, max(2, int(math.ceil(ease_out(ramp(t, self.BUILD_START, CHART_BUILD_SECONDS)) * n))))
        self.pane.update(k, t)
        build_end = self.BUILD_START + CHART_BUILD_SECONDS
        ra = ease_out(ramp(t, build_end, 0.25))
        self.ref_line.set_alpha(0.85 * ra)
        self.ref_txt.set_visible(t >= build_end)
        self.ref_txt.set_alpha(ra)
        for i, ann in enumerate(self.notes):
            t0 = build_end + 0.25 + 0.25 * i
            a = ease_out(ramp(t, t0, 0.25))
            ann.set_visible(t >= t0)
            ann.set_alpha(a)
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_alpha(a)
            if ann.get_bbox_patch() is not None:
                ann.get_bbox_patch().set_alpha(0.9 * a)


# --- 2. STOCK DASHBOARD ------------------------------------------------
DASH_ALIASES = [
    ("pe_ratio", ["p e ratio", "p e", "pe ratio", "price to earnings", "earnings multiple", "multiple"]),
    ("market_cap", ["market capitalization", "market capitalisation", "market cap", "valuation", "trillion"]),
    ("dividend_yield", ["dividend", "yield", "payout"]),
    ("week_change", ["this week", "the week", "weekly", "week", "five sessions", "five days"]),
    ("change", ["today", "daily", "single day", "last session", "one day", "the day", "session"]),
    ("price", ["share price", "price", "trading at", "closed at"]),
]


class StockDashboardLayout(LayoutBase):
    """Card grid for ONE company: a wide ticker card, then 2-column metric cards. Only cards that have data exist.
    The card the narration is talking about glows while it is mentioned, the others dim."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        d = data
        up = d["change_pct"] >= 0
        wk = d.get("week_pct")
        metrics = [dict(key="price", label="PRICE", text=f"${d['price']:,.2f}", color=C_WHITE),
                   dict(key="change", label="TODAY", text=f"{d['change_pct']:+.2f}%", color=C_GREEN if up else C_RED)]
        if wk is not None:
            metrics.append(dict(key="week_change", label="THIS WEEK", text=f"{wk:+.2f}%",
                                color=C_GREEN if wk >= 0 else C_RED))
        if d.get("market_cap"):
            metrics.append(dict(key="market_cap", label="MARKET CAP", text=fmt_big(d["market_cap"]), color=C_GOLD))
        if d.get("pe"):
            metrics.append(dict(key="pe_ratio", label="P/E RATIO", text=f"{d['pe']:.1f}", color=C_CYAN))
        if d.get("div_yield") is not None:
            metrics.append(dict(key="dividend_yield", label="DIVIDEND YIELD", text=f"{d['div_yield']:.2f}%",
                                color=C_GREEN))
        m, gap, head_h = 26.0, 22.0, 210.0
        rows = (len(metrics) + 1) // 2
        row_h = min(200.0, (H - 2 * m - head_h - gap * rows) / rows)
        y0 = (H - (head_h + gap * rows + row_h * rows)) / 2
        self.cards, self.glows, self.logo = [], [], None
        head = FancyBboxPatch((m, y0), W - 2 * m, head_h, boxstyle="round,pad=0,rounding_size=26", fc=C_PANEL,
                              ec=C_BORDER, lw=2.5)
        ax.add_patch(head)
        grp = Reveal(0.0, 0.25).add(head)
        self._ticker_card(ax, grp, m, y0, W - 2 * m, head_h, d)
        self.cards.append(("ticker", grp))
        slot = 1
        for ri in range(rows):
            cnt = 2 if (ri < rows - 1 or len(metrics) % 2 == 0) else 1
            cw = (W - 2 * m - gap * (cnt - 1)) / cnt
            y = y0 + head_h + gap + ri * (row_h + gap)
            for ci in range(cnt):
                spec = metrics[ri * 2 + ci]
                x = m + ci * (cw + gap)
                box = FancyBboxPatch((x, y), cw, row_h, boxstyle="round,pad=0,rounding_size=26", fc=C_PANEL,
                                     ec=C_BORDER, lw=2.5)
                ax.add_patch(box)
                grp = Reveal(0.05 * slot, 0.25).add(box)
                grp.add(ax.text(x + 28, y + 40, spec["label"], fontsize=30 * REF_PT, fontweight="bold", color=C_MUTED,
                                ha="left", va="center"))
                size = fit_one(spec["text"], cw - 56, row_h * 0.48, 120, 28)
                grp.add(ax.text(x + cw / 2, y + row_h * 0.62, mpl_safe(spec["text"]), fontsize=size * REF_PT,
                                fontweight="bold", color=spec["color"], ha="center", va="center"))
                glow = []
                for g, lw, a_ in ((6, 6.0, 1.0), (14, 3.6, 0.65), (24, 2.0, 0.4)):
                    gp = FancyBboxPatch((x - g, y - g), cw + 2 * g, row_h + 2 * g,
                                        boxstyle=f"round,pad=0,rounding_size={26 + g}", fc="none",
                                        ec=spec["color"] if spec["key"] != "price" else self.accent, lw=lw, alpha=0.0)
                    ax.add_patch(gp)
                    glow.append((gp, a_))
                self.glows.append((spec["key"], glow))
                self.cards.append((spec["key"], grp))
                slot += 1
        present = [c["key"] for c in metrics]
        patterns = [(p, key) for key, plist in DASH_ALIASES if key in present for p in plist]
        fallback = seg.get("highlight_metric") if seg.get("highlight_metric") in present else None
        self.mentions = self.track(patterns, fallback)
        self.header_sub = f"{d['ticker']}  |  SNAPSHOT"
        self.info_label = d["name"].upper()
        self.header_right = (f"${d['price']:,.2f}", f"{d['change_pct']:+.2f}% today", C_GREEN if up else C_RED)

    def _ticker_card(self, ax, grp: Reveal, x, y, cw, ch, d) -> None:
        cx, cy, r = x + 135, y + ch / 2, min(82.0, ch * 0.38)
        ticker = d["ticker"]
        if d.get("logo"):
            try:
                size = px(r * 2)
                with Image.open(d["logo"]) as im:
                    arr = np.asarray(im.convert("RGBA").resize((size, size)))
                self.logo = (arr, STAGE_BOX[0] + px(cx - r), STAGE_BOX[1] + px(cy - r), grp)
            except Exception as e:
                logger.warning(f"Logo for {ticker} unreadable: {e}")
        if self.logo is None:  # coloured placeholder disc with initials
            rnd = random.Random(ticker)
            hue = "#%02x%02x%02x" % (rnd.randint(40, 200), rnd.randint(90, 220), rnd.randint(120, 255))
            disc = Circle((cx, cy), r, fc=hue, ec=C_WHITE, lw=3)
            ax.add_patch(disc)
            grp.add(disc, ax.text(cx, cy, mpl_safe(ticker[:3]), fontsize=r * 0.6 * REF_PT, fontweight="bold",
                                  color=C_BG, ha="center", va="center"))
        tx = x + 255
        size = fit_one(ticker, cw - 290, ch * 0.42, 96, 30)
        grp.add(ax.text(tx, y + ch * 0.40, mpl_safe(ticker), fontsize=size * REF_PT, fontweight="bold",
                        color=C_WHITE, ha="left", va="center"))
        size2 = fit_one(d["name"], cw - 290, ch * 0.2, 36, 14)
        grp.add(ax.text(tx, y + ch * 0.76, mpl_safe(d["name"]), fontsize=size2 * REF_PT, fontweight="bold",
                        color=C_MUTED, ha="left", va="center"))

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        for ck, grp in self.cards:
            grp.apply(t, 1.0 if key is None or ck == key or ck == "ticker" else 1 - 0.62 * s)
        pulse = 0.7 + 0.3 * math.sin(2 * math.pi * t * 1.2)
        for gk, glow in self.glows:
            on = s * pulse if gk == key else 0.0
            for gp, a_ in glow:
                gp.set_alpha(on * a_)

    def paint(self, frame: np.ndarray, t: float) -> None:
        if self.logo is not None:
            arr, x, y, grp = self.logo
            blend_rgba(frame, arr, x, y, ease_out(ramp(t, grp.start, grp.length)))


# --- 3. SECTOR HEATMAP -------------------------------------------------
def treemap_rects(weights: List[float], x: float, y: float, w: float, h: float) -> List[
    Tuple[float, float, float, float]]:
    """Binary-partition treemap. `weights` must be sorted descending. Returns one rect per weight."""
    n = len(weights)
    if n == 1:
        return [(x, y, w, h)]
    total, acc, k = sum(weights), 0.0, n - 1
    for i in range(1, n):
        acc += weights[i - 1]
        if acc >= total / 2:
            k = i
            break
    frac = sum(weights[:k]) / total
    if w >= h:
        w1 = w * frac
        return treemap_rects(weights[:k], x, y, w1, h) + treemap_rects(weights[k:], x + w1, y, w - w1, h)
    h1 = h * frac
    return treemap_rects(weights[:k], x, y, w, h1) + treemap_rects(weights[k:], x, y + h1, w, h - h1)


class SectorHeatmapLayout(LayoutBase):
    """The real 11-sector map. Blocks pop in within a fraction of a second; the sector the narration names flashes
    (gold outline) while the others dim. All text is fitted to its block."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        items = sorted(data["sectors"], key=lambda s: -float(s["weight"]))
        weights = [float(s["weight"]) ** 0.55 for s in items]  # compressed so small sectors stay readable
        rects = treemap_rects(weights, 18, 18, W - 36, H - 36)
        maxabs = max(1.0, max(abs(float(s["pct"])) for s in items))
        self.blocks, patterns = [], []
        for i, (s, (x, y, w, h)) in enumerate(zip(items, rects)):
            chg = float(s["pct"])
            col = C_GREEN if chg >= 0 else C_RED
            strength = min(1.0, abs(chg) / maxabs)
            g = 4
            bx, by, bw, bh = x + g, y + g, w - 2 * g, h - 2 * g
            box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=14",
                                 fc=blend_hex(col, C_BG, 0.80 - 0.42 * strength), ec=col, lw=2.5)
            ax.add_patch(box)
            name_lines, name_size = fit_wrap(s["name"], bw - 24, bh * 0.40, 44, 11)
            pct_size = fit_one(f"{chg:+.2f}%", bw - 24, bh * 0.30, 84, 11)
            name_arts = draw_lines(ax, name_lines, bx + bw / 2, by + bh * 0.30, name_size, C_WHITE)
            pct = ax.text(bx + bw / 2, by + bh * 0.70, "", fontsize=pct_size * REF_PT, fontweight="bold", color=col,
                          ha="center", va="center")
            flash = FancyBboxPatch((bx + 2, by + 2), bw - 4, bh - 4, boxstyle="round,pad=0,rounding_size=14",
                                   fc="none", ec=C_GOLD, lw=7, alpha=0.0)
            ax.add_patch(flash)
            grp = Reveal(0.04 * i, 0.25).add(box, name_arts, pct)
            self.blocks.append(dict(grp=grp, pct=pct, chg=chg, flash=flash, t0=0.04 * i))
            for alias in ETF_ALIASES.get(s["etf"], []) + [s["name"].lower()]:
                patterns.append((alias, i))
        self.mentions = self.track(patterns)
        top = max(items, key=lambda s: abs(float(s["pct"])))
        self.header_sub = "S&P 500 SECTORS  |  TODAY"
        self.info_label = "BIGGEST SECTOR MOVE"
        self.header_right = (shorten(top["name"], 18), f"{float(top['pct']):+.2f}%",
                             C_GREEN if float(top["pct"]) >= 0 else C_RED)

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        pulse = 0.75 + 0.25 * math.sin(2 * math.pi * t * 1.4)
        for i, b in enumerate(self.blocks):
            b["grp"].apply(t, 1.0 if key is None or i == key else 1 - 0.55 * s)
            p = ease_out(ramp(t, b["t0"], 0.5))
            b["pct"].set_text(f"{b['chg'] * p:+.2f}%")
            b["flash"].set_alpha(s * pulse if i == key else 0.0)


# --- 4. TOP MOVERS (gainers over losers, or a single-side ranking) -------
class TopMoversLayout(LayoutBase):
    """movers_side = both: gainers board over losers board. gainers / losers: one ranking with up to 5 rows."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        side = seg.get("movers_side", "both")
        gain = [(r["ticker"], float(r["pct"]), r.get("name", "")) for r in data["gainers"]]
        loss = [(r["ticker"], float(r["pct"]), r.get("name", "")) for r in data["losers"]]
        if side == "both":
            panels = [(gain[:3], C_GREEN, "TOP GAINERS", 18, 392), (loss[:3], C_RED, "TOP LOSERS", 408, 782)]
        elif side == "gainers":
            panels = [(gain[:5], C_GREEN, "TOP GAINERS TODAY", 18, 782)]
        else:
            panels = [(loss[:5], C_RED, "TOP LOSERS TODAY", 18, 782)]
        self.vmax = max([abs(p) for ent, *_ in panels for _, p, _ in ent] + [0.5])
        self.bars, self.titles = [], Reveal(0.0, 0.25)
        for entries, color, title, y0, y1 in panels:
            self._panel(ax, W, entries, color, title, y0, y1)
        patterns = []
        for i, b in enumerate(self.bars):
            patterns.append((b["ticker"].lstrip("^").lower(), i))
            if b["name"]:
                patterns.append((b["name"].lower(), i))
        self.mentions = self.track(patterns)
        best = max(gain + loss, key=lambda x: abs(x[1]))
        self.header_sub = "TODAY'S MOVERS" if side == "both" else ("TOP GAINERS" if side == "gainers" else "TOP LOSERS")
        self.info_label = "BIGGEST MOVE"
        self.header_right = (best[0].lstrip("^"), f"{best[1]:+.1f}%", C_GREEN if best[1] >= 0 else C_RED)

    def _panel(self, ax, W, entries, color, title, y0, y1) -> None:
        panel = Rectangle((18, y0), W - 36, y1 - y0, facecolor=blend_hex(color, C_BG, 0.92), edgecolor=color,
                          linewidth=3)
        ax.add_patch(panel)
        self.titles.add(panel, ax.text(46, y0 + 44, title, fontsize=44 * REF_PT, fontweight="bold", color=color,
                                       ha="left", va="center"))
        n = max(len(entries), 1)
        top = y0 + 86
        row_h = (y1 - 14 - top) / n
        bar_x0, bar_max = 330.0, W - 36 - 330.0 - 250.0
        for i, (tick, pct, name) in enumerate(entries):
            cy = top + row_h * (i + 0.5)
            labs = [ax.text(46, cy - (row_h * 0.12 if name else 0), mpl_safe(tick.lstrip("^")),
                            fontsize=fit_one(tick.lstrip("^"), 270, row_h * 0.46, 62, 20) * REF_PT, fontweight="bold",
                            color=C_WHITE, ha="left", va="center")]
            if name:
                labs.append(ax.text(46, cy + row_h * 0.30, mpl_safe(name),
                                    fontsize=fit_one(name, 270, row_h * 0.27, 30, 12) * REF_PT, fontweight="bold",
                                    color=C_MUTED, ha="left", va="center"))
            rect = Rectangle((bar_x0, cy - row_h * 0.28), 0, row_h * 0.56, facecolor=color, edgecolor="none")
            ax.add_patch(rect)
            val = ax.text(bar_x0, cy, "", fontsize=fit_one("+00.0%", 230, row_h * 0.5, 56, 16) * REF_PT,
                          fontweight="bold", color=color, ha="left", va="center", zorder=6)
            t0 = 0.1 + 0.1 * len(self.bars)
            self.bars.append(dict(ticker=tick, name=name, rect=rect, val=val, lab=Reveal(t0, 0.25).add(labs), cy=cy,
                                  w=max(30.0, bar_max * abs(pct) / self.vmax), pct=pct, t0=t0, x0=bar_x0))

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        self.titles.apply(t)
        for i, b in enumerate(self.bars):
            m = 1.0 if key is None or i == key else 1 - 0.55 * s
            p = ease_out(ramp(t, b["t0"], 0.5))
            w = b["w"] * p
            b["rect"].set_width(w)
            b["val"].set_position((b["x0"] + w + 18, b["cy"]))
            b["val"].set_text(f"{b['pct'] * p:+.1f}%")
            b["val"].set_alpha(clamp01(p * 4) * m)
            b["rect"].set_alpha(m)
            b["lab"].apply(t, m)


# --- 5. NEWS FLASH -----------------------------------------------------
class NewsFlashLayout(LayoutBase):
    """Breaking-news banner + 1-3 headline cards copied from the real headlines. The card whose sentence is being
    spoken glows, the others dim."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        heads = list(seg.get("news_headlines", []))[:3] or [shorten(seg.get("script", ""), 120)]
        n = len(heads)
        banner = Rectangle((18, 18), W - 36, 92, facecolor=blend_hex(C_RED, C_BG, 0.55), edgecolor=C_RED,
                           linewidth=3)
        ax.add_patch(banner)
        self.banner = Reveal(0.0, 0.2).add(banner, ax.text(W / 2 + 30, 66, "BREAKING NEWS", fontsize=56 * REF_PT,
                                                           fontweight="bold", color=C_WHITE, ha="center", va="center"))
        self.dot, = ax.plot([88], [66], marker="o", markersize=36 * REF_PT, linestyle="none", color=C_WHITE,
                            markeredgewidth=0, zorder=8)
        gap, top, bottom = 22.0, 132.0, H - 18.0
        card_h = min(440.0, (bottom - top - gap * (n - 1)) / n)
        y0 = top + ((bottom - top) - (card_h * n + gap * (n - 1))) / 2
        self.cards, self.glows = [], []
        for i, text in enumerate(heads):
            y = y0 + i * (card_h + gap)
            box = FancyBboxPatch((18, y), W - 36, card_h, boxstyle="round,pad=0,rounding_size=22", fc=C_PANEL,
                                 ec=C_BORDER, lw=2.5)
            ax.add_patch(box)
            bar = Rectangle((18, y + 14), 14, card_h - 28, facecolor=self.accent, edgecolor="none")
            ax.add_patch(bar)
            lines, size = fit_wrap(text, W - 36 - 100, card_h - 64, 60, 22, pitch=1.22)
            arts = draw_lines(ax, lines, 62, y + card_h / 2, size, C_WHITE, pitch=1.22, ha="left")
            glow = FancyBboxPatch((18 - 6, y - 6), W - 36 + 12, card_h + 12, boxstyle="round,pad=0,rounding_size=26",
                                  fc="none", ec=self.accent, lw=6, alpha=0.0)
            ax.add_patch(glow)
            self.cards.append(Reveal(0.15 + 0.15 * i, 0.3).add(box, bar, arts))
            self.glows.append(glow)
        self.header_sub = "TODAY'S HEADLINES"
        self.info_label = "IN THE NEWS"
        self.header_right = (f"{n} HEADLINE{'S' if n > 1 else ''}", "LAST 24 HOURS", C_MUTED)

    def update(self, t: float) -> None:
        n, ns = len(self.cards), len(self.sentences)
        active = min(n - 1, int(self.sentence_at(t) * n / max(1, ns))) if ns >= n and n > 1 else None
        pulse = 0.7 + 0.3 * math.sin(2 * math.pi * t * 1.3)
        self.banner.apply(t)
        self.dot.set_alpha(0.35 + 0.65 * abs(math.sin(math.pi * t * 1.4)))
        for i, (card, glow) in enumerate(zip(self.cards, self.glows)):
            card.apply(t, 1.0 if active is None or i == active else 0.55)
            glow.set_alpha(0.85 * pulse * ease_out(ramp(t, 0.5, 0.3)) if i == active else 0.0)


# --- 6. MARKET PULSE (intraday race of 1-3 benchmarks) --------------------
PULSE_WORDS = {"SPY": ["s and p", "s&p", "spy", "s p"], "QQQ": ["nasdaq", "qqq"], "DIA": ["dow", "dia"],
               "IWM": ["russell", "small caps", "iwm"], "BTC": ["bitcoin", "btc", "crypto"]}


class MarketPulseLayout(LayoutBase):
    """Intraday % lines versus the previous close. All lines grow together in under two seconds and land exactly on
    the verified day move. The benchmark the narration names gets the thick line."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        make_stage_axes(fig, STAGE_BOX, *STAGE_REF)
        self.series = data["series"]
        n = len(self.series[0]["y"])
        self.n = n
        self.x = np.linspace(0, 1, n)
        ax = self.ax = fig.add_axes(axes_rect(px_box((130, 410, 870, 640))))
        style_axes(ax)
        ax.set_xlim(-0.02, 1.36)
        ax.set_xticks([])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}%"))
        ax.tick_params(labelsize=17)
        ax.axhline(0, color=C_MUTED, linestyle=":", linewidth=1.8, alpha=0.7)
        ax.text(0.01, 0.03, "% CHANGE VS PREVIOUS CLOSE", transform=ax.transAxes, fontsize=19 * REF_PT,
                fontweight="bold", color=C_MUTED, ha="left", va="bottom")
        self.lines, patterns = [], []
        for i, s in enumerate(self.series):
            y = np.asarray(s["y"], dtype=float)
            color = s["color"]
            line, = ax.plot([], [], color=color, linewidth=4.5, solid_capstyle="round")
            dot, = ax.plot([], [], marker="o", markersize=15, linestyle="none", color=color, markeredgewidth=0,
                           zorder=10)
            label = ax.text(0, 0, "", color=color, fontsize=24 * REF_PT, fontweight="bold", va="center",
                            visible=False, zorder=11,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=C_BG, edgecolor=color, alpha=0.9))
            ax.text(0.0 + 0.33 * i, 1.04, mpl_safe(f"{s['key']}  {s['label']}"), transform=ax.transAxes,
                    fontsize=21 * REF_PT, fontweight="bold", color=color, ha="left", va="bottom", clip_on=False)
            self.lines.append(dict(y=y, line=line, dot=dot, label=label, color=color, key=s["key"],
                                   run_min=np.minimum.accumulate(y), run_max=np.maximum.accumulate(y)))
            for w in PULSE_WORDS.get(s["key"], [s["key"].lower()]):
                patterns.append((w, i))
        self.mentions = self.track(patterns)
        self.build = min(PULSE_BUILD_SECONDS, self.dur * 0.7)
        first = self.series[0]
        self.header_sub = "TODAY'S SESSION  |  " + " \u00b7 ".join(s["key"] for s in self.series)
        self.info_label = first["label"]
        self.header_right = (f"{first['pct']:+.2f}%", "TODAY", C_GREEN if first["pct"] >= 0 else C_RED)

    def update(self, t: float) -> None:
        n = self.n
        p = max(2, int(math.ceil(ease_out(ramp(t, 0.08, self.build)) * n)))
        x_end = self.x[p - 1]
        key, s = self.mentions.active(t)
        lo, hi, ends = 0.0, 0.0, []
        for i, L in enumerate(self.lines):
            y_end = float(L["y"][p - 1])
            L["line"].set_data(self.x[:p], L["y"][:p])
            L["dot"].set_data([x_end], [y_end])
            L["label"].set_visible(True)
            L["label"].set_text(f" {L['key']} {y_end:+.2f}% ")
            hot = key == i
            L["line"].set_linewidth(4.5 + 3.0 * s * hot)
            alpha = 1.0 if key is None or hot else 1 - 0.6 * s
            L["line"].set_alpha(alpha)
            L["dot"].set_alpha(alpha)
            ends.append((y_end, i))
            lo, hi = min(lo, float(L["run_min"][p - 1])), max(hi, float(L["run_max"][p - 1]))
        span = max(hi - lo, 0.5)
        y_lo, y_hi = lo - 0.12 * span - 0.1, hi + 0.18 * span + 0.1
        self.ax.set_ylim(y_lo, y_hi)
        gap, prev = (y_hi - y_lo) * 0.09, -math.inf  # nudge the end labels apart so they never overlap
        for y_end, i in sorted(ends):
            y_lab = max(y_end, prev + gap)
            self.lines[i]["label"].set_position((x_end + 0.02, y_lab))
            prev = y_lab


# --- 7. PLAIN TEXT (also the universal fallback) ---------------------
class PlainTextLayout(LayoutBase):
    """Big centred statement; the words pop in within about 0.7 s, chosen words in the accent colour."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        for y in (46, H - 46):
            ax.plot([80, W - 80], [y, y], color=self.accent, linewidth=4, alpha=0.85, solid_capstyle="round")
        bx, by, bw, bh = STAGE_BOX
        box_x, box_y, box_w, box_h = bx + px(70), by + px(100), px(W - 140), px(H - 200)
        text = seg["plain_text"]
        highlight = {re.sub(r"[^\w$%]", "", w).lower() for w in seg.get("plain_highlight", [])} - {""}
        accent_rgb, white_rgb = _rgb(self.accent), _rgb(C_WHITE)
        size, min_size = px(96), max(14, px(34))
        while True:
            font = load_font(size)
            lines = _wrap_text(text, font, box_w)
            line_h = int(_font_height(font) * 1.25)
            widest = max((font.getlength(l) for l in lines), default=0)
            if (len(lines) * line_h <= box_h and widest <= box_w) or size <= min_size:
                break
            size -= 2
        space = font.getlength(" ")
        y = box_y + max(0, (box_h - len(lines) * line_h) // 2)
        self.words: List[dict] = []
        for line in lines:
            tokens = line.split()
            widths = [font.getlength(tok) for tok in tokens]
            x = box_x + (box_w - (sum(widths) + space * (len(tokens) - 1))) / 2
            for tok, w in zip(tokens, widths):
                mask_img = Image.new("L", (int(math.ceil(w)) + 8, line_h), 0)
                ImageDraw.Draw(mask_img).text((4, 0), tok, font=font, fill=255)
                is_hl = re.sub(r"[^\w$%]", "", tok).lower() in highlight
                self.words.append(dict(x=int(round(x)) - 4, y=y, mask=np.asarray(mask_img, dtype=np.float32) / 255.0,
                                       color=accent_rgb if is_hl else white_rgb))
                x += w + space
            y += line_h
        total = min(0.7, 0.045 * len(self.words))
        for i, w in enumerate(self.words):
            w["start"] = 0.05 + total * i / max(1, len(self.words) - 1)
        self.header_sub = "KEY TAKEAWAY"
        self.info_label = "THE BIG PICTURE"

    def paint(self, frame: np.ndarray, t: float) -> None:
        for w in self.words:
            a = ease_out(ramp(t, w["start"], 0.18))
            if a > 0:
                blend_mask(frame, w["mask"], w["x"], w["y"] + int(round((1 - a) * px(14))), w["color"], a)


# --- 8. OUTRO ---------------------------------------------------------
class OutroLayout(LayoutBase):
    """Closing screen: the character stands in the middle (painted by the scene) over one fixed block of text."""
    AVATAR_REF = 600

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        ax = self.ax = make_stage_axes(fig, (0, 0, VIDEO_W, VIDEO_H), *FULL_REF, border=False, face="none")
        title = "SEE YOU TOMORROW"
        sub = "FOLLOW  \u2022  LIKE  \u2022  COMMENT"
        self.title = ax.text(540, 250, title, fontsize=fit_one(title, 980, 150, 108, 40) * REF_PT, fontweight="bold",
                             color=C_WHITE, ha="center", va="center")
        self.sub = ax.text(540, 380, sub, fontsize=fit_one(sub, 980, 90, 54, 24) * REF_PT, fontweight="bold",
                           color=self.accent, ha="center", va="center")
        self.foot = ax.text(540, 1700, "NOT FINANCIAL ADVICE  \u2022  FOR EDUCATIONAL PURPOSES ONLY",
                            fontsize=fit_one("NOT FINANCIAL ADVICE  \u2022  FOR EDUCATIONAL PURPOSES ONLY", 980, 50, 32,
                                             16) * REF_PT, fontweight="bold", color=C_WHITE, ha="center", va="center")
        self.foot2 = ax.text(540, 1760, "Markets involve risk. Do your own research.", fontsize=28 * REF_PT,
                             fontweight="bold", color=C_MUTED, ha="center", va="center")

    def avatar_rect(self) -> Tuple[int, int, int]:
        s = px(self.AVATAR_REF)
        return (VIDEO_W - s) // 2, px(480), s

    def update(self, t: float) -> None:
        a = ease_out(ramp(t, 0.0, 0.4))
        for art in (self.title, self.sub, self.foot, self.foot2):
            art.set_alpha(a)


LAYOUT_CLASSES = {
    "market_pulse": MarketPulseLayout,
    "pro_candlestick_chart": ProCandlestickLayout,
    "stock_dashboard": StockDashboardLayout,
    "sector_heatmap": SectorHeatmapLayout,
    "top_movers": TopMoversLayout,
    "news_flash": NewsFlashLayout,
    "outro": OutroLayout,
    "plain_text": PlainTextLayout,
}


# =====================================================================
# 12. SCENES  (frame_at(t) -> RGB uint8 array; t is scene-local)
# =====================================================================
def build_info_card(fig, layout: LayoutBase) -> None:
    """Card bottom-right: what the screen is about + its key number (price, biggest move, ...)."""
    ax = make_stage_axes(fig, INFO_BOX, *INFO_REF, face=PANEL_FACE)
    label = layout.info_label or "DAILY RECAP"
    lines, size = fit_wrap(label.upper(), 592, 60, 30, 14, pitch=1.2)
    draw_lines(ax, lines, 24, 36 + (len(lines) - 1) * size * 0.6, size, C_MUTED, pitch=1.2, ha="left")
    line1, line2, col2 = layout.header_right or ("DAILY RECAP", "", C_MUTED)
    ax.text(INFO_REF[0] / 2, 140, mpl_safe(line1), fontsize=fit_one(line1, 600, 100, 96, 18) * REF_PT,
            fontweight="bold", color=C_WHITE, ha="center", va="center")
    if line2:
        ax.text(INFO_REF[0] / 2, 232, mpl_safe(line2), fontsize=fit_one(line2, 600, 64, 58, 14) * REF_PT,
                fontweight="bold", color=col2, ha="center", va="center")
    ax.text(INFO_REF[0] / 2, 304, "EDUCATIONAL ONLY  \u2022  NOT FINANCIAL ADVICE", fontsize=17 * REF_PT,
            fontweight="medium", color="#AFB9C4", ha="center", va="center")


class RecapSegmentScene:
    """Ticker + headline + one screen + info card + avatar + captions."""

    def __init__(self, a: dict, duration: float, global_start: float):
        seg = a["seg"]
        self.duration, self.global_start = duration, global_start
        self.cache = FrameCache()
        self.fig, self.canvas = new_figure()
        self.bg_im = attach_background(self.fig, global_start)
        self.ticker = NewsTickerBar(self.fig, a["ticker_text"])
        self.layout = LAYOUT_CLASSES.get(seg["layout"], PlainTextLayout)(
            self.fig, seg, duration, a["speech_dur"], a.get("data"), a.get("sentence_times"))
        head = seg["headline"]
        self.fig.text(fig_x(px(40)), fig_y(HEADER_TOP), mpl_safe(head),
                      fontsize=fit_one(head, 1000, 90, 68, 28) * REF_PT, fontweight="bold", color=C_WHITE, va="top")
        sub = self.layout.header_sub or a["date_label"]
        self.fig.text(fig_x(px(40)), fig_y(HEADER_SUB_TOP), mpl_safe(shorten(sub, 60)), fontsize=26 * REF_PT,
                      color=C_MUTED, va="top")
        build_info_card(self.fig, self.layout)
        self.captions = CaptionTrack(seg.get("caption") or seg["script"], a.get("sentence_times"), a["speech_dur"])
        self.peaks = get_audio_peaks(a["audio_path"])
        self.avatar = AvatarPanel(AVATAR_SIZE)

    def _render_chart(self, t: float) -> np.ndarray:
        self.bg_im.set_data(get_background().frame(self.global_start + t))
        self.layout.update(t)
        self.ticker.update(self.global_start + t)
        frame = grab(self.canvas)
        self.layout.paint(frame, t)
        return frame

    def frame_at(self, t: float) -> np.ndarray:
        frame = self.cache.get(t, self._render_chart).copy()
        speaking, bob = lip_sync(self.peaks, t)
        frame[AVATAR_Y:AVATAR_Y + AVATAR_SIZE, AVATAR_X:AVATAR_X + AVATAR_SIZE] = self.avatar.frame(speaking, bob)
        self.captions.overlay(frame, t)
        return frame


class OutroScene:
    """Closing screen: one fixed block of text and the character in the middle."""

    def __init__(self, a: dict, duration: float, global_start: float):
        seg = a["seg"]
        self.duration, self.global_start = duration, global_start
        self.cache = FrameCache()
        self.fig, self.canvas = new_figure()
        self.bg_im = attach_background(self.fig, global_start)
        self.layout = OutroLayout(self.fig, seg, duration, a["speech_dur"], None, a.get("sentence_times"))
        self.ax_, self.ay_, self.size = self.layout.avatar_rect()
        self.peaks = get_audio_peaks(a["audio_path"])
        self.avatar = AvatarPanel(self.size)
        self.captions = CaptionTrack(seg.get("caption") or seg["script"], a.get("sentence_times"), a["speech_dur"],
                                     OUTRO_SUB_BOX)

    def _render_chart(self, t: float) -> np.ndarray:
        self.bg_im.set_data(get_background().frame(self.global_start + t))
        self.layout.update(t)
        return grab(self.canvas)

    def frame_at(self, t: float) -> np.ndarray:
        frame = self.cache.get(t, self._render_chart).copy()
        speaking, bob = lip_sync(self.peaks, t, 60.0)
        frame[self.ay_:self.ay_ + self.size, self.ax_:self.ax_ + self.size] = self.avatar.frame(speaking, bob)
        self.captions.overlay(frame, t)
        return frame


# =====================================================================
# 13. VIDEO WRITING  (each scene -> one MP4 chunk WITH its own audio; one isolated process per chapter)
# =====================================================================
@dataclass
class SceneJob:
    kind: str  # "segment" | "outro"
    out_path: str
    start_time: float  # position on the global timeline (seconds)
    duration: float
    args: Dict[str, Any] = field(default_factory=dict)


def build_scene(job: SceneJob):
    if job.kind == "segment":
        return RecapSegmentScene(job.args, job.duration, job.start_time)
    if job.kind == "outro":
        return OutroScene(job.args, job.duration, job.start_time)
    raise ValueError(f"Unknown scene kind: {job.kind}")


def frame_range(start_time: float, duration: float) -> Tuple[int, int]:
    """Global frame indices [first, last). Cumulative rounding => no drift between scenes."""
    return round(start_time * OUTPUT_FPS), round((start_time + duration) * OUTPUT_FPS)


def build_scene_audio(job: SceneJob, n_frames: int) -> str:
    """Mono 44.1 kHz wav of exactly n_frames/FPS seconds: the TTS clip from t=0 plus the SFX at t=0."""
    sr = 44100
    total = int(round(n_frames / OUTPUT_FPS * sr))
    mix = np.zeros(total, dtype=np.float32)
    for key, gain in (("audio_path", 1.0), ("sfx_path", SFX_GAIN)):
        path = job.args.get(key)
        if path and os.path.exists(path):
            try:
                s = decode_audio(path, sr) * gain
                m = min(len(s), total)
                mix[:m] += s[:m]
            except Exception as e:
                logger.warning(f"Could not mix {key} for {job.kind}: {e}")
    peak = float(np.abs(mix).max()) if total else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    write_wav_mono(job.args["mix_path"], mix, sr)
    return job.args["mix_path"]


def render_scene_job(job: SceneJob) -> str:
    """Render one scene to an MP4 whose audio track is already embedded."""
    first, last = frame_range(job.start_time, job.duration)
    n_frames = max(1, last - first)
    mix_path = build_scene_audio(job, n_frames)
    scene = build_scene(job)
    log_memory(f"render {job.kind} begin")
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(OUTPUT_FPS), "-i", "-",
        "-i", mix_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", X264_PRESET, "-b:v", VIDEO_BITRATE, "-pix_fmt", "yuv420p",
        "-threads:v", str(FFMPEG_THREADS), "-filter_threads", "1",
        "-af", "pan=stereo|c0=c0|c1=c0", "-c:a", "aac", "-b:a", "160k", "-ar", "44100",
        "-t", f"{n_frames / OUTPUT_FPS:.6f}", job.out_path,
    ]
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                if i and i % (OUTPUT_FPS * LOG_EVERY_SECONDS) == 0:
                    log_memory(f"render {job.kind} {i / OUTPUT_FPS:.0f}s/{job.duration:.0f}s")
                t = max(0.0, (first + i) / OUTPUT_FPS - job.start_time)
                proc.stdin.write(np.ascontiguousarray(scene.frame_at(t)))
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
            raise RuntimeError(f"ffmpeg failed for {job.kind} scene: {err_file.read().decode(errors='ignore')}")
    del scene
    gc.collect()
    return job.out_path


def render_chapter_jobs(jobs: List[SceneJob]) -> List[str]:
    """All scenes of ONE chapter, sequentially. Memory is released at the end of the chapter."""
    paths = []
    for job in jobs:
        marker = Path(job.out_path + ".done")
        if marker.exists() and Path(job.out_path).exists():
            logger.info(f"Scene already rendered, skipping: {Path(job.out_path).name}")
        else:
            render_scene_job(job)
            marker.write_text("ok")
        paths.append(job.out_path)
    release_chapter_memory()
    return paths


def render_all_chapters(jobs: List[SceneJob], work_dir: Path) -> List[str]:
    """One isolated OS process per chapter: each Matplotlib scene and the avatar cache die with their process.
    RECAP_RENDER_WORKERS (default 1, the low-RAM choice) chapters are rendered at the same time."""
    by_chapter: Dict[int, List[SceneJob]] = {}
    for job in jobs:
        by_chapter.setdefault(job.args["chapter_index"], []).append(job)
    workers = max(1, RENDER_WORKERS) if RENDER_IN_SUBPROCESS else 1
    logger.info("Rendering %d scenes in %d chapter workers (%d at a time)", len(jobs), len(by_chapter), workers)

    def run_chapter(ci: int) -> int:
        chapter_jobs = by_chapter[ci]
        if RENDER_IN_SUBPROCESS:
            input_file = work_dir / f"render_chapter_{ci}.pkl"
            with input_file.open("wb") as f:
                pickle.dump(chapter_jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                run_memory_worker("render", str(input_file))
            finally:
                input_file.unlink(missing_ok=True)
        else:
            render_chapter_jobs(chapter_jobs)
        AvatarPanel._cache.clear()
        gc.collect()
        logger.info(f"Chapter {ci + 1} rendered")
        return ci

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_chapter, sorted(by_chapter)))
    return [j.out_path for j in jobs]


# =====================================================================
# 14. FINAL ASSEMBLY  (concat demuxer of A/V chunks; stream copy; optional ducked music on the audio only)
# =====================================================================
def pick_music() -> Optional[str]:
    if not USE_MUSIC:
        return None
    tracks = sorted(Path("assets/music").glob("*.mp3")) if Path("assets/music").exists() else []
    return str(random.choice(tracks)) if tracks else None


def assemble_final_video(scene_paths: List[str], total_seconds: float, output_path: str, work_dir: Path) -> None:
    list_file = work_dir / "scenes.txt"
    list_file.write_text("".join("file '{}'\n".format(Path(p).resolve().as_posix().replace("'", "'\\''"))
                                 for p in scene_paths))
    base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    music = pick_music()
    if music:
        logger.info("Mixing background music (ducked under the voice): %s", music)
        fade = max(0.0, total_seconds - BG_MUSIC_FADE_OUT)
        graph = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[vo][sc];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,atrim=duration={total_seconds:.3f},"
            f"asetpts=PTS-STARTPTS,volume={BG_MUSIC_VOLUME},afade=t=out:st={fade:.3f}:d={BG_MUSIC_FADE_OUT}[bg0];"
            "[bg0][sc]sidechaincompress=threshold=0.03:ratio=8:attack=30:release=500[bg];"
            "[vo][bg]amix=inputs=2:duration=first:dropout_transition=0,volume=2[aout]")
        cmd = base + ["-stream_loop", "-1", "-i", music, "-filter_complex", graph, "-map", "0:v:0", "-map", "[aout]",
                      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t",
                      f"{total_seconds:.3f}", str(output_path)]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            return
        logger.warning("Music mix failed, assembling without music: %s", proc.stderr.decode(errors="ignore")[-400:])
    cmd = base + ["-c", "copy", "-movflags", "+faststart", str(output_path)]
    logger.info(f"Assembling final video: {output_path}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg assembly failed: {proc.stderr.decode(errors='ignore')}")


# =====================================================================
# 14b. THUMBNAIL  (Pillow only, built AFTER the video is saved; the engine lives in thumbnail.py)
#      The Director only suggests the angle and the text. The featured stock, the candles, the colours and the
#      % badge are decided / verified here and in thumbnail.py, so the thumbnail can never promise a lie.
# =====================================================================
def plan_thumbnail(outline: DailyOutline, cache: MarketCache) -> dict:
    """Pick the featured stock and its VERIFIED move while the market cache is still in memory (tiny result)."""
    idea = outline.thumbnail
    spec = dict(template=idea.template, main_text=idea.main_text, primary_ticker="", secondary_ticker="",
                trend="neutral", avatar_emotion=idea.avatar_emotion)
    job = dict(spec=spec, pct=None)
    if idea.template != "candle_hero":
        return job
    etfs = set(SECTOR_ETFS) | {"SPY", "QQQ", "DIA", "IWM"}
    pool = {t: d for t, d in cache.days.items() if not t.startswith("^") and not t.endswith("-USD") and t not in etfs}
    wanted = clean_ticker(idea.ticker)
    perf = cache.days.get(wanted) if wanted else None
    if perf is None and wanted:
        perf = fetch_day_perf(wanted)
    chosen = wanted
    if perf is None:  # the Director named a stock we cannot verify: feature the biggest VERIFIED mover instead
        if not pool:
            spec["template"] = "massive_number"
            return job
        chosen = max(pool, key=lambda t: abs(pool[t]["day_pct"]))
        perf = pool[chosen]
        logger.warning("Thumbnail ticker '%s' unavailable; featuring %s instead", wanted, chosen)
    pct = float(perf["day_pct"])
    words = _title_word_forms(spec["main_text"])
    wrong_way = (bool(words & set(BEAR_TITLE_WORDS)) and pct > 0) or \
                (bool(words & set(BULL_TITLE_WORDS)) and pct < 0)
    if chosen != wanted or wrong_way:  # the written headline is about another stock / the wrong direction
        first = re.sub(r"[^A-Z]", "", (COMPANY_NAMES.get(chosen, chosen).split() or [chosen])[0].upper()) or chosen
        spec["main_text"] = f"{first} {'SURGES' if pct >= 0 else 'SLIDES'}?"
        logger.warning("Thumbnail headline replaced to match the verified move: %s", spec["main_text"])
    spec["primary_ticker"] = chosen
    job["pct"] = pct
    return job


def _thumbnail_candles(ticker: str):
    from thumbnail import coerce_candles
    for kwargs in (dict(period="1d", interval="5m"), dict(period="1mo", interval="1d")):
        try:
            df = _yf().Ticker(ticker).history(**kwargs).dropna()
            if len(df) >= 6:
                return coerce_candles(df, THUMB_CANDLES_KEPT)
        except Exception as e:
            logger.warning(f"Thumbnail candles failed for {ticker} ({kwargs}): {e}")
    return None


def make_daily_thumbnail(job: Optional[dict], output_filename: str) -> Optional[str]:
    """9:16 (1080x1920) thumbnail next to the video. Never raises: a thumbnail problem must not cost a render."""
    if not job:
        return None
    try:
        from thumbnail import PORTRAIT, ThumbnailEngine
        spec = job["spec"]
        candles = _thumbnail_candles(spec["primary_ticker"]) if spec["template"] == "candle_hero" and \
                                                                  spec["primary_ticker"] else None
        thumb_file = Path(output_filename).with_name(f"THUMBNAIL_DAILY_SHORT_{datetime.date.today().isoformat()}.png")
        engine = ThumbnailEngine(spec, size=PORTRAIT, candles=candles, pct=job["pct"], pct_label="TODAY",
                                 default_text="MARKET MOVERS")
        path = engine.save(thumb_file)
        logger.info("Thumbnail saved as %s (%s, '%s')", path, engine.template, engine.text)
        return path
    except Exception as e:
        logger.warning("Thumbnail generation failed (the video is unaffected): %s", e)
        return None


# =====================================================================
# 15. PIPELINE
# =====================================================================
def run_perfect_pipeline(output_filename: str = "perfect_daily_recap_shorts.mp4", upload: bool = False,
                         news_file: Optional[str] = None, save_script: Optional[str] = None,
                         load_script: Optional[str] = None, work_dir_arg: Optional[str] = None) -> str:
    logger.info("Launching Daily Shorts pipeline (director mode, %.0fs hard limit, low-memory)...", SHORTS_MAX_SECONDS)
    log_memory("pipeline start")
    persistent = bool(work_dir_arg)
    if persistent:
        work_dir = Path(work_dir_arg)
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using persistent work dir {work_dir} (finished scenes and the voiceover are reused on re-runs)")
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="daily_shorts_"))
    try:
        # ---- Phase A: data + Director -> Writers -> master timeline ----
        auto_script = work_dir / "script.json"
        if persistent and not load_script and auto_script.exists():
            load_script = str(auto_script)  # same work dir = same video: never mix scenes of two scripts
            logger.info("Work dir already holds a script: resuming that video")
        if load_script:
            master = MasterTimeline.model_validate_json(Path(load_script).read_text(encoding="utf-8"))
            logger.info(f"Loaded script from {load_script}")
        else:
            master = generate_master_timeline(gather_daily_context(news_file))
        # ---- Phase B: pre-fetch ALL market data, ground numbers, repair segments (before touching video) ----
        cache = prefetch_market_data(master)
        if persistent and not auto_script.exists():
            auto_script.write_text(master.model_dump_json(indent=2), encoding="utf-8")
        if save_script:
            Path(save_script).write_text(master.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Final, edited script saved to %s", save_script)
        outline = master.outline
        thumb_job = None
        try:
            thumb_job = plan_thumbnail(outline, cache)
        except Exception as e:
            logger.warning("Thumbnail preparation failed (the video is unaffected): %s", e)
        log_memory("market data ready")
        segs = [it.segment for it in master.items]
        logger.info("=" * 60)
        logger.info(f"{outline.date_label} | {outline.youtube_title}")
        logger.info(f"Story: {outline.story_angle}")
        for it in master.items:
            s = it.segment
            logger.info(f"  ch{it.chapter_index + 1} {s.layout:22s} '{s.headline}' :: {shorten(s.script, 90)}")
        logger.info("=" * 60)

        # ---- Phase C: voices + THE CLOCK. Cut the least valuable screens, then speed up, to stay under the limit ----
        limit = SHORTS_MAX_SECONDS - TIME_SAFETY
        speed, nudged = HOST_SPEED, False
        for attempt in range(4):
            speech_items = [{"text": s.script, "voice": HOST_VOICE, "speed": speed,
                             "path": str(work_dir / f"segment_{i}.wav")} for i, s in enumerate(segs)]
            tts = run_tts(speech_items, work_dir)
            durations = [r["duration"] for r in tts]
            pads = timeline_length([0.0] * len(segs), range(len(segs)))
            factor = sum(durations) / max(1.0, limit - pads)  # how much faster the voice would have to be
            if 1.0 < factor <= SPEEDUP_BEFORE_CUT and not nudged and speed < MAX_VOICE_SPEED:
                nudged, speed = True, min(MAX_VOICE_SPEED, speed * factor * 1.02)
                logger.info("Narration slightly long: speeding the voice up to %.2f instead of cutting screens", speed)
                continue
            keep = choose_segments_to_keep(segs, durations, limit)
            length = timeline_length(durations, keep)
            logger.info("Narration %.1fs; timeline %.1fs with %d of %d screens (voice speed %.2f)",
                        sum(durations[k] for k in keep), length, len(keep), len(segs), speed)
            if length <= limit + 1e-6 or speed >= MAX_VOICE_SPEED:
                break
            speed = min(MAX_VOICE_SPEED, speed * max(1.08, length / limit * 1.02))
            logger.warning("Still %.1fs: re-synthesising the voice faster (%.2f)", length, speed)
        log_memory("TTS completed")

        # ---- Phase D: timeline -> scene jobs ----
        jobs: List[SceneJob] = []
        cursor = 0.0
        for j, k in enumerate(keep):
            it, s = master.items[k], master.items[k].segment
            dur = durations[k] + (END_PAD if j == len(keep) - 1 else SEGMENT_PAD)
            sfx = ev(s.sfx)
            if sfx == "none" and j > 0:
                sfx = "swoosh"  # every cut gets a transition whoosh
            jobs.append(SceneJob(
                "outro" if s.layout == "outro" else "segment", str(work_dir / f"scene_{j}.mp4"), cursor, dur,
                args=dict(seg=s.model_dump(), chapter_index=it.chapter_index, date_label=outline.date_label,
                          data=data_for_segment(s, cache), speech_dur=durations[k],
                          sentence_times=tts[k]["sentences"], ticker_text=outline.news_ticker_text,
                          audio_path=speech_items[k]["path"], sfx_path=get_sfx_path(sfx, work_dir),
                          mix_path=str(work_dir / f"mix_{j}.wav"))))
            cursor += dur
        if cursor > SHORTS_MAX_SECONDS:
            logger.error("The Short is %.1fs, over the %.0fs limit (voice at maximum speed)", cursor,
                         SHORTS_MAX_SECONDS)
        logger.info(f"Final length: {cursor:.1f}s in {len(jobs)} scenes")

        final_description = outline.description.strip()
        if "#shorts" not in final_description.lower():
            final_description += "\n\n#Shorts"
        # Save even without --upload, so the uploader text can be reviewed and pasted.
        metadata_path = Path(output_filename).with_suffix(".description.txt")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(final_description + "\n", encoding="utf-8")
        Path(output_filename).with_suffix(".youtube.json").write_text(
            json.dumps({"title": outline.youtube_title, "description": final_description, "tags": outline.tags},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        youtube_title, tags = outline.youtube_title, outline.tags
        del master, segs, cache
        gc.collect()

        # ---- Phase E: render (one worker per chapter) + assemble ----
        scene_paths = render_all_chapters(jobs, work_dir)
        log_memory("scenes rendered")
        assemble_final_video(scene_paths, cursor, output_filename, work_dir)
        logger.info(f"SUCCESS! Short saved as: {output_filename} ({cursor:.1f}s)")

        # ---- Phase F: thumbnail (9:16), only after the video is safely on disk ----
        thumbnail_path = make_daily_thumbnail(thumb_job, output_filename)

        if upload:
            from uploader import upload_video  # import only when an upload is requested
            logger.info("Uploading Short to YouTube...")
            extra = {}
            if thumbnail_path:
                if "thumbnail_path" in inspect.signature(upload_video).parameters:
                    extra["thumbnail_path"] = thumbnail_path
                else:
                    logger.info("uploader.upload_video has no thumbnail_path parameter; thumbnail kept at %s",
                                thumbnail_path)
            upload_video(file_path=output_filename, title=youtube_title, description=final_description, tags=tags,
                         category_id="25", **extra)
        log_memory("pipeline finished")
        logger.info("Shorts pipeline executed successfully!")
        return output_filename
    finally:
        if persistent:
            logger.info(f"Work dir kept: {work_dir} (delete it when you are happy with the video)")
        else:
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(work_dir, ignore_errors=True)


# =====================================================================
# 16. ENTRY POINTS (parent CLI + isolated render worker)
# =====================================================================
def render_worker_main(pkl_path: str) -> None:
    # This pickle is written by this very application into its own private temp directory.
    with open(pkl_path, "rb") as f:
        jobs = pickle.load(f)
    render_chapter_jobs(jobs if isinstance(jobs, list) else [jobs])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--render-worker":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: --render-worker chapter_jobs.pkl")
        render_worker_main(sys.argv[2])
    else:
        parser = argparse.ArgumentParser(description="Generate a director-driven vertical Daily Market Short.")
        parser.add_argument("--upload", action="store_true", help="upload to YouTube when finished")
        parser.add_argument("--output", default="perfect_daily_recap_shorts.mp4", help="output mp4 path")
        parser.add_argument("--news-file", default=None, help="extra producer notes / news appended to the LLM context")
        parser.add_argument("--save-script", default=None, help="save the generated master timeline as JSON")
        parser.add_argument("--load-script", default=None, help="skip the LLM and load a saved master timeline")
        parser.add_argument("--work-dir", default=None,
                            help="keep scenes/voiceover in this folder; re-running with the same folder resumes")
        parser.add_argument("--no-music", action="store_true", help="do not mix background music")
        cli = parser.parse_args()
        if cli.no_music:
            USE_MUSIC = False
        run_perfect_pipeline(output_filename=cli.output, upload=cli.upload, news_file=cli.news_file,
                             save_script=cli.save_script, load_script=cli.load_script, work_dir_arg=cli.work_dir)