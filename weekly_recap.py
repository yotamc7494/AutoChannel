"""
        weekly_recap.py - retention-first WIDESCREEN (16:9) weekly financial recap generator.

        Pipeline
          A. Data + news + upcoming earnings + rule-based chart-pattern scan  ->  Director LLM (outline of 5-7 chapters)
             -> Writer LLM (one call PER chapter)  ->  master timeline
          B. Pre-fetch of every ticker / timeframe in the timeline (yfinance) BEFORE rendering. Numbers spoken or shown for a
             ticker come from verified data via [[TICKER:week]] style placeholders, never from the LLM's memory.
          C. TTS (Kokoro ONNX, isolated worker): Host voice + Quote voice, synthesised sentence by sentence so every screen
             knows exactly when each sentence is spoken (used to flash the right card / sector / day / level).
          D. Render: one isolated OS process per chapter (several in parallel); every scene is an MP4 chunk that already
             contains its audio (TTS + SFX at t=0).
          E. Assemble: ffmpeg concat demuxer (stream copy) + optional ducked background music.

        Usage
          python weekly_recap.py [--upload] [--output out.mp4] [--news-file notes.txt] [--work-dir DIR]
                                 [--save-script s.json] [--load-script s.json] [--no-music]
        Env
          GEMINI_API_KEY, PERFECT_VIDEO_WIDTH=960|1280|1920 (default 1280), RECAP_RENDER_WORKERS, RECAP_RENDER_FPS
        Optional assets (never fatal if missing)
          assets/avatar/open.png, assets/avatar/closed.png, assets/music/*.mp3, assets/sfx/<name>.wav|mp3,
          assets/logos/<TICKER>.png (company logos are also downloaded on a best-effort basis)
        """
from __future__ import annotations

import argparse
import datetime
import gc
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
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

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
logger = logging.getLogger("WeeklyRecap")

# =====================================================================
# 1. CONFIG
# =====================================================================
# --- LLM ---
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
LLM_MAX_ROUNDS = 5
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
CONTEXT_MAX_CHARS = 17000

# --- Video (16:9 widescreen) ---
VIDEO_W = int(os.environ.get("PERFECT_VIDEO_WIDTH", "1280"))
if VIDEO_W not in (960, 1280, 1920):
    raise ValueError("PERFECT_VIDEO_WIDTH must be 960, 1280 or 1920")
VIDEO_H = VIDEO_W * 9 // 16
UI_SCALE = VIDEO_W / 1920  # every layout pixel is authored for a 1920x1080 grid and scaled by this
DPI = VIDEO_W / 19.2
REF_PT = 0.72  # 1 reference pixel (1920 grid) in matplotlib points, at ANY size
OUTPUT_FPS = 30
RENDER_FPS = int(os.environ.get("RECAP_RENDER_FPS", "12"))  # heavy matplotlib layer redraw rate (frames repeat)
VIDEO_BITRATE = {960: "3500k", 1280: "6000k", 1920: "10000k"}[VIDEO_W]
X264_PRESET = "ultrafast"
BG_MUSIC_VOLUME = 0.10
BG_MUSIC_FADE_OUT = 3.0
USE_MUSIC = True

# --- Parallelism / memory ---
IO_WORKERS = 4
RENDER_IN_SUBPROCESS = True
RENDER_WORKERS = int(os.environ.get("RECAP_RENDER_WORKERS", max(1, min(3, (os.cpu_count() or 2) // 2))))
FFMPEG_THREADS = 2
LOG_EVERY_SECONDS = 20

# --- Retention-first pacing: the story decides the length, not a word quota. ---
CHAPTERS_MIN, CHAPTERS_MAX = 4, 7
SEG_PER_CHAPTER_MIN, SEG_PER_CHAPTER_MAX = 1, 8
SEG_WORDS_MIN, SEG_WORDS_MAX = 6, 110  # safety bounds for broken output, NOT a target
CHAPTER_MAX_ROUNDS = 3
BUMPER_SECONDS = 2.0
SEGMENT_PAD = 0.30
END_PAD = 4.0  # hold on the closing screen after the last word
SENTENCE_GAP = 0.14  # silence between sentences inside one segment
TTS_RATE_TOLERANCE = 0.08

# --- Animation speeds (seconds). Everything builds fast; the voice, not the animation, sets the pace. ---
CHART_BUILD_SECONDS = 0.5
FAST_REVEAL = 0.6  # whole text/card/table sections finish appearing within this time

# --- Voices (Kokoro) ---
HOST_VOICE, HOST_SPEED = "am_liam", 1.05
QUOTE_VOICE, QUOTE_SPEED = "am_onyx", 0.93
TTS_SYMBOL_NAMES = {
    r"\bNVDA\b": "Nvidia", r"\bTSLA\b": "Tesla", r"\bAAPL\b": "Apple", r"\bAMZN\b": "Amazon",
    r"\bMSFT\b": "Microsoft", r"\bGOOGL?\b": "Google", r"\bMETA\b": "Meta", r"\bBTC\b": "Bitcoin",
    r"\bSPY\b": "S and P 500", r"\bQQQ\b": "Nasdaq", r"\bP/E\b": "P E", r"\bCPI\b": "C P I",
    r"\bFOMC\b": "the Fed meeting", r"\bGDP\b": "G D P", r"\bETF\b": "E T F", r"\bAMC\b": "A M C",
}
SFX_GAIN = 0.2  # sound effects sit well under the voice

# --- Layouts ---
LAYOUTS = ("pro_candlestick_chart", "stock_dashboard", "sector_heatmap", "top_movers_podium", "macro_timeline",
           "bull_vs_bear", "quote_highlight", "portfolio_10k_sim", "trade_setup", "earnings_calendar", "outro",
           "plain_text")
DEFAULT_HEADLINES = {
    "pro_candlestick_chart": "THE CHART", "stock_dashboard": "BY THE NUMBERS", "sector_heatmap": "SECTOR MAP",
    "top_movers_podium": "BIGGEST MOVERS", "macro_timeline": "THE WEEK AHEAD", "bull_vs_bear": "BULL VS BEAR",
    "quote_highlight": "ON THE RECORD", "portfolio_10k_sim": "IF YOU INVESTED", "trade_setup": "CHART SETUP",
    "earnings_calendar": "EARNINGS NEXT WEEK", "outro": "THANKS FOR WATCHING", "plain_text": "THE TAKEAWAY",
}
WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI")
CANDLE_SPECS = {"1d": ("5d", "15m"), "1wk": ("5d", "1h"), "1mo": ("1mo", "1d"), "1y": ("1y", "1wk")}
CANDLE_FALLBACK_TF = "1mo"
SECTOR_ETFS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
               "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
               "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials", "XLC": "Communication Services"}
SECTOR_MATCH = (("discretionary", "XLY"), ("staples", "XLP"), ("tech", "XLK"), ("financ", "XLF"), ("energy", "XLE"),
                ("health", "XLV"), ("industrial", "XLI"), ("utilit", "XLU"), ("real estate", "XLRE"),
                ("material", "XLB"), ("communic", "XLC"))
ETF_ALIASES = {
    "XLK": ["technology", "tech"], "XLF": ["financials", "financial", "banks", "banking"], "XLE": ["energy", "oil"],
    "XLV": ["health care", "healthcare", "health"], "XLY": ["consumer discretionary", "discretionary"],
    "XLP": ["consumer staples", "staples"], "XLI": ["industrials", "industrial"], "XLU": ["utilities", "utility"],
    "XLRE": ["real estate", "property"], "XLB": ["materials"], "XLC": ["communication services", "communications",
                                                                       "communication"],
}
SECTOR_WEIGHTS = {"XLK": 32, "XLF": 13, "XLV": 10, "XLY": 10, "XLC": 9, "XLI": 8, "XLP": 6, "XLE": 3.5,
                  "XLU": 2.5, "XLRE": 2, "XLB": 2}
INDEX_SYMBOLS = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq Composite", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
                 "^VIX": "VIX volatility index", "^TNX": "10-year Treasury yield"}
NEWS_SYMBOLS = ["^GSPC", "SPY", "QQQ", "NVDA", "AAPL", "TSLA", "MSFT"]

# ticker -> spoken company name. Used for: earnings/scan universe, "stay on topic" checks, highlight keywords, logos
COMPANY_NAMES: Dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta",
    "TSLA": "Tesla", "AVGO": "Broadcom", "JPM": "JPMorgan", "XOM": "Exxon Mobil", "LLY": "Eli Lilly",
    "WMT": "Walmart", "COST": "Costco", "NFLX": "Netflix", "AMD": "AMD", "BA": "Boeing", "DIS": "Disney",
    "KO": "Coca Cola", "PFE": "Pfizer", "INTC": "Intel", "PLTR": "Palantir", "UBER": "Uber", "NKE": "Nike",
    "ORCL": "Oracle", "V": "Visa", "MA": "Mastercard", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "WFC": "Wells Fargo", "C": "Citigroup", "CVX": "Chevron", "JNJ": "Johnson and Johnson", "UNH": "UnitedHealth",
    "MRK": "Merck", "ABBV": "AbbVie", "PEP": "PepsiCo", "MCD": "McDonald's", "HD": "Home Depot",
    "CRM": "Salesforce", "ADBE": "Adobe", "QCOM": "Qualcomm", "TXN": "Texas Instruments", "CSCO": "Cisco",
    "IBM": "IBM", "T": "AT and T", "VZ": "Verizon", "CAT": "Caterpillar", "GE": "GE Aerospace", "UPS": "UPS",
    "SBUX": "Starbucks", "TGT": "Target", "LOW": "Lowe's", "PYPL": "PayPal", "SHOP": "Shopify",
    "SNOW": "Snowflake", "COIN": "Coinbase", "MU": "Micron", "ABNB": "Airbnb", "F": "Ford", "GM": "General Motors",
    "RIVN": "Rivian", "SOFI": "SoFi", "HOOD": "Robinhood", "ARM": "Arm Holdings", "SMCI": "Super Micro",
    "CRWD": "CrowdStrike", "PANW": "Palo Alto Networks", "MRVL": "Marvell", "LRCX": "Lam Research",
    "AMAT": "Applied Materials", "LMT": "Lockheed Martin", "RTX": "RTX", "DE": "Deere", "HON": "Honeywell",
    "MMM": "3M", "AXP": "American Express", "BLK": "BlackRock", "TMO": "Thermo Fisher", "NOW": "ServiceNow",
    "INTU": "Intuit", "SPOT": "Spotify", "ROKU": "Roku", "SNAP": "Snap", "PINS": "Pinterest", "ZM": "Zoom",
}
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "XOM", "LLY", "WMT", "COST",
             "NFLX", "AMD", "BA", "DIS", "KO", "PFE", "INTC", "PLTR", "UBER", "NKE", "ORCL"]
SCAN_UNIVERSE = WATCHLIST + ["V", "MA", "BAC", "GS", "CVX", "JNJ", "UNH", "MRK", "ABBV", "PEP", "MCD", "HD", "CRM",
                             "ADBE", "QCOM", "TXN", "CSCO", "IBM", "CAT", "GE", "SBUX", "PYPL", "SHOP", "MU", "CRWD",
                             "PANW", "AMAT", "LRCX", "LMT", "AXP", "NOW", "INTU", "ARM", "SMCI", "COIN", "HOOD"]
EARNINGS_UNIVERSE = list(COMPANY_NAMES)
AMBIGUOUS_NAMES = {"Target", "Visa", "Ford", "Snap", "Zoom"}  # common English words: not used for topic checks

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


def px(v: float) -> int:
    return int(round(v * UI_SCALE))


# --- Layout boxes, authored for 1920x1080 ---
TICKER_BOX = tuple(px(v) for v in (40, 22, 1840, 56))  # x, y_top, w, h
HEADER_TOP = px(100)
HEADER_SUB_TOP = px(160)  # beneath the headline, above the stage panel
STAGE_BOX = tuple(px(v) for v in (40, 210, 1440, 830))
STAGE_REF = (1440, 830)  # stage size in reference pixels
INFO_BOX = tuple(px(v) for v in (1520, 210, 360, 250))
INFO_REF = (360, 250)
AVATAR_X, AVATAR_Y, AVATAR_SIZE = px(1520), px(680), px(360)
FULL_REF = (1920, 1080)


def axes_rect(box: Tuple[int, int, int, int]) -> List[float]:
    """Pixel box (x, y_top, w, h) -> matplotlib figure-fraction rect [left, bottom, w, h]."""
    x, y, w, h = box
    return [x / VIDEO_W, 1 - (y + h) / VIDEO_H, w / VIDEO_W, h / VIDEO_H]


def fig_x(x_px: float) -> float:
    return x_px / VIDEO_W


def fig_y(y_px_from_top: float) -> float:
    return 1 - y_px_from_top / VIDEO_H


# =====================================================================
# 1b. SCHEMAS  (Director -> Outline, Writer -> ChapterScript, master timeline)
# =====================================================================
class _Model(BaseModel):
    model_config = ConfigDict(use_enum_values=True, validate_default=True)


class LayoutType(str, Enum):
    pro_candlestick_chart = "pro_candlestick_chart"
    stock_dashboard = "stock_dashboard"
    sector_heatmap = "sector_heatmap"
    top_movers_podium = "top_movers_podium"
    macro_timeline = "macro_timeline"
    bull_vs_bear = "bull_vs_bear"
    quote_highlight = "quote_highlight"
    portfolio_10k_sim = "portfolio_10k_sim"
    trade_setup = "trade_setup"
    earnings_calendar = "earnings_calendar"
    outro = "outro"
    plain_text = "plain_text"


class SfxType(str, Enum):
    none = "none"
    cash_register = "cash_register"
    swoosh = "swoosh"
    buzzer = "buzzer"
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
    d1 = "1d"  # last session, 15-minute candles
    w1 = "1wk"  # one week, 1-hour candles
    m1 = "1mo"  # one month, daily candles
    y1 = "1y"  # one year, weekly candles


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


class Stance(str, Enum):
    bull = "bull"
    bear = "bear"
    neutral = "neutral"


class Weekday(str, Enum):
    MON = "MON"
    TUE = "TUE"
    WED = "WED"
    THU = "THU"
    FRI = "FRI"


class ChartNote(_Model):
    anchor: NoteAnchor = Field(default=NoteAnchor.high)
    text: str = Field(description="Annotation text, max 4 words")


class SectorItem(_Model):
    name: str = Field(description="Sector name, e.g. 'Technology'")
    change_pct: float = Field(description="Weekly % change as a plain number, e.g. -1.8")
    weight: float = Field(default=5.0, description="Approx. share of the S&P 500 in percent (1-35), sizes the block")


class MoverItem(_Model):
    ticker: str = Field(description="Real stock ticker, e.g. NVDA")
    change_pct: float = Field(description="Weekly % change as a plain number")


class MacroEvent(_Model):
    day: Weekday = Field(default=Weekday.MON)
    title: str = Field(description="Event name, max 3 words, e.g. 'CPI Print'")
    detail: str = Field(default="", description="One short line about the outcome, max 12 words")


class Segment(_Model):
    layout: LayoutType = Field(description="Which screen to show for this segment")
    headline: str = Field(description="Short on-screen title, UPPERCASE, max 22 characters")
    script: str = Field(description="Concise, engaging spoken narration: use only the sentences needed")
    accent: AccentColor = Field(default=AccentColor.gold)
    sfx: SfxType = Field(default=SfxType.none, description="Sound effect played at the start of the segment")

    # --- pro_candlestick_chart ---
    chart_ticker: str = Field(default="", description="Real liquid ticker or index, e.g. NVDA, SPY, ^GSPC")
    chart_timeframe: CandleTimeframe = Field(default=CandleTimeframe.m1)
    chart_notes: List[ChartNote] = Field(default_factory=list, description="0 to 3 annotations")
    # --- stock_dashboard ---
    dash_ticker: str = Field(default="", description="Real INDIVIDUAL company ticker (never an index)")
    highlight_metric: HighlightMetric = Field(default=HighlightMetric.none,
                                              description="Fallback card to glow if the script names none")
    # --- sector_heatmap ---
    sectors: List[SectorItem] = Field(default_factory=list, description="4 or 5 sectors")
    # --- top_movers_podium ---
    movers_side: MoversSide = Field(default=MoversSide.both, description="both = split screen, or one side only")
    movers_gainers: List[MoverItem] = Field(default_factory=list, description="2 to 5 top gainers")
    movers_losers: List[MoverItem] = Field(default_factory=list, description="2 to 5 top losers")
    # --- macro_timeline ---
    macro_events: List[MacroEvent] = Field(default_factory=list, description="3 or 4 events, chronological")
    # --- bull_vs_bear ---
    bull_title: str = Field(default="BULL CASE", description="Max 3 words")
    bear_title: str = Field(default="BEAR CASE", description="Max 3 words")
    bull_points: List[str] = Field(default_factory=list, description="2 or 3 points, max 8 words each")
    bear_points: List[str] = Field(default_factory=list, description="2 or 3 points, max 8 words each")
    sentence_stances: List[Stance] = Field(default_factory=list,
                                           description="One entry per sentence of `script`, in order")
    # --- quote_highlight ---
    quote_text: str = Field(default="", description="The quote with some context, max 55 words")
    quote_highlight: str = Field(default="", description="The ONE sentence of quote_text that is read aloud")
    quote_source: str = Field(default="", description="Publication or role, max 6 words")
    # --- portfolio_10k_sim ---
    sim_tickers: List[str] = Field(default_factory=list, description="1 to 4 real tickers")
    # --- trade_setup ---
    setup_ticker: str = Field(default="", description="A ticker from the TECHNICAL SCAN section only")
    # --- plain_text ---
    plain_text: str = Field(default="", description="Strong statement shown centred (max 22 words)")
    plain_highlight: List[str] = Field(default_factory=list, description="Up to 3 words from plain_text")


class ChapterPlan(_Model):
    title: str = Field(description="Chapter title, max 5 words")
    summary: str = Field(description="What this chapter covers and which story it tells, 1-2 sentences")
    key_tickers: List[str] = Field(default_factory=list, description="Tickers this chapter is about")
    talking_points: List[str] = Field(description="3 to 6 facts or angles taken from the supplied data/news")
    target_words: Optional[int] = Field(default=None, description="Legacy JSON compatibility only; ignored")


class WeeklyOutline(_Model):
    week_label: str = Field(description="e.g. 'WEEK ENDING OCT 24'")
    hook: str = Field(description="Specific, irresistible cold-open idea for the first 30 seconds")
    chapters: List[ChapterPlan]
    news_ticker_text: str = Field(
        description="3 or 4 UPPERCASE items 'TOPIC: ultra short summary' joined by ' \u2022 '")
    youtube_title: str
    description: str
    tags: str


class ChapterScript(_Model):
    segments: List[Segment]


class EarningsItem(_Model):
    ticker: str
    name: str
    date: str  # YYYY-MM-DD
    session: str = Field(default="TBD")  # BMO (before market open), AMC (after market close) or TBD


class SetupItem(_Model):
    ticker: str
    name: str
    pattern: str
    direction: str = Field(default="bullish")
    score: float = Field(default=0.0)
    entry: float
    stop: float
    target: float
    rr: float = Field(default=2.0)
    note: str = Field(default="", description="spoken-friendly traits, no digits")
    metrics: str = Field(default="", description="on-screen metrics such as 'RSI 62 | VOL 1.6x'")


class TimelineItem(_Model):
    chapter_index: int
    chapter_title: str
    segment: Segment


class MasterTimeline(_Model):
    outline: WeeklyOutline
    items: List[TimelineItem]
    earnings: List[EarningsItem] = Field(default_factory=list)
    setups: List[SetupItem] = Field(default_factory=list)


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


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def parallel_map(fn: Callable, items: Sequence, max_workers: int = IO_WORKERS) -> list:
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
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
    """Normalised RMS volume per video frame (vectorised; used for lip sync and animation pacing)."""
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


def ease_out_back(p: float, k: float = 1.70158) -> float:
    p -= 1
    return 1 + (k + 1) * p ** 3 + k * p ** 2


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
# 4. MARKET DATA  (pre-fetch, local cache, strict fallbacks)
# =====================================================================
def normalize_timeframe(tf: Any) -> str:
    t = re.sub(r"\s+", "", ev(tf or "").lower())
    t = {"1w": "1wk", "1week": "1wk", "5d": "1wk", "1m": "1mo", "1month": "1mo", "1yr": "1y", "12mo": "1y",
         "1day": "1d", "day": "1d", "week": "1wk", "month": "1mo", "year": "1y"}.get(t, t)
    return t if t in CANDLE_SPECS else "1mo"


def clean_ticker(t: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.\-^=]", "", str(t or "")).upper()[:12]


def _clean_df(df, need_volume: bool = True):
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if need_volume and "Volume" in df.columns:
        df = df.assign(Volume=df["Volume"].fillna(0))
    return df


def _fetch_candles_once(ticker: str, tf: str) -> Optional[dict]:
    period, interval = CANDLE_SPECS[tf]

    def _go():
        df = _clean_df(_yf().Ticker(ticker).history(period=period, interval=interval))
        if tf == "1d" and len(df):  # keep only the most recent session of the intraday window
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
    fmt = {"1d": "%H:%M", "1wk": "%a %H:%M", "1mo": "%b %d", "1y": "%b %Y"}[tf]
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
    """1d -> 15m, 1wk -> 1h, 1mo -> 1d, 1y -> 1wk. Anything empty falls back to 1mo daily candles.
            The reference close for the on-screen % change is chosen so it matches the numbers the narration uses
            (same definition as the weekly / daily placeholders)."""
    chain = [requested_tf] + ([CANDLE_FALLBACK_TF] if requested_tf != CANDLE_FALLBACK_TF else [])
    for tf in chain:
        data = _fetch_candles_once(ticker, tf)
        if not data:
            continue
        data["requested_timeframe"] = requested_tf
        data["fallback"] = tf != requested_tf
        week = fetch_week_perf(ticker)
        if tf == "1d" and week:
            data["ref_close"], data["ref_label"] = week["prev_close"], "vs previous close"
        elif tf == "1wk" and week:
            data["ref_close"], data["ref_label"] = week["base"], "vs last week's close"
        else:
            data["ref_close"], data["ref_label"] = data["open"][0], "over the period"
        if data["fallback"]:
            logger.warning(f"{ticker}: {requested_tf} data unavailable, fell back to {tf} daily candles")
        return data
    return None


_WEEK_MEMO: Dict[str, dict] = {}


def fetch_week_perf(ticker: str) -> Optional[dict]:
    """Memoised wrapper: only successful downloads are cached, failures are retried on the next call."""
    if ticker not in _WEEK_MEMO:
        data = _fetch_week_perf_uncached(ticker)
        if data:
            _WEEK_MEMO[ticker] = data
        return data
    return _WEEK_MEMO[ticker]


def _fetch_week_perf_uncached(ticker: str) -> Optional[dict]:
    """Last 5 sessions from one daily-history call. 'base' is the close BEFORE the window (the 'start of week')."""

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
    return {"ticker": ticker, "base": base, "last": last, "prev_close": prev,
            "pct": (last / base - 1) * 100, "day_pct": (last / prev - 1) * 100,
            "start_date": str(df.index[-win].date()), "end_date": str(df.index[-1].date())}


def fetch_stock_info(ticker: str) -> Optional[dict]:
    """Price / changes from daily history (reliable); market cap, P/E, dividend yield from Yahoo (best effort).
            Anything Yahoo does not provide stays None, and the dashboard simply leaves that card out."""
    week = fetch_week_perf(ticker)
    if not week:
        return None
    info: dict = {}
    try:
        info = with_retries(lambda: dict(_yf().Ticker(ticker).info or {}), attempts=2, base_delay=1.5,
                            what=f"{ticker} info")
    except Exception as e:
        logger.warning(f"No fundamentals for {ticker}: {e}")
    price = week["last"]
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
    return {"ticker": ticker, "name": name, "price": price, "change_pct": week["day_pct"],
            "week_pct": week["pct"], "market_cap": mcap, "pe": pe, "div_yield": dy}


# ---------------------------------------------------------------------
# Logos (best effort; a coloured placeholder is drawn when nothing is found)
# ---------------------------------------------------------------------
_LOGO_MISS: set = set()


def ensure_logo(ticker: str) -> Optional[str]:
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
# Upcoming earnings
# ---------------------------------------------------------------------
def earnings_window(today: Optional[datetime.date] = None) -> List[datetime.date]:
    """The next five weekdays: Monday-Friday of next week when run on a weekend, otherwise starting today."""
    today = today or datetime.date.today()
    start = today if today.weekday() < 5 else today + datetime.timedelta(days=7 - today.weekday())
    days = [start + datetime.timedelta(days=i) for i in range(9)]
    return [d for d in days if d.weekday() < 5][:5]


def _earnings_for(args: Tuple[str, Tuple[datetime.date, ...]]) -> List[dict]:
    ticker, dates = args
    dset, out = set(dates), []
    tk = _yf().Ticker(ticker)
    try:
        df = tk.get_earnings_dates(limit=6)
        for ts in (df.index if df is not None else []):
            if ts.date() in dset:
                session = "BMO" if ts.hour < 10 else "AMC" if ts.hour >= 15 else "TBD"
                if ts.hour == 0 and ts.minute == 0:
                    session = "TBD"
                out.append({"ticker": ticker, "date": ts.date().isoformat(), "session": session})
    except Exception:
        pass
    if not out:
        try:
            cal = tk.calendar
            raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
            for d in (raw or []):
                d = d.date() if hasattr(d, "date") else d
                if d in dset:
                    out.append({"ticker": ticker, "date": d.isoformat(), "session": "TBD"})
        except Exception:
            pass
    return out[:1]


def fetch_upcoming_earnings(dates: List[datetime.date]) -> List[dict]:
    logger.info(f"Checking upcoming earnings for {len(EARNINGS_UNIVERSE)} companies ({dates[0]} to {dates[-1]})")
    results = parallel_map(_earnings_for, [(t, tuple(dates)) for t in EARNINGS_UNIVERSE])
    order = {t: i for i, t in enumerate(EARNINGS_UNIVERSE)}
    rows = [dict(r, name=COMPANY_NAMES.get(r["ticker"], r["ticker"])) for res in results for r in res]
    rows.sort(key=lambda r: (r["date"], {"BMO": 0, "AMC": 1, "TBD": 2}[r["session"]], order[r["ticker"]]))
    logger.info(f"Found {len(rows)} reports in the window")
    return rows


# ---------------------------------------------------------------------
# Rule-based chart-pattern scan (indicators computed from real prices; the LLM only narrates the result)
# ---------------------------------------------------------------------
_DAILY_MEMO: Dict[str, dict] = {}


def fetch_daily(ticker: str) -> Optional[dict]:
    if ticker in _DAILY_MEMO:
        return _DAILY_MEMO[ticker]

    def _go():
        df = _clean_df(_yf().Ticker(ticker).history(period="1y", interval="1d"))
        if len(df) < 120:
            raise RuntimeError(f"only {len(df)} rows")
        return df

    try:
        df = with_retries(_go, attempts=2, base_delay=1.5, what=f"{ticker} 1y daily")
    except Exception as e:
        logger.warning(f"No daily history for {ticker}: {e}")
        return None
    data = {"ticker": ticker, "open": df["Open"].astype(float).tolist(), "high": df["High"].astype(float).tolist(),
            "low": df["Low"].astype(float).tolist(), "close": df["Close"].astype(float).tolist(),
            "volume": df["Volume"].astype(float).tolist(), "dates": [d.strftime("%b %d") for d in df.index]}
    _DAILY_MEMO[ticker] = data
    return data


def _sma(a, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        out[w - 1:] = np.convolve(a, np.ones(w) / w, mode="valid")
    return out


def _ema(a, span: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.zeros(len(a))
    out[0] = a[0]
    k = 2 / (span + 1)
    for i in range(1, len(a)):
        out[i] = k * a[i] + (1 - k) * out[i - 1]
    return out


def _rsi(c, n: int = 14) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au, ad = up[:n].mean(), dn[:n].mean()
    out[n] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    for i in range(n, len(d)):
        au, ad = (au * (n - 1) + up[i]) / n, (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100 - 100 / (1 + au / ad) if ad > 0 else 100.0
    return out


def _atr(h, l, c, n: int = 14) -> float:
    h, l, c = (np.asarray(x, dtype=float) for x in (h, l, c))
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    if len(tr) <= n:
        return float(tr.mean())
    a = float(tr[:n].mean())
    for x in tr[n:]:
        a = (a * (n - 1) + x) / n
    return a


def analyze_setup(ticker: str, d: dict) -> Optional[dict]:
    """Looks for classic bullish daily-chart setups. Returns the best one with entry / stop / target levels."""
    o, h, l, c, v = (np.asarray(d[k], dtype=float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    if n < 210:
        return None
    s50, s200 = _sma(c, 50), _sma(c, 200)
    r, atr = _rsi(c), _atr(h, l, c)
    last = float(c[-1])
    if not (np.isfinite(s50[-1]) and np.isfinite(s200[-1]) and np.isfinite(r[-1])) or atr <= 0:
        return None
    avg_v = float(v[-21:-1].mean()) if n > 21 else 0.0
    vr = float(v[-1]) / avg_v if avg_v > 0 else 1.0
    uptrend = last > s200[-1] and s50[-1] > s200[-1]
    above50 = last > s50[-1]
    macd = _ema(c, 12) - _ema(c, 26)
    sig = _ema(macd, 9)
    cands: List[Tuple[str, float]] = []
    if last > float(h[-21:-1].max()) and above50 and vr >= 1.2 and r[-1] < 80:
        cands.append(("Breakout above the 20 day high", 70 + min(10.0, (vr - 1.2) * 10) + (8 if uptrend else 0)))
    if (uptrend and abs(last / s50[-1] - 1) <= 0.03 and last >= s50[-1] * 0.99 and float(c[-8:].max()) > last * 1.02
            and 38 <= r[-1] <= 60):
        cands.append(("Pullback to the 50 day average", 66 + (6 if vr < 1.0 else 0)))
    if any(s50[i - 1] <= s200[i - 1] and s50[i] > s200[i] for i in range(n - 8, n)):
        cands.append(("Golden cross of the 50 and 200 day averages", 60 + (5 if above50 else 0)))
    if above50 and any(macd[i - 1] <= sig[i - 1] and macd[i] > sig[i] for i in range(n - 3, n)):
        cands.append(("Bullish MACD crossover", 55))
    if float(np.nanmin(r[-5:])) < 30 and r[-1] > r[-2] and last > c[-2]:
        cands.append(("Oversold bounce", 50))
    if not cands:
        return None
    pattern, score = max(cands, key=lambda x: x[1])
    stop = max(float(l[-10:].min()), last - 2.5 * atr)
    if stop >= last:
        stop = last - 1.5 * atr
    risk = last - stop
    t90 = float(h[-90:].max())
    target = t90 if t90 > last + 1.5 * risk else last + 2 * risk
    rr = (target - last) / risk
    if rr < 1.5:
        return None
    score += min(rr, 3.0) * 3
    traits = ["trading above its fifty day average" if above50 else "still below its fifty day average",
              "inside a long term uptrend" if uptrend else "without a clear long term uptrend",
              "with volume well above normal" if vr >= 1.5 else "with volume above normal" if vr >= 1.1
              else "on quieter than normal volume"]
    return {"ticker": ticker, "name": COMPANY_NAMES.get(ticker, ticker), "pattern": pattern, "direction": "bullish",
            "score": round(score, 1), "entry": last, "stop": stop, "target": target, "rr": round(rr, 2),
            "note": ", ".join(traits), "metrics": f"RSI {r[-1]:.0f} | VOL {vr:.1f}x | ATR {atr / last * 100:.1f}%"}


def scan_setups(limit: int = 3) -> List[dict]:
    logger.info(f"Scanning {len(SCAN_UNIVERSE)} charts for setups...")
    found = []
    for t, d in zip(SCAN_UNIVERSE, parallel_map(fetch_daily, SCAN_UNIVERSE)):
        if d:
            s = analyze_setup(t, d)
            if s and s["score"] >= 55:
                found.append(s)
    found.sort(key=lambda s: -s["score"])
    logger.info("Setups found: " + (", ".join(f"{s['ticker']} ({s['pattern']})" for s in found[:limit]) or "none"))
    return found[:limit]


def setup_chart_data(ticker: str) -> Optional[dict]:
    """Last ~70 daily candles plus the 20 and 50 day averages, for the trade_setup screen."""
    d = fetch_daily(ticker)
    if not d:
        return None
    c = np.asarray(d["close"], dtype=float)
    keep = 70
    s20, s50 = _sma(c, 20)[-keep:], _sma(c, 50)[-keep:]
    n = min(keep, len(c))
    idxs = np.unique(np.linspace(0, n - 1, 5).astype(int))
    dates = d["dates"][-n:]
    return {"ticker": ticker, "timeframe": "3mo", "interval": "1d", "open": d["open"][-n:], "high": d["high"][-n:],
            "low": d["low"][-n:], "close": d["close"][-n:], "volume": d["volume"][-n:],
            "sma20": [None if not np.isfinite(x) else float(x) for x in s20],
            "sma50": [None if not np.isfinite(x) else float(x) for x in s50],
            "tick_idx": [int(i) for i in idxs], "tick_labels": [dates[i] for i in idxs]}


# ---------------------------------------------------------------------
# News + context for the LLM
# ---------------------------------------------------------------------
@dataclass
class MarketCache:
    candles: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    info: Dict[str, dict] = field(default_factory=dict)
    weeks: Dict[str, dict] = field(default_factory=dict)
    setup_charts: Dict[str, dict] = field(default_factory=dict)
    logos: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class WeeklyContext:
    text: str
    earnings: List[dict] = field(default_factory=list)
    setups: List[dict] = field(default_factory=list)


def _news_item_text(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = ascii_clean(str(c.get("title") or ""))
    if not title:
        return None
    summary = re.sub(r"<[^>]+>", " ", str(c.get("summary") or c.get("description") or ""))
    provider = c.get("provider") if isinstance(c.get("provider"), dict) else {}
    pub = ascii_clean(str(provider.get("displayName") or c.get("publisher") or ""))
    line = f"[{pub}] {title}" if pub else title
    summary = shorten(ascii_clean(summary), 220)
    return f"{line} - {summary}" if summary else line


def gather_weekly_context(news_file: Optional[str] = None) -> WeeklyContext:
    """Everything the Director/Writers may use: real index, sector and mega-cap moves, news headlines,
            next week's earnings and the rule-based chart-pattern scan."""
    logger.info("Gathering weekly market data and news...")
    symbols = list(INDEX_SYMBOLS) + list(SECTOR_ETFS) + WATCHLIST
    weeks = {s: w for s, w in zip(symbols, parallel_map(fetch_week_perf, symbols)) if w}
    if not weeks and not news_file:
        raise RuntimeError("Could not download any market data and no --news-file was given.")

    lines: List[str] = []
    ref = weeks.get("^GSPC") or next(iter(weeks.values()), None)
    if ref:
        end = datetime.date.fromisoformat(ref["end_date"])
        lines.append(
            f"WEEK: {ref['start_date']} to {ref['end_date']} (last 5 trading sessions, ending {end.strftime('%A %B %d %Y')})")
    lines.append("INDEXES (weekly % change; use placeholders like [[^GSPC:week]] in scripts):")
    for s, name in INDEX_SYMBOLS.items():
        if s in weeks:
            lines.append(
                f"- {name} ({s}): {weeks[s]['pct']:+.2f}% week, {weeks[s]['day_pct']:+.2f}% last day, level {weeks[s]['last']:,.2f}")
    lines.append("SECTORS via sector ETFs (weekly % change):")
    for s, name in SECTOR_ETFS.items():
        if s in weeks:
            lines.append(f"- {name} ({s}): {weeks[s]['pct']:+.2f}%")
    movers = sorted(((s, weeks[s]["pct"]) for s in WATCHLIST if s in weeks), key=lambda x: x[1], reverse=True)
    if movers:
        lines.append("MOVERS among large caps (weekly % change):")
        lines.append("- Gainers: " + ", ".join(f"{COMPANY_NAMES.get(s, s)} ({s}) {p:+.2f}%" for s, p in movers[:6]))
        lines.append(
            "- Losers: " + ", ".join(f"{COMPANY_NAMES.get(s, s)} ({s}) {p:+.2f}%" for s, p in movers[::-1][:6]))

    def _news(sym):
        try:
            return _yf().Ticker(sym).news or []
        except Exception as e:
            logger.warning(f"No news for {sym}: {e}")
            return []

    seen, headlines = set(), []
    for items in parallel_map(_news, NEWS_SYMBOLS):
        for it in items:
            text = _news_item_text(it)
            key = (text or "")[:60].lower()
            if text and key not in seen:
                seen.add(key)
                headlines.append(text)
    if headlines:
        lines.append("HEADLINES AND SUMMARIES (the only quotable sources):")
        lines += [f"- {h}" for h in headlines[:28]]

    earnings: List[dict] = []
    try:
        dates = earnings_window()
        earnings = fetch_upcoming_earnings(dates)
    except Exception as e:
        logger.warning(f"Earnings lookup failed: {e}")
    if earnings:
        lines.append(
            "UPCOMING EARNINGS next week (BMO = before market open, AMC = after market close, TBD = time unconfirmed):")
        for d in sorted({r["date"] for r in earnings}):
            day = datetime.date.fromisoformat(d)
            parts = []
            for sess in ("BMO", "AMC", "TBD"):
                names = [f"{r['name']} ({r['ticker']})" for r in earnings if r["date"] == d and r["session"] == sess]
                if names:
                    parts.append(f"{sess}: {', '.join(names[:8])}")
            lines.append(f"- {day.strftime('%A %b %d')}: " + "; ".join(parts))

    setups: List[dict] = []
    try:
        setups = scan_setups()
    except Exception as e:
        logger.warning(f"Pattern scan failed: {e}")
    if setups:
        lines.append(
            "TECHNICAL SCAN (rule-based daily chart patterns, educational; trade_setup may ONLY use these tickers; "
            "refer to their levels with placeholders [[TICKER:entry]] [[TICKER:stop]] [[TICKER:target]]):")
        for s in setups:
            lines.append(f"- {s['ticker']} ({s['name']}): {s['pattern']}; {s['note']}; entry {s['entry']:.2f}, "
                         f"stop {s['stop']:.2f}, target {s['target']:.2f}, reward to risk {s['rr']:.1f}; {s['metrics']}")
    if news_file:
        try:
            lines.append("ADDITIONAL NOTES FROM THE PRODUCER:")
            lines.append(Path(news_file).read_text(encoding="utf-8", errors="ignore")[:6000])
        except Exception as e:
            logger.warning(f"Could not read {news_file}: {e}")
    return WeeklyContext("\n".join(lines)[:CONTEXT_MAX_CHARS], earnings, setups)


# =====================================================================
# 5. SCRIPT NORMALISATION / VALIDATION / LLM PROMPTS
# =====================================================================
def sector_etf(name: str) -> Optional[str]:
    n = (name or "").lower()
    return next((etf for key, etf in SECTOR_MATCH if key in n), None)


def best_sentence_index(sentences: List[str], wanted: str) -> int:
    """Index of the sentence that best matches `wanted` (the writer is asked to copy it, but may paraphrase)."""
    target = set(re.findall(r"[a-z0-9']+", (wanted or "").lower()))
    if not target:
        return int(np.argmax([len(s) for s in sentences])) if sentences else 0
    scores = [len(target & set(re.findall(r"[a-z0-9']+", s.lower()))) / max(1, len(target)) for s in sentences]
    return int(np.argmax(scores))


def downgrade_to_plain(seg: Segment, reason: str, apology: Optional[str] = None) -> None:
    """A layout that lacks its data becomes a centred statement taken from the narration."""
    logger.warning(f"Segment '{seg.headline}': {ev(seg.layout)} -> plain_text ({reason})")
    summary = shorten(first_sentence(seg.script), 140)
    if apology:
        seg.script = f"{apology} {seg.script}".strip()
        summary = shorten(f"{apology} {summary}", 150)
    seg.layout = "plain_text"
    seg.plain_text = summary
    seg.plain_highlight = []


def _payload_problem(seg: Segment) -> Optional[str]:
    L = seg.layout
    if L == "pro_candlestick_chart" and not seg.chart_ticker:
        return "no ticker"
    if L == "stock_dashboard" and not seg.dash_ticker:
        return "no ticker"
    if L == "sector_heatmap" and len(seg.sectors) < 3:
        return "needs at least 3 sectors"
    if L == "top_movers_podium":
        need = {"both": 2, "gainers": 2, "losers": 2}[seg.movers_side]
        have = {"both": len(seg.movers_gainers) + len(seg.movers_losers), "gainers": len(seg.movers_gainers),
                "losers": len(seg.movers_losers)}[seg.movers_side]
        if have < need:
            return "needs movers"
    if L == "macro_timeline" and len(seg.macro_events) < 2:
        return "needs at least 2 events"
    if L == "bull_vs_bear" and (not seg.bull_points or not seg.bear_points):
        return "needs bull and bear points"
    if L == "quote_highlight" and (len(seg.quote_text) < 20 or not seg.quote_highlight):
        return "quote too short"
    if L == "portfolio_10k_sim" and not seg.sim_tickers:
        return "no tickers"
    if L == "trade_setup" and not seg.setup_ticker:
        return "no setup ticker"
    if L == "plain_text" and len(seg.plain_text) < 3:
        seg.plain_text = shorten(first_sentence(seg.script), 140)
    return None


DISCLAIMER_RE = re.compile(r"not\s+(a\s+)?(financial|investment)\s+advice", re.I)


def normalize_segment(seg: Segment) -> None:
    """Cleans every field and repairs / rejects what the renderer cannot draw. Raising triggers an LLM retry."""
    seg.layout = ev(seg.layout) if ev(seg.layout) in LAYOUTS else "plain_text"
    L = seg.layout
    seg.quote_text = shorten(ascii_clean(seg.quote_text), 330)
    seg.quote_highlight = shorten(ascii_clean(seg.quote_highlight), 220)
    if L == "quote_highlight" and seg.quote_text:
        sents = split_sentences(seg.quote_text) or [seg.quote_text]
        seg.quote_highlight = sents[best_sentence_index(sents, seg.quote_highlight)]
        seg.script = seg.quote_highlight  # the voice reads exactly what is highlighted
    seg.script = ascii_clean(seg.script)
    words = len(seg.script.split())
    if words < (6 if L == "quote_highlight" else SEG_WORDS_MIN) or words > SEG_WORDS_MAX:
        raise ValueError(f"segment narration has {words} words: {seg.script[:60]!r}")
    seg.headline = shorten(ascii_clean(seg.headline).upper(), 26) or DEFAULT_HEADLINES[L]
    seg.accent = norm_color_name(seg.accent)
    seg.sfx = ev(seg.sfx) if ev(seg.sfx) in {s.value for s in SfxType} else "none"

    seg.chart_ticker = clean_ticker(seg.chart_ticker)
    seg.chart_timeframe = normalize_timeframe(seg.chart_timeframe)
    for n in seg.chart_notes:
        n.text, n.anchor = shorten(ascii_clean(n.text), 32), ev(n.anchor)
    seg.chart_notes = [n for n in seg.chart_notes if n.text][:3]
    seg.dash_ticker = clean_ticker(seg.dash_ticker)
    seg.highlight_metric = ev(seg.highlight_metric)
    for s in seg.sectors:
        s.name = shorten(ascii_clean(s.name), 26)
        s.weight = float(s.weight) if _num(s.weight) and s.weight > 0 else 5.0
        s.change_pct = float(s.change_pct) if _num(s.change_pct) is not None else 0.0
    seg.sectors = [s for s in seg.sectors if s.name][:5]
    seg.movers_side = ev(seg.movers_side) if ev(seg.movers_side) in ("both", "gainers", "losers") else "both"
    for m in seg.movers_gainers + seg.movers_losers:
        m.ticker = clean_ticker(m.ticker)
    seg.movers_gainers = [m for m in seg.movers_gainers if m.ticker][:5]
    seg.movers_losers = [m for m in seg.movers_losers if m.ticker][:5]
    for e in seg.macro_events:
        e.day, e.title, e.detail = ev(e.day), shorten(ascii_clean(e.title), 28), shorten(ascii_clean(e.detail), 90)
    seg.macro_events = [e for e in seg.macro_events if e.title][:4]
    seg.bull_title = shorten(ascii_clean(seg.bull_title).upper(), 18) or "BULL CASE"
    seg.bear_title = shorten(ascii_clean(seg.bear_title).upper(), 18) or "BEAR CASE"
    seg.bull_points = [shorten(ascii_clean(p), 70) for p in seg.bull_points if ascii_clean(p)][:3]
    seg.bear_points = [shorten(ascii_clean(p), 70) for p in seg.bear_points if ascii_clean(p)][:3]
    seg.sentence_stances = [ev(s) for s in seg.sentence_stances]
    seg.quote_source = shorten(ascii_clean(seg.quote_source), 50)
    seg.sim_tickers = [t for t in dict.fromkeys(clean_ticker(t) for t in seg.sim_tickers) if t][:4]
    seg.setup_ticker = clean_ticker(seg.setup_ticker)
    seg.plain_text = shorten(ascii_clean(seg.plain_text), 150)
    seg.plain_highlight = [ascii_clean(w) for w in seg.plain_highlight if ascii_clean(w)][:3]

    problem = _payload_problem(seg)
    if problem:
        downgrade_to_plain(seg, problem)


def clean_ticker_text(text: str) -> str:
    return re.sub(r"[^\x20-\x7E\u2022]+", "", text or "").strip()


def normalize_outline(o: WeeklyOutline) -> WeeklyOutline:
    if not CHAPTERS_MIN <= len(o.chapters) <= CHAPTERS_MAX:
        raise ValueError(f"expected {CHAPTERS_MIN}-{CHAPTERS_MAX} chapters, got {len(o.chapters)}")
    if not o.youtube_title.strip():
        raise ValueError("empty youtube_title")
    o.week_label = shorten(ascii_clean(o.week_label).upper(), 30) or "WEEKLY RECAP"
    o.hook = ascii_clean(o.hook)
    for ch in o.chapters:
        ch.title = shorten(ascii_clean(ch.title), 34)
        if not ch.title or not ch.talking_points:
            raise ValueError("chapter without title or talking points")
        ch.summary = ascii_clean(ch.summary)
        ch.key_tickers = [t for t in (clean_ticker(t) for t in ch.key_tickers) if t][:6]
        ch.talking_points = [ascii_clean(p) for p in ch.talking_points if ascii_clean(p)][:6]
    # Never scale chapters to a fixed word count. Omit deprecated target_words for new episodes.
    for ch in o.chapters:
        ch.target_words = None
    o.news_ticker_text = clean_ticker_text(
        o.news_ticker_text) or "WEEKLY RECAP  \u2022  MARKETS  \u2022  NOT FINANCIAL ADVICE"
    return o


def parse_outline(response) -> WeeklyOutline:
    return normalize_outline(WeeklyOutline.model_validate_json(response.text))


def _normalize_segments(segments: List[Segment]) -> List[Segment]:
    """Normalise every segment; a segment that cannot be repaired is dropped instead of failing the chapter."""
    good = []
    for seg in segments[:SEG_PER_CHAPTER_MAX]:
        try:
            normalize_segment(seg)
            good.append(seg)
        except ValueError as e:
            logger.warning(f"Dropping unusable segment: {e}")
    return good


DRY_JARGON = re.compile(
    r"\b(?:market participants|sustained liquidation|institutional sponsorship|"
    r"risk assets|amidst|impending trouble|broader index uncertainty|"
    r"reassessed their risk exposure|navigating a very choppy environment|"
    r"downward pressure|on the other side of the ledger|"
    r"relative strength index reading)\b", re.I)


def _narration_signature(script: str) -> set:
    return set(re.findall(r"[a-z0-9]+", PLACEHOLDER_RE.sub("", script.lower())))


def _nearly_same_story(a: str, b: str) -> bool:
    aa, bb = _narration_signature(a), _narration_signature(b)
    return bool(aa and bb and len(aa & bb) / len(aa | bb) >= 0.80)


def _screen_subjects(seg: Segment) -> frozenset:
    """What this screen covers; different stocks do NOT constitute duplicate sections."""
    layout = ev(seg.layout)
    if layout == "trade_setup":
        return frozenset([seg.setup_ticker]) if seg.setup_ticker else frozenset()
    if layout == "stock_dashboard":
        return frozenset([seg.dash_ticker]) if seg.dash_ticker else frozenset()
    if layout == "pro_candlestick_chart":
        return frozenset([seg.chart_ticker]) if seg.chart_ticker else frozenset()
    if layout == "top_movers_podium":
        return frozenset(m.ticker for m in seg.movers_gainers + seg.movers_losers if m.ticker)
    if layout == "sector_heatmap":
        return frozenset(sector_etf(x.name) or x.name.lower() for x in seg.sectors)
    if layout == "portfolio_10k_sim":
        return frozenset(seg.sim_tickers)
    return frozenset()


def _duplicated_story(a: Segment, b: Segment) -> bool:
    """Reject substantially identical narration about the same subject, not reusable visual layouts."""
    aa, bb = _screen_subjects(a), _screen_subjects(b)
    if aa and bb and aa != bb:
        return False
    return _nearly_same_story(a.script, b.script)


def parse_chapter(response, plan: ChapterPlan, near_misses: Optional[list] = None,
                  previous_layout: str = "", first_chapter: bool = False,
                  source_context: str = "") -> ChapterScript:
    """Judge story quality and screen rhythm; never judge length against a quota."""
    cs = ChapterScript.model_validate_json(response.text)
    cs.segments = _normalize_segments(cs.segments)
    if len(cs.segments) < SEG_PER_CHAPTER_MIN:
        raise ValueError("no usable chapter segments")
    issues = []
    seen_segments: List[Segment] = []
    for s in cs.segments:
        if DRY_JARGON.search(s.script):
            issues.append("corporate jargon")
        if s.layout == "quote_highlight" and (
                not s.quote_source.strip() or
                not s.quote_highlight or
                ascii_clean(s.quote_highlight).lower() not in ascii_clean(source_context).lower()):
            issues.append("unverified attributed quotation")
        if any(_duplicated_story(s, old) for old in seen_segments):
            issues.append("repeated narration about the same subject")
        seen_segments.append(s)
    if first_chapter:
        first_two = " ".join(split_sentences(cs.segments[0].script)[:2])
        if not re.search(r"\b(you|your|we|our)\b|\?", first_two, re.I):
            issues.append("weak first-30-seconds hook")
    if issues:
        if near_misses is not None:
            near_misses.append((len(issues), cs))
        raise ValueError("; ".join(issues[:5]))
    return cs


def _gen_config(types, schema, temperature: float):
    kwargs = dict(response_mime_type="application/json", response_schema=schema, temperature=temperature)
    try:  # the SDK's automatic function calling is not needed here and only produces warnings
        return types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **kwargs)
    except Exception:
        return types.GenerateContentConfig(**kwargs)


LAYOUT_GUIDE = """
        SCREENS. Fill ONLY the fields for the selected layout. Make each screen earn its place.
        - pro_candlestick_chart: ONE ticker or index (chart_ticker, chart_timeframe 1d/1wk/1mo/1y, chart_notes).
        - stock_dashboard: metric cards for ONE company (dash_ticker, highlight_metric), not an ETF/index.
        - sector_heatmap: 4-5 verified sector ETFs (sectors with name, change_pct and weight).
        - top_movers_podium: verified gainers/losers; use the 'both' view no more than once per episode.
        - macro_timeline: 2-4 REAL dated events only (macro_events), never generic invented market events.
        - bull_vs_bear: supported competing interpretations (bull_points, bear_points, sentence_stances).
        - quote_highlight: only if you can COPY an exact attributed sentence from the supplied news. quote_source required.
        - portfolio_10k_sim: verified comparative returns (sim_tickers). Never invent returns.
        - trade_setup: the COMPLETE stock analysis on ONE screen, for one ticker from the supplied TECHNICAL SCAN.
          Include pattern, entry [[TICKER:entry]], stop [[TICKER:stop]], target [[TICKER:target]], and why it matters.
          Show each setup ticker EXACTLY ONCE in the entire setup chapter. NEVER follow or precede it with a second chart,
          dashboard, or other analysis of the same ticker. One setup screen per stock is enough.
        - earnings_calendar: show the full next-week calendar visually but use ONE screen in the ENTIRE episode.
          In narration mention only the 2-3 biggest reporting days and 1-2 headline companies per day.
          Name the weekdays so the right columns highlight. NEVER read the entire list of companies.
          Mention what to watch (guidance, revenue growth, reaction) instead of company roll calls.
        - outro: one brief, genuine final takeaway + invite a comment/subscribe, only at the end.
        - plain_text: a short, substantial insight, not a transition or a filler summary.

        ABSOLUTE EDITING RULES:
        1. Reuse a layout whenever it helps explain a DIFFERENT stock or genuinely different insight.
           Two trade_setup screens for two different tickers can appear one after another.
        2. ONE earnings_calendar in the episode, at most. ONE trade_setup per setup ticker.
        3. Combine redundant insights about the SAME subject, rather than repeating the same facts.
        4. A chapter can have only 1-3 segments if that is all the story needs. Do not fill empty time.
        """

STYLE_GUIDE = """
        VOICE: You are a smart, charismatic, high-energy FINANCIAL YOUTUBER talking to a friend, NOT a news anchor.
        Use 'you', 'your', 'we' naturally. Short, punchy sentences. Ask meaningful rhetorical questions.
        Start with surprise, stakes or contrast, then quickly show the evidence and tell viewers WHY THEY SHOULD CARE.
        Contractions and everyday English are welcome. Energy comes from sharp insights, not fake hype or shouting.
        NEVER write like a financial press release. Avoid: 'market participants', 'sustained liquidation',
        'institutional sponsorship', 'risk assets', 'downward pressure', 'amidst', 'strategic positioning',
        'or vague claims about traders, analysts, market sentiment or volume with no evidence.
        No unsubstantiated causes, fake quotes, invented indicators, guaranteed outcomes or invented catalysts.
        Cut every sentence that does not introduce a new fact, useful interpretation or concrete takeaway.
        Never repeat the same fact in different words, restate the chapter heading, or give empty risk-management advice.
        Never read a list of prices when the screen can show them. Translate numbers into meaning.

        FIRST 30 SECONDS: No hello, no channel intro, no week-ending date, no dry index rundown.
        Open in the FIRST SPOKEN SENTENCE with a data-backed twist, sharp question or surprising contrast.
        Use the next two or three sentences to explain the stakes and tease what viewers will learn.
        The hook must feel worthwhile within 20-30 seconds, not promise information you'll never deliver.

        NUMBERS: Never invent prices, percent moves, or trading levels for tickers, indices or sector ETFs.
        Use verified placeholders, e.g. [[NVDA:week]] for a complete spoken move INCLUDING direction.
        Write 'Nvidia finished the week [[NVDA:week]].' NOT 'Nvidia rose [[NVDA:week]].'
        Also available: [[NVDA:day]], [[NVDA:price]], and scan tickers [[NVDA:entry]], [[NVDA:stop]], [[NVDA:target]].
        Talk only about the displayed company during a single-ticker screen.

        TTS: Use natural spoken English, company names instead of ticker codes, and simple punctuation
        (periods, commas, question marks). No emojis, bullet symbols or markdown in spoken scripts.
        Each segment may be as short as 1-2 sentences or as long as needed for ONE interesting idea, but don't pad.
        Keep headlines concise, UPPERCASE, ASCII. SFX only when meaningful, not for every screen.
        Stay grounded and balanced; do not insert or speak financial-advice disclaimers.
        """


def generate_outline(ctx: WeeklyContext) -> WeeklyOutline:
    from google.genai import types
    logger.info("Director: planning a retention-first story...")
    extras = []
    if ctx.setups:
        extras.append(
            "If the technical scan has compelling setups, plan ONE brief chart-setup chapter with ONE screen per ticker.")
    if ctx.earnings:
        extras.append(
            "Finish with the 2-3 most important earnings days, highlighting only the biggest names and what to watch.")
    prompt = f"""
            You are the DIRECTOR of a fast, high-retention financial YouTube recap.
            Your priority is information density and viewer satisfaction, NOT runtime.
            An excellent 5-8 minute video beats a boring 10-minute video. An even shorter video is fine.
            Use ONLY verified information below. Find a strong central tension, surprising contrast or big question.
            Build a narrative, NOT a list of disconnected market facts. Prioritize what's truly newsworthy.

            Return {CHAPTERS_MIN} to {CHAPTERS_MAX} tight chapters, each with title, summary, key_tickers and
            2-6 distinct talking_points grounded in the supplied facts. No target_words, no word-count goals.
            Make room for indexes, sector story, movers, compelling chart setups (if available), and key next-week catalysts.
            You may skip redundant deep dives. Do not make up a bull/bear argument just to fill a chapter.
            {' '.join(extras)}

            The HOOK must work in the FIRST 30 SECONDS, beginning with an unexpected verified contrast or sharp question,
            speaking directly to the viewer and teasing the real payoff. Avoid greetings, generic 'volatile week' statements,
            background explanations and week-ending dates before the hook.
            Also return: week_label, hook, news_ticker_text (3-4 short UPPERCASE items),
            youtube_title (max 70 characters, intriguing but truthful),
            description (a detailed, useful 150-230 word YouTube summary in conversational language, with concrete
            stories from the data, key tickers, questions to watch, and no disclaimer),
            tags (6-10 comma-separated relevant tags).
            Do not make claims that aren't supported by the context. No filler, no empty generalizations.

            {STYLE_GUIDE}

            VERIFIED MARKET DATA AND NEWS
            {ctx.text}
            """
    return ask_llm(prompt, config=_gen_config(types, WeeklyOutline, 0.9), parse=parse_outline)


def _chapter_prompt_core(outline: WeeklyOutline, plan: ChapterPlan, context: str) -> str:
    overview = "\n".join(f"  {i + 1}. {c.title}: {c.summary}" for i, c in enumerate(outline.chapters))
    return f"""
            EPISODE OVERVIEW ({outline.week_label})
        {overview}
            THIS CHAPTER: {plan.title}. {plan.summary}
            Key tickers: {', '.join(plan.key_tickers) or 'none'}
            Talking points: {' | '.join(plan.talking_points)}

            {LAYOUT_GUIDE}
            {STYLE_GUIDE}

            MARKET DATA AND NEWS (the only source of facts)
            {context}
            """


def write_chapter(outline: WeeklyOutline, index: int, context: str, previous_tail: str,
                  previous_layout: str = "") -> ChapterScript:
    from google.genai import types
    plan = outline.chapters[index]
    last = index == len(outline.chapters) - 1
    logger.info("Writer: chapter %d/%d '%s' (no word quota)", index + 1, len(outline.chapters), plan.title)
    roles = []
    if index == 0:
        roles.append(f"FIRST CHAPTER: put the gripping cold open IMMEDIATELY in the first segment's spoken script. "
                     f"Use this hook idea, rewording it to sound natural and direct: {outline.hook}")
    if last:
        roles.append("LAST CHAPTER: your final segment MUST be one short outro, with a punchy closing observation "
                     "and a comment or subscription invitation. NO spoken financial disclaimer.")
    prompt = f"""
            You are a charismatic finance YouTuber writing ONE chapter, not a news anchor filling airtime.
            Return only the needed segments, usually 2-5, but as few as ONE if the story is simple.
            NO exact segment count. NO minimum words. NO chapter length goal.
            Every sentence must advance the story with new information, a concrete takeaway, or a genuinely useful question.
            This chapter is allowed to be VERY short. Do not repeat prior points to make it longer.
            {' '.join(roles)}
            {('Prior chapter ended: ' + previous_tail) if previous_tail else ''}
            You MAY repeat the same layout on adjacent segments when covering different companies or substantive ideas.
            Do not repeat the same facts about the same stock just to fill extra screens.
            For any stock in the chart-setups chapter use ONLY its ONE trade_setup screen, never a second chart.
            For upcoming earnings use ONE earnings_calendar in the episode. Highlight 2-3 important weekdays and
            the top 1-2 companies on each, then explain what numbers or management comments viewers should watch.
            Cut routine transition screens, generic summaries, redundant closing advice, and hypothetical narratives.
            {_chapter_prompt_core(outline, plan, context)}
            """
    near_misses: list = []
    try:
        return ask_llm(prompt, config=_gen_config(types, ChapterScript, 0.85),
                       parse=lambda r: parse_chapter(r, plan, near_misses, previous_layout,
                                                     index == 0, context),
                       max_rounds=CHAPTER_MAX_ROUNDS)
    except LLMUnavailableError:
        if not near_misses:
            raise
        score, cs = min(near_misses, key=lambda x: x[0])
        logger.warning("Chapter '%s': accepting best available draft (%d editorial issues); final pass will repair it",
                       plan.title, score)
        return cs


def generate_master_timeline(ctx: WeeklyContext) -> MasterTimeline:
    outline = generate_outline(ctx)
    items: List[TimelineItem] = []
    tail = ""
    for ci, plan in enumerate(outline.chapters):
        prev_layout = ev(items[-1].segment.layout) if items else ""
        script = write_chapter(outline, ci, ctx.text, tail, prev_layout)
        for seg in script.segments:
            items.append(TimelineItem(chapter_index=ci, chapter_title=plan.title, segment=seg))
        tail = shorten(script.segments[-1].script, 160) if script.segments else tail
    return MasterTimeline(outline=outline, items=items,
                          earnings=[EarningsItem(**e) for e in ctx.earnings],
                          setups=[SetupItem(**s) for s in ctx.setups])


# =====================================================================
# 6. DATA PRE-FETCH  (before any video processing; zero-crash fallbacks)
#    + verified-number placeholders, topic checks, de-duplication and auto-inserted screens
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
        n = int(round(v))
        return f"{int_words(n)} dollars"
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


PLACEHOLDER_RE = re.compile(r"\[\[\s*([A-Za-z0-9.\-^=]{1,12})\s*:\s*(week|day|price|entry|stop|target)\s*\]\]", re.I)


def placeholder_tickers(seg: Segment) -> set:
    out = set()
    for text in _text_fields(seg):
        out.update(clean_ticker(m.group(1)) for m in PLACEHOLDER_RE.finditer(text))
    return out


def _text_fields(seg: Segment) -> List[str]:
    return [seg.script, seg.plain_text] + list(seg.bull_points) + list(seg.bear_points)


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


def resolve_outline_metadata(outline: WeeklyOutline,
                             lookup: Callable[[str, str], Optional[float]]) -> None:
    """Resolve the director's metadata too; YouTube descriptions must never expose [[...]].

            An unverified claim is dropped, not substituted with an invented number or 'n/a'.
            In particular, outline talking_points are NOT passed through the narration resolver
            until all their tickers have been fetched.
            """

    def verified(text: str, context: str) -> str:
        missing = [(m.group(1), m.group(2)) for m in PLACEHOLDER_RE.finditer(text)
                   if lookup(clean_ticker(m.group(1)), m.group(2).lower()) is None]
        if missing:
            logger.warning("Omitting unverified metadata text in %s: %s", context, missing)
            return ""
        return resolve_text(text, lookup, spoken=False)

    # Description is free-form, so drop just the unsupported sentences, not a whole paragraph.
    paragraphs = []
    for paragraph in outline.description.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
        good = [verified(sentence, "description") for sentence in sentences if sentence.strip()]
        paragraphs.append(" ".join(sentence for sentence in good if sentence))
    outline.description = "\n".join(paragraphs).strip()
    if not outline.description:
        outline.description = "Weekly market recap: " + "; ".join(c.title for c in outline.chapters) + "."
    for chapter in outline.chapters:
        chapter.summary = verified(chapter.summary, chapter.title + " summary")
        chapter.talking_points = [point for item in chapter.talking_points
                                  if (point := verified(item, chapter.title + " point"))]


def resolve_segment(seg: Segment, lookup: Callable[[str, str], Optional[float]]) -> None:
    seg.script = resolve_text(seg.script, lookup, True)
    seg.plain_text = resolve_text(seg.plain_text, lookup, False)
    seg.bull_points = [resolve_text(p, lookup, False) for p in seg.bull_points]
    seg.bear_points = [resolve_text(p, lookup, False) for p in seg.bear_points]


def scrub_offtopic(seg: Segment) -> None:
    """Chart / dashboard segments are about ONE company: sentences that talk about another company are removed."""
    focus = seg.chart_ticker if seg.layout == "pro_candlestick_chart" else seg.dash_ticker
    if not focus or focus.startswith("^"):
        return
    focus_name = COMPANY_NAMES.get(focus, "").lower()
    others = [n.lower() for t, n in COMPANY_NAMES.items()
              if t != focus and len(n) >= 4 and n not in AMBIGUOUS_NAMES and n.lower() != focus_name
              and n.lower() not in focus_name and focus_name not in n.lower()]
    sentences = split_sentences(seg.script)
    keep = [s for s in sentences if
            not any(re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", s.lower()) for n in others)]
    if keep and len(keep) < len(sentences) and len(" ".join(keep).split()) >= 15:
        logger.warning(
            f"'{seg.headline}': removed {len(sentences) - len(keep)} off-topic sentence(s) about other companies")
        seg.script = " ".join(keep)


def join_names(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# Editorial importance is a heuristic, not a claim about market cap or expected price impact.
EARNINGS_PRIORITY = {
    t: 100 - i * 4 for i, t in enumerate((
        "MSFT", "NVDA", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD",
        "XOM", "CVX", "LLY", "JPM", "NFLX", "V", "MA", "COIN", "INTC",
        "BA", "UPS", "CAT", "NOW", "MRK", "WMT", "COST"))
}


def auto_earnings_segment(earnings: List[EarningsItem]) -> Optional[Segment]:
    """One calendar, three important days max, no company roll call; the visual shows the full list."""
    by_date: Dict[str, List[EarningsItem]] = {}
    for e in earnings:
        by_date.setdefault(e.date, []).append(e)
    if not by_date:
        return None
    ranking = sorted(by_date, key=lambda d: (
        -max([EARNINGS_PRIORITY.get(x.ticker, 0) for x in by_date[d]], default=0),
        -len(by_date[d]), d))
    dates = sorted(ranking[:3])
    sentences = ["Next week's earnings? You don't need every ticker. Just circle these days."]
    for d in dates:
        rows = sorted(by_date[d], key=lambda e: -EARNINGS_PRIORITY.get(e.ticker, 0))
        highlighted = [e.name for e in rows if EARNINGS_PRIORITY.get(e.ticker, 0) > 0][:2]
        highlighted = highlighted or [e.name for e in rows[:1]]
        day = datetime.date.fromisoformat(d).strftime("%A")
        names = join_names(highlighted)
        sentences.append(f"{day}, watch {names}.")
    sentences.append(
        "The big question isn't just who beats estimates. It's what management says about the months ahead.")
    try:
        seg = Segment(layout="earnings_calendar", headline="EARNINGS THAT MATTER",
                      script=" ".join(sentences), accent="cyan")
        normalize_segment(seg)
        return seg
    except Exception as e:
        logger.warning("Could not build the focused earnings segment: %s", e)
        return None


def auto_setup_segment(s: SetupItem) -> Optional[Segment]:
    t = s.ticker
    script = (f"Here's {s.name}. It has a {s.pattern.lower()} on the daily chart. "
              f"The level to watch is [[{t}:entry]], with a stop near [[{t}:stop]] "
              f"and a target around [[{t}:target]]. "
              "What matters is whether the setup actually holds when price tests it.")
    try:
        seg = Segment(layout="trade_setup", headline=shorten(f"{s.name} SETUP".upper(), 22),
                      script=script, accent="cyan", setup_ticker=t)
        normalize_segment(seg)
        return seg
    except Exception as e:
        logger.warning("Could not build setup segment for %s: %s", t, e)
        return None


def insert_auto_segments(master: MasterTimeline) -> None:
    """Insert a missing verified setup/calendar only when the episode has none; never force padding."""
    items = master.items
    if not items:
        return
    layouts = {ev(it.segment.layout) for it in items}
    insertions = []
    # Do not force a chart setup into the earnings chapter just because the writer skipped it.
    # The editor chooses relevance over filling every available screen type.
    if master.earnings and "earnings_calendar" not in layouts:
        one = auto_earnings_segment(master.earnings)
        if one:
            insertions.append(one)
    if insertions:
        last = items[-1]
        new = [TimelineItem(chapter_index=last.chapter_index,
                            chapter_title=last.chapter_title, segment=s) for s in insertions]
        # Keep an existing outro at the end.
        master.items = items[:-1] + new + items[-1:] if last.segment.layout == "outro" else items + new


def _strip_spoken_disclaimer(text: str) -> str:
    return " ".join(s for s in split_sentences(text) if not DISCLAIMER_RE.search(s)).strip()


def enforce_editorial_rules(master: MasterTimeline) -> None:
    """Deterministic editor for new OR --load-script JSON, including across chapter boundaries.

            Remove duplicated narration/stock deep-dives and collapse the entire earnings rundown
            into ONE real calendar. Repair remaining adjacent layouts without touching the renderer.
            """
    first_calendar = False
    seen_setup_tickers = set()
    movers_by_chapter: Dict[int, set] = {}
    for it in master.items:
        s = it.segment
        if s.layout == "top_movers_podium":
            movers_by_chapter.setdefault(it.chapter_index, set()).update(
                m.ticker for m in s.movers_gainers + s.movers_losers)
    setup_by_chapter: Dict[int, set] = {}
    for it in master.items:
        s = it.segment
        if s.layout == "trade_setup" and s.setup_ticker:
            setup_by_chapter.setdefault(it.chapter_index, set()).add(s.setup_ticker)

    staged: List[TimelineItem] = []
    for it in master.items:
        s = it.segment
        s.script = _strip_spoken_disclaimer(s.script)
        if not s.script:
            logger.info("Editor removed disclaimer-only segment: %s", s.headline)
            continue
        if s.layout == "quote_highlight" and (
                not s.quote_source.strip() or
                re.search(r"\b(?:traders should monitor|investors should watch|market participants)\b",
                          s.quote_text, re.I)):
            logger.info("Editor removed unattributed or generic quotation: %s", s.headline)
            continue
        if s.layout == "earnings_calendar":
            if first_calendar or not master.earnings:
                logger.info("Editor removed redundant earnings-calendar screen: %s", s.headline)
                continue
            first_calendar = True
            s = auto_earnings_segment(master.earnings) or s
            it.segment = s
        if s.layout == "trade_setup":
            if not s.setup_ticker or s.setup_ticker in seen_setup_tickers:
                logger.info("Editor removed repeated trade setup for %s", s.setup_ticker)
                continue
            seen_setup_tickers.add(s.setup_ticker)
        if s.layout == "stock_dashboard" and s.dash_ticker in movers_by_chapter.get(it.chapter_index, set()):
            # Keep a genuine second angle (earnings/valuation); drop a second telling of the same weekly move.
            if not re.search(r"\b(earnings|revenue|guidance|margin|market cap|valuation|p e ratio|dividend)\b",
                             s.script, re.I):
                logger.info("Editor cut repeat dashboard for mover %s", s.dash_ticker)
                continue
        if s.layout in ("pro_candlestick_chart", "stock_dashboard"):
            ticker = s.chart_ticker if s.layout == "pro_candlestick_chart" else s.dash_ticker
            if ticker and ticker in setup_by_chapter.get(it.chapter_index, set()):
                logger.info("Editor removed second analysis screen for setup ticker %s", ticker)
                continue
        if any(_duplicated_story(s, old.segment) for old in staged):
            logger.info("Editor removed duplicated narration screen: %s", s.headline)
            continue
        staged.append(it)

    if not staged:
        raise ValueError("No content left after editorial cleanup")

    # The final outro is written intentionally; never relabel an earnings screen to 'outro'.
    if staged[-1].segment.layout != "outro":
        last = staged[-1]
        outro = Segment(layout="outro", headline="THE BOTTOM LINE", accent="gold",
                        script=("So what would change your mind about this market next week? "
                                "Tell me the one stock or story you're watching in the comments. "
                                "And subscribe for the next recap."))
        staged.append(TimelineItem(chapter_index=last.chapter_index,
                                   chapter_title=last.chapter_title, segment=outro))

    # A repeated visual template is fine when its content differs. Don't collapse two
    # PayPal/Adobe setup charts or two stock dashboards just because they're adjacent.
    master.items = staged
    logger.info("Editorial cleanup retained %d screen(s), unique trade setups per ticker",
                len(master.items))


def calendar_columns(event_dates: List[str]) -> List[str]:
    """Five weekday columns for the earnings table (Monday-Friday of the first reporting week when possible)."""
    ds = sorted({datetime.date.fromisoformat(d) for d in event_dates})
    if not ds:
        return []
    monday = ds[0] - datetime.timedelta(days=ds[0].weekday())
    week = [monday + datetime.timedelta(days=i) for i in range(5)]
    if all(d in week for d in ds):
        return [d.isoformat() for d in week]
    cols = ds[:5]
    d = cols[-1]
    while len(cols) < 5:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5 and d not in cols:
            cols.append(d)
    return [c.isoformat() for c in sorted(cols)]


def prefetch_market_data(master: MasterTimeline) -> MarketCache:
    """Collect every (ticker, timeframe) the timeline asks for, fetch once, ground the numbers, repair the segments."""
    cache = MarketCache()

    # --- structural repairs that need no network ---
    if not master.earnings:
        for it in list(master.items):
            if ev(it.segment.layout) == "earnings_calendar":
                logger.info("No earnings found: dropping the earnings_calendar segment")
                master.items.remove(it)
    insert_auto_segments(master)
    enforce_editorial_rules(master)
    segs = [it.segment for it in master.items]
    for seg in segs:  # an index/ETF has no P/E or market cap: show a chart instead of a grid of "N/A"
        if seg.layout == "stock_dashboard" and seg.dash_ticker.startswith("^"):
            logger.info(f"Dashboard for index {seg.dash_ticker} -> pro_candlestick_chart")
            seg.layout, seg.chart_ticker, seg.chart_timeframe = "pro_candlestick_chart", seg.dash_ticker, "1wk"

    setups = {s.ticker: s for s in master.setups}
    candle_req = sorted({(s.chart_ticker, s.chart_timeframe) for s in segs if s.layout == "pro_candlestick_chart"})
    info_req = sorted({s.dash_ticker for s in segs if s.layout == "stock_dashboard"})
    setup_req = sorted({s.setup_ticker for s in segs if s.layout == "trade_setup" and s.setup_ticker in setups})
    week_req = set(info_req)
    for s in segs:
        week_req.update(placeholder_tickers(s))
        if s.layout == "top_movers_podium":
            week_req.update(m.ticker for m in s.movers_gainers + s.movers_losers)
        elif s.layout == "portfolio_10k_sim":
            week_req.update(s.sim_tickers)
        elif s.layout == "sector_heatmap":
            week_req.update(e for e in (sector_etf(x.name) for x in s.sectors) if e)
    outline_texts = [master.outline.description]
    for chapter in master.outline.chapters:
        outline_texts.extend([chapter.summary, *chapter.talking_points])
    for metadata_text in outline_texts:
        for match in PLACEHOLDER_RE.finditer(metadata_text):
            ticker, metric = clean_ticker(match.group(1)), match.group(2).lower()
            if metric in ("week", "day", "price"):
                week_req.add(ticker)
    logger.info(
        f"Prefetching {len(candle_req)} chart(s), {len(info_req)} dashboard(s), {len(setup_req)} setup chart(s), "
        f"{len(week_req)} weekly series")

    for key, data in zip(candle_req, parallel_map(lambda k: fetch_candles(*k), candle_req)):
        if data:
            cache.candles[key] = data
    for t, w in zip(sorted(week_req), parallel_map(fetch_week_perf, sorted(week_req))):
        if w:
            cache.weeks[t] = w
    for t, info in zip(info_req, parallel_map(fetch_stock_info, info_req)):
        if info:
            cache.info[t] = info
    for t, d in zip(setup_req, parallel_map(setup_chart_data, setup_req)):
        if d:
            cache.setup_charts[t] = d
    logo_tickers = sorted(set(info_req) | {e.ticker for e in master.earnings})
    for t, path in zip(logo_tickers, parallel_map(ensure_logo, logo_tickers)):
        cache.logos[t] = path

    # --- ground every data-driven screen on the verified numbers ---
    for seg in segs:
        L = seg.layout
        if L == "pro_candlestick_chart":
            if (seg.chart_ticker, seg.chart_timeframe) not in cache.candles:
                downgrade_to_plain(seg, f"no price data for {seg.chart_ticker}",
                                   f"We could not load the chart for {COMPANY_NAMES.get(seg.chart_ticker, seg.chart_ticker)} right now, so here is the quick summary.")
        elif L == "stock_dashboard":
            if seg.dash_ticker not in cache.info:
                downgrade_to_plain(seg, f"no data for {seg.dash_ticker}",
                                   f"We could not load the numbers for {COMPANY_NAMES.get(seg.dash_ticker, seg.dash_ticker)} right now, so here is the quick summary.")
        elif L == "top_movers_podium":
            side = seg.movers_side
            pool = {m.ticker: cache.weeks[m.ticker]["pct"] for m in seg.movers_gainers + seg.movers_losers
                    if m.ticker in cache.weeks}
            gains = sorted(((t, p) for t, p in pool.items() if p > 0), key=lambda x: -x[1])
            losses = sorted(((t, p) for t, p in pool.items() if p < 0), key=lambda x: x[1])
            need_g, need_l = side in ("both", "gainers"), side in ("both", "losers")
            floor = 1 if side == "both" else 2
            if (need_g and len(gains) < floor) or (need_l and len(losses) < floor):
                downgrade_to_plain(seg, "could not verify the movers",
                                   "We could not verify this week's movers, so here is the quick summary.")
            else:  # replace the LLM's numbers with the real ones
                cap = 3 if side == "both" else 5
                seg.movers_gainers = [MoverItem(ticker=t, change_pct=round(p, 2)) for t, p in
                                      gains[:cap]] if need_g else []
                seg.movers_losers = [MoverItem(ticker=t, change_pct=round(p, 2)) for t, p in
                                     losses[:cap]] if need_l else []
        elif L == "sector_heatmap":
            matched = 0
            for sec in seg.sectors:
                etf = sector_etf(sec.name)
                if etf and etf in cache.weeks:
                    sec.change_pct = round(cache.weeks[etf]["pct"], 2)
                    matched += 1
                if etf and sec.weight == 5.0:
                    sec.weight = SECTOR_WEIGHTS.get(etf, 5.0)
            if not matched:
                logger.warning("Sector ETFs unavailable: keeping the writer's sector numbers")
        elif L == "portfolio_10k_sim":
            seg.sim_tickers = [t for t in seg.sim_tickers if t in cache.weeks]
            if not seg.sim_tickers:
                downgrade_to_plain(seg, "no price data for the simulation",
                                   "We could not run the ten thousand dollar simulation right now, so here is the quick summary.")
        elif L == "trade_setup":
            if seg.setup_ticker not in setups or seg.setup_ticker not in cache.setup_charts:
                downgrade_to_plain(seg, f"no verified setup for {seg.setup_ticker}")

    # --- reject only the SAME movers selection; different tickers deserve another screen ---
    seen_movers_views = set()
    for seg in segs:
        if seg.layout != "top_movers_podium":
            continue
        key = (ev(seg.movers_side),
               tuple(sorted(m.ticker for m in seg.movers_gainers)),
               tuple(sorted(m.ticker for m in seg.movers_losers)))
        if key in seen_movers_views:
            downgrade_to_plain(seg, "exact movers selection was already shown")
        else:
            seen_movers_views.add(key)

    # --- verified numbers into the text, then keep single-company screens on topic ---
    def lookup(t: str, kind: str) -> Optional[float]:
        if kind in ("entry", "stop", "target"):
            s = setups.get(t)
            return float(getattr(s, kind)) if s else None
        w = cache.weeks.get(t)
        return None if not w else {"week": w["pct"], "day": w["day_pct"], "price": w["last"]}[kind]

    for seg in segs:
        resolve_segment(seg, lookup)
        if seg.layout in ("pro_candlestick_chart", "stock_dashboard"):
            scrub_offtopic(seg)
    resolve_outline_metadata(master.outline, lookup)
    # A failed data fetch can downgrade screens to plain_text after initial editing.
    # Run the editor again to keep layout diversity even in these fallback conditions.
    enforce_editorial_rules(master)
    return cache


def data_for_segment(seg: Segment, cache: MarketCache, master: MasterTimeline) -> Optional[dict]:
    L = seg.layout
    if L == "pro_candlestick_chart":
        return cache.candles.get((seg.chart_ticker, seg.chart_timeframe))
    if L == "stock_dashboard":
        info = dict(cache.info.get(seg.dash_ticker) or {})
        info["logo"] = cache.logos.get(seg.dash_ticker)
        return info
    if L == "portfolio_10k_sim":
        return {"rows": [cache.weeks[t] for t in seg.sim_tickers if t in cache.weeks]}
    if L == "trade_setup":
        setup = next((s for s in master.setups if s.ticker == seg.setup_ticker), None)
        return {"chart": cache.setup_charts.get(seg.setup_ticker), "setup": setup.model_dump() if setup else None}
    if L == "earnings_calendar":
        items = [e.model_dump() for e in master.earnings]
        return {"items": items, "columns": calendar_columns([e["date"] for e in items]),
                "logos": {e["ticker"]: cache.logos.get(e["ticker"]) for e in items}}
    return None


# =====================================================================
# 7. TEXT-TO-SPEECH (Kokoro-82M ONNX, isolated worker, Host + Quote voices, sentence-level timings)
# =====================================================================
KOKORO_DIR = Path(__file__).resolve().parent / "kokoro_assets"
_KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_NAME = "kokoro-v1.0.int8.onnx"
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
    tmp.replace(dest)


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
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.replace("$", "").replace("%", " percent")
    text = re.sub(r"\s+[-\u2013\u2014]+\s+|[\u2013\u2014]", ", ", text).replace(";", ",").replace("...", ".")
    text = re.sub(r"\^([A-Z]+)", r"\1", text)
    for pattern, replacement in TTS_SYMBOL_NAMES.items():
        text = re.sub(pattern, replacement, text)
    return text.replace("\n", " ").strip()


def voice_for(seg: Segment) -> Tuple[str, float]:
    """quote_highlight is read by the deeper Quote voice, everything else by the Host."""
    return (QUOTE_VOICE, QUOTE_SPEED) if seg.layout == "quote_highlight" else (HOST_VOICE, HOST_SPEED)


def generate_voiceovers(items: List[dict]) -> List[dict]:
    """items = [{text, voice, speed, path}] -> [{duration, sentences: [[start, end], ...]}]. Every sentence is
            synthesised on its own and joined with a short pause, so each screen knows when each sentence is spoken.
            Runs inside the isolated TTS worker."""
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
    in_file, out_file = work_dir / "tts_in.json", work_dir / "tts_out.json"
    payload = json.dumps(items)
    if (in_file.exists() and out_file.exists() and in_file.read_text(encoding="utf-8") == payload
            and all(Path(i["path"]).exists() for i in items)):
        logger.info("Reusing the voiceover from the previous run")
        return json.loads(out_file.read_text(encoding="utf-8"))
    in_file.write_text(payload, encoding="utf-8")
    out_file.unlink(missing_ok=True)
    run_memory_worker("tts", str(in_file), str(out_file))
    return json.loads(out_file.read_text(encoding="utf-8"))


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

    def swoosh(d=0.9):
        t = tt(d)
        a = 0.03 + 0.55 * np.sin(np.pi * t / d) ** 2  # cut-off sweeps up then down
        noise, y, acc = rng.standard_normal(len(t)), np.zeros(len(t)), 0.0
        for i in range(len(t)):
            acc += a[i] * (noise[i] - acc)
            y[i] = acc
        return norm(y * np.sin(np.pi * t / d) ** 2)

    def boom(d=1.4):
        t = tt(d)
        phase = 2 * np.pi * np.cumsum(38 + 70 * np.exp(-t * 5)) / sr
        thump = np.sin(phase) * np.exp(-t * 3.2)
        grit = np.convolve(rng.standard_normal(len(t)), np.ones(40) / 40, mode="same") * np.exp(-t * 9)
        return norm(thump + 0.6 * grit)

    def bell(f, d=0.9):
        t = tt(d)
        return (np.sin(2 * np.pi * f * t) * np.exp(-t * 6) + 0.5 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 9))

    def place(buf, clip, at):
        i = int(at * sr)
        n = max(0, min(len(clip), len(buf) - i))
        buf[i:i + n] += clip[:n]

    if name == "swoosh":
        return swoosh()
    if name == "swoosh_boom":
        buf = np.zeros(int(2.0 * sr))
        place(buf, swoosh(0.9), 0.0)
        place(buf, boom(1.5), 0.45)
        return norm(buf, 0.9)
    if name == "cash_register":
        buf = np.zeros(int(1.3 * sr))
        click = np.convolve(rng.standard_normal(int(0.03 * sr)), np.ones(6) / 6, mode="same")
        place(buf, click * np.linspace(1, 0, len(click)), 0.0)
        place(buf, bell(1568.0) * 0.8, 0.10)
        place(buf, bell(2093.0), 0.22)
        return norm(buf)
    if name == "buzzer":
        t = tt(0.6)
        sq = np.sign(np.sin(2 * np.pi * 150 * t)) * 0.6 + np.sign(np.sin(2 * np.pi * 156 * t)) * 0.4
        env = np.minimum(1, t / 0.01) * np.minimum(1, (0.6 - t) / 0.05)
        return norm(np.tanh(1.8 * sq) * env)
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
        n = 46
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
    def __init__(self, fig, raw_text: str, label: str = "WEEKLY RECAP", scroll_speed: float = 0.07, fontsize: int = 18):
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
        self.period = max((w2 - w1) / ax.bbox.width, 1e-3)
        self._text.set_text(unit * (int(math.ceil(1.0 / self.period)) + 2))

    def update(self, global_t: float) -> None:
        self._text.set_x(0.02 - (global_t * self.speed) % self.period)


# ---------------------------------------------------------------------
# Avatar. The mouth-state cache is CLASS level so it can be cleared (and RAM reclaimed) between chapters.
# ---------------------------------------------------------------------
class AvatarPanel:
    _cache: Dict[Tuple[int, bool, int], np.ndarray] = {}

    def __init__(self, size: int):
        self.size = size
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
                d.line((self.size * 0.40, self.size * 0.62, self.size * 0.60, self.size * 0.62), fill=(11, 14, 20, 255),
                       width=max(2, self.size // 40))
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
    except TypeError:
        return ImageFont.load_default()


@lru_cache(maxsize=64)
def load_serif(size: int):
    """Newspaper-style bold serif; falls back to the regular bold font."""
    for p in (r"C:\Windows\Fonts\georgiab.ttf", r"C:\Windows\Fonts\timesbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
              "/Library/Fonts/Georgia Bold.ttf", "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return load_font(size)


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


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    while len(text) > 1 and draw.textlength(text + "...", font=font) > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


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


# =====================================================================
# 11. LAYOUTS ("screens")
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
            sector / day / level can flash exactly while the voice talks about it."""

    def __init__(self, sentences: List[str], times: List[Tuple[float, float]], patterns: List[Tuple[str, Any]],
                 default_key: Any = None, hold: float = 0.35, lead: float = 0.12):
        regs = [(re.compile(r"(?<![a-z0-9])" + r"\s+".join(re.escape(w) for w in p.split()) + r"(?![a-z0-9])"), key)
                for p, key in sorted(patterns, key=lambda x: -len(x[0]))]
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


def px_box(ref_box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    return tuple(px(v) for v in ref_box)


# ---------------------------------------------------------------------
# Candle pane (shared by the pro chart and the trade-setup chart)
# ---------------------------------------------------------------------
class CandlePane:
    """Candles + volume + moving averages + live price tag. Everything is drawn once; update(k) reveals k candles."""

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
            ax.text(0.0125 * len(title) + 0.04 + 0.075 * i, 1.03, name, transform=ax.transAxes, fontsize=24 * REF_PT,
                    fontweight="bold", color=color, ha="left", va="bottom", clip_on=False)
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
        done = k >= n
        self.tag.set_visible(done)  # the price label box only exists once the chart is built
        self.tag.set_text(mpl_safe(f" {self.cur}{last:,.2f} "))
        self.tag.set_y(min(max(last, self.y_lo), self.y_hi))
        self.tag.get_bbox_patch().set_facecolor(C_UP if last >= self.closes[0] else C_DOWN)


# --- 1. PRO CANDLESTICK CHART ------------------------------------------
class ProCandlestickLayout(LayoutBase):
    """TradingView-style chart. The whole chart builds in half a second; annotations pop up right after."""
    NOTE_COLORS = {"high": C_GREEN, "low": C_RED, "biggest_drop": C_RED, "start": C_CYAN, "end": C_GOLD}
    BUILD_START = 0.08

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
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
        label = {"1d": "1D", "1wk": "1W", "1mo": "1M", "1y": "1Y"}.get(data["timeframe"], data["timeframe"].upper())
        cur = "" if data["ticker"].startswith("^") else "$"
        self.pane = CandlePane(fig, data, (40, 268, 1340, 500), (40, 780, 1340, 200),
                               [(f"MA{fast}", C_AMBER, sma(c, fast)), (f"MA{slow}", C_CYAN, sma(c, slow))],
                               f"{data['ticker']} \u00b7 {data['interval']}", cur)
        ax, pane = self.pane.ax, self.pane
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
            tx = min(max(idx + side * 0.2 * span, -1 + 0.16 * span), ax.get_xlim()[1] - 0.16 * span)
            ty = y + (0.16 if anchor in ("high", "start", "end") else -0.16) * (pane.y_hi - pane.y_lo)
            ty = min(max(ty, pane.y_lo + 0.07 * (pane.y_hi - pane.y_lo)), pane.y_hi - 0.10 * (pane.y_hi - pane.y_lo))
            ann = ax.annotate(mpl_safe("\n".join(_wrap_text(note["text"], ref_font(22), 280))), xy=(idx, y),
                              xytext=(tx, ty), arrowprops=dict(arrowstyle="-|>", color=col, lw=3), fontsize=22 * REF_PT,
                              fontweight="bold", color=col, ha="center", va="center", zorder=12, annotation_clip=False,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BG, edgecolor=col, alpha=0.9))
            ann.set_visible(False)  # a hidden annotation also hides its box and arrow
            self.notes.append(ann)
        ref = float(data.get("ref_close") or o[0])
        pct = (c[-1] / ref - 1) * 100
        self.header_sub = f"{data['ticker']}  |  {label} {data.get('ref_label', '')}"
        self.info_label = f"{data['ticker']} \u00b7 {label}" + (" (daily fallback)" if data.get("fallback") else "")
        self.header_right = (f"{cur}{c[-1]:,.2f}", f"{pct:+.2f}%", C_GREEN if pct >= 0 else C_RED)

    def update(self, t: float) -> None:
        n = self.pane.n
        k = min(n, max(2, int(math.ceil(ease_out(ramp(t, self.BUILD_START, CHART_BUILD_SECONDS)) * n))))
        self.pane.update(k, t)
        build_end = self.BUILD_START + CHART_BUILD_SECONDS
        for i, ann in enumerate(self.notes):
            t0 = build_end + 0.15 + 0.2 * i
            a = ease_out(ramp(t, t0, 0.25))
            ann.set_visible(t >= t0)
            ann.set_alpha(a)
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_alpha(a)
            if ann.get_bbox_patch() is not None:
                ann.get_bbox_patch().set_alpha(0.9 * a)


# --- 2. STOCK DASHBOARD ------------------------------------------------
DASH_ROWS = {1: [1], 2: [2], 3: [3], 4: [2, 2], 5: [3, 2], 6: [3, 3], 7: [4, 3]}
DASH_ALIASES = [
    ("pe_ratio", ["p e ratio", "p e", "pe ratio", "price to earnings", "earnings multiple", "multiple"]),
    ("market_cap", ["market capitalization", "market capitalisation", "market cap", "valuation", "trillion"]),
    ("dividend_yield", ["dividend", "yield", "payout"]),
    ("week_change", ["this week", "the week", "weekly", "week", "five sessions", "five days"]),
    ("change", ["today", "daily", "single day", "last session", "friday", "one day", "the day"]),
    ("price", ["share price", "price", "trading at", "closed at"]),
]


class StockDashboardLayout(LayoutBase):
    """Card grid for ONE company. Only cards that have data exist; the rest of the space is shared out evenly.
            The card the narration is talking about glows while it is mentioned, the others dim."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        d = data
        up = d["change_pct"] >= 0
        wk = d.get("week_pct")
        cards = [dict(key="ticker")]
        cards.append(dict(key="price", label="PRICE", text=f"${d['price']:,.2f}", color=C_WHITE))
        cards.append(dict(key="change", label="TODAY", text=f"{d['change_pct']:+.2f}%",
                          color=C_GREEN if up else C_RED))
        if wk is not None:
            cards.append(
                dict(key="week_change", label="THIS WEEK", text=f"{wk:+.2f}%", color=C_GREEN if wk >= 0 else C_RED))
        if d.get("market_cap"):
            cards.append(dict(key="market_cap", label="MARKET CAP", text=fmt_big(d["market_cap"]), color=C_GOLD))
        if d.get("pe"):
            cards.append(dict(key="pe_ratio", label="P/E RATIO", text=f"{d['pe']:.1f}", color=C_CYAN))
        if d.get("div_yield") is not None:
            cards.append(
                dict(key="dividend_yield", label="DIVIDEND YIELD", text=f"{d['div_yield']:.2f}%", color=C_GREEN))
        rows = DASH_ROWS[len(cards)]
        m, gap = 28.0, 24.0
        row_h = min(380.0, (H - 2 * m - gap * (len(rows) - 1)) / len(rows))
        y0 = (H - (row_h * len(rows) + gap * (len(rows) - 1))) / 2
        self.cards, self.glows, self.logo = [], [], None
        slot = 0
        for ri, cnt in enumerate(rows):
            cw = (W - 2 * m - gap * (cnt - 1)) / cnt
            y = y0 + ri * (row_h + gap)
            for ci in range(cnt):
                x = m + ci * (cw + gap)
                spec = cards[slot]
                box = FancyBboxPatch((x, y), cw, row_h, boxstyle="round,pad=0,rounding_size=26", fc=C_PANEL,
                                     ec=C_BORDER, lw=2.5)
                ax.add_patch(box)
                grp = Reveal(0.04 * slot, 0.25).add(box)
                if spec["key"] == "ticker":
                    self._ticker_card(ax, grp, x, y, cw, row_h, d)
                else:
                    grp.add(ax.text(x + 30, y + 46, spec["label"], fontsize=30 * REF_PT, fontweight="bold",
                                    color=C_MUTED, ha="left", va="center"))
                    size = fit_one(spec["text"], cw - 70, row_h * 0.42, 130, 28)
                    grp.add(ax.text(x + cw / 2, y + row_h * 0.60, mpl_safe(spec["text"]), fontsize=size * REF_PT,
                                    fontweight="bold", color=spec["color"], ha="center", va="center"))
                    glow = []
                    for g, lw, a_ in ((6, 6.0, 1.0), (14, 3.6, 0.65), (24, 2.0, 0.4)):
                        gp = FancyBboxPatch((x - g, y - g), cw + 2 * g, row_h + 2 * g,
                                            boxstyle=f"round,pad=0,rounding_size={26 + g}", fc="none",
                                            ec=spec["color"] if spec["key"] != "price" else self.accent, lw=lw,
                                            alpha=0.0)
                        ax.add_patch(gp)
                        glow.append((gp, a_))
                    self.glows.append((spec["key"], glow))
                self.cards.append((spec["key"], grp))
                slot += 1
        present = [c["key"] for c in cards if c["key"] != "ticker"]
        patterns = [(p, key) for key, plist in DASH_ALIASES if key in present for p in plist]
        fallback = seg.get("highlight_metric") if seg.get("highlight_metric") in present else None
        self.mentions = self.track(patterns, fallback)
        cur_col = C_GREEN if up else C_RED
        self.header_sub = f"{d['ticker']}  |  SNAPSHOT"
        self.info_label = d["name"].upper()
        self.header_right = (f"${d['price']:,.2f}", f"{d['change_pct']:+.2f}% today", cur_col)

    def _ticker_card(self, ax, grp: Reveal, x, y, cw, ch, d) -> None:
        cx, cy, r = x + cw / 2, y + ch * 0.36, min(80.0, ch * 0.22)
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
            grp.add(disc, ax.text(cx, cy, mpl_safe(ticker[:3]), fontsize=r * 0.8 * REF_PT, fontweight="bold",
                                  color=C_BG, ha="center", va="center"))
        size = fit_one(ticker, cw - 50, ch * 0.2, 84, 26)
        grp.add(ax.text(cx, y + ch * 0.72, mpl_safe(ticker), fontsize=size * REF_PT, fontweight="bold", color=C_WHITE,
                        ha="center", va="center"))
        size2 = fit_one(d["name"], cw - 50, ch * 0.12, 32, 14)
        grp.add(ax.text(cx, y + ch * 0.89, mpl_safe(d["name"]), fontsize=size2 * REF_PT, fontweight="bold",
                        color=C_MUTED, ha="center", va="center"))

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        for ck, grp in self.cards:
            grp.apply(t, 1.0 if key is None or ck == key else 1 - 0.62 * s)
        pulse = 0.7 + 0.3 * math.sin(2 * math.pi * t * 1.2)
        for gk, glow in self.glows:
            on = s * pulse if gk == key else 0.0
            for gp, a_ in glow:
                gp.set_alpha(on * a_)

    def paint(self, frame: np.ndarray, t: float) -> None:
        if self.logo is not None:
            arr, x, y, grp = self.logo
            key, s = self.mentions.active(t)
            blend_rgba(frame, arr, x, y, ease_out(ramp(t, grp.start, grp.length)) * (1 - 0.62 * s if key else 1.0))


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
    """Blocks pop in within a fraction of a second. The sector the narration names flashes (gold outline) while the
            others dim. All text is fitted to its block."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        items = sorted(seg["sectors"], key=lambda s: -float(s["weight"]))[:5]
        wmax = max(float(s["weight"]) for s in items)
        weights = [max(float(s["weight"]), wmax * 0.16) for s in items]  # tiny sectors stay readable
        rects = treemap_rects(weights, 18, 18, W - 36, H - 36)
        maxabs = max(2.0, max(abs(float(s["change_pct"])) for s in items))
        self.blocks, patterns = [], []
        for i, (s, (x, y, w, h)) in enumerate(zip(items, rects)):
            chg = float(s["change_pct"])
            col = C_GREEN if chg >= 0 else C_RED
            strength = min(1.0, abs(chg) / maxabs)
            g = 5
            bx, by, bw, bh = x + g, y + g, w - 2 * g, h - 2 * g
            box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=18",
                                 fc=blend_hex(col, C_BG, 0.80 - 0.42 * strength), ec=col, lw=3)
            ax.add_patch(box)
            name_lines, name_size = fit_wrap(s["name"].upper(), bw - 50, bh * 0.38, 56, 12)
            pct_size = fit_one(f"{chg:+.2f}%", bw - 50, bh * 0.28, 110, 12)
            name_y = by + bh * 0.30
            name_arts = draw_lines(ax, name_lines, bx + bw / 2, name_y, name_size, C_WHITE)
            pct = ax.text(bx + bw / 2, by + bh * 0.68, "", fontsize=pct_size * REF_PT, fontweight="bold", color=col,
                          ha="center", va="center")
            flash = FancyBboxPatch((bx + 2, by + 2), bw - 4, bh - 4, boxstyle="round,pad=0,rounding_size=18",
                                   fc="none", ec=C_GOLD, lw=8, alpha=0.0)
            ax.add_patch(flash)
            grp = Reveal(0.05 * i, 0.25).add(box, name_arts, pct)
            self.blocks.append(dict(grp=grp, pct=pct, chg=chg, flash=flash, t0=0.05 * i))
            etf = sector_etf(s["name"])
            for alias in ETF_ALIASES.get(etf, []) + [s["name"].lower()]:
                patterns.append((alias, i))
        self.mentions = self.track(patterns)
        top = max(items, key=lambda s: abs(float(s["change_pct"])))
        self.header_sub = "S&P 500 SECTORS  |  WEEKLY"
        self.info_label = "BIGGEST SECTOR MOVE"
        self.header_right = (shorten(top["name"].upper(), 22), f"{float(top['change_pct']):+.2f}%",
                             C_GREEN if float(top["change_pct"]) >= 0 else C_RED)

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        pulse = 0.75 + 0.25 * math.sin(2 * math.pi * t * 1.4)
        for i, b in enumerate(self.blocks):
            b["grp"].apply(t, 1.0 if key is None or i == key else 1 - 0.55 * s)
            p = ease_out(ramp(t, b["t0"], 0.5))
            b["pct"].set_text(f"{b['chg'] * p:+.2f}%")
            b["flash"].set_alpha(s * pulse if i == key else 0.0)


# --- 4. TOP MOVERS (split podium, or a single-side ranking) -------------
class TopMoversPodiumLayout(LayoutBase):
    """movers_side = both: split screen (green bars rising on the left, red bars hanging on the right).
            movers_side = gainers / losers: a ranking of that side only, so the same chart is never shown twice."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        side = seg.get("movers_side", "both")
        gain = [(m["ticker"], float(m["change_pct"])) for m in seg["movers_gainers"]]
        loss = [(m["ticker"], float(m["change_pct"])) for m in seg["movers_losers"]]
        self.bars, self.titles = [], Reveal(0.0, 0.25)
        if side == "both":
            self._split(ax, W, H, gain[:3], loss[:3])
        else:
            self._board(ax, W, H, gain if side == "gainers" else loss, side == "gainers")
        patterns = []
        for i, b in enumerate(self.bars):
            tick = b["ticker"].lstrip("^").lower()
            patterns.append((tick, i))
            name = COMPANY_NAMES.get(b["ticker"])
            if name:
                patterns.append((name.lower(), i))
        self.mentions = self.track(patterns)
        allm = gain + loss
        best = max(allm, key=lambda x: abs(x[1]))
        self.header_sub = "WEEKLY MOVERS" if side == "both" else ("TOP GAINERS" if side == "gainers" else "TOP LOSERS")
        self.info_label = "BIGGEST MOVE"
        self.header_right = (best[0].lstrip("^"), f"{best[1]:+.1f}%", C_GREEN if best[1] >= 0 else C_RED)

    def _split(self, ax, W, H, gain, loss) -> None:
        vmax = max([abs(p) for _, p in gain + loss] + [0.5])
        max_h, floor, ceil_ = 470.0, 740.0, 165.0
        ax.plot([W / 2, W / 2], [28, H - 28], color=C_BORDER, linewidth=3)
        ax.plot([40, W / 2 - 40], [floor, floor], color=C_GREEN, linewidth=3, alpha=0.7)
        ax.plot([W / 2 + 40, W - 40], [ceil_, ceil_], color=C_RED, linewidth=3, alpha=0.7)
        self.titles.add(ax.text(W / 4, 58, "TOP GAINERS", fontsize=50 * REF_PT, fontweight="bold", color=C_GREEN,
                                ha="center", va="center"),
                        ax.text(3 * W / 4, 58, "TOP LOSERS", fontsize=50 * REF_PT, fontweight="bold", color=C_RED,
                                ha="center", va="center"))
        for entries, x_left, color, rising in ((gain, 40.0, C_GREEN, True), (loss, W / 2 + 40.0, C_RED, False)):
            k = max(len(entries), 1)
            slot = (W / 2 - 80) / k
            bw = min(slot * 0.62, 200.0)
            for i, (tick, pct) in enumerate(entries):
                cx = x_left + slot * (i + 0.5)
                h = max(44.0, max_h * abs(pct) / vmax)
                rect = Rectangle((cx - bw / 2, floor if rising else ceil_), bw, 0, facecolor=color, edgecolor="none")
                ax.add_patch(rect)
                lab = ax.text(cx, floor + 44 if rising else ceil_ - 38, mpl_safe(tick.lstrip("^")),
                              fontsize=fit_one(tick.lstrip("^"), slot - 20, 56, 54, 18) * REF_PT, fontweight="bold",
                              color=C_WHITE, ha="center", va="center")
                val = ax.text(cx, floor, "", fontsize=fit_one("+00.0%", slot - 10, 60, 58, 16) * REF_PT,
                              fontweight="bold", color=color, ha="center", va="center", zorder=6)
                t0 = 0.1 + 0.1 * i + (0.0 if rising else 0.1)
                self.bars.append(dict(kind="col", ticker=tick, rect=rect, val=val, lab=Reveal(t0, 0.25).add(lab),
                                      cx=cx, h=h, pct=pct, rising=rising, t0=t0, floor=floor, ceil=ceil_))

    def _board(self, ax, W, H, entries, gainers: bool) -> None:
        color = C_GREEN if gainers else C_RED
        entries = entries[:5]
        vmax = max([abs(p) for _, p in entries] + [0.5])
        self.titles.add(ax.text(W / 2, 64, "TOP GAINERS THIS WEEK" if gainers else "TOP LOSERS THIS WEEK",
                                fontsize=56 * REF_PT, fontweight="bold", color=color, ha="center", va="center"))
        n = max(len(entries), 1)
        row_h = min(124.0, (H - 190.0) / n)
        bar_x0, bar_max = 430.0, W - 430.0 - 260.0
        for i, (tick, pct) in enumerate(entries):
            cy = 150 + row_h * (i + 0.5)
            lab = [ax.text(60, cy - row_h * 0.12, mpl_safe(tick.lstrip("^")),
                           fontsize=fit_one(tick.lstrip("^"), 330, row_h * 0.5, 64, 22) * REF_PT, fontweight="bold",
                           color=C_WHITE, ha="left", va="center")]
            name = COMPANY_NAMES.get(tick)
            if name:
                lab.append(ax.text(60, cy + row_h * 0.30, mpl_safe(name),
                                   fontsize=fit_one(name, 330, row_h * 0.25, 30, 14) * REF_PT,
                                   fontweight="bold", color=C_MUTED, ha="left", va="center"))
            rect = Rectangle((bar_x0, cy - row_h * 0.30), 0, row_h * 0.60, facecolor=color, edgecolor="none")
            ax.add_patch(rect)
            val = ax.text(bar_x0, cy, "", fontsize=fit_one("+00.0%", 240, row_h * 0.5, 60, 18) * REF_PT,
                          fontweight="bold", color=color, ha="left", va="center", zorder=6)
            t0 = 0.1 + 0.1 * i
            self.bars.append(dict(kind="row", ticker=tick, rect=rect, val=val, lab=Reveal(t0, 0.25).add(lab), cy=cy,
                                  w=max(30.0, bar_max * abs(pct) / vmax), pct=pct, t0=t0, x0=bar_x0))

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        self.titles.apply(t)
        for i, b in enumerate(self.bars):
            m = 1.0 if key is None or i == key else 1 - 0.55 * s
            p = ease_out(ramp(t, b["t0"], 0.5))
            if b["kind"] == "col":
                h = b["h"] * p
                if b["rising"]:
                    b["rect"].set_y(b["floor"] - h)
                    b["val"].set_position((b["cx"], b["floor"] - h - 34))
                else:
                    b["rect"].set_y(b["ceil"])
                    b["val"].set_position((b["cx"], b["ceil"] + h + 36))
                b["rect"].set_height(h)
            else:
                w = b["w"] * p
                b["rect"].set_width(w)
                b["val"].set_position((b["x0"] + w + 20, b["cy"]))
            b["val"].set_text(f"{b['pct'] * p:+.1f}%")
            b["val"].set_alpha(clamp01(p * 4) * m)
            b["rect"].set_alpha(m)
            b["lab"].apply(t, m)


# --- 5. MACRO TIMELINE -------------------------------------------------
class MacroTimelineLayout(LayoutBase):
    """Horizontal Mon-Fri line; 3-4 event nodes light up one by one as the narration reaches them."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        line_y, x0, x1 = 415.0, 90.0, 1350.0
        day_x = {d: 190.0 + i * 265.0 for i, d in enumerate(WEEKDAYS)}
        events = sorted(seg["macro_events"][:4], key=lambda e: WEEKDAYS.index(e["day"]) if e["day"] in WEEKDAYS else 0)
        xs = []
        for e in events:
            x = day_x.get(e["day"], 190.0)
            if xs and x < xs[-1] + 215:
                x = min(xs[-1] + 215, 1330.0)
            xs.append(x)
        for d, x in day_x.items():
            ax.plot([x, x], [70, 765], color=C_BORDER, linewidth=1.6, linestyle="--", alpha=0.5)
            ax.text(x, 800, d, fontsize=32 * REF_PT, fontweight="bold", color=C_MUTED, ha="center", va="center")
        ax.plot([x0, x1], [line_y, line_y], color=C_BORDER, linewidth=9, solid_capstyle="round")
        self.fill, = ax.plot([], [], color=self.accent, linewidth=9, solid_capstyle="round", zorder=3)
        self.x0, self.line_y, self.nodes = x0, line_y, []
        n, ns = len(events), len(self.sentence_times)
        for i, (e, x) in enumerate(zip(events, xs)):
            above = i % 2 == 0
            si = int(round(i * (ns - 1) / max(1, n - 1))) if n > 1 else 0
            t0 = self.sentence_times[min(si, ns - 1)][0] + 0.1
            cx = min(max(x, 215.0), 1225.0)
            cw_, chh = 400.0, 200.0
            cy_top = 95.0 if above else 535.0
            box = FancyBboxPatch((cx - cw_ / 2, cy_top), cw_, chh, boxstyle="round,pad=0,rounding_size=22", fc=C_PANEL,
                                 ec=self.accent, lw=3.5)
            ax.add_patch(box)
            t_lines, t_size = fit_wrap(e["title"].upper(), cw_ - 40, 78, 44, 18)
            title = draw_lines(ax, t_lines, cx, cy_top + 52, t_size, self.accent)
            d_lines, d_size = fit_wrap(e["detail"], cw_ - 40, 92, 30, 14, pitch=1.2) if e["detail"] else ([], 14)
            detail = draw_lines(ax, d_lines, cx, cy_top + 140, d_size, C_WHITE, pitch=1.2)
            c_y0, c_y1 = (cy_top + chh, line_y) if above else (line_y, cy_top)
            conn, = ax.plot([x, x], [c_y0, c_y1], color=self.accent, linewidth=3.5)
            halo, = ax.plot([x], [line_y], marker="o", markersize=84 * REF_PT, linestyle="none",
                            markerfacecolor=self.accent, markeredgewidth=0, alpha=0.0, zorder=4)
            dot, = ax.plot([x], [line_y], marker="o", markersize=44 * REF_PT, linestyle="none", markerfacecolor=C_BG,
                           markeredgecolor=C_BORDER, markeredgewidth=5, zorder=6)
            self.nodes.append(dict(x=x, t0=t0, dot=dot, halo=halo, card=Reveal(t0, 0.3).add(box, title, detail, conn)))
        self.header_sub = "MACRO CALENDAR  |  THIS WEEK"
        self.info_label = "EVENTS"
        self.header_right = (f"{n} KEY EVENTS", f"{events[0]['day']} - {events[-1]['day']}", C_MUTED)

    def update(self, t: float) -> None:
        x_now = self.x0
        for i, nd in enumerate(self.nodes):
            prev_x, prev_t = (self.x0, 0.0) if i == 0 else (self.nodes[i - 1]["x"], self.nodes[i - 1]["t0"])
            if t >= nd["t0"]:
                x_now = nd["x"]
            elif t > prev_t:
                x_now = lerp(prev_x, nd["x"], ease_in_out(ramp(t, prev_t, nd["t0"] - prev_t)))
            lit = t >= nd["t0"]
            nd["card"].apply(t)
            nd["dot"].set_markerfacecolor(self.accent if lit else C_BG)
            nd["dot"].set_markeredgecolor(C_WHITE if lit else C_BORDER)
            active = lit and (i == len(self.nodes) - 1 or t < self.nodes[i + 1]["t0"])
            nd["halo"].set_alpha(
                (0.25 + 0.2 * math.sin(2 * math.pi * t * 1.6)) * ramp(t, nd["t0"], 0.3) if active else 0.0)
        self.fill.set_data([self.x0, x_now], [self.line_y, self.line_y])


# --- 6. BULL VS BEAR ---------------------------------------------------
BULL_WORDS = ("surge", "rally", "beat", "record", "strong", "gain", "growth", "upside", "bullish", "soar", "jump",
              "rise", "optimis", "outperform", "positive", "win", "boom", "climb", "momentum", "recover", "profit")
BEAR_WORDS = ("fall", "drop", "plunge", "miss", "weak", "loss", "risk", "bearish", "concern", "warn", "decline",
              "slump",
              "overvalued", "downside", "cut", "fear", "worry", "crash", "sell", "debt", "slow", "inflation", "but",
              "however", "trouble", "pressure", "volatil")


def auto_stances(script: str) -> List[str]:
    out = []
    for s in split_sentences(script):
        low = s.lower()
        b, r = sum(w in low for w in BULL_WORDS), sum(w in low for w in BEAR_WORDS)
        out.append("bull" if b > r else "bear" if r > b else "neutral")
    return out


class BullVsBearLayout(LayoutBase):
    """Vertical split: green left, red right. The side being argued brightens while its sentence is spoken."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        stances = [s for s in seg.get("sentence_stances", []) if s in ("bull", "bear", "neutral")]
        self.stances = stances if len(stances) == len(self.sentences) else auto_stances(seg["script"])
        if len(self.stances) != len(self.sentences):
            self.stances = ["neutral"] * len(self.sentences)
        self.sides = []
        for side, (x, col, title, pts, cx) in enumerate((
                (24.0, C_GREEN, seg["bull_title"], seg["bull_points"], 369.0),
                (726.0, C_RED, seg["bear_title"], seg["bear_points"], 1071.0))):
            panel = Rectangle((x, 24), 690, H - 48, facecolor=blend_hex(col, C_BG, 0.9), edgecolor=col, linewidth=3)
            ax.add_patch(panel)
            head = ax.text(cx, 90, mpl_safe(title), fontsize=fit_one(title, 620, 80, 64, 28) * REF_PT,
                           fontweight="bold", color=col, ha="center", va="center")
            glyph = ax.text(cx, 185, "\u25B2" if side == 0 else "\u25BC", fontsize=70 * REF_PT, fontweight="bold",
                            color=col, ha="center", va="center")
            texts = [head, glyph]
            k = max(1, len(pts))
            slot = 520.0 / k
            for i, p in enumerate(pts):
                lines, size = fit_wrap(p, 600, slot - 36, 46, 18)
                texts += draw_lines(ax, lines, cx, 270 + slot * (i + 0.5), size, C_WHITE)
                if i:
                    texts.append(ax.plot([x + 70, x + 620], [270 + slot * i, 270 + slot * i], color=col, linewidth=2,
                                         alpha=0.5)[0])
            self.sides.append(dict(panel=panel, texts=texts, col=col))
        vs_dot, = ax.plot([W / 2], [H / 2], marker="o", markersize=84 * REF_PT, linestyle="none", markerfacecolor=C_BG,
                          markeredgecolor=C_GOLD, markeredgewidth=4, zorder=9)
        vs = ax.text(W / 2, H / 2, "VS", fontsize=40 * REF_PT, fontweight="bold", color=C_GOLD, ha="center",
                     va="center", zorder=10)
        self.vs = Reveal(0.0, 0.25).add(vs_dot, vs)
        self.header_sub = "BULL VS BEAR"
        self.info_label = "THE DEBATE"

    def _levels(self, t: float) -> Tuple[float, float]:
        target = {"bull": (1.0, 0.0), "bear": (0.0, 1.0), "neutral": (0.5, 0.5)}
        idx = None
        for i, (a, _b) in enumerate(self.sentence_times):
            if t >= a:
                idx = i
        if idx is None:
            return 0.5, 0.5
        cur = target[self.stances[idx]]
        prev = target[self.stances[idx - 1]] if idx > 0 else (0.5, 0.5)
        e = ease_out(ramp(t, self.sentence_times[idx][0], 0.3))
        return lerp(prev[0], cur[0], e), lerp(prev[1], cur[1], e)

    def update(self, t: float) -> None:
        appear = ease_out(ramp(t, 0.0, 0.3))
        self.vs.apply(t)
        for side, lvl in zip(self.sides, self._levels(t)):
            side["panel"].set_facecolor(blend_hex(side["col"], C_BG, 0.92 - 0.44 * lvl))
            side["panel"].set_linewidth(2 + 6 * lvl)
            side["panel"].set_alpha(appear)
            for art in side["texts"]:
                art.set_alpha(appear * (0.40 + 0.60 * lvl))


# --- 7. QUOTE HIGHLIGHT ------------------------------------------------
class QuoteHighlightLayout(LayoutBase):
    """Newspaper clipping. The sentence the voice reads is marked in yellow and the camera slowly zooms onto it.
            The pipeline uses the deeper Quote voice and hides the avatar for this layout."""
    S = 2  # the card is rendered at 2x the stage size, which is the zoom headroom

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        make_stage_axes(fig, STAGE_BOX, W, H)
        self.sx, self.sy, self.sw, self.sh = STAGE_BOX
        S = self.S
        cw, ch = self.sw * S, self.sh * S
        self.cw, self.ch = cw, ch
        quote = seg["quote_text"]
        sentences = split_sentences(quote) or [quote]
        hl_i = best_sentence_index(sentences, seg.get("quote_highlight", ""))
        words = [(tok, i) for i, s in enumerate(sentences) for tok in s.split()]
        margin, top_h, bot_h = int(cw * 0.07), int(ch * 0.19), int(ch * 0.13)
        avail_w, avail_h = cw - 2 * margin, ch - top_h - bot_h
        size, min_size = int(px(72) * S), max(14, int(px(28) * S))
        while True:
            font = load_serif(size)
            space = font.getlength(" ")
            lines, cur, x = [], [], 0.0
            for tok, si in words:
                w = font.getlength(tok)
                if cur and x + w > avail_w:
                    lines.append(cur)
                    cur, x = [], 0.0
                cur.append((tok, si, x, w))
                x += w + space
            if cur:
                lines.append(cur)
            line_h = int(_font_height(font) * 1.22)
            if len(lines) * line_h <= avail_h or size <= min_size:
                break
            size -= 2
        y0 = top_h + max(0, (avail_h - len(lines) * line_h) // 2)
        pad = int(line_h * 0.12)
        hl_rects = []
        for li, line in enumerate(lines):
            hl_words = [w for w in line if w[1] == hl_i]
            if hl_words:
                y = y0 + li * line_h
                hl_rects.append((margin + hl_words[0][2] - pad, y + int(line_h * 0.10),
                                 margin + hl_words[-1][2] + hl_words[-1][3] + pad, y + line_h - int(line_h * 0.02)))
        src = seg.get("quote_source", "")
        self.base = self._render(lines, font, margin, y0, line_h, hl_i, hl_rects, False, src, top_h, bot_h)
        self.focus = self._render(lines, font, margin, y0, line_h, hl_i, hl_rects, True, src, top_h, bot_h)
        if hl_rects:
            bx0, by0 = min(r[0] for r in hl_rects), min(r[1] for r in hl_rects)
            bx1, by1 = max(r[2] for r in hl_rects), max(r[3] for r in hl_rects)
        else:
            bx0, by0, bx1, by1 = cw * 0.2, ch * 0.4, cw * 0.8, ch * 0.6
        self.hcx, self.hcy = (bx0 + bx1) / 2, (by0 + by1) / 2
        self.zmax = float(np.clip(min(cw * 0.92 / max(bx1 - bx0, 1), ch * 0.8 / max(by1 - by0, 1)), 1.15, 1.6))
        self.header_sub = shorten((src or "ON THE RECORD").upper(), 40)
        self.info_label = "QUOTE"
        self.header_right = (shorten((src or "THE RECORD").upper(), 18), "", C_MUTED)

    def _render(self, lines, font, margin, y0, line_h, hl_i, hl_rects, focus: bool, source: str, top_h: int,
                bot_h: int):
        cw, ch, S = self.cw, self.ch, self.S
        paper, ink, faded = (244, 241, 232), (28, 28, 28), (150, 146, 138)
        img = Image.new("RGB", (cw, ch), paper)
        d = ImageDraw.Draw(img)
        d.text((cw // 2, int(top_h * 0.38)), "THE WEEKLY WIRE", font=load_serif(max(12, int(top_h * 0.34))), fill=ink,
               anchor="mm")
        for off, wd in ((0.66, 3 * S), (0.74, S)):
            d.line([(margin, int(top_h * off)), (cw - margin, int(top_h * off))], fill=ink, width=int(wd))
        if focus:
            for r in hl_rects:
                d.rounded_rectangle(r, radius=int(line_h * 0.12), fill=(255, 226, 41))
        for li, line in enumerate(lines):
            for tok, si, x, _w in line:
                d.text((margin + x, y0 + li * line_h), tok, font=font, fill=ink if (not focus or si == hl_i) else faded)
        d.line([(margin, ch - int(bot_h * 0.92)), (cw - margin, ch - int(bot_h * 0.92))], fill=ink, width=S)
        if source:
            d.text((cw - margin, ch - int(bot_h * 0.45)), "- " + source, font=load_serif(max(12, int(bot_h * 0.36))),
                   fill=(70, 70, 70), anchor="rm")
        return img

    def paint(self, frame: np.ndarray, t: float) -> None:
        p = ease_in_out(ramp(t, 0.25, self.dur * 0.9))
        z = 1 + (self.zmax - 1) * p
        vw, vh = self.cw / z, self.ch / z
        cx, cy = lerp(self.cw / 2, self.hcx, p), lerp(self.ch / 2, self.hcy, p)
        x0 = min(max(cx - vw / 2, 0), self.cw - vw)
        y0 = min(max(cy - vh / 2, 0), self.ch - vh)
        box = (x0, y0, x0 + vw, y0 + vh)
        size = (self.sw, self.sh)
        img = self.base.resize(size, Image.BILINEAR, box=box)
        mix = ease_out(ramp(t, 0.3, 0.6))
        if mix > 0:
            img = Image.blend(img, self.focus.resize(size, Image.BILINEAR, box=box), mix)
        ImageDraw.Draw(img).rectangle((0, 0, self.sw - 1, self.sh - 1), outline=_rgb(self.accent), width=max(2, px(4)))
        frame[self.sy:self.sy + self.sh, self.sx:self.sx + self.sw] = np.asarray(img)


# --- 8. PORTFOLIO $10K SIMULATOR --------------------------------------
class Portfolio10kLayout(LayoutBase):
    """Cash-register screen: an LCD total on top, a receipt-style ledger below. $10,000 per stock at the start of
            the week (the prior close) versus the latest close; the numbers count up within about a second."""
    STAKE = 10000.0
    COUNT_SECONDS = 0.9

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        rows = data["rows"][:4]
        n = len(rows)
        self.ends = [self.STAKE * r["last"] / r["base"] for r in rows]
        self.total_start, self.total_end = self.STAKE * n, sum(self.ends)
        lcd_bg, lcd_col = "#06210F", C_GREEN
        self.lcd = FancyBboxPatch((30, 30), W - 60, 250, boxstyle="round,pad=0,rounding_size=22", fc=lcd_bg, ec=lcd_col,
                                  lw=5)
        ax.add_patch(self.lcd)
        lbl = (f"${self.STAKE:,.0f} IN EACH  |  START OF WEEK > FRIDAY CLOSE" if n > 1
               else f"${self.STAKE:,.0f}  |  START OF WEEK > FRIDAY CLOSE")
        ax.text(70, 68, lbl, fontsize=fit_one(lbl, W - 140, 40, 32, 16) * REF_PT, family="monospace", fontweight="bold",
                color=blend_hex(lcd_col, lcd_bg, 0.35), ha="left", va="center")
        size = fit_one(f"${self.total_end:,.0f}", W - 200, 130, 150, 40)
        self.total_txt = ax.text(W / 2, 158, "", fontsize=size * REF_PT, family="monospace", fontweight="bold",
                                 color=lcd_col, ha="center", va="center")
        self.total_pill = ax.text(W / 2, 244, "", fontsize=40 * REF_PT, family="monospace", fontweight="bold",
                                  color=lcd_col, ha="center", va="center")
        self.rows = []
        top, avail = 312.0, H - 312.0 - 24.0
        rh = min(118.0, avail / max(n, 1) - 12)
        for i, (r, end) in enumerate(zip(rows, self.ends)):
            y = top + i * (rh + 12)
            col = C_GREEN if end >= self.STAKE else C_RED
            box = FancyBboxPatch((30, y), W - 60, rh, boxstyle="round,pad=0,rounding_size=18", fc=C_PANEL, ec=C_BORDER,
                                 lw=2.5)
            ax.add_patch(box)
            cy = y + rh / 2
            tick = ax.text(70, cy, mpl_safe(r["ticker"].lstrip("^")),
                           fontsize=fit_one(r["ticker"].lstrip("^"), 340, rh * 0.6, 60, 20) * REF_PT, fontweight="bold",
                           color=C_WHITE, ha="left", va="center")
            start = ax.text(560, cy, f"${self.STAKE:,.0f}", fontsize=min(46, rh * 0.42) * REF_PT, fontweight="bold",
                            color=C_MUTED, ha="center", va="center")
            arrow = ax.text(750, cy, "\u2192", fontsize=min(58, rh * 0.55) * REF_PT, fontweight="bold", color=C_MUTED,
                            ha="center", va="center")
            val = ax.text(990, cy, "", fontsize=min(64, rh * 0.58) * REF_PT, fontweight="bold", color=col, ha="center",
                          va="center")
            pill_box = FancyBboxPatch((1160, cy - rh * 0.32), 230, rh * 0.64, boxstyle="round,pad=0,rounding_size=24",
                                      fc=blend_hex(col, C_BG, 0.82), ec=col, lw=3)
            ax.add_patch(pill_box)
            pill = ax.text(1275, cy, "", fontsize=min(40, rh * 0.38) * REF_PT, fontweight="bold", color=col,
                           ha="center", va="center")
            t0 = 0.1 + 0.12 * i
            self.rows.append(
                dict(grp=Reveal(t0, 0.25).add(box, tick, start, arrow, pill_box), val=val, pill=pill, end=end, t0=t0))
        gain = (self.total_end / self.total_start - 1) * 100
        self.header_sub = "THE $10,000 TEST"
        self.info_label = "PORTFOLIO RESULT"
        self.header_right = (f"${self.total_end:,.0f}", f"{gain:+.2f}%", C_GREEN if gain >= 0 else C_RED)

    def update(self, t: float) -> None:
        total_now = 0.0
        for r in self.rows:
            a = r["grp"].apply(t)
            p = ease_out(ramp(t, r["t0"], self.COUNT_SECONDS))
            v = lerp(self.STAKE, r["end"], p)
            total_now += v
            r["val"].set_text(f"${v:,.0f}")
            r["val"].set_alpha(a)
            r["pill"].set_text(f"{(v / self.STAKE - 1) * 100:+.1f}%")
            r["pill"].set_alpha(a)
        self.total_txt.set_text(f"${total_now:,.0f}")
        delta = total_now - self.total_start
        self.total_pill.set_text(
            f"{'+' if delta >= 0 else '-'}${abs(delta):,.0f}  ({delta / self.total_start * 100:+.2f}%)")
        done = ramp(t, 0.4 + 0.12 * len(self.rows) + self.COUNT_SECONDS, 0.3)
        self.lcd.set_linewidth(5 + 4 * done * abs(math.sin(2 * math.pi * t * 1.5)))


# --- 9. EARNINGS CALENDAR ---------------------------------------------
class EarningsCalendarLayout(LayoutBase):
    """Big table of next week's reports: one column per day, before-market and after-market rows, company logos
            (a coloured disc with the ticker when no logo file exists). The weekday the narration names lights up."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        items, cols, logos = data["items"], data["columns"], data.get("logos", {})
        nc = max(1, len(cols))
        has_tbd = any(i["session"] == "TBD" for i in items)
        label_w, gap = 150.0, 14.0
        x0, x1 = 18 + label_w, W - 18.0
        col_w = (x1 - x0) / nc
        head_h = 96.0
        top = 18 + head_h + 12
        avail = H - 18 - top
        tbd_h = 120.0 if has_tbd else 0.0
        main_h = (avail - tbd_h - gap * (2 if has_tbd else 1)) / 2
        rows = [("BMO", ("BEFORE", "MARKET"), top, main_h, 2, 3),
                ("AMC", ("AFTER", "MARKET"), top + main_h + gap, main_h, 2, 3)]
        if has_tbd:
            rows.append(("TBD", ("TIME", "TBD"), top + 2 * (main_h + gap), tbd_h, 4, 1))
        by = {}
        for it in items:
            by.setdefault((it["date"], it["session"]), []).append(it)
        self.cols = [Reveal(0.07 * ci, 0.3) for ci in range(nc)]
        self.col_alpha = [0.0] * nc
        self.logo_items: List[dict] = []
        self.heads = []
        patterns = []
        for ci, dstr in enumerate(cols):
            day = datetime.date.fromisoformat(dstr)
            cx = x0 + col_w * (ci + 0.5)
            hbox = FancyBboxPatch((x0 + col_w * ci + 5, 18), col_w - 10, head_h,
                                  boxstyle="round,pad=0,rounding_size=16",
                                  fc=C_PANEL, ec=C_BORDER, lw=2.5)
            ax.add_patch(hbox)
            glow = FancyBboxPatch((x0 + col_w * ci + 5, 18), col_w - 10, head_h,
                                  boxstyle="round,pad=0,rounding_size=16",
                                  fc="none", ec=C_GOLD, lw=7, alpha=0.0)
            ax.add_patch(glow)
            self.heads.append(glow)
            self.cols[ci].add(hbox,
                              ax.text(cx, 52, day.strftime("%a").upper(), fontsize=46 * REF_PT, fontweight="bold",
                                      color=C_WHITE, ha="center", va="center"),
                              ax.text(cx, 90, day.strftime("%b %d").upper(), fontsize=28 * REF_PT, fontweight="bold",
                                      color=C_MUTED, ha="center", va="center"))
            patterns.append((day.strftime("%A").lower(), ci))
            for sess, _lbl, ry, rh, ncols, nrows in rows:
                cell = FancyBboxPatch((x0 + col_w * ci + 5, ry), col_w - 10, rh,
                                      boxstyle="round,pad=0,rounding_size=16",
                                      fc=C_PANEL, ec=C_BORDER, lw=2)
                ax.add_patch(cell)
                self.cols[ci].add(cell)
                self._place(ax, by.get((dstr, sess), []), x0 + col_w * ci + 5, ry, col_w - 10, rh, ncols, nrows, ci,
                            logos)
        for sess, lbl, ry, rh, _nc, _nr in rows:
            self.cols[0].add(draw_lines(ax, list(lbl), 18 + label_w / 2, ry + rh / 2, 28, C_MUTED, pitch=1.3))
        self.mentions = self.track(patterns)
        self.header_sub = "UPCOMING REPORTS"
        self.info_label = "NEXT WEEK"
        first, last = datetime.date.fromisoformat(cols[0]), datetime.date.fromisoformat(cols[-1])
        self.header_right = (f"{len(items)} REPORTS", f"{first:%b %d} - {last:%b %d}".upper(), C_CYAN)

    def _place(self, ax, items, x, y, w, h, ncols, nrows, ci, logos) -> None:
        cap = ncols * nrows
        shown = items if len(items) <= cap else items[:cap - 1]
        extra = len(items) - len(shown)
        slot_w, slot_h = w / ncols, h / nrows
        d = min(slot_w * 0.62, slot_h * 0.58, 92.0)
        for k, it in enumerate(shown):
            cx, cy = x + (k % ncols + 0.5) * slot_w, y + (k // ncols + 0.5) * slot_h - slot_h * 0.08
            path = logos.get(it["ticker"])
            arr = None
            if path:
                try:
                    with Image.open(path) as im:
                        arr = np.asarray(im.convert("RGBA").resize((px(d), px(d))))
                except Exception as e:
                    logger.warning(f"Logo for {it['ticker']} unreadable: {e}")
            if arr is not None:
                self.logo_items.append(
                    dict(arr=arr, x=STAGE_BOX[0] + px(cx - d / 2), y=STAGE_BOX[1] + px(cy - d / 2), col=ci))
            else:
                rnd = random.Random(it["ticker"])
                hue = "#%02x%02x%02x" % (rnd.randint(40, 200), rnd.randint(90, 220), rnd.randint(120, 255))
                disc = Circle((cx, cy), d / 2, fc=hue, ec=C_WHITE, lw=2)
                ax.add_patch(disc)
                self.cols[ci].add(disc, ax.text(cx, cy, mpl_safe(it["ticker"][:3]), fontsize=d * 0.34 * REF_PT,
                                                fontweight="bold", color=C_BG, ha="center", va="center"))
            self.cols[ci].add(ax.text(cx, cy + d / 2 + slot_h * 0.15, mpl_safe(it["ticker"]),
                                      fontsize=fit_one(it["ticker"], slot_w - 8, slot_h * 0.26, 34, 11) * REF_PT,
                                      fontweight="bold", color=C_WHITE, ha="center", va="center"))
        if extra:
            k = len(shown)
            cx, cy = x + (k % ncols + 0.5) * slot_w, y + (k // ncols + 0.5) * slot_h
            self.cols[ci].add(
                ax.text(cx, cy, f"+{extra}", fontsize=fit_one("+00", slot_w - 8, slot_h * 0.4, 40, 14) * REF_PT,
                        fontweight="bold", color=C_GOLD, ha="center", va="center"))

    def update(self, t: float) -> None:
        key, s = self.mentions.active(t)
        pulse = 0.7 + 0.3 * math.sin(2 * math.pi * t * 1.3)
        for ci, grp in enumerate(self.cols):
            m = 1.0 if key is None or ci == key else 1 - 0.55 * s
            self.col_alpha[ci] = grp.apply(t, m)
            self.heads[ci].set_alpha(s * pulse if ci == key else 0.0)

    def paint(self, frame: np.ndarray, t: float) -> None:
        for it in self.logo_items:
            blend_rgba(frame, it["arr"], it["x"], it["y"], self.col_alpha[it["col"]])


# --- 10. TRADE SETUP (rule-based pattern, entry / stop / target) -------
class TradeSetupLayout(LayoutBase):
    """Daily chart of a pattern found by the scanner with the entry, stop and target levels drawn on it, and a
            permanent disclaimer banner. The level the narration mentions is emphasised."""
    BUILD_START = 0.08

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        chart, setup = data["chart"], data["setup"]
        W, H = STAGE_REF
        ax0 = self.ax0 = make_stage_axes(fig, STAGE_BOX, W, H)
        banner = FancyBboxPatch((30, H - 78), W - 60, 58, boxstyle="round,pad=0,rounding_size=14",
                                fc=blend_hex(C_AMBER, C_BG, 0.86), ec=C_AMBER, lw=3)
        ax0.add_patch(banner)
        ax0.text(W / 2, H - 49, "EDUCATIONAL PATTERN ANALYSIS. NOT FINANCIAL ADVICE. DO YOUR OWN RESEARCH.",
                 fontsize=fit_one("EDUCATIONAL PATTERN ANALYSIS. NOT FINANCIAL ADVICE. DO YOUR OWN RESEARCH.", W - 120,
                                  44, 30, 14) * REF_PT,
                 fontweight="bold", color=C_AMBER, ha="center", va="center")
        self.pane = CandlePane(fig, chart, (40, 268, 1340, 470), (40, 748, 1340, 150),
                               [("SMA20", C_AMBER, chart["sma20"]), ("SMA50", C_CYAN, chart["sma50"])],
                               f"{chart['ticker']} \u00b7 1D")
        entry, stop, target = float(setup["entry"]), float(setup["stop"]), float(setup["target"])
        self.pane.include_levels([stop, target, entry])
        ax, n = self.pane.ax, self.pane.n
        ax.set_xlim(-1, n + n * 0.34)  # free space on the right for the level labels
        self.zones = [ax.axhspan(entry, target, facecolor=C_GREEN, alpha=0.0, zorder=1),
                      ax.axhspan(stop, entry, facecolor=C_RED, alpha=0.0, zorder=1)]
        self.levels = []
        for i, (key, val, col, label, va) in enumerate((("target", target, C_GREEN, "TARGET", "bottom"),
                                                        ("entry", entry, C_CYAN, "ENTRY", "bottom"),
                                                        ("stop", stop, C_RED, "STOP", "top"))):
            line = ax.axhline(val, color=col, linestyle="--", linewidth=3, alpha=0.0, zorder=7)
            txt = ax.text(n + 0.6, val, f" {label} ${val:,.2f} ", color=C_BG, fontsize=22 * REF_PT, fontweight="bold",
                          ha="left", va=va, zorder=12, visible=False,
                          bbox=dict(boxstyle="round,pad=0.25", facecolor=col, edgecolor="none"))
            self.levels.append(
                dict(key=key, line=line, txt=txt, t0=self.BUILD_START + CHART_BUILD_SECONDS + 0.2 + 0.25 * i))
        self.mentions = self.track([(p, k) for k, plist in (
            ("entry", ["entry", "enter", "buy zone"]), ("stop", ["stop", "risk", "invalidat"]),
            ("target", ["target", "upside", "reward", "profit"])) for p in plist])
        self.header_sub = f"{chart['ticker']}  |  {setup['pattern'].upper()}"
        self.info_label = setup["pattern"].upper()
        self.header_right = (f"R:R {setup['rr']:.1f}", setup.get("metrics", ""), C_CYAN)

    def update(self, t: float) -> None:
        n = self.pane.n
        k = min(n, max(2, int(math.ceil(ease_out(ramp(t, self.BUILD_START, CHART_BUILD_SECONDS)) * n))))
        self.pane.update(k, t)
        key, s = self.mentions.active(t)
        for lv in self.levels:
            a = ease_out(ramp(t, lv["t0"], 0.25))
            hot = s if lv["key"] == key else 0.0
            lv["line"].set_alpha(a * (0.8 + 0.2 * hot))
            lv["line"].set_linewidth(3 + 3.5 * hot)
            lv["txt"].set_visible(t >= lv["t0"])
            lv["txt"].set_alpha(a)
            lv["txt"].get_bbox_patch().set_alpha(a)
        zone_a = ease_out(ramp(t, self.levels[0]["t0"], 0.4))
        self.zones[0].set_alpha(0.12 * zone_a)
        self.zones[1].set_alpha(0.12 * zone_a)


# --- 11. OUTRO ---------------------------------------------------------
class OutroLayout(LayoutBase):
    """Closing screen: the character stands in the middle (painted by the scene) over one fixed block of text."""
    AVATAR_REF = 460

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        ax = self.ax = make_stage_axes(fig, (0, 0, VIDEO_W, VIDEO_H), *FULL_REF, border=False, face="none")
        self.title = ax.text(960, 170, "THANKS FOR WATCHING", fontsize=104 * REF_PT, fontweight="bold", color=C_WHITE,
                             ha="center", va="center")
        self.sub = ax.text(960, 290, "LIKE  \u2022  SUBSCRIBE  \u2022  SEE YOU NEXT WEEK", fontsize=48 * REF_PT,
                           fontweight="bold", color=self.accent, ha="center", va="center")
        self.foot = ax.text(960, 925, "NOT FINANCIAL ADVICE  \u2022  FOR EDUCATIONAL PURPOSES ONLY",
                            fontsize=44 * REF_PT,
                            fontweight="bold", color=C_WHITE, ha="center", va="center")
        self.foot2 = ax.text(960, 1000, "Markets involve risk. Do your own research.", fontsize=32 * REF_PT,
                             fontweight="bold", color=C_MUTED, ha="center", va="center")

    def avatar_rect(self) -> Tuple[int, int, int]:
        s = px(self.AVATAR_REF)
        return (VIDEO_W - s) // 2, px(370), s

    def update(self, t: float) -> None:
        a = ease_out(ramp(t, 0.0, 0.5))
        for art in (self.title, self.sub, self.foot, self.foot2):
            art.set_alpha(a)


# --- 12. CHAPTER BUMPER ------------------------------------------------
class ChapterBumperLayout(LayoutBase):
    """2-second interstitial: just the chapter title over the animated background. No TTS; the pipeline plays a
            heavy swoosh/boom SFX under it."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        ax = self.ax = make_stage_axes(fig, (0, 0, VIDEO_W, VIDEO_H), *FULL_REF, border=False, face="none")
        lines, size = fit_wrap(seg["chapter_title"].upper(), 1500, 520, 150, 56, pitch=1.18)
        half = len(lines) * size * 1.18 / 2
        self.title = draw_lines(ax, lines, 960, 520, size, C_WHITE, pitch=1.18)
        self.bar = Rectangle((960, 520 + half + 40), 0, 12, facecolor=self.accent, edgecolor="none")
        ax.add_patch(self.bar)
        self.bar_y = 520 + half + 40
        self.rings = []
        for _ in range(3):
            ring = Circle((960, 540), 10, fill=False, ec=self.accent, lw=3, alpha=0.0)
            ax.add_patch(ring)
            self.rings.append(ring)

    def update(self, t: float) -> None:
        a = ease_out(ramp(t, 0.1, 0.4)) * (1 - 0.7 * ramp(t, self.duration - 0.35, 0.35))
        for art in self.title:
            art.set_alpha(a)
        w = 900 * ease_out(ramp(t, 0.2, 0.7))
        self.bar.set_x(960 - w / 2)
        self.bar.set_width(w)
        for i, ring in enumerate(self.rings):
            p = ramp(t, 0.30 + 0.16 * i, 1.0)
            ring.set_radius(40 + 1100 * ease_out(p))
            ring.set_alpha(0.55 * (1 - p) if 0 < p < 1 else 0.0)


# --- 13. PLAIN TEXT (also the universal fallback) ---------------------
class PlainTextLayout(LayoutBase):
    """Big centred statement; the words pop in within about 0.7 s, chosen words in the accent colour."""

    def __init__(self, fig, seg, duration, speech, data=None, sentence_times=None):
        super().__init__(fig, seg, duration, speech, data, sentence_times)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        for y in (46, H - 46):
            ax.plot([100, W - 100], [y, y], color=self.accent, linewidth=4, alpha=0.85, solid_capstyle="round")
        bx, by, bw, bh = STAGE_BOX
        box_x, box_y, box_w, box_h = bx + px(110), by + px(110), px(W - 220), px(H - 220)
        text = seg["plain_text"]
        highlight = {re.sub(r"[^\w$%]", "", w).lower() for w in seg.get("plain_highlight", [])} - {""}
        accent_rgb, white_rgb = _rgb(self.accent), _rgb(C_WHITE)
        size, min_size = px(110), max(14, px(36))
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
        self.info_label = "KEY TAKEAWAY"

    def paint(self, frame: np.ndarray, t: float) -> None:
        for w in self.words:
            a = ease_out(ramp(t, w["start"], 0.18))
            if a > 0:
                blend_mask(frame, w["mask"], w["x"], w["y"] + int(round((1 - a) * px(14))), w["color"], a)


LAYOUT_CLASSES = {
    "pro_candlestick_chart": ProCandlestickLayout,
    "stock_dashboard": StockDashboardLayout,
    "sector_heatmap": SectorHeatmapLayout,
    "top_movers_podium": TopMoversPodiumLayout,
    "macro_timeline": MacroTimelineLayout,
    "bull_vs_bear": BullVsBearLayout,
    "quote_highlight": QuoteHighlightLayout,
    "portfolio_10k_sim": Portfolio10kLayout,
    "trade_setup": TradeSetupLayout,
    "earnings_calendar": EarningsCalendarLayout,
    "outro": OutroLayout,
    "chapter_bumper": ChapterBumperLayout,
    "plain_text": PlainTextLayout,
}


# =====================================================================
# 12. SCENES  (frame_at(t) -> RGB uint8 array; t is scene-local)
# =====================================================================
def build_info_card(fig, layout: LayoutBase, week_label: str) -> None:
    """Small card in the right column: what the screen is about + its key number (price, biggest move, R:R ...)."""
    ax = make_stage_axes(fig, INFO_BOX, *INFO_REF, face=PANEL_FACE)
    label = layout.info_label or week_label
    lines, size = fit_wrap(label.upper(), 312, 64, 26, 12, pitch=1.2)
    draw_lines(ax, lines, 24, 30 + (len(lines) - 1) * size * 0.6, size, C_MUTED, pitch=1.2, ha="left")
    line1, line2, col2 = layout.header_right or (shorten(week_label, 22), "", C_MUTED)
    ax.text(INFO_REF[0] / 2, 128, mpl_safe(line1), fontsize=fit_one(line1, 316, 96, 88, 18) * REF_PT, fontweight="bold",
            color=C_WHITE, ha="center", va="center")
    if line2:
        ax.text(INFO_REF[0] / 2, 205, mpl_safe(line2), fontsize=fit_one(line2, 316, 62, 54, 14) * REF_PT,
                fontweight="bold", color=col2, ha="center", va="center")


class RecapSegmentScene:
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
        n = len(head)
        size = 46 if n <= 14 else 40 if n <= 18 else 36 if n <= 22 else 30
        self.fig.text(fig_x(px(60)), fig_y(HEADER_TOP), mpl_safe(head), fontsize=size, fontweight="bold",
                      color=C_WHITE, va="top")
        sub = self.layout.header_sub or a["week_label"]
        self.fig.text(fig_x(px(60)), fig_y(HEADER_SUB_TOP), mpl_safe(shorten(sub, 70)), fontsize=20,
                      color=C_MUTED, va="top")
        build_info_card(self.fig, self.layout, a["week_label"])
        # The 220 px gap between info card (y=460) and avatar (y=680) is centered.
        self.fig.text(fig_x(px(1700)), fig_y(px(568)), "EDUCATIONAL ONLY\nNOT FINANCIAL ADVICE",
                      fontsize=21, fontweight="medium", color="#AFB9C4", ha="center", va="center",
                      linespacing=1.5)
        self.show_avatar = seg["layout"] != "quote_highlight"  # the quote is read by the other voice: no avatar
        if self.show_avatar:
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
        if self.show_avatar:
            idx = int(t * OUTPUT_FPS)
            vol = float(self.peaks[idx]) if idx < len(self.peaks) else 0.0
            speaking = vol > 0.08
            bob = px(min(max(vol * 50, 4), 20)) if speaking else 0
            frame[AVATAR_Y:AVATAR_Y + AVATAR_SIZE, AVATAR_X:AVATAR_X + AVATAR_SIZE] = self.avatar.frame(speaking, bob)
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
        self.ax_, self.ay_, size = self.layout.avatar_rect()
        self.size = size
        self.peaks = get_audio_peaks(a["audio_path"])
        self.avatar = AvatarPanel(size)

    def _render_chart(self, t: float) -> np.ndarray:
        self.bg_im.set_data(get_background().frame(self.global_start + t))
        self.layout.update(t)
        return grab(self.canvas)

    def frame_at(self, t: float) -> np.ndarray:
        frame = self.cache.get(t, self._render_chart).copy()
        idx = int(t * OUTPUT_FPS)
        vol = float(self.peaks[idx]) if idx < len(self.peaks) else 0.0
        speaking = vol > 0.08
        bob = px(min(max(vol * 60, 4), 24)) if speaking else 0
        frame[self.ay_:self.ay_ + self.size, self.ax_:self.ax_ + self.size] = self.avatar.frame(speaking, bob)
        return frame


class ChapterBumperScene:
    def __init__(self, a: dict, duration: float, global_start: float):
        self.duration, self.global_start = duration, global_start
        self.cache = FrameCache()
        self.fig, self.canvas = new_figure()
        self.bg_im = attach_background(self.fig, global_start)
        self.layout = ChapterBumperLayout(self.fig, a, duration, duration)

    def _render(self, t: float) -> np.ndarray:
        self.bg_im.set_data(get_background().frame(self.global_start + t))
        self.layout.update(t)
        return grab(self.canvas)

    def frame_at(self, t: float) -> np.ndarray:
        return self.cache.get(t, self._render)


# =====================================================================
# 13. VIDEO WRITING  (each scene -> one MP4 chunk WITH its own audio; one isolated process per chapter)
# =====================================================================
@dataclass
class SceneJob:
    kind: str  # "segment" | "bumper" | "outro"
    out_path: str
    start_time: float  # position on the global timeline (seconds)
    duration: float
    args: Dict[str, Any] = field(default_factory=dict)


def build_scene(job: SceneJob):
    if job.kind == "segment":
        return RecapSegmentScene(job.args, job.duration, job.start_time)
    if job.kind == "outro":
        return OutroScene(job.args, job.duration, job.start_time)
    if job.kind == "bumper":
        return ChapterBumperScene(job.args, job.duration, job.start_time)
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
    """One isolated OS process per chapter (several at once): each Matplotlib scene and the avatar cache die with
            their process. Chapters are independent, so rendering them in parallel is safe."""
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
        logger.info(f"Chapter {ci + 1}/{len(by_chapter)} rendered")
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
                      f"{total_seconds:.3f}",
                      str(output_path)]
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
# 15. PIPELINE
# =====================================================================
def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def build_youtube_description(outline: WeeklyOutline,
                              chapter_starts: List[Tuple[float, str]]) -> str:
    """Longer description + chapter markers from the *actual* TTS timeline, no guessed timestamps."""
    intro = outline.description.strip()
    intro = re.sub(r"(?i)\s*(?:this is )?(?:educational (?:and )?)?not\s+(?:a\s+)?financial advice\.?", "",
                   intro).strip()
    # Ground any lengthening exclusively in the supplied director outline, not generic AI boilerplate.
    details = []
    for chapter in outline.chapters:
        points = [p.strip().rstrip(".") for p in chapter.talking_points[:2] if p.strip()]
        details.append(f"{chapter.title}: {chapter.summary.strip()}" +
                       (" Key points: " + "; ".join(points) + "." if points else ""))
    if len(intro.split()) < 145:
        intro += "\n\nWHAT WE COVER\n" + "\n".join(details)
    if PLACEHOLDER_RE.search(intro):
        raise ValueError(
            "YouTube description contains unresolved prices. Resolve outline metadata after market prefetch.")
    if chapter_starts:
        # YouTube recognizes chapter links only when the first one begins at 00:00.
        chapters = "\n".join(f"{fmt_ts(t)} - {title}" for t, title in chapter_starts)
        intro += "\n\nCHAPTERS\n" + chapters
    return intro.strip()


def run_weekly_recap_pipeline(output_filename: Optional[str] = None, upload: bool = False,
                              news_file: Optional[str] = None, save_script: Optional[str] = None,
                              load_script: Optional[str] = None, work_dir_arg: Optional[str] = None) -> str:
    logger.info("Launching retention-first Weekly Recap pipeline (widescreen, low-memory mode)...")
    log_memory("pipeline start")
    persistent = bool(work_dir_arg)
    if persistent:
        work_dir = Path(work_dir_arg)
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using persistent work dir {work_dir} (finished scenes and the voiceover are reused on re-runs)")
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="weekly_recap_"))
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
            master = generate_master_timeline(gather_weekly_context(news_file))
        # ---- Phase B: pre-fetch ALL market data, ground numbers, repair segments (before touching video) ----
        cache = prefetch_market_data(master)
        if persistent and not auto_script.exists():
            auto_script.write_text(master.model_dump_json(indent=2), encoding="utf-8")
        if save_script:
            Path(save_script).write_text(master.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Final, edited script saved to %s", save_script)
        log_memory("market data ready")
        outline = master.outline
        n_chapters = len(outline.chapters)
        segs = [it.segment for it in master.items]
        logger.info("=" * 60)
        logger.info(f"{outline.week_label} | {outline.youtube_title}")
        for it in master.items:
            s = it.segment
            logger.info(
                f"  ch{it.chapter_index + 1} {s.layout:22s} sfx={s.sfx:13s} '{s.headline}' :: {shorten(s.script, 90)}")
        logger.info("=" * 60)

        # ---- Phase C: voices, at natural speaking speed (no artificial 10-minute target) ----
        speech_items = []
        for i, s in enumerate(segs):
            voice, speed = voice_for(s)
            speech_items.append({"text": s.script, "voice": voice, "speed": speed,
                                 "path": str(work_dir / f"segment_{i}.wav")})
        tts = run_tts(speech_items, work_dir)
        durations = [r["duration"] for r in tts]
        pads = [SEGMENT_PAD] * len(segs)
        pads[-1] = END_PAD
        seg_durs = [d + p for d, p in zip(durations, pads)]
        # Start speaking at 00:00. Chapter bumpers appear only between chapters.
        total = sum(seg_durs) + max(0, n_chapters - 1) * BUMPER_SECONDS
        logger.info("Narration %.1fs; natural video runtime %s (no minimum)", sum(durations), fmt_ts(total))
        log_memory("TTS completed")

        # ---- Phase D: timeline -> scene jobs (bumper + segments for every chapter) ----
        jobs: List[SceneJob] = []
        chapter_starts: List[Tuple[float, str]] = []
        cursor = 0.0
        for ci, plan in enumerate(outline.chapters):
            chapter_idx = [k for k, it in enumerate(master.items) if it.chapter_index == ci]
            if not chapter_idx:
                continue
            chapter_starts.append((cursor, plan.title))
            if ci > 0:  # Cold open starts with the first word, not an opening logo bumper.
                jobs.append(SceneJob("bumper", str(work_dir / f"scene_c{ci}_bumper.mp4"),
                                     cursor, BUMPER_SECONDS, args=dict(
                        chapter_index=ci, n_chapters=n_chapters, chapter_title=plan.title, accent="gold",
                        audio_path=None, sfx_path=get_sfx_path("swoosh_boom", work_dir),
                        mix_path=str(work_dir / f"mix_c{ci}_bumper.wav"))))
                cursor += BUMPER_SECONDS
            for k in chapter_idx:
                s = segs[k]
                jobs.append(
                    SceneJob("outro" if s.layout == "outro" else "segment", str(work_dir / f"scene_{k}.mp4"), cursor,
                             seg_durs[k], args=dict(
                            seg=s.model_dump(), chapter_index=ci, n_chapters=n_chapters, chapter_title=plan.title,
                            week_label=outline.week_label, data=data_for_segment(s, cache, master),
                            speech_dur=durations[k],
                            sentence_times=tts[k]["sentences"], ticker_text=outline.news_ticker_text,
                            audio_path=speech_items[k]["path"], sfx_path=get_sfx_path(ev(s.sfx), work_dir),
                            mix_path=str(work_dir / f"mix_{k}.wav"))))
                cursor += seg_durs[k]
        logger.info(f"Final length: {cursor:.1f}s ({fmt_ts(cursor)}) in {len(jobs)} scenes")

        if not output_filename:
            output_filename = f"WEEKLY_RECAP_{datetime.date.today().isoformat()}.mp4"
        description = build_youtube_description(outline, chapter_starts)
        tags = [t.strip() for t in outline.tags.split(",") if t.strip()]
        youtube_title = outline.youtube_title
        # Save even without --upload, so the uploader text can be reviewed and pasted.
        metadata_path = Path(output_filename).with_suffix(".description.txt")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(description + "\n", encoding="utf-8")
        logger.info("YouTube description saved to %s", metadata_path)
        metadata_json = Path(output_filename).with_suffix(".youtube.json")
        metadata_json.write_text(json.dumps({"title": youtube_title, "description": description, "tags": tags},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
        del master, segs, cache
        gc.collect()

        # ---- Phase E: render (one worker per chapter, several at once) + assemble ----
        scene_paths = render_all_chapters(jobs, work_dir)
        log_memory("scenes rendered")
        assemble_final_video(scene_paths, cursor, output_filename, work_dir)
        logger.info(f"SUCCESS! Video saved as: {output_filename} ({fmt_ts(cursor)})")

        if upload:
            from uploader import upload_video
            logger.info("Uploading to YouTube...")
            upload_video(file_path=output_filename, title=youtube_title, description=description, tags=tags,
                         category_id="27")
            logger.info(f"Uploaded to YouTube: {youtube_title}")
        log_memory("pipeline finished")
        return output_filename
    finally:
        if persistent:
            logger.info(f"Work dir kept: {work_dir} (delete it when you are happy with the video)")
        else:
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(work_dir, ignore_errors=True)


# =====================================================================
# 16. ENTRY POINTS (parent CLI + isolated workers)
# =====================================================================
def tts_worker_main(in_json: str, out_json: str) -> None:
    items = json.loads(Path(in_json).read_text(encoding="utf-8"))
    Path(out_json).write_text(json.dumps(generate_voiceovers(items)), encoding="utf-8")


def render_worker_main(pkl_path: str) -> None:
    # This pickle is written by this very application into its own private temp directory.
    with open(pkl_path, "rb") as f:
        jobs = pickle.load(f)
    render_chapter_jobs(jobs if isinstance(jobs, list) else [jobs])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tts-worker":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: --tts-worker input.json durations.json")
        tts_worker_main(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--render-worker":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: --render-worker chapter_jobs.pkl")
        render_worker_main(sys.argv[2])
    else:
        parser = argparse.ArgumentParser(description="Generate a retention-first widescreen Weekly Market Recap.")
        parser.add_argument("--upload", action="store_true", help="upload to YouTube when finished")
        parser.add_argument("--output", default=None, help="output mp4 path (default: WEEKLY_RECAP_<date>.mp4)")
        parser.add_argument("--news-file", default=None, help="extra producer notes / news appended to the LLM context")
        parser.add_argument("--save-script", default=None, help="save the generated master timeline as JSON")
        parser.add_argument("--load-script", default=None, help="skip the LLM and load a saved master timeline")
        parser.add_argument("--work-dir", default=None,
                            help="keep scenes/voiceover in this folder; re-running with the same folder and "
                                 "--load-script resumes where a crashed run stopped")
        parser.add_argument("--no-music", action="store_true", help="do not mix background music")
        cli = parser.parse_args()
        if cli.no_music:
            USE_MUSIC = False
        run_weekly_recap_pipeline(output_filename=cli.output, upload=cli.upload, news_file=cli.news_file,
                                  save_script=cli.save_script, load_script=cli.load_script, work_dir_arg=cli.work_dir)