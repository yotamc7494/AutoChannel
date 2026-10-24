from __future__ import annotations

import gc
import json
import datetime
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

try:
    import httpx  # installed with google-genai; used to recognise network errors
except ImportError:  # pragma: no cover
    httpx = None

load_dotenv()
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MythBusters")


# =====================================================================
# 1. CONFIG
# =====================================================================
# --- LLM ---
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]   # tried in order, per round
LLM_MAX_ROUNDS = 5
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# --- Video (9:16 vertical) ---
# Low-RAM default (2.25x fewer pixels than 1080x1920). PERFECT_VIDEO_WIDTH=1080 restores full resolution.
VIDEO_W = int(os.environ.get("PERFECT_VIDEO_WIDTH", "720"))
if VIDEO_W not in (540, 720, 1080):
    raise ValueError("PERFECT_VIDEO_WIDTH must be 540, 720 or 1080")
VIDEO_H = VIDEO_W * 16 // 9
UI_SCALE = VIDEO_W / 1080              # every layout pixel below is authored for 1080 wide and scaled by this
DPI = VIDEO_W / 10.8                   # keeps every font/layout proportion identical at any size
REF_PT = 0.72                          # 1 reference pixel (1080 grid) expressed in matplotlib points, at ANY size
OUTPUT_FPS = 30
RENDER_FPS = 15                        # heavy matplotlib layer is redrawn at this rate, frames are repeated
VIDEO_BITRATE = "4500k" if VIDEO_W <= 720 else "8000k"
X264_PRESET = "ultrafast"
BG_MUSIC_VOLUME = 0.12
BG_MUSIC_FADE_OUT = 2.0
SHORTS_MAX_SECONDS = 59.0

# --- Parallelism / memory (same strategy as the recap script) ---
IO_WORKERS = 1
RENDER_IN_SUBPROCESS = True            # TTS and every segment render run in an isolated, sequential OS process
FFMPEG_THREADS = 1
LOG_EVERY_SECONDS = 8

# --- Pacing ---
SEG_MIN, SEG_MAX = 3, 4
SCRIPT_WORDS_MIN, SCRIPT_WORDS_MAX = 90, 135     # ~35-52 s of speech at speed 1.1
TTS_VOICE, TTS_SPEED, TTS_RATE = "am_liam", 1.1, 24000
TTS_MAX_SPEED = 1.35                   # if the Short would exceed 59 s, narration is re-recorded faster (up to this)
SEGMENT_PAD = 0.25                     # silence after each segment's last word (s)
END_PAD = 0.9                          # extra hold at the very end (s)
CAPTION_MAX_WORDS = 12

# --- Popups ---
POPUP_SECONDS = 2.8
POPUP_MIN_SECONDS = 1.2
POPUP_IN = 0.28
POPUP_OUT = 0.22
MAX_POPUPS_PER_SEGMENT = 1
POPUP_POSITIONS = ("top", "center", "bottom", "left", "right")

from myth_topics import TOPIC_AREAS

# --- Data ---
TIMEFRAME_INTERVALS = {"1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1d", "2y": "1wk", "5y": "1wk", "10y": "1mo"}
TTS_SYMBOL_NAMES = {
    r"\bNVDA\b": "Nvidia", r"\bTSLA\b": "Tesla", r"\bAAPL\b": "Apple", r"\bAMZN\b": "Amazon",
    r"\bMSFT\b": "Microsoft", r"\bGOOGL\b": "Google", r"\bMETA\b": "Meta", r"\bBTC\b": "Bitcoin",
    r"\bSPY\b": "S and P 500", r"\bQQQ\b": "Nasdaq", r"\bP/E\b": "P E",
}

# --- Layouts ---
LAYOUTS = ("myth_vs_reality", "big_calculator", "comparison_bars", "annotated_chart", "metric_breakdown", "plain_text")
LAYOUT_ALIASES = {
    "myth": "myth_vs_reality", "mythvsreality": "myth_vs_reality", "split": "myth_vs_reality",
    "calculator": "big_calculator", "bigcalculator": "big_calculator", "calc": "big_calculator",
    "bars": "comparison_bars", "bar": "comparison_bars", "comparison": "comparison_bars", "comparisonbars": "comparison_bars",
    "chart": "annotated_chart", "annotatedchart": "annotated_chart", "stock_chart": "annotated_chart",
    "metric": "metric_breakdown", "metricbreakdown": "metric_breakdown", "formula": "metric_breakdown",
    "text": "plain_text", "plaintext": "plain_text", "statement": "plain_text",
}
DEFAULT_HEADLINES = {
    "myth_vs_reality": "MYTH VS REALITY", "big_calculator": "DO THE MATH", "comparison_bars": "SIDE BY SIDE",
    "annotated_chart": "LOOK AT THE CHART", "metric_breakdown": "THE FORMULA", "plain_text": "THE TRUTH",
}
OPERATOR_GLYPHS = {"/": "\u00f7", "\u00f7": "\u00f7", "x": "\u00d7", "*": "\u00d7", "\u00d7": "\u00d7",
                   "-": "\u2212", "\u2212": "\u2212", "+": "+"}

# --- Palette (same as the daily recap) ---
C_BG, C_PANEL, C_BORDER = "#0B0E14", "#161B22", "#30363D"
C_WHITE, C_MUTED = "#FFFFFF", "#8B949E"
C_GREEN, C_RED, C_CYAN, C_GOLD, C_AMBER = "#00FFA3", "#FF3366", "#00E5FF", "#FFD700", "#FFB800"
COLOR_THEMES = {
    "red": C_RED, "green": C_GREEN, "gold": C_GOLD, "cyan": C_CYAN, "amber": C_AMBER, "white": C_WHITE,
    "purple": "#B388FF", "blue": "#448AFF", "orange": "#FF9100",
}

AVATAR_OPEN_PATH = "assets/avatar/open.png"
AVATAR_CLOSED_PATH = "assets/avatar/closed.png"


def px(v: float) -> int:
    return int(round(v * UI_SCALE))


# --- Layout boxes (authored for 1080x1920, identical to the recap) ---
TICKER_BOX = tuple(px(v) for v in (40, 100, 1000, 65))      # x, y_top, w, h
HEADER_TOP = px(190)
CHART_BOX = tuple(px(v) for v in (140, 335, 900, 605))      # candlestick chart (leaves room for the y axis labels)
STAGE_BOX = tuple(px(v) for v in (40, 335, 1000, 605))      # full-width stage for every other layout
STAGE_REF = (1000, 605)                                     # stage size in reference (1080-grid) pixels
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
# 1b. SCHEMAS  (what the AI director returns)
# =====================================================================
class PopupSpec(BaseModel):
    title: str = Field(description="Very short label, max 3 words, e.g. 'TRUE COST'")
    value: str = Field(description="The punchy value shown big, max 10 characters, e.g. '-40%', '$24,000', '10x', 'TRAP'")
    color: str = Field(default="gold", description="red, green, gold, cyan, amber, white or purple")
    position: str = Field(default="center", description="top, center, bottom, left or right")
    trigger: float = Field(default=0.5, description="When it pops up, as a fraction 0.0-1.0 of THIS segment's duration")


class MetricPart(BaseModel):
    label: str = Field(description="What the number is, max 3 words, e.g. 'Share price'")
    value: str = Field(description="Display string, e.g. '$100'")


class ChartNote(BaseModel):
    anchor: str = Field(description="Where on the chart: high, low, start, end or biggest_drop")
    text: str = Field(description="Annotation text, max 4 words")


class Segment(BaseModel):
    layout: str = Field(description="One of: myth_vs_reality, big_calculator, comparison_bars, annotated_chart, "
                                    "metric_breakdown, plain_text")
    headline: str = Field(description="Short on-screen title for this segment, UPPERCASE, max 22 characters")
    script: str = Field(description="Spoken narration for this segment: 1 to 3 short sentences")
    accent: str = Field(default="gold", description="Accent colour: red, green, gold, cyan, amber, white or purple")
    popups: List[PopupSpec] = Field(default_factory=list, description="1 popups for this segment")

    # --- myth_vs_reality ---
    myth_text: str = Field(default="", description="What people think (max 14 words)")
    reality_text: str = Field(default="", description="The actual truth (max 18 words)")
    # --- big_calculator ---
    calc_label: str = Field(default="", description="What is being calculated (max 8 words)")
    calc_start: float = Field(default=0.0, description="Starting number, plain digits (0 if none)")
    calc_end: float = Field(default=0.0, description="Final number the big counter ends on")
    calc_prefix: str = Field(default="", description="Prefix such as '$' or ''")
    calc_suffix: str = Field(default="", description="Suffix such as '%', 'x' or ''")
    calc_caption: str = Field(default="", description="One line under the number (max 12 words)")
    # --- comparison_bars ---
    bar_labels: List[str] = Field(default_factory=list, description="Exactly 2 labels, max 3 words each")
    bar_values: List[float] = Field(default_factory=list, description="Exactly 2 positive numbers")
    bar_prefix: str = Field(default="", description="Prefix for the values, e.g. '$'")
    bar_suffix: str = Field(default="", description="Suffix for the values, e.g. '%'")
    bar_caption: str = Field(default="", description="Question or takeaway above the bars (max 10 words)")
    bar_colors: List[str] = Field(default_factory=list, description="2 colour names, one per bar")
    # --- annotated_chart ---
    chart_ticker: str = Field(default="", description="A real, liquid ticker, e.g. SPY, QQQ, AAPL, KO, ^GSPC")
    chart_timeframe: str = Field(default="1y", description="One of: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y")
    chart_notes: List[ChartNote] = Field(default_factory=list, description="0 to 3 annotations")
    # --- metric_breakdown ---
    metric_title: str = Field(default="", description="Name of the metric, e.g. 'P/E RATIO'")
    metric_parts: List[MetricPart] = Field(default_factory=list, description="2 or 3 inputs of the formula")
    metric_operator: str = Field(default="/", description="Operator between the inputs: '/', 'x', '+' or '-'")
    metric_result_label: str = Field(default="", description="Name of the result, max 3 words")
    metric_result_value: str = Field(default="", description="The result as a display string, e.g. '20'")
    metric_verdict: str = Field(default="", description="What it means in plain English (max 12 words)")
    # --- plain_text ---
    plain_text: str = Field(default="", description="Strong statement shown centred on screen (max 22 words)")
    plain_highlight: List[str] = Field(default_factory=list, description="Up to 3 words from plain_text to highlight")


class MythBusterScript(BaseModel):
    topic: str = Field(description="The myth being busted, as a catchy title, e.g. 'The 10% Dividend Trap'")
    news_ticker_text: str = Field(description=(
        "Single line, 3 or 4 punchy UPPERCASE items separated by '  \u2022  ', each 'TOPIC: ultra short summary'"))
    segments: List[Segment]
    youtube_title: str
    description: str
    tags: str


# Heavy dependencies are imported only where used (yfinance in the parent, Matplotlib in the render worker,
# Kokoro / Torch only in the TTS worker).
@lru_cache(maxsize=1)
def _yf():
    import yfinance as yf
    return yf


def load_render_dependencies() -> None:
    global Figure, FigureCanvasAgg, Rectangle, FancyBboxPatch, FuncFormatter
    if "Figure" not in globals():
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyBboxPatch, Rectangle
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
        cur = Path("/sys/fs/cgroup/memory.current")
        lim = Path("/sys/fs/cgroup/memory.max")
        total = int(cur.read_text()) / (1024 ** 2) if cur.exists() else None
        raw_lim = lim.read_text().strip() if lim.exists() else "max"
        max_mib = int(raw_lim) / (1024 ** 2) if raw_lim.isdecimal() else None
        fmt = lambda n: f"{n:.0f} MiB" if n is not None else "N/A"
        logger.info("RAM [%s] process=%s container=%s limit=%s", stage, fmt(rss_mib), fmt(total), fmt(max_mib))
    except Exception as e:
        logger.debug("RAM logging unavailable: %s", e)


def run_memory_worker(kind: str, *args: str) -> None:
    """Start one worker, wait for exit, reclaim its RAM before the next stage."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
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


def parallel_map(fn: Callable, items: Sequence, max_workers: int = IO_WORKERS) -> list:
    """ThreadPool map that preserves order (use for network-bound work)."""
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
    text = text or ""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        text = text.replace(src, dst)
    text = text.replace("\\", "")
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", "", text)).strip()


def shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:max(1, n - 3)].rstrip() + "..."


def first_sentence(text: str) -> str:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return parts[0] if parts else (text or "").strip()


def first_sentence_fraction(script: str, lo: float = 0.22, hi: float = 0.62) -> float:
    """Share of the narration taken by the first sentence (the myth); the 'reality' appears right after it."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (script or "").strip()) if s]
    if len(sentences) < 2:
        return 0.4
    return min(hi, max(lo, len(sentences[0].split()) / max(1, len(script.split()))))


def theme_color(name: Any, default: str = C_GOLD) -> str:
    return COLOR_THEMES.get(str(name or "").strip().lower(), default)


def norm_color_name(name: Any, default: str = "gold") -> str:
    n = str(name or "").strip().lower()
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


def ramp(t: float, start: float, length: float) -> float:
    return clamp01((t - start) / max(length, 1e-6))


def ease_out(p: float) -> float:
    return 1 - (1 - p) ** 3


def ease_out_back(p: float, k: float = 1.70158) -> float:
    p -= 1
    return 1 + (k + 1) * p ** 3 + k * p ** 2


def _decimals(*vals: float) -> int:
    for d in (0, 1, 2):
        if all(abs(v - round(v, d)) < 1e-6 for v in vals):
            return d
    return 2


def fmt_value(v: float, prefix: str, suffix: str, dec: int) -> str:
    sign = "-" if v < 0 and round(abs(v), dec) != 0 else ""
    return f"{sign}{prefix}{abs(v):,.{dec}f}{suffix}"


def _suffix(s: str) -> str:
    s = s or ""
    return " " + s if len(s) > 1 and s[0].isalpha() else s


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
# 4. TOPIC MEMORY + SCRIPT VALIDATION
# =====================================================================


def normalize_layout(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    if key in LAYOUTS:
        return key
    return LAYOUT_ALIASES.get(key.replace("_", ""), LAYOUT_ALIASES.get(key, "plain_text"))


def normalize_timeframe(tf: str) -> str:
    t = re.sub(r"\s+", "", (tf or "").lower())
    t = {"1m": "1mo", "1month": "1mo", "3m": "3mo", "6m": "6mo", "1yr": "1y", "12mo": "1y", "2yr": "2y",
         "5yr": "5y", "10yr": "10y", "ytd": "1y", "max": "10y"}.get(t, t)
    return t if t in TIMEFRAME_INTERVALS else "1y"


def downgrade_to_plain(seg: Segment, reason: str) -> None:
    """A layout that lacks its data becomes a centred statement taken from the narration."""
    logger.warning(f"Segment '{seg.headline}': {seg.layout} -> plain_text ({reason})")
    seg.layout = "plain_text"
    seg.plain_text = shorten(first_sentence(seg.script), 140)
    seg.plain_highlight = []


def _payload_problem(seg: Segment) -> Optional[str]:
    L = seg.layout
    if L == "myth_vs_reality" and (len(seg.myth_text) < 3 or len(seg.reality_text) < 3):
        return "missing myth/reality text"
    if L == "big_calculator" and seg.calc_end == seg.calc_start:
        return "calculator start equals end"
    if L == "comparison_bars":
        if len(seg.bar_labels) < 2 or len(seg.bar_values) < 2:
            return "needs 2 labels and 2 values"
        if min(seg.bar_values[:2]) < 0 or max(seg.bar_values[:2]) <= 0:
            return "bar values must be positive"
    if L == "annotated_chart" and not seg.chart_ticker:
        return "no ticker"
    if L == "metric_breakdown" and (not 2 <= len(seg.metric_parts) <= 3 or not seg.metric_result_value):
        return "needs 2-3 parts and a result"
    if L == "plain_text" and len(seg.plain_text) < 3:
        seg.plain_text = shorten(first_sentence(seg.script), 140)
    return None


def _clean_popups(popups: List[PopupSpec]) -> List[PopupSpec]:
    out = []
    for p in popups:
        title, value = shorten(ascii_clean(p.title).upper(), 22), shorten(ascii_clean(p.value), 12)
        if not value:
            continue
        pos = (p.position or "").strip().lower()
        p.title, p.value = title, value
        p.color = norm_color_name(p.color)
        p.position = pos if pos in POPUP_POSITIONS else "center"
        try:
            p.trigger = min(0.9, max(0.05, float(p.trigger)))
        except (TypeError, ValueError):
            p.trigger = 0.5
        out.append(p)
    return out[:MAX_POPUPS_PER_SEGMENT]


def normalize_script(script: MythBusterScript) -> MythBusterScript:
    """Cleans every field and repairs / rejects what the renderer cannot draw. Raising triggers an LLM retry."""
    segs = script.segments
    if not SEG_MIN <= len(segs) <= SEG_MAX:
        raise ValueError(f"expected {SEG_MIN}-{SEG_MAX} segments, got {len(segs)}")
    script.topic = shorten(ascii_clean(script.topic), 60)
    if len(script.topic) < 5:
        raise ValueError("empty topic")
    if not script.youtube_title.strip():
        raise ValueError("empty youtube_title")

    for seg in segs:
        seg.layout = normalize_layout(seg.layout)
        seg.script = ascii_clean(seg.script)
        if len(seg.script.split()) < 6:
            raise ValueError(f"segment script too short: {seg.script!r}")
        seg.headline = shorten(ascii_clean(seg.headline).upper(), 26) or DEFAULT_HEADLINES[seg.layout]
        seg.accent = norm_color_name(seg.accent)
        seg.popups = _clean_popups(seg.popups)

        seg.myth_text = shorten(ascii_clean(seg.myth_text), 120)
        seg.reality_text = shorten(ascii_clean(seg.reality_text), 140)
        seg.calc_label = shorten(ascii_clean(seg.calc_label), 70)
        seg.calc_prefix, seg.calc_suffix = ascii_clean(seg.calc_prefix)[:4], ascii_clean(seg.calc_suffix)[:10]
        seg.calc_caption = shorten(ascii_clean(seg.calc_caption), 90)
        seg.bar_labels = [shorten(ascii_clean(x), 26) for x in seg.bar_labels[:2]]
        seg.bar_values = [float(v) for v in seg.bar_values[:2]]
        seg.bar_prefix, seg.bar_suffix = ascii_clean(seg.bar_prefix)[:4], ascii_clean(seg.bar_suffix)[:10]
        seg.bar_caption = shorten(ascii_clean(seg.bar_caption), 80)
        seg.bar_colors = [norm_color_name(c, d) for c, d in zip(list(seg.bar_colors) + ["", ""], ("red", "green"))][:2]
        seg.chart_ticker = re.sub(r"[^A-Za-z0-9.\-^=]", "", seg.chart_ticker).upper()[:12]
        seg.chart_timeframe = normalize_timeframe(seg.chart_timeframe)
        for note in seg.chart_notes:
            note.text = shorten(ascii_clean(note.text), 32)
        seg.chart_notes = [n for n in seg.chart_notes if n.text][:3]
        seg.metric_title = shorten(ascii_clean(seg.metric_title).upper(), 30)
        for part in seg.metric_parts:
            part.label, part.value = shorten(ascii_clean(part.label), 28), shorten(ascii_clean(part.value), 14)
        seg.metric_parts = [p for p in seg.metric_parts if p.value][:3]
        seg.metric_result_label = shorten(ascii_clean(seg.metric_result_label), 28)
        seg.metric_result_value = shorten(ascii_clean(seg.metric_result_value), 14)
        seg.metric_verdict = shorten(ascii_clean(seg.metric_verdict), 90)
        seg.plain_text = shorten(ascii_clean(seg.plain_text), 150)
        seg.plain_highlight = [ascii_clean(w) for w in seg.plain_highlight if ascii_clean(w)][:3]

        problem = _payload_problem(seg)
        if problem:
            downgrade_to_plain(seg, problem)

    words = sum(len(s.script.split()) for s in segs)
    if not SCRIPT_WORDS_MIN <= words <= SCRIPT_WORDS_MAX:
        raise ValueError(f"narration has {words} words, need {SCRIPT_WORDS_MIN}-{SCRIPT_WORDS_MAX}")
    script.news_ticker_text = clean_ticker_text(script.news_ticker_text) or (
        f"MYTH BUSTED: {script.topic.upper()}  \u2022  FACTS OVER HYPE  \u2022  NOT FINANCIAL ADVICE")
    return script


def clean_ticker_text(text: str) -> str:
    return re.sub(r"[^\x20-\x7E\u2022]+", "", text or "").strip()


def parse_myth_script(response) -> MythBusterScript:
    return normalize_script(MythBusterScript.model_validate_json(response.text))


def generate_myth_script() -> MythBusterScript:
    logger.info("The AI director is inventing a myth and planning the video...")
    from google.genai import types
    areas = TOPIC_AREAS[datetime.date.today().timetuple().tm_yday - 1] if datetime.date.today().timetuple().tm_yday <= len(TOPIC_AREAS) else random.choice(TOPIC_AREAS)
    prompt = f"""
    You are the DIRECTOR of a viral YouTube Shorts channel called "Financial MythBusters".
    Invent ONE evergreen financial myth to bust, then direct the whole vertical video: the spoken script,
    the visual layout of every segment and the on-screen popup. The video must stay under 50 seconds.

    TOPIC
    - A widespread, genuinely wrong or misleading belief about money or investing that you can debunk with
      CORRECT, conservative facts. Evergreen only: no dates, no "today", no news, no live prices.
    - Inspiration (pick one, or something equally viral): 
    {areas}

    NARRATIVE ARC (THE "VERITASIUM" STRUCTURE)
    To maximize viewer retention, you must delay the final answer. Follow this exact flow across your {SEG_MIN} to {SEG_MAX} segments:
    - THE HOOK (Segment 1): Start with a scroll-stopping question that states the misconception. Make the viewer wonder if the myth might actually be true. DO NOT debunk it yet.
    - THE SETUP (Segment 2): Explain the intuition behind the myth. Why do so many people believe this? (Build the mystery).
    - THE EXPLANATION (Middle Segments): Break down the actual mechanics, math, or hidden reality. 
    - THE VERDICT & CTA (Last Segment): Give the conclusive, finalizing argument of why the myth is wrong based on your explanation. End with a punchy call to action (e.g., comment your thoughts / subscribe for more).

    SCRIPT FORMAT
    - All "script" fields combined must be exactly {SCRIPT_WORDS_MIN + 5} to {SCRIPT_WORDS_MAX - 5} words.
    - Keep every segment to 1 to 3 SHORT, punchy sentences.
    - Use at least 3 DIFFERENT layouts, never the same layout twice in a row, "annotated_chart" at most once.
    - Fill ONLY the fields of the chosen layout; leave the others empty / default.

    THE 6 LAYOUTS
    1. myth_vs_reality: split screen. myth_text = what people think (max 14 words), reality_text = the actual truth
       (max 18 words). In "script" the FIRST sentence states the myth; the reality panel appears right after it.
    2. big_calculator: massive counting number. calc_label (max 8 words), calc_start and calc_end as plain numbers
       (no symbols, no commas), calc_prefix ("$" or ""), calc_suffix ("%", "x" or ""), calc_caption (max 12 words).
       The arithmetic MUST be correct. Make clear in the script that figures are examples or illustrations.
    3. comparison_bars: animated 2-bar chart. bar_labels (exactly 2, max 3 words), bar_values (exactly 2 positive
       numbers), bar_prefix, bar_suffix, bar_caption (max 10 words), bar_colors (2 colour names).
    4. annotated_chart: a real stock chart fetched from Yahoo Finance. chart_ticker (a real, liquid ticker or index
       such as SPY, QQQ, AAPL, MSFT, KO, JNJ, ^GSPC), chart_timeframe (1mo, 3mo, 6mo, 1y, 2y, 5y or 10y),
       chart_notes: 1 to 3 annotations, anchor = high, low, start, end or biggest_drop, text max 4 words.
       Only choose it when a real price history genuinely supports the point. Never state prices in the script.
    5. metric_breakdown: deconstructs a formula such as P/E. metric_title, metric_parts (2 or 3 items with label and
       display value, e.g. "Share price" / "$100"), metric_operator ("/", "x", "+" or "-"), metric_result_label,
       metric_result_value, metric_verdict (max 12 words). The result MUST follow from the parts.
    6. plain_text: a strong centred statement. plain_text (max 22 words) and plain_highlight (up to 3 words from it).

    POPUPS (Only One)
    - title (max 3 words), value (max 10 characters, the punchy number or word being said at that moment),
      color (red, green, gold, cyan, amber, white or purple), position (top, center, bottom, left or right),
      trigger (0.0 to 1.0 = when inside THAT segment it appears; spread them between 0.15 and 0.85).
    - Vary colours and positions. Popups add emphasis, they never repeat the headline.

    TTS-FRIENDLY SCRIPT
    - Natural spoken English. Only periods, commas and question marks. NO dashes, ellipses, semicolons, parentheses,
      emojis or symbols. Write money in words ("ten thousand dollars"), percentages as "ten percent",
      and avoid decimal points. Use plain names, not ticker symbols.

    ON-SCREEN TEXT
    - Plain ASCII only, no emojis. Dollar signs are allowed ONLY in calc_prefix, bar_prefix, popup values and
      metric values. "headline" is UPPERCASE, max 22 characters.

    TICKER AND SEO
    - "news_ticker_text": 3 or 4 UPPERCASE items 'TOPIC: ultra short summary', separated by ' \u2022 ',
      evergreen myth-busting style, e.g. "MYTH ALERT: HIGH YIELD IS NOT FREE MONEY \u2022 FACTS OVER HYPE".
    - "youtube_title": max 70 characters. Talk directly to the audience, using the misconception to tell them they are wrong (e.g., "Why You're WRONG About [Topic]"). Emojis allowed here only.
    - "description": 2 sentences with key financial terms, a call to action, and the line "Not financial advice."
    - "tags": 5 to 8 comma separated tags.

    ACCURACY: educational, conservative, factually correct. No guarantees, no predictions, no personal advice.
    """
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=MythBusterScript,
                                         temperature=1.0)
    return ask_llm(prompt, config=config, parse=parse_myth_script)


# =====================================================================
# 5. MARKET DATA  (only for annotated_chart segments)
# =====================================================================
def fetch_chart_data(ticker: str, timeframe: str) -> Optional[dict]:
    """Compact, picklable price history for one chart segment. None if Yahoo has nothing usable."""
    tf = normalize_timeframe(timeframe)
    interval = TIMEFRAME_INTERVALS[tf]
    cols = ["Open", "High", "Low", "Close"]

    def _go():
        df = _yf().Ticker(ticker).history(period=tf, interval=interval).dropna(subset=cols)
        if len(df) < 8:
            raise RuntimeError(f"only {len(df)} rows")
        return df

    try:
        logger.info(f"Fetching chart data: {ticker} {tf} ({interval})")
        df = with_retries(_go, attempts=2, base_delay=1.5, what=f"{ticker} history")
    except Exception as e:
        logger.warning(f"No chart data for {ticker}: {e}")
        return None

    n = len(df)
    idxs = np.unique(np.linspace(0, n - 1, min(5, n)).astype(int))
    fmt = "%b %d" if tf in ("1mo", "3mo", "6mo") else "%b %Y"
    try:
        labels = [df.index[i].strftime(fmt) for i in idxs]
    except Exception:
        labels = [str(i) for i in idxs]
    return {
        "ticker": ticker, "timeframe": tf,
        "open": df["Open"].astype(float).tolist(), "high": df["High"].astype(float).tolist(),
        "low": df["Low"].astype(float).tolist(), "close": df["Close"].astype(float).tolist(),
        "tick_idx": [int(i) for i in idxs], "tick_labels": labels,
    }


# =====================================================================
# 6. TEXT-TO-SPEECH (Kokoro-82M, local, isolated worker, one clip per segment)
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


def generate_voiceovers(items: List[Tuple[str, str]], speed: float = TTS_SPEED) -> List[float]:
    """In-process ONNX TTS (no Torch, so no separate worker is needed)."""
    import soundfile as sf
    kokoro = get_kokoro_onnx()
    durations = []
    logger.info("Generating %d Kokoro clips via ONNX (speed %.2f)", len(items), speed)
    for text, path in items:
        samples, sample_rate = kokoro.create(clean_script_for_tts(text), voice=TTS_VOICE,
                                             speed=speed, lang="en-us")
        if len(samples) == 0:
            raise RuntimeError(f"Kokoro returned empty audio for {path}")
        sf.write(str(path), samples, sample_rate, subtype="PCM_16")
        durations.append(len(samples) / sample_rate)
        logger.info("Created %s (%.2fs)", Path(path).name, durations[-1])
    return durations


# =====================================================================
# 7. DRAWING PRIMITIVES
# =====================================================================
def new_figure(width_px: int = VIDEO_W, height_px: int = VIDEO_H, facecolor: str = C_BG):
    """Pyplot-free figure (no global state, safe in threads/processes).
    The +1e-3 guards against int() truncation giving a 1079/1919 px canvas."""
    load_render_dependencies()
    fig = Figure(figsize=((width_px + 1e-3) / DPI, (height_px + 1e-3) / DPI), dpi=DPI, facecolor=facecolor)
    return fig, FigureCanvasAgg(fig)


def grab(canvas, width_px: int = VIDEO_W, height_px: int = VIDEO_H) -> np.ndarray:
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


def blend_rgba(frame: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alpha-composite an (h, w, 4) uint8 image onto the frame in place (clipped to the frame)."""
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


def blend_mask(frame: np.ndarray, mask: np.ndarray, x: int, y: int, color: Tuple[int, int, int], alpha: float = 1.0) -> None:
    """Paint a solid colour through an (h, w) float32 0..1 mask, in place (clipped to the frame)."""
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
# Scrolling ticker (one text artist; scrolling = moving its x position)
# ---------------------------------------------------------------------
class NewsTickerBar:
    def __init__(self, fig, raw_text: str, label: str = "MYTH BUSTER", scroll_speed: float = 0.12, fontsize: int = 18):
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
    """Quantized bob states and Pillow compositing avoid a float64 cache explosion."""

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
    """The exact bold font matplotlib draws with, so text measured with Pillow fits what matplotlib renders."""
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


def fit_wrap(text: str, max_w: float, max_h: float, max_size: int, min_size: int = 16,
             pitch: float = 1.25) -> Tuple[List[str], int]:
    """Largest font size (reference px) at which `text` wraps into max_w x max_h. Returns (lines, size)."""
    size = int(max_size)
    while True:
        lines = _wrap_text(text, ref_font(size), max_w) or [""]
        if len(lines) * size * pitch <= max_h or size <= min_size:
            return lines, size
        size -= 2


def draw_lines(ax, lines: List[str], cx: float, cy: float, size: float, color: str, pitch: float = 1.25,
               ha: str = "center", **kw) -> List:
    """One text artist per line (explicit line pitch => exact positions for strike-throughs and fades)."""
    n = len(lines)
    return [ax.text(cx, cy + (i - (n - 1) / 2) * size * pitch, mpl_safe(line), fontsize=size * REF_PT,
                    fontweight="bold", color=color, ha=ha, va="center", **kw)
            for i, line in enumerate(lines)]


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
        """segments = [(text, start, end)]. Cues are contiguous (no flicker in the pauses)."""
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
# Bottom panel: avatar (bottom-left) + "THE BREAKDOWN" progress panel (bottom-right)
# ---------------------------------------------------------------------
def render_breakdown_panel(headlines: List[str], current: int, accent: str,
                           width_px: int = SIDEBAR_W, height_px: int = PANEL_SIZE) -> np.ndarray:
    fig, canvas = new_figure(width_px, height_px, facecolor=C_PANEL)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C_PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=C_BORDER, linewidth=6, transform=ax.transAxes))
    ax.text(0.5, 0.925, "THE BREAKDOWN", fontsize=22, fontweight="bold", color=C_WHITE, ha="center", va="center")
    ax.plot([0.08, 0.92], [0.865, 0.865], color=C_BORDER, linewidth=1.5)

    n = max(1, len(headlines))
    row_h = min(0.16, 0.66 / n)
    for i, head in enumerate(headlines):
        yc = 0.865 - row_h * (i + 0.5) - 0.02
        done, active = i < current, i == current
        col = accent if active else (C_GREEN if done else C_MUTED)
        filled = active or done
        ax.plot([0.11], [yc], marker="o", markersize=26, linestyle="none", markerfacecolor=col if filled else C_PANEL,
                markeredgecolor=col, markeredgewidth=2.5)
        ax.text(0.11, yc, str(i + 1), fontsize=13, fontweight="bold", color=C_BG if filled else C_MUTED,
                ha="center", va="center")
        ax.text(0.20, yc, mpl_safe(shorten(head, 22)), fontsize=15, fontweight="bold",
                color=C_WHITE if active else C_MUTED, va="center")
    ax.text(0.5, 0.06, "Educational only. Not financial advice.", fontsize=11, color=C_MUTED, ha="center", va="center")
    return grab(canvas, width_px, height_px)


class BottomPanel:
    SPEAKING_THRESHOLD = 0.08  # lower = mouth moves more easily (useful range 0.02 - 0.15)

    def __init__(self, sidebar_img: np.ndarray, audio_path: str):
        self.sidebar = sidebar_img
        self.avatar = AvatarPanel(PANEL_SIZE)
        self.peaks = get_audio_peaks(audio_path)

    def paint_sidebar(self, frame: np.ndarray) -> None:
        frame[PANEL_Y:PANEL_Y + PANEL_SIZE, SIDEBAR_X:SIDEBAR_X + SIDEBAR_W] = self.sidebar

    def paint_avatar(self, frame: np.ndarray, t: float) -> None:
        idx = int(t * OUTPUT_FPS)
        vol = float(self.peaks[idx]) if idx < len(self.peaks) else 0.0
        speaking = vol > self.SPEAKING_THRESHOLD
        bob_px = px(min(max(vol * 50, 4), 20)) if speaking else 0
        frame[PANEL_Y:PANEL_Y + PANEL_SIZE, AVATAR_X:AVATAR_X + PANEL_SIZE] = self.avatar.frame(speaking, bob_px)


# =====================================================================
# 8. POPUPS  (AI-scheduled: pop in, float, fade out)
# =====================================================================
POPUP_REGION = (px(40), px(335), px(1000), px(605))                        # popups live on the stage
POPUP_SIZES_REF = {"top": (560, 190), "center": (560, 190), "bottom": (560, 190), "left": (400, 190), "right": (400, 190)}


@dataclass
class PopupInstance:
    start: float
    end: float
    x: int
    y: int
    img: Any   # PIL RGBA image


def popup_origin(position: str, w: int, h: int) -> Tuple[int, int]:
    rx, ry, rw, rh = POPUP_REGION
    m = px(18)
    if position == "top":
        return rx + (rw - w) // 2, ry + m
    if position == "bottom":
        return rx + (rw - w) // 2, ry + rh - h - m
    if position == "left":
        return rx + m, ry + (rh - h) // 2
    if position == "right":
        return rx + rw - w - m, ry + (rh - h) // 2
    return rx + (rw - w) // 2, ry + (rh - h) // 2


def render_popup_card(title: str, value: str, color_hex: str, w: int, h: int) -> Optional[Image.Image]:
    """Dark rounded card with a coloured border, small title and a big value (2x supersampled for smooth edges)."""
    try:
        S = 2
        W, H = w * S, h * S
        col = _rgb(color_hex)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, W - 1, H - 1), radius=int(H * 0.18), fill=(11, 14, 20, 244), outline=col,
                            width=max(4, int(H * 0.04)))
        pad = int(W * 0.07)
        title_font = load_font(max(10, int(H * 0.16)))
        d.text((W // 2, int(H * 0.22)), _fit_text(d, title, title_font, W - 2 * pad), font=title_font,
               fill=_rgb(blend_hex(C_WHITE, C_BG, 0.25)), anchor="mm")
        size = int(H * 0.46)
        while size > 10 and d.textlength(value, font=load_font(size)) > W - 2 * pad:
            size -= 2
        d.text((W // 2, int(H * 0.62)), value, font=load_font(size), fill=col, anchor="mm")
        return img.resize((w, h), Image.LANCZOS)
    except Exception as e:
        logger.warning(f"Could not render popup '{title}/{value}': {e}")
        return None


def plan_popups(specs: List[dict], duration: float) -> List[PopupInstance]:
    """Turns the AI's popup specs into timed, positioned cards; popups that would collide are pushed back."""
    placed: List[PopupInstance] = []
    last_end = duration - 0.05
    for sp in sorted(specs, key=lambda s: s["trigger"])[:MAX_POPUPS_PER_SEGMENT]:
        position = sp.get("position", "center")
        w, h = (px(v) for v in POPUP_SIZES_REF.get(position, (560, 190)))
        img = render_popup_card(sp.get("title", ""), sp.get("value", ""), theme_color(sp.get("color")), w, h)
        if img is None:
            continue
        x, y = popup_origin(position, w, h)
        start = clamp01(float(sp["trigger"])) * duration
        end = min(start + POPUP_SECONDS, last_end)
        if end - start < POPUP_MIN_SECONDS:
            start = max(0.0, end - POPUP_MIN_SECONDS)
        for _ in range(2):  # push back behind any earlier popup that overlaps in time AND on screen
            for q in placed:
                qw, qh = q.img.size
                overlap_xy = x < q.x + qw and q.x < x + w and y < q.y + qh and q.y < y + h
                if overlap_xy and start < q.end and end > q.start:
                    start = q.end + 0.12
                    end = min(start + POPUP_SECONDS, last_end)
        if end - start >= POPUP_MIN_SECONDS:
            placed.append(PopupInstance(start, end, x, y, img))
    return placed


class PopupLayer:
    def __init__(self, popups: List[PopupInstance]):
        self.popups = popups

    def paint(self, frame: np.ndarray, t: float) -> None:
        for p in self.popups:
            if not p.start <= t < p.end:
                continue
            age, left = t - p.start, p.end - t
            s = ease_out_back(clamp01(age / POPUP_IN))
            alpha = clamp01(age / (POPUP_IN * 0.8)) * clamp01(left / POPUP_OUT)
            scale = 0.55 + 0.45 * s
            w, h = p.img.size
            img = p.img
            if abs(scale - 1.0) > 0.015:
                img = p.img.resize((max(2, int(w * scale)), max(2, int(h * scale))), Image.BILINEAR)
            arr = np.asarray(img)
            bob = int(round(px(5) * math.sin(2 * math.pi * age * 0.9)))
            cx, cy = p.x + w // 2, p.y + h // 2 + bob
            blend_rgba(frame, arr, cx - arr.shape[1] // 2, cy - arr.shape[0] // 2, alpha)


# =====================================================================
# 9. THE SIX LAYOUTS
#    Each layout builds its artists ONCE, then update(t) tweaks them and paint(frame, t) may add Pillow overlays.
#    `t` is segment-local; animations are paced to `self.dur` (the speech length, not the silent tail).
# =====================================================================
def make_stage_axes(fig, box: Tuple[int, int, int, int], ref_w: float, ref_h: float):
    """Axes whose data units are reference pixels, y pointing DOWN (so you can author like a web page)."""
    ax = fig.add_axes(axes_rect(box))
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, ref_w)
    ax.set_ylim(ref_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(C_BORDER)
        spine.set_linewidth(2)
    return ax


class Reveal:
    """A group of artists that fades in together."""

    def __init__(self, start: float, length: float = 0.35):
        self.start, self.length, self.artists = start, length, []

    def add(self, *arts):
        for a in arts:
            if isinstance(a, (list, tuple)):
                self.artists.extend(a)
            else:
                self.artists.append(a)
        return self

    def apply(self, t: float, mult: float = 1.0) -> float:
        a = ease_out(ramp(t, self.start, self.length)) * mult
        for art in self.artists:
            art.set_alpha(a)
        return a


class LayoutBase:
    header_sub: Optional[str] = None                      # replaces the muted line under the headline
    header_right: Optional[Tuple[str, str, str]] = None   # (line 1, line 2, line-2 colour)

    def __init__(self, fig, seg: dict, duration: float, speech: float, data: Optional[dict] = None):
        self.fig, self.seg = fig, seg
        self.duration = duration
        self.dur = max(speech, 0.5)
        self.accent = theme_color(seg.get("accent"), C_GOLD)

    def update(self, t: float) -> None:
        pass

    def paint(self, frame: np.ndarray, t: float) -> None:
        pass


# --- 1. MYTH VS REALITY ------------------------------------------------
class MythVsRealityLayout(LayoutBase):
    """Top panel 'WHAT PEOPLE THINK' gets struck through and dimmed; the green 'TRUTH' panel appears after the
    first spoken sentence."""

    def __init__(self, fig, seg, duration, speech, data=None):
        super().__init__(fig, seg, duration, speech, data)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        ph, margin = 268, 14
        y_myth, y_real = margin, H - margin - ph
        self.reveal_at = self.dur * first_sentence_fraction(seg["script"])

        rounding = "round,pad=0,rounding_size=26"
        myth_box = FancyBboxPatch((margin, y_myth), W - 2 * margin, ph, boxstyle=rounding,
                                  fc=blend_hex(C_RED, C_BG, 0.86), ec=C_RED, lw=3)
        real_box = FancyBboxPatch((margin, y_real), W - 2 * margin, ph, boxstyle=rounding,
                                  fc=blend_hex(C_GREEN, C_BG, 0.88), ec=C_GREEN, lw=3)
        ax.add_patch(myth_box)
        ax.add_patch(real_box)

        myth_lbl = ax.text(48, y_myth + 34, "WHAT PEOPLE THINK", fontsize=24 * REF_PT, fontweight="bold",
                           color=C_RED, ha="left", va="center")
        real_lbl = ax.text(48, y_real + 34, "THE ACTUAL TRUTH", fontsize=24 * REF_PT, fontweight="bold",
                           color=C_GREEN, ha="left", va="center")

        text_h = ph - 78
        m_lines, m_size = fit_wrap(seg["myth_text"], 880, text_h, 54, 22)
        r_lines, r_size = fit_wrap(seg["reality_text"], 880, text_h, 54, 22)
        m_cy, r_cy = y_myth + 58 + text_h / 2, y_real + 58 + text_h / 2
        myth_arts = draw_lines(ax, m_lines, W / 2, m_cy, m_size, C_WHITE)
        real_arts = draw_lines(ax, r_lines, W / 2, r_cy, r_size, C_WHITE)

        # one red strike per line of the myth text, drawn progressively
        self.strikes = []
        n = len(m_lines)
        for i, line in enumerate(m_lines):
            hw = ref_font(m_size).getlength(line) / 2 + 12
            y = m_cy + (i - (n - 1) / 2) * m_size * 1.25
            ln, = ax.plot([], [], color=C_RED, linewidth=5, solid_capstyle="round", zorder=8)
            self.strikes.append((ln, W / 2 - hw, W / 2 + hw, y))

        vs_y = (y_myth + ph + y_real) / 2
        vs_dot, = ax.plot([W / 2], [vs_y], marker="o", markersize=44, linestyle="none", markerfacecolor=C_BG,
                          markeredgecolor=C_GOLD, markeredgewidth=3, zorder=9)
        vs_txt = ax.text(W / 2, vs_y, "VS", fontsize=26 * REF_PT, fontweight="bold", color=C_GOLD,
                         ha="center", va="center", zorder=10)

        self.myth = Reveal(0.05, 0.45).add(myth_box, myth_lbl, myth_arts)
        self.real = Reveal(self.reveal_at, 0.45).add(real_box, real_lbl, real_arts)
        self.vs = Reveal(self.reveal_at, 0.3).add(vs_dot, vs_txt)

    def update(self, t: float) -> None:
        dim = 1 - 0.5 * ease_out(ramp(t, self.reveal_at, 0.45))
        self.myth.apply(t, dim)
        self.real.apply(t)
        self.vs.apply(t)
        p = ease_out(ramp(t, self.reveal_at + 0.05, 0.4))
        for ln, x0, x1, y in self.strikes:
            ln.set_data([x0, x0 + (x1 - x0) * p], [y, y])


# --- 2. BIG CALCULATOR -------------------------------------------------
class BigCalculatorLayout(LayoutBase):
    """Huge counter that counts from calc_start to calc_end, with an automatic % change pill."""

    def __init__(self, fig, seg, duration, speech, data=None):
        super().__init__(fig, seg, duration, speech, data)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        self.start, self.end = float(seg["calc_start"]), float(seg["calc_end"])
        self.prefix, self.suffix = seg["calc_prefix"], _suffix(seg["calc_suffix"])
        self.dec = _decimals(self.start, self.end)
        has_start = self.start != 0
        self.reveal = []

        label_lines, label_size = fit_wrap(seg["calc_label"].upper(), 920, 84, 34, 20)
        lbl = draw_lines(ax, label_lines, W / 2, 52, label_size, C_MUTED)
        self.reveal.append(Reveal(0.0, 0.4).add(lbl))

        big_cy = 355 if has_start else 215
        pill_y = big_cy + 112
        cap_cy = pill_y + 82 if has_start else pill_y + 88

        if has_start:
            s_txt = ax.text(W / 2, 156, mpl_safe(fmt_value(self.start, self.prefix, self.suffix, self.dec)),
                            fontsize=64 * REF_PT, fontweight="bold", color=C_MUTED, ha="center", va="center")
            arrow = ax.annotate("", xy=(W / 2, 258), xytext=(W / 2, 204),
                                arrowprops=dict(arrowstyle="-|>", color=self.accent, lw=5, mutation_scale=34))
            self.reveal.append(Reveal(0.12, 0.4).add(s_txt, arrow.arrow_patch))

        widest = max((fmt_value(v, self.prefix, self.suffix, self.dec) for v in (self.start, self.end)), key=len)
        size = 210
        while size > 40 and ref_font(size).getlength(widest) > 900:
            size -= 4
        self.big_size = size
        self.big = ax.text(W / 2, big_cy, "", fontsize=size * REF_PT, fontweight="bold", color=self.accent,
                           ha="center", va="center", zorder=5)

        self.pill = None
        if has_start:
            pct = (self.end - self.start) / abs(self.start) * 100
            pill_box = FancyBboxPatch((W / 2 - 150, pill_y - 30), 300, 60, boxstyle="round,pad=0,rounding_size=30",
                                      fc=blend_hex(self.accent, C_BG, 0.82), ec=self.accent, lw=3)
            ax.add_patch(pill_box)
            pill_txt = ax.text(W / 2, pill_y, f"{pct:+,.0f}%", fontsize=34 * REF_PT, fontweight="bold",
                               color=self.accent, ha="center", va="center")
            self.pill = Reveal(self.dur * 0.68, 0.35).add(pill_box, pill_txt)

        cap_lines, cap_size = fit_wrap(seg["calc_caption"], 920, 96, 34, 20)
        cap = draw_lines(ax, cap_lines, W / 2, cap_cy, cap_size, C_WHITE)
        self.reveal.append(Reveal(self.dur * 0.55, 0.45).add(cap))
        self.count_start, self.count_len = self.dur * 0.06, self.dur * 0.62

    def update(self, t: float) -> None:
        for r in self.reveal:
            r.apply(t)
        if self.pill:
            self.pill.apply(t)
        p = ease_out(ramp(t, self.count_start, self.count_len))
        self.big.set_text(mpl_safe(fmt_value(self.start + (self.end - self.start) * p, self.prefix, self.suffix, self.dec)))
        self.big.set_alpha(ease_out(ramp(t, 0.0, 0.25)))
        pop = 1 + 0.07 * math.sin(math.pi * ramp(t, self.count_start + self.count_len, 0.35))
        self.big.set_fontsize(self.big_size * REF_PT * pop)


# --- 3. COMPARISON BARS ------------------------------------------------
class ComparisonBarsLayout(LayoutBase):
    """Two bars that grow one after the other, counting their values up; a ratio pill appears at the end."""

    def __init__(self, fig, seg, duration, speech, data=None):
        super().__init__(fig, seg, duration, speech, data)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        self.values = [float(v) for v in seg["bar_values"][:2]]
        labels = seg["bar_labels"][:2]
        cols = [theme_color(c, d) for c, d in zip((list(seg.get("bar_colors") or []) + ["", ""])[:2], (C_RED, C_GREEN))]
        self.prefix, self.suffix = seg["bar_prefix"], _suffix(seg["bar_suffix"])
        self.dec = _decimals(*self.values)
        self.base, self.max_h, self.bar_w = 466, 290, 280
        centers = [290, 710]
        vmax = max(self.values)

        cap_lines, cap_size = fit_wrap(seg["bar_caption"], 920, 84, 38, 22)
        self.cap = Reveal(0.0, 0.4).add(draw_lines(ax, cap_lines, W / 2, 48, cap_size, C_WHITE))
        ax.plot([50, 950], [self.base, self.base], color=C_BORDER, linewidth=3)

        self.bars = []
        for i in range(2):
            rect = Rectangle((centers[i] - self.bar_w / 2, self.base), self.bar_w, 0, facecolor=cols[i], edgecolor="none")
            ax.add_patch(rect)
            val = ax.text(centers[i], self.base - 12, "", fontsize=52 * REF_PT, fontweight="bold", color=cols[i],
                          ha="center", va="bottom", zorder=6)
            l_lines, l_size = fit_wrap(labels[i], self.bar_w + 90, 96, 32, 18)
            lab = draw_lines(ax, l_lines, centers[i], 528, l_size, C_WHITE)
            t0 = (0.06, 0.42)[i] * self.dur
            self.bars.append(dict(rect=rect, val=val, label=Reveal(t0, 0.4).add(lab), t0=t0, t1=t0 + 0.34 * self.dur,
                                  h=max(14.0, self.max_h * self.values[i] / vmax), v=self.values[i]))

        self.pill = None
        lo, hi = min(self.values), max(self.values)
        if lo > 0 and hi / lo >= 1.15:
            ratio = hi / lo
            text = f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"
            box = FancyBboxPatch((W / 2 - 66, self.base - 190), 132, 56, boxstyle="round,pad=0,rounding_size=28",
                                 fc=C_BG, ec=C_GOLD, lw=3, zorder=7)
            ax.add_patch(box)
            txt = ax.text(W / 2, self.base - 162, text, fontsize=30 * REF_PT, fontweight="bold", color=C_GOLD,
                          ha="center", va="center", zorder=8)
            self.pill = Reveal(self.dur * 0.82, 0.3).add(box, txt)

    def update(self, t: float) -> None:
        self.cap.apply(t)
        for b in self.bars:
            p = ease_out(ramp(t, b["t0"], b["t1"] - b["t0"]))
            h = b["h"] * p
            b["rect"].set_y(self.base - h)
            b["rect"].set_height(h)
            b["val"].set_position((b["rect"].get_x() + self.bar_w / 2, self.base - h - 12))
            b["val"].set_text(mpl_safe(fmt_value(b["v"] * p, self.prefix, self.suffix, self.dec)))
            b["val"].set_alpha(clamp01(p * 4))
            b["label"].apply(t)
        if self.pill:
            self.pill.apply(t)


# --- 4. ANNOTATED CHART ------------------------------------------------
class AnnotatedChartLayout(LayoutBase):
    """Real yfinance data. Candles (<= 90 bars) or a glowing line (long timeframes) draw in, with a live price tag,
    a pulsing start-of-period line and AI-chosen annotations that pop when the chart reaches them."""
    CANDLE_MAX = 90
    NOTE_COLORS = {"high": C_GREEN, "low": C_RED, "biggest_drop": C_RED, "drop": C_RED, "start": C_CYAN, "end": C_GOLD}

    def __init__(self, fig, seg, duration, speech, data=None):
        super().__init__(fig, seg, duration, speech, data)
        o = np.asarray(data["open"], dtype=float)
        h = np.asarray(data["high"], dtype=float)
        lo = np.asarray(data["low"], dtype=float)
        c = np.asarray(data["close"], dtype=float)
        n = self.n = len(c)
        self.closes = c
        self.candles = n <= self.CANDLE_MAX
        self.trend = C_GREEN if c[-1] >= c[0] else C_RED
        x_idx = np.arange(n)

        ax = self.ax = fig.add_axes(axes_rect(CHART_BOX))
        style_axes(ax)
        y_min, y_max = float(lo.min()), float(h.max())
        rng = max(y_max - y_min, y_max * 0.002, 0.01)
        self.level = float(c[0])
        view_lo, view_hi = min(y_min, self.level), max(y_max, self.level)
        self.view_range = max(view_hi - view_lo, 0.01)

        if self.candles:
            colors = list(np.where(c >= o, C_GREEN, C_RED))
            self.segs = [[(float(i), lo[i]), (float(i), h[i])] for i in range(n)]
            self.wicks = ax.vlines(x_idx, lo, h, color=colors, linewidth=1.4)
            body = c - o
            tiny = rng * 0.002
            body = np.where(np.abs(body) < tiny, tiny, body)          # doji candles stay visible
            self.bodies = ax.bar(x_idx, body, bottom=o, color=colors, width=0.7)
        else:
            self.line_glow, = ax.plot([], [], color=self.trend, linewidth=9, alpha=0.18, solid_capstyle="round")
            self.line, = ax.plot([], [], color=self.trend, linewidth=3.6, solid_capstyle="round")
        self.x_min = -1.0
        self.x_max = n - 1 + max(2, int(n * 0.05))
        span = self.x_max - self.x_min
        self.y_lo = view_lo - self.view_range * 0.22
        self.y_hi = view_hi + self.view_range * 0.22
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_lo, self.y_hi)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
        ax.set_xticks(data["tick_idx"])
        ax.set_xticklabels(data["tick_labels"])

        self.cur = "" if data["ticker"].startswith("^") else "$"
        label_x = self.x_max - span * 0.02
        self._label_x = label_x
        self.level_line = ax.axhline(y=self.level, color=C_GOLD, linestyle="--", linewidth=2.2)
        self.level_label = ax.text(label_x, self.level, mpl_safe(f"Start: {self.cur}{self.level:,.2f}"), color=C_GOLD,
                                   fontsize=14, fontweight="bold", ha="right", va="top",
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=C_BG, edgecolor=C_GOLD, alpha=0.85))
        self.live_line = ax.axhline(y=c[0], color=self.trend, linestyle=":", linewidth=1.6, alpha=0.6)
        self.halo, = ax.plot([], [], marker="o", markersize=20, linestyle="none", color=self.trend,
                             markeredgewidth=0, alpha=0.3, zorder=9)
        self.dot, = ax.plot([], [], marker="o", markersize=8, linestyle="none", color=self.trend,
                            markeredgewidth=0, zorder=10)
        self.tag = ax.text(label_x, c[0], "", color=C_BG, fontsize=15, fontweight="bold", ha="right", va="bottom",
                           zorder=11, bbox=dict(boxstyle="round,pad=0.25", facecolor=self.trend, edgecolor="none", alpha=0.95))

        self.notes = []
        used = set()
        for note in seg.get("chart_notes", []):
            anchor = note["anchor"].strip().lower()
            idx, y = self._anchor(anchor, h, lo, c)
            if idx in used:
                continue
            used.add(idx)
            col = self.NOTE_COLORS.get(anchor, C_GOLD)
            side = -1 if idx > n * 0.5 else 1
            tx = min(max(idx + side * 0.2 * span, self.x_min + 0.16 * span), self.x_max - 0.16 * span)
            up = anchor in ("high", "start", "end")
            ty = y + (0.15 if up else -0.15) * self.view_range
            ty = min(max(ty, self.y_lo + 0.07 * self.view_range), self.y_hi - 0.07 * self.view_range)
            ann = ax.annotate(mpl_safe("\n".join(_wrap_text(note["text"], ref_font(16), 190))), xy=(idx, y),
                              xytext=(tx, ty), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5),
                              fontsize=15, fontweight="bold", color=col, ha="center", va="center", zorder=12,
                              annotation_clip=False,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BG, edgecolor=col, alpha=0.9))
            ann.set_alpha(0.0)
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_alpha(0.0)
            self.notes.append(dict(ann=ann, idx=idx, t0=None))

        pct = (c[-1] / c[0] - 1) * 100
        self.header_sub = f"{data['ticker']}  |  {data['timeframe'].upper()}"
        self.header_right = (f"{self.cur}{c[-1]:,.2f}", f"{pct:+.2f}%", self.trend)
        self._shown = None

    @staticmethod
    def _anchor(anchor: str, highs, lows, closes) -> Tuple[int, float]:
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

    def _visible(self, t: float) -> int:
        p = ease_out(min(t / max(self.dur * 0.55, 1e-6), 1.0))
        return min(self.n, max(2, int(round(p * self.n))))

    def update(self, t: float) -> None:
        k = self._visible(t)
        if k != self._shown:
            if self.candles:
                self.wicks.set_segments(self.segs[:k])
                for i, rect in enumerate(self.bodies):
                    rect.set_visible(i < k)
            else:
                xs = np.arange(k)
                self.line.set_data(xs, self.closes[:k])
                self.line_glow.set_data(xs, self.closes[:k])
            self._shown = k
        last = float(self.closes[k - 1])
        above = last >= self.level
        off = self.view_range * 0.015

        self.live_line.set_ydata([last, last])
        self.dot.set_data([k - 1], [last])
        self.halo.set_data([k - 1], [last])
        self.halo.set_alpha(0.12 + 0.28 * abs(math.sin(math.pi * t * 1.2)))
        self.tag.set_text(mpl_safe(f" {self.cur}{last:,.2f} "))
        self.tag.set_position((self._label_x, last + (off if above else -off)))
        self.tag.set_va("bottom" if above else "top")
        self.level_line.set_alpha(0.45 + 0.45 * abs(math.sin(math.pi * t * 1.5)))
        self.level_label.set_position((self._label_x, self.level + (-off if above else off)))
        self.level_label.set_va("top" if above else "bottom")

        for note in self.notes:
            if note["t0"] is None and k - 1 >= note["idx"]:
                note["t0"] = t + 0.15
            a = 0.0 if note["t0"] is None else ease_out(ramp(t, note["t0"], 0.3))
            note["ann"].set_alpha(a)
            if note["ann"].arrow_patch is not None:
                note["ann"].arrow_patch.set_alpha(a)


# --- 5. METRIC BREAKDOWN -----------------------------------------------
class MetricBreakdownLayout(LayoutBase):
    """Formula deconstruction: input cards appear one by one with the operator between them, then '=' and the result."""

    def __init__(self, fig, seg, duration, speech, data=None):
        super().__init__(fig, seg, duration, speech, data)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        parts = seg["metric_parts"][:3]
        n = len(parts)
        glyph = OPERATOR_GLYPHS.get(str(seg.get("metric_operator", "/")).strip().lower(), "\u00f7")
        round_style = "round,pad=0,rounding_size=22"

        self.groups: List[Reveal] = []
        title = ax.text(W / 2, 46, mpl_safe(seg["metric_title"] or "THE FORMULA"), fontsize=36 * REF_PT,
                        fontweight="bold", color=C_CYAN, ha="center", va="center")
        self.groups.append(Reveal(0.0, 0.4).add(title))

        margin, op_w = 30, (90 if n == 2 else 60)
        card_w = (W - 2 * margin - (n - 1) * op_w) / n
        row_y, card_h = 96, 196
        fracs = {2: (0.06, 0.30), 3: (0.05, 0.23, 0.41)}.get(n, (0.06,) * n)
        for i, part in enumerate(parts):
            x = margin + i * (card_w + op_w)
            box = FancyBboxPatch((x, row_y), card_w, card_h, boxstyle=round_style, fc=C_PANEL, ec=C_CYAN, lw=2.5)
            ax.add_patch(box)
            v_lines, v_size = fit_wrap(part["value"], card_w - 34, 84, 72, 24)
            vals = draw_lines(ax, v_lines, x + card_w / 2, row_y + 76, v_size, C_GOLD)
            l_lines, l_size = fit_wrap(part["label"], card_w - 34, 58, 28, 16)
            labs = draw_lines(ax, l_lines, x + card_w / 2, row_y + 156, l_size, C_MUTED)
            grp = Reveal(self.dur * fracs[i], 0.35).add(box, vals, labs)
            if i > 0:
                grp.add(ax.text(x - op_w / 2, row_y + card_h / 2, glyph, fontsize=66 * REF_PT, fontweight="bold",
                                color=C_WHITE, ha="center", va="center"))
            self.groups.append(grp)

        res_t = self.dur * 0.62
        eq = ax.text(W / 2, 326, "=", fontsize=60 * REF_PT, fontweight="bold", color=C_MUTED, ha="center", va="center")
        res_box = FancyBboxPatch((170, 366), 660, 150, boxstyle=round_style, fc=blend_hex(self.accent, C_BG, 0.85),
                                 ec=self.accent, lw=3.5)
        ax.add_patch(res_box)
        r_lines, r_size = fit_wrap(seg["metric_result_value"], 600, 84, 84, 28)
        res_val = draw_lines(ax, r_lines, W / 2, 366 + 62, r_size, self.accent)
        rl_lines, rl_size = fit_wrap(seg["metric_result_label"] or seg["metric_title"], 600, 44, 28, 16)
        res_lab = draw_lines(ax, rl_lines, W / 2, 366 + 122, rl_size, C_WHITE)
        self.groups.append(Reveal(res_t, 0.4).add(eq, res_box, res_val, res_lab))

        v_lines, v_size = fit_wrap(seg["metric_verdict"], 940, 72, 32, 18)
        self.groups.append(Reveal(self.dur * 0.8, 0.45).add(draw_lines(ax, v_lines, W / 2, 565, v_size, C_WHITE)))

    def update(self, t: float) -> None:
        for g in self.groups:
            g.apply(t)


# --- 6. PLAIN TEXT -----------------------------------------------------
class PlainTextLayout(LayoutBase):
    """Big centred, word-wrapped statement. Words fade in one by one; chosen words are highlighted in the accent
    colour. Drawn with Pillow (per-word colours + exact wrapping) and composited in paint()."""

    def __init__(self, fig, seg, duration, speech, data=None):
        super().__init__(fig, seg, duration, speech, data)
        W, H = STAGE_REF
        ax = self.ax = make_stage_axes(fig, STAGE_BOX, W, H)
        for y in (34, H - 34):
            ax.plot([70, W - 70], [y, y], color=self.accent, linewidth=3, alpha=0.85, solid_capstyle="round")

        bx, by, bw, bh = STAGE_BOX
        box_x, box_y, box_w, box_h = bx + px(70), by + px(70), px(860), px(465)
        text = seg["plain_text"]
        highlight = {re.sub(r"[^\w$%]", "", w).lower() for w in seg.get("plain_highlight", [])} - {""}
        accent_rgb, white_rgb = _rgb(self.accent), _rgb(C_WHITE)

        size, min_size = px(84), max(14, px(30))
        while True:
            font = load_font(size)
            lines = _wrap_text(text, font, box_w)
            line_h = int(_font_height(font) * 1.25)
            if len(lines) * line_h <= box_h or size <= min_size:
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
        n_words = len(self.words)
        for i, w in enumerate(self.words):
            w["start"] = self.dur * (0.03 + 0.62 * i / max(1, n_words - 1))

    def paint(self, frame: np.ndarray, t: float) -> None:
        for w in self.words:
            a = ease_out(ramp(t, w["start"], 0.22))
            if a > 0:
                blend_mask(frame, w["mask"], w["x"], w["y"] + int(round((1 - a) * px(16))), w["color"], a)


LAYOUT_CLASSES = {
    "myth_vs_reality": MythVsRealityLayout,
    "big_calculator": BigCalculatorLayout,
    "comparison_bars": ComparisonBarsLayout,
    "annotated_chart": AnnotatedChartLayout,
    "metric_breakdown": MetricBreakdownLayout,
    "plain_text": PlainTextLayout,
}


# =====================================================================
# 10. THE SEGMENT SCENE  (header + ticker + AI layout + popups + subtitles + avatar + breakdown panel)
#     frame_at(t) -> RGB uint8 array (H, W, 3); t is scene-local. Treat the result as read-only.
# =====================================================================
class MythSegmentScene:
    def __init__(self, seg: dict, index: int, n_segments: int, topic: str, headlines: List[str],
                 chart_data: Optional[dict], duration: float, speech_dur: float, global_start: float,
                 ticker_text: str, audio_path: str):
        self.duration = duration
        self.global_start = global_start
        self.cache = FrameCache()

        self.fig, self.canvas = new_figure()
        self.ticker = NewsTickerBar(self.fig, ticker_text)
        layout_cls = LAYOUT_CLASSES.get(seg["layout"], PlainTextLayout)
        self.layout = layout_cls(self.fig, seg, duration, speech_dur, chart_data)

        # --- header ---
        head = seg["headline"]
        n = len(head)
        size = 42 if n <= 14 else 36 if n <= 18 else 30 if n <= 22 else 26
        self.fig.text(fig_x(px(40)), fig_y(HEADER_TOP), mpl_safe(head), fontsize=size, fontweight="bold",
                      color=C_WHITE, va="top")
        sub = self.layout.header_sub or f"MYTH BUSTER  |  {shorten(topic, 34)}"
        self.fig.text(fig_x(px(40)), fig_y(HEADER_TOP + px(76)), mpl_safe(sub), fontsize=20, color=C_MUTED, va="top")
        line1, line2, col2 = self.layout.header_right or (f"{index + 1}/{n_segments}", "STEP", C_MUTED)
        self.fig.text(fig_x(px(1040)), fig_y(HEADER_TOP), mpl_safe(line1), fontsize=38, fontweight="bold",
                      color=C_WHITE, ha="right", va="top")
        self.fig.text(fig_x(px(1040)), fig_y(HEADER_TOP + px(70)), mpl_safe(line2), fontsize=26, fontweight="bold",
                      color=col2, ha="right", va="top")

        # --- overlays ---
        self.subtitles = SubtitleTrack([(seg["script"], 0.0, speech_dur)], duration)
        accent = theme_color(seg.get("accent"), C_GOLD)
        self.panel = BottomPanel(render_breakdown_panel(headlines, index, accent), audio_path)
        self.popups = PopupLayer(plan_popups(seg.get("popups", []), duration))

    def _render_chart(self, t: float) -> np.ndarray:
        self.layout.update(t)
        self.ticker.update(self.global_start + t)
        frame = grab(self.canvas)
        self.layout.paint(frame, t)
        self.panel.paint_sidebar(frame)
        return frame

    def frame_at(self, t: float) -> np.ndarray:
        frame = self.cache.get(t, self._render_chart).copy()   # we paint on it, so own a copy
        self.popups.paint(frame, t)
        self.panel.paint_avatar(frame, t)
        return self.subtitles.overlay(frame, t, inplace=True)


# =====================================================================
# 11. VIDEO WRITING  (frames are piped straight to ffmpeg; one scene per isolated process)
# =====================================================================
@dataclass
class SceneJob:
    kind: str                      # "segment"
    out_path: str
    start_time: float              # position on the global timeline (seconds)
    duration: float
    ffmpeg_threads: int = 2
    args: Dict[str, Any] = field(default_factory=dict)


def build_scene(job: SceneJob):
    a = job.args
    if job.kind == "segment":
        return MythSegmentScene(a["seg"], a["index"], a["n_segments"], a["topic"], a["headlines"], a.get("chart_data"),
                                job.duration, a["speech_dur"], job.start_time, a["ticker_text"], a["audio_path"])
    raise ValueError(f"Unknown scene kind: {job.kind}")


def frame_range(start_time: float, duration: float) -> Tuple[int, int]:
    """Global frame indices [first, last). Using cumulative rounding avoids A/V drift between scenes."""
    return round(start_time * OUTPUT_FPS), round((start_time + duration) * OUTPUT_FPS)


def render_scene_job(job: SceneJob) -> str:
    """Top-level (picklable) worker: build one scene and encode it to a silent mp4."""
    scene = build_scene(job)
    log_memory(f"render scene {job.kind} begin")
    first, last = frame_range(job.start_time, job.duration)
    n_frames = max(1, last - first)

    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(OUTPUT_FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", X264_PRESET, "-b:v", VIDEO_BITRATE,
        "-pix_fmt", "yuv420p", "-threads:v", str(FFMPEG_THREADS),
        "-filter_threads", "1", job.out_path,
    ]
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                if i and i % (OUTPUT_FPS * LOG_EVERY_SECONDS) == 0:
                    log_memory(f"render {job.kind} {i / OUTPUT_FPS:.0f}s/{job.duration:.0f}s")
                g = first + i
                t = max(0.0, g / OUTPUT_FPS - job.start_time)
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
    log_memory(f"render scene {job.kind} finished")
    return job.out_path


def render_scenes(jobs: List[SceneJob], work_dir: Path) -> List[str]:
    """One subprocess at a time: each Matplotlib scene is fully freed on process exit."""
    logger.info("Rendering %d scenes sequentially, each in its own process", len(jobs))
    paths = []
    for index, job in enumerate(jobs):
        job.ffmpeg_threads = FFMPEG_THREADS
        if RENDER_IN_SUBPROCESS:
            input_file = work_dir / f"render_job_{index}.pkl"
            with input_file.open("wb") as f:
                pickle.dump(job, f, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                run_memory_worker("render", str(input_file))
            finally:
                input_file.unlink(missing_ok=True)
        else:
            render_scene_job(job)
        paths.append(job.out_path)
        log_memory(f"scene {index + 1}/{len(jobs)} complete")
    return paths


# =====================================================================
# 12. FINAL ASSEMBLY  (concat with stream copy + a single audio mix; no video re-encode)
# =====================================================================
def assemble_final_video(scene_paths: List[str], voice_tracks: List[Tuple[str, float]],
                         total_frames: int, output_path: str, work_dir: Path) -> None:
    total = total_frames / OUTPUT_FPS

    list_file = work_dir / "scenes.txt"
    list_file.write_text("".join(
        "file '{}'\n".format(Path(p).resolve().as_posix().replace("'", "'\\''")) for p in scene_paths))

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    for path, _ in voice_tracks:
        cmd += ["-i", str(path)]
    music_path = f"assets/music/{random.choice([1,2,3,4,5,6])}.mp3"
    have_music = bool(music_path) and os.path.exists(music_path)
    if have_music:
        logger.info("Mixing background music...")
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

    logger.info(f"Assembling final video: {output_path}")
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg assembly failed: {proc.stderr.decode(errors='ignore')}")


# =====================================================================
# 13. PIPELINE
# =====================================================================
def _timeline(durations: List[float]) -> List[float]:
    """Video length of every segment: its speech + a short pad (+ a longer hold on the last one)."""
    return [d + (END_PAD if i == len(durations) - 1 else SEGMENT_PAD) for i, d in enumerate(durations)]


def run_myth_buster_pipeline(
        output_filename: Optional[str] = None,
        upload: bool = False,
) -> str:
    logger.info("Launching Financial MythBusters pipeline (low-memory mode)...")
    log_memory("pipeline start")
    work_dir = Path(tempfile.mkdtemp(prefix="myth_short_"))

    try:
        # ---- Phase A: the AI director + (only if needed) market data ----
        script = generate_myth_script()
        segs = script.segments
        logger.info("=" * 60)
        logger.info(f"TOPIC: {script.topic}   |   TITLE: {script.youtube_title}")
        for i, s in enumerate(segs):
            logger.info(f"  [{i + 1}] {s.layout:16s} '{s.headline}' popups={len(s.popups)} :: {s.script}")
        logger.info("=" * 60)

        chart_idx = [i for i, s in enumerate(segs) if s.layout == "annotated_chart"]
        results = parallel_map(lambda i: fetch_chart_data(segs[i].chart_ticker, segs[i].chart_timeframe), chart_idx)
        chart_data: Dict[int, dict] = {}
        for i, data in zip(chart_idx, results):
            if data:
                chart_data[i] = data
            else:
                downgrade_to_plain(segs[i], f"no price data for {segs[i].chart_ticker}")
        log_memory("market data ready")

        # ---- Phase B: voice (re-recorded faster once if the Short would exceed the limit) ----
        speech_items = [(s.script, work_dir / f"segment_{i}.wav") for i, s in enumerate(segs)]
        speed = TTS_SPEED
        for attempt in range(2):
            durations = generate_voiceovers(speech_items, speed)
            seg_durs = _timeline(durations)
            total = sum(seg_durs)
            logger.info(f"Voiceover at speed {speed:.2f}: speech {sum(durations):.1f}s, video {total:.1f}s")
            if total <= SHORTS_MAX_SECONDS or attempt == 1 or speed >= TTS_MAX_SPEED:
                break
            budget = SHORTS_MAX_SECONDS - (total - sum(durations)) - 0.5
            speed = min(TTS_MAX_SPEED, speed * sum(durations) / max(budget, 1.0))
            logger.warning(f"Too long ({total:.1f}s); re-recording at speed {speed:.2f}")
        if total > SHORTS_MAX_SECONDS:
            logger.warning(f"Video is {total:.1f}s, longer than the {SHORTS_MAX_SECONDS:.0f}s Shorts limit!")
        log_memory("TTS completed / model unloaded")

        # ---- Phase C: timeline -> one render job per segment ----
        headlines = [s.headline for s in segs]
        jobs: List[SceneJob] = []
        voice_tracks: List[Tuple[str, float]] = []
        cursor = 0.0
        for i, (s, speech, seg_dur) in enumerate(zip(segs, durations, seg_durs)):
            audio_path = str(speech_items[i][1])
            jobs.append(SceneJob("segment", str(work_dir / f"scene_{i}.mp4"), cursor, seg_dur, args=dict(
                seg=s.model_dump(), index=i, n_segments=len(segs), topic=script.topic, headlines=headlines,
                chart_data=chart_data.get(i), speech_dur=speech, ticker_text=script.news_ticker_text,
                audio_path=audio_path)))
            voice_tracks.append((audio_path, seg_dur))
            cursor += seg_dur

        total_frames = frame_range(0.0, cursor)[1]
        logger.info(f"Final length: {cursor:.1f}s")

        if not output_filename:
            slug = re.sub(r"[^a-z0-9]+", "_", script.topic.lower()).strip("_")[:40] or "myth"
            output_filename = f"MYTH_{slug}.mp4"
        description = script.description.strip()
        if "not financial advice" not in description.lower():
            description += "\n\nNot financial advice."
        if "#shorts" not in description.lower():
            description += "\n\n#Shorts"
        tags = [t.strip() for t in script.tags.split(",") if t.strip()]
        topic = script.topic
        youtube_title = script.youtube_title
        del script, segs
        gc.collect()

        # ---- Phase D: render + assemble ----
        scene_paths = render_scenes(jobs, work_dir)
        log_memory("scenes rendered")
        assemble_final_video(scene_paths, voice_tracks, total_frames, output_filename, work_dir)
        logger.info(f"SUCCESS! Video saved as: {output_filename} ({cursor:.1f}s)")

        # ---- Phase E: upload ----
        if upload:
            from uploader import upload_video  # import only when an upload is requested
            logger.info("Uploading Short to YouTube...")
            upload_video(file_path=output_filename, title=youtube_title, description=description, tags=tags, category_id="27")
            logger.info(f"Uploaded To YouTube: {youtube_title}")
        log_memory("pipeline finished")
        return output_filename
    finally:
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tts-worker":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: --tts-worker input.json durations.json")
        job_data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    elif len(sys.argv) > 1 and sys.argv[1] == "--render-worker":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: --render-worker job.pkl")
        # This pickle file is written by this application into its private temp dir.
        with open(sys.argv[2], "rb") as f:
            render_job = pickle.load(f)
        render_scene_job(render_job)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Generate an Evergreen Financial MythBusters Short.")
        parser.add_argument("--upload", action="store_true", help="upload to YouTube when finished")
        parser.add_argument("--test", action="store_true", help="do not record the topic in the history file")
        parser.add_argument("--output", default=None, help="output mp4 path (default: MYTH_<topic>.mp4)")
        cli = parser.parse_args()
        run_myth_buster_pipeline(output_filename=cli.output, upload=cli.upload)