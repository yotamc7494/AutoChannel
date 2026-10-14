"""
Top-moving-stock vertical short (1080x1920)  -  refactored for speed

Stages
    1. Config
    2. Helpers          (retry, ffmpeg, audio decoding)
    3. LLM              (Gemini with model rotation + backoff)
    4. Stock selection  (Redis "used today" set, one batch download, news, company name)
    5. Market metrics   (market cap / S&P 500 / BTC, fetched concurrently, multiple fallbacks)
    6. Script           (Gemini JSON: title, headline, voiceover, description, tags)
    7. TTS              (Piper, local)
    8. Captions         (Whisper word timestamps -> 2-word captions)
    9. Graphics         (headline/price overlay PNG, caption sprites)
   10. Animated chart   (built once, redrawn with blitting)
   11. Video render     (frames piped to ffmpeg, which does the background/dim/overlay/encode)
   12. Pipeline + entry points

Dependencies: numpy, scipy, matplotlib, Pillow, yfinance, openai-whisper, google-genai, upstash-redis,
python-dotenv, requests (+ the `piper` CLI and an ffmpeg binary). moviepy is no longer needed.
"""
import json
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
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
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
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

try:
    import httpx  # installed with google-genai; used to recognise network errors
except ImportError:  # pragma: no cover
    httpx = None

from uploader import upload_video

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StockShort")


# =====================================================================
# 1. CONFIG
# =====================================================================
# --- Video ---
VIDEO_W, VIDEO_H = 1080, 1920
FPS = 24
CHART_RENDER_FPS = 12              # the matplotlib chart is redrawn at this rate, frames are repeated
BG_DIM = 0.22                      # brightness multiplier for the background video
BG_VIDEO_DIR = "assets/trading_floor_videos"
X264_PRESET = "ultrafast"
CHART_POINTS = 600                 # smooth-path resolution (PCHIP interpolation)

# --- LLM ---
GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]   # tried in order, per round
LLM_MAX_ROUNDS = 5
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# --- TTS (Piper) - keep the values you already use ---
PIPER_MODEL_FILE = "en_US-ryan-high.onnx"
PIPER_CONFIG_FILE = "en_US-ryan-high.onnx.json"
PIPER_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
PIPER_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
        # <1 = faster speech (~ the old "+15%"). Set to None to use the voice default.

# --- Data ---
TICKERS_TO_CHECK = [
    "NVDA", "AVGO", "AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "META",
    "AMD", "PLTR", "SMCI", "ARM", "NFLX", "COIN", "MSTR", "INTC",
    "BRK-B", "JPM", "LLY", "QQQ", "SPY",
]
ETF_TICKERS = {"QQQ", "SPY"}       # ETFs have no market cap; "net assets" is shown instead
COMPANY_NAMES = {                  # avoids a slow and flaky yfinance `.info` call for the name
    "NVDA": "Nvidia", "AVGO": "Broadcom", "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft",
    "AMZN": "Amazon", "GOOGL": "Alphabet Google", "META": "Meta Platforms", "AMD": "Advanced Micro Devices",
    "PLTR": "Palantir", "SMCI": "Super Micro Computer", "ARM": "Arm Holdings", "NFLX": "Netflix",
    "COIN": "Coinbase", "MSTR": "Strategy MicroStrategy", "INTC": "Intel", "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase", "LLY": "Eli Lilly", "QQQ": "Invesco QQQ Nasdaq 100 ETF", "SPY": "SPDR S&P 500 ETF",
}
TTS_SYMBOL_NAMES = {               # pronunciations used before text-to-speech
    r"\bNVDA\b": "Nvidia", r"\bTSLA\b": "Tesla", r"\bAAPL\b": "Apple", r"\bAMZN\b": "Amazon",
    r"\bMSFT\b": "Microsoft", r"\bGOOGL\b": "Google", r"\bMETA\b": "Meta", r"\bAVGO\b": "Broadcom",
    r"\bPLTR\b": "Palantir", r"\bSMCI\b": "Super Micro", r"\bMSTR\b": "MicroStrategy", r"\bCOIN\b": "Coinbase",
    r"\bNFLX\b": "Netflix", r"\bINTC\b": "Intel", r"\bJPM\b": "JP Morgan", r"\bLLY\b": "Eli Lilly",
    r"\bBRK-B\b": "Berkshire Hathaway", r"\bBTC\b": "Bitcoin", r"\bSPY\b": "S and P 500", r"\bQQQ\b": "Nasdaq",
}

# --- Palette ---
C_UP, C_DOWN, C_GRID, C_BANNER, C_BANNER_EDGE = "#00E676", "#FF1744", "#2A3447", "#141A26", "#2A3447"


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


def decode_audio(path: str, sr: int = 22050) -> np.ndarray:
    """Decode any audio file to mono float32 samples via ffmpeg."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {proc.stderr.decode(errors='ignore')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def audio_duration(path: str) -> float:
    return len(decode_audio(path)) / 22050


def mpl_safe(text: str) -> str:
    """A pair of bare '$' makes matplotlib switch to math mode ('$6,800 | BTC: $70,000' breaks)."""
    return text.replace("$", r"\$")


# =====================================================================
# 3. LLM  (model rotation + exponential backoff; same logic as the daily-recap pipeline)
# =====================================================================
class LLMUnavailableError(RuntimeError):
    """Every model kept failing after all retry rounds."""


@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("🚨 GEMINI_API_KEY is missing!")
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
    """Each round tries every model once; transient errors (503/429/5xx/network) move on to the next model,
    then back off and retry. Permanent errors drop the model. `parse` failures are retried too."""
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
                    logger.warning(f"⚠️ {model} temporarily unavailable (round {round_no}/{max_rounds}): {e}")
                    continue
                logger.error(f"❌ {model} failed with a non-retryable error, removing it from rotation: {e}")
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
                logger.warning(f"⚠️ {model} returned unusable output (round {round_no}/{max_rounds}): {e}")

        if round_no < max_rounds:
            delay = min(LLM_MAX_DELAY, LLM_BASE_DELAY * 2 ** (round_no - 1)) * random.uniform(0.5, 1.0)
            logger.info(f"⏳ All models busy; retrying in {delay:.1f}s (round {round_no + 1}/{max_rounds})...")
            time.sleep(delay)

    raise LLMUnavailableError(
        f"Gemini still failing after {max_rounds} rounds across {GEMINI_MODELS}. Last error: {last_exc}") from last_exc


# =====================================================================
# 4. STOCK SELECTION
# =====================================================================
def _get_redis_client():
    from upstash_redis import Redis
    url, token = get_secret("UPSTASH_REDIS_REST_URL"), get_secret("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        raise ValueError("❌ Missing Redis credentials! Check your .env file or Streamlit secrets.")
    return Redis(url=url, token=token)


def get_used_stocks_today() -> set:
    try:
        members = _get_redis_client().smembers(f"used_stocks:{date.today()}")
        return set(members) if members else set()
    except Exception as e:
        logger.warning(f"⚠️ Error fetching used stocks from Redis: {e}")
        return set()


def mark_stock_as_used(ticker_symbol: str) -> None:
    try:
        redis = _get_redis_client()
        key = f"used_stocks:{date.today()}"
        redis.sadd(key, ticker_symbol)
        redis.expire(key, 172800)  # 48h
        logger.info(f"✅ Stock {ticker_symbol} marked as used for {date.today()}")
    except Exception as e:
        logger.warning(f"⚠️ Error marking stock {ticker_symbol} as used in Redis: {e}")


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


def get_top_moving_stock(test: bool = False) -> dict:
    """The ticker with the biggest move (in %) in the latest session, skipping ones already used today."""
    used_today = get_used_stocks_today()
    logger.info(f"📋 Stocks already used today ({date.today()}): {sorted(used_today) or 'None'}")

    candidates = [t for t in TICKERS_TO_CHECK if t not in used_today]
    if not candidates:
        logger.warning("⚠️ All tickers were already used today; allowing repeats.")
        candidates = list(TICKERS_TO_CHECK)

    logger.info(f"🔍 Batch fetching {len(candidates)} tickers in one request...")
    batch_df = _download_batch(candidates)

    best = None
    for symbol in candidates:
        try:
            df = _ticker_frame(batch_df, symbol)
            if df is None or df.empty or len(df) < 2:
                continue
            latest_date = df.index.date[-1]
            today_hist = df[df.index.date == latest_date]
            if today_hist.empty:
                continue
            curr_price = today_hist["Close"].iloc[-1]
            prev_day = df[df.index.date < latest_date]
            prev_close = prev_day["Close"].iloc[-1] if not prev_day.empty else today_hist["Open"].iloc[0]
            if not prev_close or prev_close == 0:
                continue
            raw_pct = (curr_price - prev_close) / prev_close * 100
            if best is None or abs(raw_pct) > abs(best["change_pct"]):
                best = {"symbol": symbol, "current_price": round(float(curr_price), 2),
                        "change_pct": round(float(raw_pct), 2), "history": today_hist["Close"].dropna()}
        except Exception as e:
            logger.warning(f"Error analyzing batch data for {symbol}: {e}")

    if best is None:
        raise RuntimeError("No valid market data found for any ticker.")

    logger.info(f"🎯 Top Moving Stock: {best['symbol']} ({best['change_pct']}%)")
    if not test:
        mark_stock_as_used(best["symbol"])
    return best


def extract_news_context(news_data, max_articles: int = 3) -> str:
    """Headlines + summaries from yfinance's news structure."""
    if not news_data or not isinstance(news_data, list):
        return "No recent breaking news available."
    articles = []
    for item in news_data:
        if len(articles) >= max_articles:
            break
        content = item.get("content", {}) or {}
        title = (content.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or "").strip()
        provider = (content.get("provider") or {}).get("displayName", "Market News")
        text = f"• [{provider}] {title}"
        if summary:
            text += f"\n  Context: {summary}"
        articles.append(text)
    return "\n\n".join(articles) if articles else "No breaking news found for this ticker today. "


def fetch_news_context(symbol: str) -> str:
    try:
        return extract_news_context(yf.Ticker(symbol).news)
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch news for {symbol}: {e}")
        return "No recent breaking news available."


def get_company_name(symbol: str) -> str:
    if symbol in COMPANY_NAMES:
        return COMPANY_NAMES[symbol]
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol


# =====================================================================
# 5. MARKET METRICS  (market cap was always "N/A" - see _fetch_market_cap)
# =====================================================================
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


def _fetch_market_cap(symbol: str) -> Tuple[float, str]:
    """
    Returns (value, label). Tries several sources, because Yahoo's `.info` scrape (the original code's
    only source) is frequently rate-limited/blocked and then comes back WITHOUT 'marketCap', which the old
    `.get('marketCap', 0)` silently turned into "N/A". ETFs never have a market cap at all.
    Raises LookupError when every source fails (the caller retries).
    """
    t = yf.Ticker(symbol)
    fi = t.fast_info  # lightweight endpoints - the same ones that already work for ^GSPC / BTC

    def from_fast_info():
        return _positive_number(_safe_attr(fi, "market_cap"))

    def from_info():
        info = t.info or {}
        cap = _positive_number(info.get("marketCap"))
        if cap:
            return cap
        shares, price = _positive_number(info.get("sharesOutstanding")), _positive_number(info.get("currentPrice"))
        return shares * price if shares and price else None

    def from_shares_times_price():
        shares = t.get_shares_full(start=(date.today().replace(day=1)).isoformat())
        shares = _positive_number(shares.dropna().iloc[-1]) if shares is not None and len(shares) else None
        price = _positive_number(_safe_attr(fi, "last_price"))
        return shares * price if shares and price else None

    def from_etf_assets():
        info = t.info or {}
        return _positive_number(info.get("totalAssets") or info.get("netAssets"))

    quote_type = str(_safe_attr(fi, "quote_type") or "").upper()
    if symbol in ETF_TICKERS or quote_type == "ETF":
        strategies = [("info.totalAssets", from_etf_assets, "NET ASSETS")]
    else:
        strategies = [("fast_info.market_cap", from_fast_info, "MARKET CAP"),
                      ("info.marketCap", from_info, "MARKET CAP"),
                      ("shares x price", from_shares_times_price, "MARKET CAP")]

    for source, getter, label in strategies:
        try:
            value = getter()
        except Exception as e:
            logger.info(f"   market cap source '{source}' failed for {symbol}: {e}")
            continue
        if value:
            logger.info(f"   market cap for {symbol} = {value:,.0f} (source: {source})")
            return value, label
    raise LookupError(f"no market cap source worked for {symbol}")


def _fetch_last_price(symbol: str) -> float:
    price = _positive_number(_safe_attr(yf.Ticker(symbol).fast_info, "last_price"))
    if price is None:  # fallback: last daily close
        hist = yf.Ticker(symbol).history(period="5d")
        price = _positive_number(hist["Close"].dropna().iloc[-1]) if not hist.empty else None
    if price is None:
        raise LookupError(f"no price for {symbol}")
    return price


def fetch_market_metrics(symbol: str) -> Dict[str, Optional[str]]:
    """Market cap, S&P 500 and BTC prices fetched concurrently (no more fixed sleeps)."""
    def attempt(fn, what):
        try:
            return with_retries(fn, attempts=3, base_delay=1.0, what=what)
        except Exception as e:
            logger.warning(f"⚠️ {what} unavailable: {e}")
            return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        cap_f = ex.submit(attempt, lambda: _fetch_market_cap(symbol), f"{symbol} market cap")
        sp_f = ex.submit(attempt, lambda: _fetch_last_price("^GSPC"), "S&P 500 price")
        btc_f = ex.submit(attempt, lambda: _fetch_last_price("BTC-USD"), "BTC price")
        cap, sp, btc = cap_f.result(), sp_f.result(), btc_f.result()

    return {
        "mcap": format_money(cap[0]) if cap else None,
        "mcap_label": cap[1] if cap else "MARKET CAP",
        "sp500": f"${sp:,.0f}" if sp else None,
        "btc": f"${btc:,.0f}" if btc else None,
    }


# =====================================================================
# 6. SCRIPT + TITLES
# =====================================================================
def _parse_script(response) -> dict:
    data = json.loads(response.text)
    if not isinstance(data, dict) or not data.get("voiceover_text") or not data.get("youtube_title"):
        raise ValueError("JSON is missing 'voiceover_text' or 'youtube_title'")
    return data


def generate_script_and_titles_with_ai(stock_data: dict) -> dict:
    prompt = f"""
        You are a high-energy financial content creator and YouTube SEO expert for TikTok, YouTube Shorts, and Instagram Reels.
        Create a fast-paced 45-second video script and SEO-optimized metadata about this stock movement:
        - Symbol: {stock_data['symbol']}
        - Company Name: {stock_data['company_name']}
        - Current Price: ${stock_data['current_price']}
        - Today's Change: {stock_data['change_pct']}%
        - News Context: {stock_data['news']}

        CRITICAL SCRIPTING & SEO RULES:
        1. YOUTUBE SEARCH SEO OPTIMIZATION (MAXIMIZE SEARCH TRAFFIC):
           - "youtube_title" MUST explicitly contain BOTH the Ticker Symbol AND the Full Company Name (e.g., "AVGO Broadcom Shocking Drop! Buy Or Pass? 🚀") so it ranks high in YouTube Search.
           - "description" MUST naturally incorporate both the ticker and company name alongside key financial terms.
           - "tags" MUST include the ticker, company name, specific stock tags (e.g., AVGO, Broadcom, BroadcomStock), and general investing tags.

        2. SCROLL-STOPPING HOOK:
           - Start instantly with a high-stakes, dramatic statement in the first 2 seconds that calls out the specific stock and its movement.

        3. POLARIZING STORYTELLING & MARKET CONTROVERSY (BULL VS. BEAR):
           - Show the tension: Explain why the Bears think this stock is heading for a crash, while the Bulls argue it is a lifetime opportunity.
           - Focus on big market forces, macro trends, and major institutions (Wall Street, Big Tech, The Fed) rather than obscure individuals.

        4. OPTIMIZED FOR TTS (RETENTION & FLOW):
           - Write purely in natural, spoken English.
           - ABSOLUTELY NO dashes (-), ellipses (...), semicolons (;), or complex formatting that causes Text-To-Speech (TTS) engines to pause awkwardly.
           - Keep punctuation strictly to simple periods and commas for a relentless, fast pace.
           - Do not use ndecimal points and number after for example insted of (+1.64% say the stock is up by more the 1 percent, same for prices)

        5. HIGH-ENGAGEMENT BINARY CTA (COMMENT MAGNET):
           - End the script immediately with an intense, forced binary choice question that triggers emotional responses.
           - Force the viewer to pick a side in the comments using a simple word choice (e.g., "Is this a massive trap or an easy bargain? Type BUY or SELL in the comments right now!").

        6. STRICT OUTPUT FORMAT:
           - Return ONLY a valid JSON object with the following 5 fields:
        {{
          "youtube_title": "High-CTR & Search-Optimized title with TICKER + COMPANY NAME + Emojis",
          "overlay_headline": "2 to 4 WORDS IN ALL CAPS FOR ON-SCREEN OVERLAY (e.g., AVGO HUGE CRASH!)",
          "voiceover_text": "The full fluid script for narration with a controversial story beat and binary CTA",
          "description": "SEO-rich summary with Ticker and Company Name, ending with a strong call to debate in the comments",
          "tags": "5-7 relevant tags separated by commas including Ticker and Full Company Name"
        }}
        """
    return ask_llm(prompt, config=types.GenerateContentConfig(response_mime_type="application/json"),
                   parse=_parse_script)


# =====================================================================
# 7. TEXT-TO-SPEECH  (Piper, local)
# =====================================================================
def ensure_piper_model():
    """Downloads the voice model once if it is not already in the local folder."""
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
    text = text.replace('\n', ' ').strip()
    return text


def generate_voiceovers(items: List[Tuple[str, str]]) -> List[float]:
    """items = [(script_text, output_path), ...]  ->  durations in seconds (same order)."""
    logger.info(f"Generating {len(items)} TTS clips locally with Piper...")
    durations = []
    ensure_piper_model()

    for text, path in items:
        clean_text = clean_script_for_tts(text)
        path_str = str(path)
        logger.info(f"Generating TTS audio: {Path(path_str).name}")

        # the text is piped through stdin, so no shell quoting problems
        cmd = ["piper", "--model", PIPER_MODEL_FILE, "--output_file", path_str]
        proc = subprocess.run(cmd, input=clean_text.encode('utf-8'), capture_output=True)

        if proc.returncode != 0:
            raise RuntimeError(f"Piper TTS failed for {path_str}: {proc.stderr.decode(errors='ignore')}")

        dur = audio_duration(path_str)
        durations.append(dur)
        logger.info(f"✅ Created: {Path(path_str).name} ({dur:.2f}s)")

    return durations


# =====================================================================
# 8. CAPTIONS (Whisper word timestamps)
# =====================================================================
@lru_cache(maxsize=1)
def get_whisper_model():
    """Import + load once (the pipeline starts this in the background at second zero)."""
    import whisper
    return whisper.load_model("tiny.en")


def get_exact_captions_with_whisper(audio_file: str, words_per_caption: int = 2) -> List[dict]:
    logger.info("🎧 Aligning captions with Whisper...")
    # Whisper accepts a 16 kHz float32 array, so it never needs an ffmpeg executable on PATH.
    audio = decode_audio(audio_file, sr=16000)
    result = get_whisper_model().transcribe(audio, word_timestamps=True)

    words = [{"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
             for seg in result.get("segments", []) for w in seg.get("words", [])]
    captions = []
    for i in range(0, len(words), words_per_caption):
        group = words[i:i + words_per_caption]
        start, end = group[0]["start"], group[-1]["end"]
        if i + words_per_caption < len(words):
            end = min(end + 0.05, words[i + words_per_caption]["start"])
        captions.append({"start": start, "end": end, "text": " ".join(w["word"] for w in group)})
    return captions


# =====================================================================
# 9. GRAPHICS  (overlay PNG + caption sprites, pre-rendered once)
# =====================================================================
@lru_cache(maxsize=16)
def load_system_font(size: int):
    for font_file in ["Roboto-Bold.ttf", "Montserrat-Bold.ttf", "arial.ttf"]:          # local project fonts
        if os.path.exists(font_file):
            try:
                return ImageFont.truetype(font_file, size)
            except Exception:
                pass
    for font_path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",          # Linux
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                      "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    for font_name in ["impact.ttf", "ariblk.ttf", "arialbd.ttf", "trebucbd.ttf"]:       # Windows
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def create_overlay_graphics(script_data: dict, stock_data: dict, output_path: str,
                            canvas_size: Tuple[int, int] = (VIDEO_W, VIDEO_H)) -> str:
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    headline = script_data.get("overlay_headline", f"{stock_data['symbol']} MOVES!").upper()
    font = load_system_font(75 if len(headline) < 15 else 60)
    color = random.choice(["#FFD700", "#00FFFF", "#FF3366", "#33FF57", "#FF9900", "#FFFFFF"])
    angle = random.uniform(-6.0, 6.0)

    box = (950, 250)
    text_img = Image.new("RGBA", box, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_img)
    bbox = draw.textbbox((0, 0), headline, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((box[0] - tw) / 2, (box[1] - th) / 2), headline, font=font, fill=color,
              stroke_width=10, stroke_fill="black")
    rotated = text_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(rotated, (int((canvas_size[0] - rotated.width) / 2), 380), rotated)  # just under the top banners

    pct = stock_data["change_pct"]
    pct_text = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
    pct_color = "#00E676" if pct >= 0 else "#FF1744"
    price_text = f"${stock_data['current_price']:.2f}"

    badge = Image.new("RGBA", (900, 400), (0, 0, 0, 0))
    draw_b = ImageDraw.Draw(badge)
    pct_font, price_font = load_system_font(120), load_system_font(70)
    pw = draw_b.textbbox((0, 0), pct_text, font=pct_font)
    draw_b.text(((900 - (pw[2] - pw[0])) / 2, 20), pct_text, font=pct_font, fill=pct_color,
                stroke_width=12, stroke_fill="black")
    prw = draw_b.textbbox((0, 0), price_text, font=price_font)
    draw_b.text(((900 - (prw[2] - prw[0])) / 2, 220), price_text, font=price_font, fill="white",
                stroke_width=8, stroke_fill="black")
    canvas.paste(badge, (90, 750), badge)

    canvas.save(output_path)
    return output_path


@dataclass
class CaptionSprite:
    start: float
    end: float
    x: int
    y: int
    rgba: np.ndarray   # (h, w, 4) uint8, straight alpha, cropped to the text


def build_caption_sprites(captions: List[dict], box: Tuple[int, int] = (1000, 220), top: int = 1420) -> List[CaptionSprite]:
    """One small sprite per caption, rendered once (no temp PNG files, no per-caption clips)."""
    font = load_system_font(52)
    box_x = (VIDEO_W - box[0]) // 2
    sprites = []
    for cap in captions:
        if cap["end"] - cap["start"] <= 0.05:
            continue
        img = Image.new("RGBA", box, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        text = cap["text"].upper()
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((box[0] - (bbox[2] - bbox[0])) / 2, (box[1] - (bbox[3] - bbox[1])) / 2), text, font=font,
                  fill="#FFE600", stroke_width=8, stroke_fill="black")
        arr = np.asarray(img)
        rows, cols = np.where(arr[:, :, 3] > 0)
        if len(rows) == 0:
            continue
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        sprites.append(CaptionSprite(cap["start"], cap["end"], box_x + int(x0), top + int(y0),
                                     np.ascontiguousarray(arr[y0:y1, x0:x1])))
    return sprites


def blend_sprite_over(frame_rgba: np.ndarray, sprite: CaptionSprite) -> None:
    """Straight-alpha 'over' of a small sprite onto the RGBA chart frame (in place, region only)."""
    h, w = sprite.rgba.shape[:2]
    dst = frame_rgba[sprite.y:sprite.y + h, sprite.x:sprite.x + w].astype(np.float32)
    src = sprite.rgba.astype(np.float32)
    sa, da = src[:, :, 3:4] / 255.0, dst[:, :, 3:4] / 255.0
    out_a = sa + da * (1 - sa)
    out_rgb = (src[:, :, :3] * sa + dst[:, :, :3] * da * (1 - sa)) / np.maximum(out_a, 1e-6)
    frame_rgba[sprite.y:sprite.y + h, sprite.x:sprite.x + w, :3] = (out_rgb + 0.5).astype(np.uint8)
    frame_rgba[sprite.y:sprite.y + h, sprite.x:sprite.x + w, 3:4] = (out_a * 255 + 0.5).astype(np.uint8)


# =====================================================================
# 10. ANIMATED CHART  (figure built ONCE; only the moving artists are redrawn - matplotlib blitting)
# =====================================================================
class ChartRenderer:
    def __init__(self, stock_data: dict, duration: float, market_metrics: Optional[dict] = None):
        self.duration = duration
        prices = np.asarray(stock_data["history"].values, dtype=float)
        if len(prices) < 2:
            prices = np.repeat(prices, 2)
        n = len(prices)

        color = C_UP if stock_data["change_pct"] >= 0 else C_DOWN
        self.x_dense = np.linspace(0, n - 1, CHART_POINTS)
        self.y_dense = PchipInterpolator(np.arange(n), prices)(self.x_dense)  # natural path, no overshoot

        y_min, y_max = prices.min(), prices.max()
        y_range = y_max - y_min if y_max != y_min else 1.0
        self.y_padding = y_range * 0.30
        self.floor = y_min - self.y_padding

        self.fig = Figure(figsize=((VIDEO_W + 1e-3) / 100, (VIDEO_H + 1e-3) / 100), dpi=100, facecolor="none")
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.ax = self.fig.subplots()   # same default margins as plt.subplots()
        self.fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)

        # ---- static layers ----
        ax.set_xticks(np.linspace(0, n, 6))
        ax.set_yticks(np.linspace(self.floor, y_max + self.y_padding, 8))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(True, color=C_GRID, linestyle="--", linewidth=1.5, alpha=0.5)
        ax.set_axisbelow(True)

        m = market_metrics or {}
        index_parts = [p for p in (f"S&P 500: {m['sp500']}" if m.get("sp500") else None,
                                   f"BTC: {m['btc']}" if m.get("btc") else None) if p]
        if index_parts:
            ax.text(0.5, 0.96, mpl_safe("   |   ".join(index_parts)), transform=ax.transAxes, color="#FFFFFF",
                    fontsize=24, fontweight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="square,pad=0.5", facecolor=C_BANNER, edgecolor=C_BANNER_EDGE,
                              alpha=0.9, linewidth=1.5))
        if m.get("mcap"):   # hidden (instead of showing "N/A") when no source could provide it
            ax.text(0.5, 0.89, mpl_safe(f"{m.get('mcap_label', 'MARKET CAP')}: {m['mcap']}"),
                    transform=ax.transAxes, color="white", fontsize=24, fontweight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=C_BANNER, edgecolor=color, alpha=0.9,
                              linewidth=1.5))

        # ---- animated layers (excluded from the static draw, redrawn every frame) ----
        self.fill = Polygon(np.zeros((3, 2)), closed=True, color=color, alpha=0.2, animated=True)
        ax.add_patch(self.fill)
        self.line, = ax.plot([], [], color=color, linewidth=8, animated=True)
        self.dot, = ax.plot([], [], marker="o", markersize=14, color=color, animated=True)
        self.label = ax.text(0, 0, "", color="white", fontsize=28, fontweight="bold", ha="center", va="bottom",
                             animated=True,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none", alpha=0.95))

        ax.set_xlim(-1, n + 1)
        ax.set_ylim(self.floor, y_max + self.y_padding)
        for spine in ax.spines.values():
            spine.set_visible(False)

        self.canvas.draw()
        self._static = self.canvas.copy_from_bbox(self.fig.bbox)   # grid + banners, kept as pixels

    def render(self, t: float) -> np.ndarray:
        """RGBA (straight alpha) frame of shape (H, W, 4) for time t (a private copy)."""
        idx = int(t / self.duration * (CHART_POINTS - 1))
        idx = max(1, min(idx, CHART_POINTS - 1))
        sub_x, sub_y = self.x_dense[:idx + 1], self.y_dense[:idx + 1]
        curr_x, curr_price = sub_x[-1], sub_y[-1]

        verts = np.empty((len(sub_x) + 2, 2))
        verts[0] = (sub_x[0], self.floor)
        verts[1:-1, 0], verts[1:-1, 1] = sub_x, sub_y
        verts[-1] = (curr_x, self.floor)
        self.fill.set_xy(verts)
        self.line.set_data(sub_x, sub_y)
        self.dot.set_data([curr_x], [curr_price])
        self.label.set_position((curr_x, curr_price + self.y_padding * 0.12))
        self.label.set_text(mpl_safe(f" ${curr_price:.2f} "))

        self.canvas.restore_region(self._static)
        for artist in (self.fill, self.line, self.dot, self.label):
            self.ax.draw_artist(artist)
        return np.array(self.canvas.buffer_rgba())


# =====================================================================
# 11. VIDEO RENDER  (python only draws the chart; ffmpeg composites, dims, encodes)
# =====================================================================
def pick_background_video() -> Optional[str]:
    files = sorted(Path(BG_VIDEO_DIR).glob("*.mp4"))
    if not files:
        logger.warning(f"No background videos in {BG_VIDEO_DIR}; using a plain dark background.")
        return None
    return str(random.choice(files))


def render_video(chart: ChartRenderer, sprites: List[CaptionSprite], overlay_png: str, bg_path: Optional[str],
                 audio_path: str, duration: float, out_path: str) -> None:
    n_frames = max(1, round(duration * FPS))

    # which caption is visible on each output frame (-1 = none)
    cap_for_frame = np.full(n_frames, -1, dtype=np.int32)
    for i, s in enumerate(sprites):
        f0 = int(math.ceil(s.start * FPS - 1e-9))
        f1 = int(math.ceil(s.end * FPS - 1e-9))
        cap_for_frame[max(0, f0):max(0, min(n_frames, f1))] = i

    bg_input = (["-stream_loop", "-1", "-i", bg_path] if bg_path
                else ["-f", "lavfi", "-i", f"color=c=0x0B0E14:s={VIDEO_W}x{VIDEO_H}:r={FPS}"])
    # background: cover-fit to 1080x1920 (scale to height, centre crop), dim; then chart layer, then overlay PNG
    graph = (f"[0:v]fps={FPS},scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
             f"crop={VIDEO_W}:{VIDEO_H},colorchannelmixer=rr={BG_DIM}:gg={BG_DIM}:bb={BG_DIM},setsar=1[bg];"
             "[bg][1:v]overlay=format=auto[a];[a][2:v]overlay=format=auto[v]")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", *bg_input,
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(FPS), "-i", "-",
           "-loop", "1", "-framerate", str(FPS), "-i", overlay_png,
           "-i", audio_path,
           "-filter_complex", graph, "-map", "[v]", "-map", "3:a", "-t", f"{n_frames / FPS:.6f}",
           "-c:v", "libx264", "-preset", X264_PRESET, "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]

    base_tick, base, last_key, frame = None, None, None, None
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                tick = int(i / FPS * CHART_RENDER_FPS + 1e-6)
                key = (tick, int(cap_for_frame[i]))
                if key != last_key:                       # otherwise the previous frame is sent again
                    if tick != base_tick:
                        base, base_tick = chart.render(tick / CHART_RENDER_FPS), tick
                    if key[1] >= 0:
                        frame = base.copy()
                        blend_sprite_over(frame, sprites[key[1]])
                    else:
                        frame = base
                    last_key = key
                proc.stdin.write(frame)
            proc.stdin.close()
            proc.wait()
        except BrokenPipeError:
            proc.wait()
        except Exception:
            proc.kill()
            raise
        if proc.returncode != 0:
            err_file.seek(0)
            raise RuntimeError(f"ffmpeg failed: {err_file.read().decode(errors='ignore')}")


def extract_frame(video_path: str, t: float, out_png: str) -> Optional[str]:
    proc = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(t), "-i", video_path,
                           "-frames:v", "1", out_png], capture_output=True)
    return out_png if proc.returncode == 0 and os.path.exists(out_png) else None


# =====================================================================
# 12. PIPELINE + ENTRY POINTS
# =====================================================================
def create_stock_short(upload: bool = False, test: bool = False,
                       on_preview: Optional[Callable[[str], None]] = None) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="stock_short_"))
    try:
        # ---- Phase A: data + audio. Independent network calls overlap; Whisper loads in the background ----
        with ThreadPoolExecutor(max_workers=6) as pool:
            whisper_future = pool.submit(get_whisper_model)

            stock_data = get_top_moving_stock(test=test)
            symbol = stock_data["symbol"]
            news_f = pool.submit(fetch_news_context, symbol)
            name_f = pool.submit(get_company_name, symbol)
            metrics_f = pool.submit(fetch_market_metrics, symbol)   # runs while the LLM + TTS work below
            stock_data["news"], stock_data["company_name"] = news_f.result(), name_f.result()

            logger.info("🤖 Generating script...")
            script_data = generate_script_and_titles_with_ai(stock_data)
            logger.info("=" * 60)
            logger.info(f"📌 YOUTUBE TITLE: {script_data.get('youtube_title')}")
            logger.info(f"🎬 OVERLAY HEADLINE: {script_data.get('overlay_headline')}")
            logger.info("=" * 60)

            audio_file = work_dir / "voiceover.wav"   # Piper writes WAV
            duration = generate_voiceovers([(script_data["voiceover_text"], audio_file)])[0]

            whisper_future.result()
            captions = get_exact_captions_with_whisper(str(audio_file), words_per_caption=2)
            overlay_png = create_overlay_graphics(script_data, stock_data, str(work_dir / "overlay.png"))
            market_metrics = metrics_f.result()
        logger.info(f"Market metrics: {market_metrics}")

        # ---- Phase B: render ----
        logger.info(f"📊 Rendering {len(captions)} captions + animated chart ({duration:.1f}s)...")
        chart = ChartRenderer(stock_data, duration, market_metrics)
        sprites = build_caption_sprites(captions)
        output_filename = f"STOCK_{symbol}_{int(stock_data['change_pct'])}pct.mp4"
        render_video(chart, sprites, overlay_png, pick_background_video(), str(audio_file), duration, output_filename)
        logger.info(f"✅ SUCCESS! Video saved as: {output_filename}")

        if on_preview:
            png = extract_frame(output_filename, min(2.0, duration / 2), str(work_dir / "preview.png"))
            if png:
                on_preview(png)

        # ---- Phase C: upload ----
        if upload:
            tags = script_data.get("tags")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            elif not isinstance(tags, list):
                tags = []
            upload_video(file_path=output_filename, title=script_data.get("youtube_title"),
                         description=script_data.get("description"), tags=tags)
            logger.info(f"🚀 Uploaded To YouTube: {script_data.get('youtube_title')}")
        return output_filename
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def render_final_video() -> str:
    """Render and upload to YouTube."""
    return create_stock_short(upload=True)


def view_final_video() -> str:
    """Render only (no upload) and show a preview frame in Streamlit."""
    def show(png_path: str):
        import streamlit as st
        st.image(png_path, caption="📸 תמונה מקדימה מתוך הסרטון", use_container_width=True)
    return create_stock_short(upload=False, on_preview=show)


if __name__ == "__main__":
    create_stock_short(upload=False)