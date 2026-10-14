# Patched for low-memory Streamlit deployments. v2 fixes cgroup page-cache accounting.
# Optional environment settings:
#   STOCK_APPROX_TIMINGS=1     skips Whisper (estimated subtitle timings)
#   STOCK_DISABLE_BG_VIDEO=1   eliminates background-video decode cost
#   STOCK_MEMORY_GUARD_FRACTION=0.92  terminates ML worker before cgroup OOM

"""
Top-moving-stock vertical short (1080x1920)  -  refactored for speed

Stages
    1. Config
    2. Helpers          (retry, ffmpeg, audio decoding)
    3. LLM              (Gemini with model rotation + backoff)
    4. Stock selection  (Redis "used today" set, one batch download, Google News RSS, company name)
    5. Market metrics   (market cap / S&P 500 / BTC, fetched concurrently, multiple fallbacks)
    6. Script           (Gemini JSON: title, headline, voiceover, description, tags, catalyst_sentences)
    7. TTS              (Piper, local)
    8. Word timings     (Whisper word timestamps)
    9. Graphics         (header with logo + headline/price overlay PNG, word-highlight caption sprites,
                         news headline popup card PNG)
   10. Animated chart   (built once, redrawn with blitting)
   11. Video render     (frames piped to ffmpeg, which does the background/dim/overlays/news-card slide/encode)
   12. Pipeline + entry points

Dependencies: numpy, scipy, matplotlib, Pillow, yfinance, openai-whisper, google-genai, upstash-redis,
python-dotenv, requests (+ the `piper` CLI and an ffmpeg binary). moviepy is no longer needed.
"""
import gc
import html
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
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# IMPORTANT: Workers must be dispatched BEFORE importing yfinance, SciPy,
# matplotlib or Gemini. Otherwise subprocesses duplicate the Streamlit process's
# scientific stack even though only Kokoro / Whisper is actually needed.
def _run_lightweight_ml_worker(kind: str, argv: List[str]) -> None:
    import traceback

    def report(stage: str) -> None:
        try:
            kb = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                      if line.startswith('VmRSS:'))
            cur = Path('/sys/fs/cgroup/memory.current')
            used = f", container={int(cur.read_text()) / 2**20:.0f} MiB" if cur.exists() else ''
            print(f"ML WORKER [{kind}/{stage}] RSS={kb / 1024:.0f} MiB{used}", flush=True)
        except Exception:
            pass

    report('start')
    if kind == 'tts':
        import numpy as np
        import soundfile as sf
        import torch
        from kokoro import KPipeline
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        job_file, result_file = map(Path, argv)
        items = json.loads(job_file.read_text(encoding='utf-8'))
        report('before model')
        pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        report('model loaded')
        durations = []
        with torch.inference_mode():
            for phrase, out_file in items:
                chunks = []
                for _, _, audio in pipeline(phrase, voice='am_liam', speed=1.1,
                                            split_pattern=r'\n+'):
                    if hasattr(audio, 'detach'):
                        audio = audio.detach().cpu().numpy()
                    chunks.append(np.asarray(audio, dtype=np.float32))
                if not chunks:
                    raise RuntimeError('Kokoro returned empty speech')
                sf.write(out_file, np.concatenate(chunks), 24000)
                durations.append(float(sf.info(out_file).duration))
                chunks.clear()
                report('wrote audio')
        result_file.write_text(json.dumps(durations), encoding='utf-8')
        report('complete')
    elif kind == 'whisper':
        import numpy as np
        import whisper
        audio_file, result_file = argv
        executable = shutil.which('ffmpeg')
        if not executable:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        decoded = subprocess.run([executable, '-v', 'error', '-i', audio_file,
                                  '-f', 'f32le', '-ac', '1', '-ar', '16000', '-'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        audio = np.frombuffer(decoded.stdout, dtype=np.float32)
        report('before model')
        model = whisper.load_model('tiny.en', device='cpu')
        report('model loaded')
        result = model.transcribe(audio, word_timestamps=True, fp16=False,
                                  beam_size=1, best_of=1, verbose=False)
        words = []
        for seg in result.get('segments', []):
            for w in seg.get('words', []):
                raw = w['word'].strip()
                clean = re.sub(r'^[^\w$%]+|[^\w$%]+$', '', raw)
                words.append({'raw': raw, 'text': clean,
                              'start': float(w['start']), 'end': float(w['end'])})
        Path(result_file).write_text(json.dumps(words), encoding='utf-8')
        report('complete')
    else:
        raise ValueError(f'Unknown worker kind: {kind}')


if __name__ == '__main__' and len(sys.argv) >= 2 and sys.argv[1] in ('--tts-worker', '--whisper-worker'):
    kind = sys.argv[1][2:].removesuffix('-worker')
    expected_args = 2
    if len(sys.argv) != expected_args + 2:
        raise SystemExit(f'{kind} worker expects {expected_args} file arguments')
    try:
        _run_lightweight_ml_worker(kind, sys.argv[2:])
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    raise SystemExit(0)

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

# ML models are loaded only in lightweight isolated workers.
try:
    import httpx  # installed with google-genai; used to recognise network errors
except ImportError:  # pragma: no cover
    httpx = None


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("StockShort")


def _cgroup_memory_snapshot() -> dict:
    """Read Linux cgroup v2 accounting, distinguishing page cache from held RAM.

    memory.current includes file-backed cache that the kernel can usually reclaim.
    It must *not* be treated as the same thing as anonymous/process RAM.
    All derived cache figures are estimates, not a promise of free memory.
    """
    path = Path('/sys/fs/cgroup')
    result = {'raw': None, 'limit': None, 'reclaimable': 0,
              'guarded': None, 'anon': 0, 'file': 0, 'shmem': 0,
              'inactive_file': 0, 'oom_kill': None}
    try:
        raw_path, limit_path = path / 'memory.current', path / 'memory.max'
        if not raw_path.exists():
            return result
        result['raw'] = int(raw_path.read_text().strip())
        max_value = limit_path.read_text().strip() if limit_path.exists() else 'max'
        result['limit'] = int(max_value) if max_value.isdecimal() else None
        stat_path = path / 'memory.stat'
        stats = {}
        if stat_path.exists():
            for line in stat_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    try:
                        stats[parts[0]] = int(parts[1])
                    except ValueError:
                        continue
        for name in ('anon', 'file', 'shmem', 'inactive_file'):
            result[name] = stats.get(name, 0)
        # 'file' also accounts for tmpfs/shmem, which should not be deducted.
        # Dirty/writeback pages are not immediately reclaimable either.
        clean_file_cache = max(0, stats.get('file', 0) - stats.get('shmem', 0)
                               - stats.get('file_dirty', 0) - stats.get('file_writeback', 0))
        result['reclaimable'] = min(stats.get('inactive_file', 0), clean_file_cache)
        # Conservative margin: credit only 85% of the estimated inactive file cache.
        result['guarded'] = max(0, result['raw'] - int(result['reclaimable'] * 0.85))
        events_path = path / 'memory.events'
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == 'oom_kill':
                    result['oom_kill'] = int(parts[1])
    except (OSError, ValueError):
        # If cgroup stats are absent, fail open only when no limit can be read;
        # ordinary Linux OOM protection still applies.
        if result['raw'] is not None:
            result['guarded'] = result['raw']
    return result


def log_memory(stage: str):
    """Log main-process RSS and container raw vs estimated pressured memory."""
    try:
        rss_kb = next((int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('VmRSS:')), None)
        m = _cgroup_memory_snapshot()
        def mb(n):
            return f'{n / 2**20:.0f}' if n is not None else 'N/A'
        logger.info(
            'MEMORY [%s]: main RSS=%s MiB, cgroup=%s/%s MiB, '
            'anon=%s MiB, file=%s MiB, shmem=%s MiB, '
            'inactive_file=%s MiB, estimated_reclaimable=%s MiB, '
            'guarded_usage=%s MiB, cgroup_oom_kill=%s',
            stage, f'{rss_kb / 1024:.0f}' if rss_kb is not None else 'N/A',
            mb(m['raw']), mb(m['limit']), mb(m['anon']), mb(m['file']),
            mb(m['shmem']), mb(m['inactive_file']), mb(m['reclaimable']),
            mb(m['guarded']), m['oom_kill'] if m['oom_kill'] is not None else 'N/A',
        )
    except Exception as e:
        logger.debug('Memory measurement unavailable: %s', e)


def run_memory_worker(kind: str, *args: str) -> None:
    """Run the ML task in its own lean process and monitor true memory pressure.

    Previously this guard checked memory.current directly, so a cgroup at
    3072/3072 MiB due to *reclaimable file cache* was falsely rejected.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), f'--{kind}-worker', *map(str, args)]
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['MALLOC_ARENA_MAX'] = '2'
    logger.info('Starting lightweight %s worker...', kind)
    log_memory(f'before {kind}')
    before = _cgroup_memory_snapshot()
    limit = before['limit']
    fraction = min(0.98, max(0.50, float(os.getenv('STOCK_MEMORY_GUARD_FRACTION', '0.92'))))
    threshold = min(limit * fraction, limit - 96 * 2**20) if limit else None
    current_guarded = before['guarded']
    if threshold is not None and current_guarded is not None and current_guarded >= threshold:
        raise MemoryError(
            f'Not enough non-reclaimable RAM before {kind}: '
            f'cgroup raw={before["raw"] / 2**20:.0f} / {limit / 2**20:.0f} MiB, '
            f'file cache estimate={before["reclaimable"] / 2**20:.0f} MiB, '
            f'guarded usage={current_guarded / 2**20:.0f} MiB '
            f'(threshold={threshold / 2**20:.0f} MiB). '
            'This is actual memory pressure, not merely raw page cache. '
            'See the MEMORY log for main RSS/anon/shmem and restart the app '
            'if there are stuck processes.')

    raw_peak = before['raw'] or 0
    guarded_peak = current_guarded or 0
    killed_by_guard = False
    proc = subprocess.Popen(cmd, env=env)
    try:
        while proc.poll() is None:
            stats = _cgroup_memory_snapshot()
            if stats['raw'] is not None:
                raw_peak = max(raw_peak, stats['raw'])
            if stats['guarded'] is not None:
                guarded_peak = max(guarded_peak, stats['guarded'])
            if (threshold is not None and stats['guarded'] is not None
                    and stats['guarded'] > threshold):
                logger.error('MEMORY GUARD in %s: guarded=%.0f MiB / limit %.0f MiB '
                             '(raw=%.0f, reclaimable cache=%.0f); stopping worker',
                             kind, stats['guarded']/2**20, limit/2**20,
                             (stats['raw'] or 0)/2**20, stats['reclaimable']/2**20)
                killed_by_guard = True
                proc.kill()
                break
            time.sleep(0.15)
        retcode = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    gc.collect()
    log_memory(f'after {kind}; raw peak {raw_peak / 2**20:.0f} MiB; '
               f'guarded peak {guarded_peak / 2**20:.0f} MiB')
    if killed_by_guard:
        raise MemoryError(f'{kind} exceeded guarded memory threshold: '
                          f'guarded peak {guarded_peak / 2**20:.0f} MiB / '
                          f'limit {limit / 2**20:.0f} MiB. '
                          'If this was Whisper, use STOCK_APPROX_TIMINGS=1. '
                          'If Kokoro truly needs more RAM, move TTS to an API or upgrade memory.')
    if retcode != 0:
        detail = 'OS killed the worker (likely OOM).' if retcode in (-9, 137) else 'See worker traceback above.'
        raise RuntimeError(f'{kind} worker exited {retcode}. {detail} '
                           f'Raw peak {raw_peak / 2**20:.0f} MiB; '
                           f'guarded peak {guarded_peak / 2**20:.0f} MiB.')


# =====================================================================
# 1. CONFIG
# =====================================================================
# --- Video ---
VIDEO_W, VIDEO_H = 1080, 1920   # design coordinates (do not change)
OUTPUT_W, OUTPUT_H = 720, 1280  # lower-memory final output; use 1080, 1920 if you have more RAM
FPS = 24
CHART_RENDER_FPS = 12              # the matplotlib chart is redrawn at this rate, frames are repeated
BG_DIM = 0.22                      # brightness multiplier for the background video
BG_VIDEO_DIR = "assets/trading_floor_videos"
X264_PRESET = "ultrafast"
CHART_POINTS = 600                 # smooth-path resolution (PCHIP interpolation)

# --- Logos ---
LOGOS_DIR = "assets/logos"         # downloaded logos are cached here as <TICKER>.png

# --- Captions (active-word highlight) ---
CAPTION_FONT_SIZE = 52
CAPTION_INACTIVE_COLOR = "#FFFFFF"  # words not being spoken right now
CAPTION_ACTIVE_COLOR = "#FFE600"    # the word being spoken right now
CAPTION_ACTIVE_BY_DIRECTION = False  # True -> active word is green (stock up) / red (stock down) instead

# --- News headline popup card ---
NEWS_CARD_LABEL = "LATEST NEWS"    # change to "BREAKING NEWS" if your feed is always fresh
NEWS_CARD_X = 60                   # resting x position (card is 960px wide, slides in from the right)
NEWS_CARD_Y = 1110                 # sits between the price badge and the captions
NEWS_CARD_SLIDE = 0.35             # slide in / out duration in seconds
NEWS_CARD_START = 0.35             # card appears this many seconds into the video
NEWS_CARD_MIN_LEN, NEWS_CARD_MAX_LEN = 3.5, 12.0

# --- LLM ---
GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]   # tried in order, per round
LLM_MAX_ROUNDS = 5
LLM_BASE_DELAY = 2.0
LLM_MAX_DELAY = 30.0
LLM_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

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


def ascii_clean(text: str) -> str:
    """Normalises typographic characters and drops anything the bold fonts may not render (emoji etc.)."""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", "", text)).strip()


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


# Google News RSS, based on the RSS method in the daily-recap video.
# Keep the original (prompt_context, news_items) interface unchanged so Gemini
# and the existing animated news card require no changes.
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
RSS_NEWS_TIMEOUT = 7  # seconds per request; three searches run concurrently


def _rss_company_terms(symbol: str) -> List[str]:
    """Recognizable company names to search and verify in Google News."""
    company = COMPANY_NAMES.get(symbol, symbol).strip()
    choices = list(COMPANY_NEWS_ALIASES.get(symbol, ()))
    if company and company != symbol:
        choices.append(company)
    if company and company != symbol and len(company.split()[0]) >= 4:
        choices.append(company.split()[0])
    return list(dict.fromkeys(c for c in choices if c))


def _rss_queries(symbol: str) -> List[str]:
    """Company-specific searches (unlike the broad market RSS used by the other video)."""
    terms = _rss_company_terms(symbol)
    queries = [f'"{symbol}" stock']
    if terms:
        queries.append(f'"{terms[0]}" stock')
        queries.append(f'"{terms[0]}" shares OR earnings')
    return list(dict.fromkeys(queries))


def _fetch_rss_headlines(query: str, days: int = 1) -> List[dict]:
    """Read Google News RSS like the second script; include publisher and timestamp."""
    url = (
        'https://news.google.com/rss/search?q='
        + urllib.parse.quote_plus(f'{query} when:{days}d')
        + '&hl=en-US&gl=US&ceid=US:en'
    )
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
            # No fabricated RSS summaries: Google News' <description> is
            # often an HTML list of links, not a real article synopsis.
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
    """Prevent Google News search from slipping in unrelated market headlines."""
    for term in [symbol, *_rss_company_terms(symbol)]:
        # Word/phrase boundaries are important for symbols such as ARM / META.
        if re.search(r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])',
                     title, flags=re.IGNORECASE):
            return True
    return False


def _fetch_stock_rss_items(symbol: str, days: int) -> List[dict]:
    queries = _rss_queries(symbol)
    # RSS requests are low-memory I/O; run them concurrently instead of serially.
    with ThreadPoolExecutor(max_workers=min(3, len(queries))) as pool:
        lists = list(pool.map(lambda q: _fetch_rss_headlines(q, days), queries))

    now_ts = datetime.now(timezone.utc).timestamp()
    # Google occasionally serves stale results, even with when:1d/7d.
    # Allow a small cushion for timezone/feed indexing differences.
    max_age = (days * 24 + 8) * 3600
    seen = set()
    articles = []
    for results in lists:
        for item in results:
            title = item['title']
            if not _rss_title_matches_company(title, symbol):
                continue
            published_ts = item.get('_published_ts') or 0.0
            if published_ts and (now_ts - published_ts > max_age or published_ts > now_ts + 7200):
                continue
            key = re.sub(r'[^a-z0-9]+', '', title.casefold())
            if key in seen:
                continue
            seen.add(key)
            articles.append(item)
    articles.sort(key=lambda item: item.get('_published_ts') or 0.0, reverse=True)
    return articles


def format_news_context(items: List[dict]) -> str:
    """Prompt text in the original structure consumed by the existing script."""
    if not items:
        return 'No recent company-specific news found. Do not invent a cause for today\'s move.'
    blocks = []
    for it in items:
        published = it.get('published')
        date_note = f', published {published[:10]} UTC' if published else ''
        line = f"• [{it['provider']}{date_note}] {it['title']}"
        if it.get('summary'):
            line += f"\n  Context: {it['summary']}"
        blocks.append(line)
    return ('Company news headlines from Google News RSS (headlines are NOT proof '
            'that an event caused today\'s stock move):\n' + '\n\n'.join(blocks))


def fetch_news_context(symbol: str) -> Tuple[str, List[dict]]:
    """Return the SAME (prompt text, card items) as before, now via Google RSS."""
    symbol = symbol.strip().upper()
    logger.info('📰 Fetching %s news via Google News RSS (past 24h)...', symbol)
    fresh = _fetch_stock_rss_items(symbol, days=1)
    if len(fresh) >= 3:
        articles = fresh[:RSS_NEWS_MAX_ARTICLES]
    else:
        # Weekend / slow news cycle fallback; dated in the Gemini context.
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

    # Drop private ranking metadata; the card needs provider/title/summary only.
    items = [{k: v for k, v in item.items() if not k.startswith('_')}
             for item in articles[:RSS_NEWS_MAX_ARTICLES]]
    logger.info('✅ %s: received %d relevant RSS headlines.', symbol, len(items))
    if items:
        logger.info('Top RSS headline: %s (%s)', items[0]['title'], items[0]['provider'])
    return format_news_context(items), items


def get_company_name(symbol: str) -> str:
    if symbol in COMPANY_NAMES:
        return COMPANY_NAMES[symbol]
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol


def get_ticker_logo(ticker):
    """Downloads a transparent logo automatically from a CDN, validating the file."""
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

    with ThreadPoolExecutor(max_workers=1) as ex:
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
    direction = "UP" if stock_data["change_pct"] >= 0 else "DOWN"
    prompt = f"""
        You are a high-energy financial content creator and YouTube SEO expert for TikTok, YouTube Shorts, and Instagram Reels.
        Create a fast-paced 45-second video script and SEO-optimized metadata about this stock movement:
        - Symbol: {stock_data['symbol']}
        - Company Name: {stock_data['company_name']}
        - Current Price: ${stock_data['current_price']}
        - Today's Change: {stock_data['change_pct']}% (the stock is {direction} today)
        - News Context: {stock_data['news']}

        CRITICAL SCRIPTING & SEO RULES:
        1. YOUTUBE SEARCH SEO OPTIMIZATION (MAXIMIZE SEARCH TRAFFIC):
           - "youtube_title" MUST explicitly contain BOTH the Ticker Symbol AND the Full Company Name (e.g., "AVGO Broadcom Shocking Drop! Buy Or Pass? 🚀") so it ranks high in YouTube Search.
           - "description" MUST naturally incorporate both the ticker and company name alongside key financial terms.
           - "tags" MUST include the ticker, company name, specific stock tags (e.g., AVGO, Broadcom, BroadcomStock), and general investing tags.

        2. IMMEDIATE ANSWER HOOK (FIRST 3 SECONDS):
           - The FIRST sentence must immediately say WHAT the stock did and WHY it moved today, using the real reason found in the News Context above. Example shape: "Nvidia just dropped more than 5 percent today because [the actual reason from the news]."
           - Use the company's plain name (not just the ticker) in that first sentence.
           - NO generic high-stakes filler, NO vague teasers like "you won't believe what happened". The first sentence must carry real substance.
           - NEVER invent a catalyst. If the News Context contains no clear reason, say the stock is moving sharply today and honestly note that no single headline explains it yet, then give the most relevant market-wide factor.

        3. POLARIZING STORYTELLING & MARKET CONTROVERSY (BULL VS. BEAR):
           - After the catalyst, show the tension: Explain why the Bears think this stock is heading for a crash, while the Bulls argue it is a lifetime opportunity.
           - Focus on big market forces, macro trends, and major institutions (Wall Street, Big Tech, The Fed) rather than obscure individuals.

        4. OPTIMIZED FOR TTS (RETENTION & FLOW):
           - Write purely in natural, spoken English.
           - ABSOLUTELY NO dashes (-), ellipses (...), semicolons (;), or complex formatting that causes Text-To-Speech (TTS) engines to pause awkwardly.
           - Keep punctuation strictly to simple periods, commas and the final question mark, for a relentless, fast pace.
           - Do not use decimal points. Round everything to speakable numbers. For example, instead of "+1.64%" say "the stock is up by more than 1 percent", and do the same for prices.
           - Avoid abbreviations that contain periods (write "US" not "U.S.").

        5. HIGH-ENGAGEMENT PRICE TARGET CTA (COMMENT MAGNET):
           - End the script with a direct question asking viewers for their PRICE TARGET or their AVERAGE BUY PRICE for {stock_data['symbol']}.
           - Example: "What is your price target for {stock_data['symbol']}? Drop your target or average buy price in the comments right now!"
           - Do NOT use the generic "Type BUY or SELL" call to action. You may vary the wording, but it must ask for a price target or a buy price.

        6. CATALYST SEGMENT MARKER:
           - "catalyst_sentences" is an integer (1 or 2): how many of the OPENING sentences of the voiceover explain the news catalyst. The video shows a news headline card during exactly that part.

        7. STRICT OUTPUT FORMAT:
           - Return ONLY a valid JSON object with the following 6 fields:
        {{
          "youtube_title": "High-CTR & Search-Optimized title with TICKER + COMPANY NAME + Emojis",
          "overlay_headline": "2 to 4 WORDS IN ALL CAPS FOR ON-SCREEN OVERLAY (e.g., AVGO HUGE CRASH!)",
          "voiceover_text": "The full fluid script: immediate why-it-moved hook, bull vs bear story beat, price target CTA",
          "description": "SEO-rich summary with Ticker and Company Name, ending with a call to share your price target in the comments",
          "tags": "5-7 relevant tags separated by commas including Ticker and Full Company Name",
          "catalyst_sentences": 1
        }}
        """
    return ask_llm(prompt, config=types.GenerateContentConfig(response_mime_type="application/json"),
                   parse=_parse_script)


# =====================================================================
# 7. TEXT-TO-SPEECH (Kokoro-82M, local)
# =====================================================================
def clean_script_for_tts(text: str) -> str:
    """Removes non-ASCII characters/emojis & converts symbols for clear TTS."""
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = text.replace("$", "").replace("%", " percent")
    for pattern, replacement in TTS_SYMBOL_NAMES.items():
        text = re.sub(pattern, replacement, text)
    text = text.replace('\n', ' ').strip()
    return text


def generate_voiceovers(items: List[Tuple[str, str]]) -> List[float]:
    """Compatibility wrapper: run the actual TTS model in a short-lived process."""
    with tempfile.TemporaryDirectory(prefix="stock_tts_job_") as td:
        job, result = Path(td) / "job.json", Path(td) / "durations.json"
        job.write_text(json.dumps([(clean_script_for_tts(str(txt)), str(path)) for txt, path in items]), encoding="utf-8")
        run_memory_worker("tts", str(job), str(result))
        return json.loads(result.read_text(encoding="utf-8"))


# =====================================================================
# 8. WORD TIMINGS (Whisper word timestamps)
# =====================================================================
def approximate_word_timings(script_text: str, duration: float) -> List[dict]:
    """Memory-free fallback; captions are approximate rather than ASR-aligned."""
    tokens = clean_script_for_tts(script_text).split()
    if not tokens:
        return []
    weights = [max(1.0, math.sqrt(len(re.sub(r'[^A-Za-z]', '', token)) or 1)) for token in tokens]
    unit = max(0.01, duration - 0.2) / sum(weights)
    words, cursor = [], 0.0
    for token, weight in zip(tokens, weights):
        end = min(duration, cursor + weight * unit)
        clean = re.sub(r'^[^\w$%]+|[^\w$%]+$', '', token)
        words.append({'raw': token, 'text': clean, 'start': cursor, 'end': end})
        cursor = end
    return words


def transcribe_words(audio_file: str, script_text: str = '', duration: float = 0.0) -> List[dict]:
    """Whisper in a lean process; optional low-RAM approximate timings mode."""
    if os.getenv('STOCK_APPROX_TIMINGS', '0') == '1':
        logger.warning('Skipping Whisper by STOCK_APPROX_TIMINGS=1; captions are approximate.')
        return approximate_word_timings(script_text, duration)
    with tempfile.TemporaryDirectory(prefix='stock_whisper_job_') as td:
        result = Path(td) / 'words.json'
        try:
            run_memory_worker('whisper', str(audio_file), str(result))
        except (MemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, MemoryError) or 'OS killed the worker' in str(exc)
            if not is_oom or not script_text or not duration:
                raise
            logger.warning('Whisper ran out of memory: using approximate caption timings.')
            return approximate_word_timings(script_text, duration)
        return json.loads(result.read_text(encoding='utf-8'))


def catalyst_window(words: List[dict], n_sentences: int, duration: float) -> Tuple[float, float]:
    """Time window during which the script talks about the catalyst: from the start until the end of the
    n-th sentence (detected from sentence-ending punctuation in Whisper's output)."""
    start = NEWS_CARD_START
    end = None
    count = 0
    for w in words:
        if re.search(r"[.!?][\"')\]]*$", w["raw"]):
            count += 1
            if count >= n_sentences:
                end = w["end"] + 0.2
                break
    if end is None:                                   # no punctuation found: fall back to ~7 seconds
        end = min(duration, 7.0)
    end = min(max(end, start + NEWS_CARD_MIN_LEN), start + NEWS_CARD_MAX_LEN, duration - 0.1)
    return start, end


# =====================================================================
# 9. GRAPHICS  (header + overlay PNG, news card PNG, word-highlight caption sprites)
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


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> str:
    """Shortens text with '...' until it fits max_width."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    while len(text) > 1 and draw.textlength(text + "...", font=font) > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


def prepare_logo(path: Optional[str], size: int) -> Optional[Tuple[Image.Image, float]]:
    """Trims transparent borders, pads to a square with transparent padding and resizes.
    Returns (RGBA logo of size x size, mean luminance of its visible pixels) or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        bbox = img.getchannel("A").getbbox()
        if not bbox:
            return None
        img = img.crop(bbox)
        side = max(img.size)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        square = square.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(square).astype(np.float32)
        mask = arr[:, :, 3] > 40
        lum = float((0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])[mask].mean()) \
            if mask.any() else 0.0
        return square, lum
    except Exception as e:
        logger.warning(f"⚠️ Could not process logo {path}: {e}")
        return None


def draw_header_banner(canvas: Image.Image, stock_data: dict, logo_path: Optional[str], accent: str) -> None:
    """Top header: [logo tile]  TICKER / Company name. Sits above the chart area (y 60..210)."""
    w, h = 960, 150
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=32, fill=(20, 26, 38, 238), outline=accent, width=3)

    text_x = 40
    logo = prepare_logo(logo_path, 104)
    if logo:
        sq, lum = logo
        tile_size = 132
        # white logos get a dark tile, everything else a white tile so it always stays readable
        tile_color = (30, 38, 51, 255) if lum > 215 else (255, 255, 255, 255)
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        ImageDraw.Draw(tile).rounded_rectangle((0, 0, tile_size - 1, tile_size - 1), radius=26, fill=tile_color)
        tile.alpha_composite(sq, ((tile_size - 104) // 2, (tile_size - 104) // 2))
        layer.alpha_composite(tile, (24, (h - tile_size) // 2))
        text_x = 24 + tile_size + 28

    max_w = w - text_x - 30
    ticker_font, name_font = load_system_font(68), load_system_font(34)
    ticker = _fit_text(d, stock_data["symbol"], ticker_font, max_w)
    name = _fit_text(d, ascii_clean(stock_data.get("company_name", "")), name_font, max_w)
    d.text((text_x, int(h * 0.38)), ticker, font=ticker_font, fill="white", anchor="lm")
    d.text((text_x, int(h * 0.76)), name, font=name_font, fill="#9AA7BD", anchor="lm")

    canvas.alpha_composite(layer, ((canvas.width - w) // 2, 60))


def create_overlay_graphics(script_data: dict, stock_data: dict, output_path: str,
                            canvas_size: Tuple[int, int] = (VIDEO_W, VIDEO_H)) -> str:
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    pct = stock_data["change_pct"]

    # ---- header banner: logo + ticker + company name ----
    if "logo_path" not in stock_data:            # normally prefetched in the thread pool
        stock_data["logo_path"] = get_ticker_logo(stock_data["symbol"])
    draw_header_banner(canvas, stock_data, stock_data.get("logo_path"), C_UP if pct >= 0 else C_DOWN)

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


# ---------------- News headline popup card ----------------
def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: float, max_lines: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and draw.textlength(last + "...", font=font) > max_w:
            last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
        lines[-1] = last + "..."
    return lines


def create_news_card(item: dict, change_pct: float, output_path: str, width: int = 960) -> Optional[str]:
    """Styled 'news headline' card (transparent PNG). No dates/timestamps are drawn."""
    title = ascii_clean(item.get("title", ""))
    if len(title) < 10:
        return None
    provider = ascii_clean(item.get("provider", "")) or "Market News"
    accent = C_UP if change_pct >= 0 else C_DOWN

    pad, pill_h, line_h = 36, 50, 58
    label_font, prov_font, title_font = load_system_font(28), load_system_font(28), load_system_font(46)
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines = _wrap_text(probe, title, title_font, width - 2 * pad - 8, max_lines=3)
    height = pad + pill_h + 24 + line_h * len(lines) + pad - 8

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, width - 1, height - 1), radius=30, fill=(20, 26, 38, 244), outline=accent, width=4)

    # label pill (top-left) + source (top-right)
    label_w = int(d.textlength(NEWS_CARD_LABEL, font=label_font)) + 40
    d.rounded_rectangle((pad, pad, pad + label_w, pad + pill_h), radius=pill_h // 2, fill="#FF1744")
    d.text((pad + label_w // 2, pad + pill_h // 2), NEWS_CARD_LABEL, font=label_font, fill="white", anchor="mm")
    prov = _fit_text(d, provider, prov_font, width - 2 * pad - label_w - 30)
    d.text((width - pad, pad + pill_h // 2), prov, font=prov_font, fill="#9AA7BD", anchor="rm")

    # headline
    y = pad + pill_h + 24
    for line in lines:
        d.text((pad, y), line, font=title_font, fill="white", anchor="lt")
        y += line_h

    img.save(output_path)
    return output_path


def build_news_card_plan(stock_data: dict, script_data: dict, words: List[dict], duration: float,
                         out_path: str) -> Optional[Tuple[str, float, float]]:
    """(png_path, start, end) for the news card, or None when there is no usable headline."""
    items = stock_data.get("news_items") or []
    if not items:
        logger.info("📰 No news headline available - skipping the news card.")
        return None
    png = create_news_card(items[0], stock_data["change_pct"], out_path)
    if not png:
        return None
    try:
        n_sentences = max(1, min(3, int(script_data.get("catalyst_sentences") or 2)))
    except (TypeError, ValueError):
        n_sentences = 2
    start, end = catalyst_window(words, n_sentences, duration)
    logger.info(f"📰 News card: {start:.2f}s -> {end:.2f}s (catalyst = first {n_sentences} sentence(s))")
    return png, start, end


# ---------------- Word-highlight captions ----------------
@dataclass
class CaptionSprite:
    start: float
    end: float
    x: int
    y: int
    rgba: np.ndarray   # (h, w, 4) uint8, straight alpha, cropped to the text


def build_caption_sprites(words: List[dict], words_per_caption: int = 2,
                          box: Tuple[int, int] = (1000, 220), top: int = 1420,
                          active_color: str = CAPTION_ACTIVE_COLOR,
                          inactive_color: str = CAPTION_INACTIVE_COLOR) -> List[CaptionSprite]:
    """
    Groups words into captions of `words_per_caption`, then renders ONE sprite per spoken word: the same
    layout every time, with only the currently spoken word in `active_color`. Sprites are strictly
    time-ordered and non-overlapping, so the renderer can pick one per frame. Everything is pre-rendered once.
    """
    spoken = [w for w in words if w["text"]]
    box_x = (VIDEO_W - box[0]) // 2
    sprites: List[CaptionSprite] = []
    last_end = 0.0

    for g in range(0, len(spoken), words_per_caption):
        group = spoken[g:g + words_per_caption]
        texts = [w["text"].upper() for w in group]

        group_end = group[-1]["end"]
        if g + words_per_caption < len(spoken):                       # keep a tiny tail, never overlap the next
            group_end = min(group_end + 0.05, spoken[g + words_per_caption]["start"])

        # shared layout: shrink the font until the whole caption fits the box
        size = CAPTION_FONT_SIZE
        while True:
            font = load_system_font(size)
            probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
            space = probe.textlength(" ", font=font)
            widths = [probe.textlength(t, font=font) for t in texts]
            total = sum(widths) + space * (len(texts) - 1)
            if total <= box[0] - 40 or size <= 28:
                break
            size -= 4
        x = (box[0] - total) / 2
        xs = []
        for wd in widths:
            xs.append(x)
            x += wd + space

        for j, word in enumerate(group):
            t0 = max(word["start"], last_end)
            t1 = group[j + 1]["start"] if j + 1 < len(group) else group_end
            if t1 - t0 <= 0.02:
                continue

            img = Image.new("RGBA", box, (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            for k, t in enumerate(texts):
                draw.text((xs[k], box[1] / 2), t, font=font, anchor="lm",
                          fill=active_color if k == j else inactive_color,
                          stroke_width=8, stroke_fill="black")
            arr = np.asarray(img)
            rows, cols = np.where(arr[:, :, 3] > 0)
            if len(rows) == 0:
                continue
            y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            sprites.append(CaptionSprite(t0, t1, box_x + int(x0), top + int(y0),
                                         np.ascontiguousarray(arr[y0:y1, x0:x1])))
            last_end = t1
    return sprites


def blend_sprite_over(frame_rgba: np.ndarray, sprite: CaptionSprite) -> None:
    """Composite a caption over only its bounding rectangle, using Pillow's C code."""
    h, w = sprite.rgba.shape[:2]
    if h == 0 or w == 0:
        return
    region = frame_rgba[sprite.y:sprite.y + h, sprite.x:sprite.x + w]
    if region.shape != sprite.rgba.shape:
        raise ValueError('Caption sprite is outside chart bounds')
    background = Image.fromarray(region.copy(), 'RGBA')
    foreground = Image.fromarray(sprite.rgba, 'RGBA')
    blended = Image.alpha_composite(background, foreground)
    region[...] = np.asarray(blended)



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
    if os.getenv('STOCK_DISABLE_BG_VIDEO', '0') == '1':
        logger.warning('Background video disabled for low-memory rendering.')
        return None
    files = sorted(Path(BG_VIDEO_DIR).glob("*.mp4"))
    if not files:
        logger.warning(f"No background videos in {BG_VIDEO_DIR}; using a plain dark background.")
        return None
    return str(random.choice(files))


def _news_card_overlay(start: float, end: float) -> str:
    """ffmpeg overlay filter: card slides in from the right at `start`, slides back out before `end`.
    Done inside ffmpeg (no per-frame Python work)."""
    s = NEWS_CARD_SLIDE
    p_in = f"clip((t-{start:.3f})/{s},0,1)"
    p_out = f"clip((t-{end - s:.3f})/{s},0,1)"
    render_scale = OUTPUT_W / VIDEO_W
    resting_x = round(NEWS_CARD_X * render_scale)
    card_y = round(NEWS_CARD_Y * render_scale)
    x = f"{resting_x}+({OUTPUT_W}-{resting_x})*(pow(1-{p_in},2)+pow({p_out},2))"
    return f"overlay=x='{x}':y={card_y}:enable='between(t,{start:.3f},{end:.3f})':format=auto"


def render_video(chart: ChartRenderer, sprites: List[CaptionSprite], overlay_png: str, bg_path: Optional[str],
                 audio_path: str, duration: float, out_path: str,
                 news_card: Optional[Tuple[str, float, float]] = None) -> None:
    n_frames = max(1, round(duration * FPS))
    scale = OUTPUT_W / VIDEO_W
    if abs(OUTPUT_H / VIDEO_H - scale) > 1e-6:
        raise ValueError("Output and design aspect ratios must match")

    # Keep all chart/typography layout in the original 1080x1920 design coordinates;
    # scale the composited raw frames and ffmpeg's PNG assets only at render time.
    if scale != 1:
        small_overlay = str(Path(overlay_png).with_name("_render_overlay.png"))
        with Image.open(overlay_png) as im:
            im.resize((OUTPUT_W, OUTPUT_H), Image.Resampling.LANCZOS).save(small_overlay)
        overlay_png = small_overlay
        if news_card:
            card_path, c_start, c_end = news_card
            small_card = str(Path(card_path).with_name("_render_news_card.png"))
            with Image.open(card_path) as im:
                im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                          Image.Resampling.LANCZOS).save(small_card)
            news_card = (small_card, c_start, c_end)

    # which caption sprite is visible on each output frame (-1 = none)
    cap_for_frame = np.full(n_frames, -1, dtype=np.int32)
    for i, s in enumerate(sprites):
        f0 = int(math.ceil(s.start * FPS - 1e-9))
        f1 = int(math.ceil(s.end * FPS - 1e-9))
        cap_for_frame[max(0, f0):max(0, min(n_frames, f1))] = i

    bg_input = (["-stream_loop", "-1", "-i", bg_path] if bg_path
                else ["-f", "lavfi", "-i", f"color=c=0x0B0E14:s={OUTPUT_W}x{OUTPUT_H}:r={FPS}"])
    # background: cover-fit to 1080x1920 (scale to height, centre crop), dim; then chart layer, then overlay PNG,
    # then (optionally) the sliding news card
    graph = (f"[0:v]fps={FPS},scale={OUTPUT_W}:{OUTPUT_H}:force_original_aspect_ratio=increase,"
             f"crop={OUTPUT_W}:{OUTPUT_H},colorchannelmixer=rr={BG_DIM}:gg={BG_DIM}:bb={BG_DIM},setsar=1[bg];"
             "[bg][1:v]overlay=format=auto[a];")
    extra_inputs: List[str] = []
    if news_card:
        card_png, c_start, c_end = news_card
        extra_inputs = ["-loop", "1", "-framerate", str(FPS), "-i", card_png]
        graph += f"[a][2:v]overlay=format=auto[b];[b][3:v]{_news_card_overlay(c_start, c_end)}[v]"
        audio_idx = 4
    else:
        graph += "[a][2:v]overlay=format=auto[v]"
        audio_idx = 3

    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-filter_complex_threads", "1", "-threads", "1", *bg_input,
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{OUTPUT_W}x{OUTPUT_H}", "-r", str(FPS), "-i", "-",
           "-loop", "1", "-framerate", str(FPS), "-i", overlay_png,
           *extra_inputs,
           "-i", audio_path,
           "-filter_complex", graph, "-map", "[v]", "-map", f"{audio_idx}:a", "-t", f"{n_frames / FPS:.6f}",
           "-c:v", "libx264", "-threads:v", "1", "-preset", X264_PRESET,
           "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]

    base_tick, base, last_key, frame, encoded_frame = None, None, None, None, None
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
        try:
            for i in range(n_frames):
                if i and i % (FPS * 8) == 0:
                    log_memory(f"render {i / FPS:.0f}s/{duration:.0f}s")
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
                    if scale == 1:
                        encoded_frame = frame
                    else:
                        # Downsize the composite once per unique frame, not every 1/FPS repeat.
                        encoded_frame = Image.fromarray(frame).resize(
                            (OUTPUT_W, OUTPUT_H), Image.Resampling.BILINEAR).tobytes()
                    last_key = key
                proc.stdin.write(encoded_frame)
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
def _create_stock_short_unlocked(upload: bool = False, test: bool = False,
                       on_preview: Optional[Callable[[str], None]] = None) -> str:
    work_dir = Path(tempfile.mkdtemp(prefix="stock_short_"))
    try:
        # ---- Phase A: data + audio. ML models run sequentially in isolated processes. ----
        log_memory("pipeline start")
        with ThreadPoolExecutor(max_workers=1) as pool:

            stock_data = get_top_moving_stock(test=test)
            symbol = stock_data["symbol"]
            news_f = pool.submit(fetch_news_context, symbol)
            name_f = pool.submit(get_company_name, symbol)
            logo_f = pool.submit(get_ticker_logo, symbol)            # downloads/validates while the LLM works
            metrics_f = pool.submit(fetch_market_metrics, symbol)   # runs while the LLM + TTS work below
            stock_data["news"], stock_data["news_items"] = news_f.result()
            stock_data["company_name"] = name_f.result()

            logger.info("🤖 Generating script...")
            script_data = generate_script_and_titles_with_ai(stock_data)
            logger.info("=" * 60)
            logger.info(f"📌 YOUTUBE TITLE: {script_data.get('youtube_title')}")
            logger.info(f"🎬 OVERLAY HEADLINE: {script_data.get('overlay_headline')}")
            logger.info("=" * 60)

            audio_file = work_dir / "voiceover.wav"   # Kokoro writes WAV
            duration = generate_voiceovers([(script_data["voiceover_text"], audio_file)])[0]

            # Whisper is launched only after Kokoro has exited, so their RAM never overlaps.
            words = transcribe_words(str(audio_file), script_data["voiceover_text"], duration)
            stock_data["logo_path"] = logo_f.result()                # None -> header is drawn without a logo
            overlay_png = create_overlay_graphics(script_data, stock_data, str(work_dir / "overlay.png"))
            news_card = build_news_card_plan(stock_data, script_data, words, duration,
                                             str(work_dir / "news_card.png"))
            market_metrics = metrics_f.result()
        logger.info(f"Market metrics: {market_metrics}")

        # ---- Phase B: render ----
        log_memory("before video renderer")
        logger.info(f"📊 Rendering {len(words)} highlighted words + animated chart ({duration:.1f}s)...")
        chart = ChartRenderer(stock_data, duration, market_metrics)
        active_color = (C_UP if stock_data["change_pct"] >= 0 else C_DOWN) if CAPTION_ACTIVE_BY_DIRECTION \
            else CAPTION_ACTIVE_COLOR
        sprites = build_caption_sprites(words, words_per_caption=2, active_color=active_color)
        output_filename = f"STOCK_{symbol}_{int(stock_data['change_pct'])}pct.mp4"
        render_video(chart, sprites, overlay_png, pick_background_video(), str(audio_file), duration,
                     output_filename, news_card=news_card)
        chart.fig.clear()
        del chart, sprites
        gc.collect()
        log_memory("after video renderer")
        logger.info(f"✅ SUCCESS! Video saved as: {output_filename}")

        if on_preview:
            png = extract_frame(output_filename, min(2.0, duration / 2), str(work_dir / "preview.png"))
            if png:
                on_preview(png)

        # ---- Phase C: upload ----
        if upload:
            from uploader import upload_video  # defer potentially heavy dependency until actually needed
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



def create_stock_short(upload: bool = False, test: bool = False,
                       on_preview: Optional[Callable[[str], None]] = None) -> str:
    """Allow one pipeline at a time across Streamlit sessions in this container."""
    if sys.platform == 'linux':
        import fcntl
        lock_path = Path(tempfile.gettempdir()) / 'stock_short_pipeline.lock'
        with lock_path.open('w') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Another video is being generated. Try again after it finishes.') from exc
            try:
                return _create_stock_short_unlocked(upload=upload, test=test, on_preview=on_preview)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return _create_stock_short_unlocked(upload=upload, test=test, on_preview=on_preview)


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
