"""
thumbnail.py - shared YouTube thumbnail engine for the Daily Shorts (9:16) and Weekly Recap (16:9) generators.

Pure Pillow + NumPy (no Matplotlib, no Torch), so it is safe for the low-RAM Streamlit setup.

    from thumbnail import ThumbnailEngine, LANDSCAPE, PORTRAIT

    ThumbnailEngine(spec, size=LANDSCAPE).save("thumb.png")                       # weekly, 1280x720
    ThumbnailEngine(spec, size=PORTRAIT, candles=df, pct=3.2, pct_label="TODAY")   # daily, 1080x1920

`spec` is a dict or a pydantic model with: template, main_text, primary_ticker, secondary_ticker, trend,
avatar_emotion.  Templates
    landscape: divergence, chart_alert, vs_showdown, warning_sign, magnifying_glass, massive_number, candle_hero
    portrait : candle_hero, massive_number
candle_hero = real candlestick chart of the featured ticker + company logo + verified % badge + a surprised face
+ big headline. The trend colour is taken from the verified data, not from the LLM, so it never lies.

Assets (never fatal if missing)
    assets/avatar/<shocked|pointing|thinking|neutral>.png   transparent cut-outs (cropped automatically)
    assets/avatar/open.png                                   last-resort avatar (the one the videos already use)
    assets/logos/<TICKER>.png                                logos (downloaded best-effort when missing)
    assets/fonts/Anton-Regular.ttf                           headline font if Impact is not installed

Try it:  python thumbnail.py --demo out_folder
"""
from __future__ import annotations

import logging
import math
import os
import random
import re
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logger = logging.getLogger("Thumbnail")

# =====================================================================
# 1. CONFIG
# =====================================================================
LANDSCAPE = (1280, 720)  # weekly recap
PORTRAIT = (1080, 1920)  # daily Short (9:16)
THUMB_MAX_BYTES = 2_000_000  # YouTube rejects thumbnails over 2 MB
THUMB_TEXT_MAX_CHARS = 26  # 3 words max AND this many characters, so the headline stays huge

LANDSCAPE_TEMPLATES = ("divergence", "chart_alert", "vs_showdown", "warning_sign", "magnifying_glass",
                       "massive_number", "candle_hero")
PORTRAIT_TEMPLATES = ("candle_hero", "massive_number")
EMOTIONS = ("shocked", "pointing", "thinking", "neutral")
TRENDS = ("bullish", "bearish", "neutral")

AVATAR_PATHS = {e: f"assets/avatar/{e}.png" for e in EMOTIONS}
AVATAR_LAST_RESORT = ("assets/avatar/open.png", "assets/avatar/closed.png")
LOGO_DIR = Path("assets/logos")

WHITE, YELLOW, BLACK = (255, 255, 255), (255, 214, 10), (0, 0, 0)
BASE = (8, 10, 16)  # near-black every gradient starts from


def _rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


TREND_RGB = {"bullish": _rgb("#00FFA3"), "bearish": _rgb("#FF3366"), "neutral": _rgb("#FFD700")}
UP_RGB, DOWN_RGB = TREND_RGB["bullish"], TREND_RGB["bearish"]
BLUE_RGB, ORANGE_RGB, GOLD_RGB = _rgb("#448AFF"), _rgb("#FF9100"), _rgb("#FFD700")
FONT_CANDIDATES = (  # Impact-style display fonts first; a bold system font is the fallback
    "assets/fonts/Anton-Regular.ttf", "assets/fonts/Impact.ttf", r"C:\Windows\Fonts\impact.ttf",
    "/Library/Fonts/Impact.ttf", "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    r"C:\Windows\Fonts\arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", "/Library/Fonts/Arial Bold.ttf",
)
INDEX_LABELS = {"GSPC": "S&P", "IXIC": "NDQ", "DJI": "DOW", "RUT": "R2K", "VIX": "VIX", "TNX": "10Y"}
BADGE_COLORS = ((41, 98, 255), (123, 31, 162), (230, 81, 0), (0, 137, 123), (194, 24, 91), (2, 119, 189))
RESAMPLE = Image.Resampling.LANCZOS


# =====================================================================
# 2. SMALL HELPERS
# =====================================================================
def _ev(x: Any) -> str:
    return str(getattr(x, "value", x))


def clean_ticker(t: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.\-^=]", "", str(t or "")).upper()[:12]


def ascii_clean(text: str) -> str:
    text = text or ""
    for src, dst in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                     ("\u2013", "-"), ("\u2014", "-"), ("\u2026", "...")):
        text = text.replace(src, dst)
    text = text.replace("\\", "")
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", "", text)).strip()


def shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:max(1, n - 3)].rstrip() + "..."


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _get(spec: Any, key: str, default: Any = "") -> Any:
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def spec_to_dict(spec: Any) -> dict:
    keys = ("template", "main_text", "primary_ticker", "secondary_ticker", "trend", "avatar_emotion")
    return {k: _ev(_get(spec, k, "")) for k in keys}


def normalize_spec(spec: Any, portrait: bool = False, have_candles: Optional[bool] = None,
                   default_text: str = "WEEK IN REVIEW") -> dict:
    """Make an LLM-written thumbnail spec safe to draw: at most 3 short UPPERCASE words, clean tickers, a template
    that has the data it needs (and exists for this orientation), the avatar emotion the template is designed
    around. Idempotent. have_candles=None means 'not known yet' (the Director stage); True/False once it is."""
    s = spec_to_dict(spec)
    words = re.sub(r"\[\[[^\]]*\]\]", "", ascii_clean(s["main_text"])).upper().split()[:3]
    while len(words) > 1 and len(" ".join(words)) > THUMB_TEXT_MAX_CHARS:
        words.pop()
    text = shorten(" ".join(words), THUMB_TEXT_MAX_CHARS) or default_text
    primary, secondary = clean_ticker(s["primary_ticker"]), clean_ticker(s["secondary_ticker"])
    if secondary == primary:
        secondary = ""
    if secondary and not primary:
        primary, secondary = secondary, ""
    allowed = PORTRAIT_TEMPLATES if portrait else LANDSCAPE_TEMPLATES
    wanted = template = s["template"] if s["template"] in LANDSCAPE_TEMPLATES else "massive_number"
    if template in ("divergence", "vs_showdown") and not secondary:
        template = "chart_alert"
    if template in ("chart_alert", "magnifying_glass", "candle_hero") and not primary:
        template = "massive_number"
    if template == "candle_hero" and have_candles is False:
        template = "massive_number"
    if template not in allowed:  # e.g. a landscape-only template asked for a Short
        template = "candle_hero" if (primary and have_candles is not False) else "massive_number"
    if template != wanted:
        logger.warning("Thumbnail template '%s' is not possible here (missing tickers/data/orientation); using '%s'",
                       wanted, template)
    emotion = s["avatar_emotion"] if s["avatar_emotion"] in EMOTIONS else "shocked"
    emotion = {"massive_number": "shocked", "chart_alert": "pointing"}.get(template, emotion)
    trend = s["trend"] if s["trend"] in TRENDS else "neutral"
    return dict(template=template, main_text=text, primary_ticker=primary, secondary_ticker=secondary,
                trend=trend, avatar_emotion=emotion)


BEARISH_WORDS = ("CRASH", "DROP", "PLUNGE", "TANK", "DUMP", "COLLAPSE", "SINK", "MELTDOWN", "SELLOFF", "SLUMP",
                 "BLEED", "TUMBLE", "FALL", "CRUSH", "FREEFALL")
BULLISH_WORDS = ("SOAR", "SURGE", "EXPLOD", "ROCKET", "RALLY", "JUMP", "MOON", "BOOM", "PUMP", "BREAKOUT", "SPIKE",
                 "SKYROCKET", "RIP", "SOARS")


def headline_contradicts(text: str, pct: Optional[float], min_move: float = 0.5) -> bool:
    """True when the headline claims the opposite of the VERIFIED move ('CRASH?' on a +7% day)."""
    if pct is None or abs(pct) < min_move:
        return False
    up = text.upper()
    if pct > 0:
        return any(w in up for w in BEARISH_WORDS)
    return any(w in up for w in BULLISH_WORDS)


# =====================================================================
# 3. PILLOW UTILITIES
# =====================================================================
@lru_cache(maxsize=256)
def thumb_font(size: int):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, int(size))
            except OSError:
                continue
    try:
        return ImageFont.load_default(int(size))
    except TypeError:  # old Pillow
        return ImageFont.load_default()


def _shade(rgb, k: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * k))) for c in rgb)


def _lighten(rgb, k: float) -> Tuple[int, int, int]:
    return tuple(int(c + (255 - c) * k) for c in rgb)


def thumb_composite(canvas: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    """Alpha-composite `layer` onto the RGBA `canvas` with its top-left at (x, y); off-canvas parts are clipped."""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.width, x + layer.width), min(canvas.height, y + layer.height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas.alpha_composite(layer.crop((x0 - x, y0 - y, x1 - x, y1 - y)), (x0, y0))


def thumb_paste_center(canvas: Image.Image, layer: Image.Image, cx: float, cy: float) -> None:
    thumb_composite(canvas, layer, int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2)))


def vertical_gradient(size: Tuple[int, int], top, bottom) -> Image.Image:
    w, h = size
    t = np.linspace(0.0, 1.0, max(1, h), dtype=np.float32)[:, None, None]
    a, b = np.array(top, np.float32), np.array(bottom, np.float32)
    rgb = np.broadcast_to(a * (1 - t) + b * t, (h, w, 3))
    return Image.fromarray(np.dstack([rgb, np.full((h, w, 1), 255, np.float32)]).astype(np.uint8))


def sprite_from_mask(fill: Image.Image, mask: Image.Image) -> Image.Image:
    sprite = fill.convert("RGBA")
    sprite.putalpha(mask)
    return sprite


def aa_disk(size: int, center: Tuple[float, float], radius: float, scale: int = 3) -> Image.Image:
    m = Image.new("L", (size * scale, size * scale), 0)
    cx, cy, r = center[0] * scale, center[1] * scale, radius * scale
    ImageDraw.Draw(m).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    return m.resize((size, size), RESAMPLE)


def aa_polygon(size: Tuple[int, int], points: Sequence[Tuple[float, float]], scale: int = 3) -> Image.Image:
    m = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    ImageDraw.Draw(m).polygon([(x * scale, y * scale) for x, y in points], fill=255)
    return m.resize(size, RESAMPLE)


def aa_rounded(size: Tuple[int, int], box: Tuple[float, float, float, float], radius: float,
               scale: int = 3) -> Image.Image:
    m = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    ImageDraw.Draw(m).rounded_rectangle([v * scale for v in box], radius=radius * scale, fill=255)
    return m.resize(size, RESAMPLE)


def dilate_alpha(alpha: Image.Image, radius: float, steps: int = 16) -> Image.Image:
    out = alpha.copy()
    for k in (1.0, 0.66, 0.33):
        for i in range(steps):
            ang = 2 * math.pi * i / steps
            out = ImageChops.lighter(out, ImageChops.offset(
                alpha, int(round(radius * k * math.cos(ang))), int(round(radius * k * math.sin(ang)))))
    return out


def add_outline(sprite: Image.Image, width: int, color=BLACK) -> Image.Image:
    """'Sticker' outline: grows the sprite by `width` and paints a solid silhouette of `color` underneath."""
    pad = width + 3
    base = Image.new("RGBA", (sprite.width + 2 * pad, sprite.height + 2 * pad), (0, 0, 0, 0))
    base.alpha_composite(sprite, (pad, pad))
    grown = dilate_alpha(base.getchannel("A"), width).filter(ImageFilter.GaussianBlur(0.8))
    ring = Image.new("RGBA", base.size, tuple(color) + (0,))
    ring.putalpha(grown)
    ring.alpha_composite(base)
    return ring


def glow_from_alpha(layer: Image.Image, color, radius: float, strength: float = 1.0) -> Image.Image:
    a = layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
    if strength != 1.0:
        a = a.point(lambda v: min(255, int(v * strength)))
    glow = Image.new("RGBA", layer.size, tuple(color) + (0,))
    glow.putalpha(a)
    return glow


def decorate_sprite(sprite: Image.Image, shadow: Optional[Tuple[int, int]] = (8, 12), shadow_blur: float = 8,
                    shadow_opacity: float = 0.7, glow=None, glow_radius: float = 24,
                    glow_strength: float = 1.5) -> Image.Image:
    """Adds padding + (optional) coloured glow + (optional) drop shadow around a sprite. The sprite stays centred."""
    pad = int((max(abs(shadow[0]), abs(shadow[1])) if shadow else 0) + shadow_blur * 3 +
              (glow_radius * 2 if glow else 0) + 4)
    base = Image.new("RGBA", (sprite.width + 2 * pad, sprite.height + 2 * pad), (0, 0, 0, 0))
    base.alpha_composite(sprite, (pad, pad))
    out = Image.new("RGBA", base.size, (0, 0, 0, 0))
    if glow is not None:
        out.alpha_composite(glow_from_alpha(base, glow, glow_radius, glow_strength))
    if shadow:
        shifted = Image.new("RGBA", base.size, (0, 0, 0, 0))
        shifted.alpha_composite(base, (shadow[0], shadow[1]))
        out.alpha_composite(glow_from_alpha(shifted, BLACK, shadow_blur, shadow_opacity))
    out.alpha_composite(base)
    return out


# ---------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------
def make_gradient(w: int, h: int, color, spot=(0.5, 0.4), strength: float = 0.4, base=BASE) -> Image.Image:
    """Dark, rich radial gradient: `color` glows around `spot` (fractions of w/h) and fades into near-black."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ref = min(w, h)  # landscape: identical to the original weekly look; portrait: same glow size relative to the width
    dx, dy = (xx - spot[0] * w) / ref, (yy - spot[1] * h) / ref
    k = (np.clip(1.0 - np.sqrt(dx * dx + dy * dy) / 0.95, 0.0, 1.0) ** 1.5 * strength)[..., None]
    rgb = np.array(base, np.float32) * (1 - k) + np.array(color, np.float32) * k
    rgb *= (1.0 - 0.35 * (yy / h))[..., None]
    return Image.fromarray(np.dstack([rgb, np.full((h, w, 1), 255, np.float32)]).clip(0, 255).astype(np.uint8))


def make_grid(w: int, h: int, color, spacing: int = 64, alpha: int = 28) -> Image.Image:
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    line = tuple(color) + (alpha,)
    for x in range(0, w + 1, spacing):
        d.line([(x, 0), (x, h)], fill=line, width=2)
    for y in range(0, h + 1, spacing):
        d.line([(0, y), (w, y)], fill=line, width=2)
    fade = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None]
    layer.putalpha(Image.fromarray((np.asarray(layer.getchannel("A"), np.float32) * fade).astype(np.uint8)))
    return layer


def draw_trend_background(w: int, h: int, color, grid: bool = True, spot=(0.5, 0.4),
                          strength: float = 0.4) -> Image.Image:
    bg = make_gradient(w, h, color, spot, strength)
    if grid:
        bg.alpha_composite(make_grid(w, h, _lighten(color, 0.3), spacing=max(48, min(w, h) // 11)))
    return bg


def make_sunburst(w: int, h: int, center: Tuple[float, float], color, rays: int = 22, alpha: int = 38) -> Image.Image:
    layer = Image.new("RGBA", (w, h), tuple(color) + (0,))
    d = ImageDraw.Draw(layer)
    reach = math.hypot(w, h)
    for i in range(rays):
        a0, a1 = 2 * math.pi * i / rays, 2 * math.pi * (i + 0.5) / rays
        d.polygon([center, (center[0] + reach * math.cos(a0), center[1] + reach * math.sin(a0)),
                   (center[0] + reach * math.cos(a1), center[1] + reach * math.sin(a1))],
                  fill=tuple(color) + (alpha,))
    return layer


def make_candle_layer(w: int, h: int, trend: str, seed: int, n: int = 5, blur: float = 8,
                      opacity: float = 0.8) -> Image.Image:
    """Giant, blurred DECORATIVE candlesticks staircasing with the trend (deterministic per episode)."""
    rnd = random.Random(seed)
    step = {"bullish": 0.11, "bearish": -0.11}.get(trend, 0.0)
    up_c, down_c = ((UP_RGB, DOWN_RGB) if trend != "neutral" else (GOLD_RGB, (150, 120, 40)))
    price, candles = 0.5 - step * n / 2, []
    for _ in range(n):
        o = price
        c = o + (step * 1.7 + rnd.uniform(-0.05, 0.05) if step else rnd.uniform(-0.18, 0.18))
        candles.append((o, c, max(o, c) + rnd.uniform(0.05, 0.14), min(o, c) - rnd.uniform(0.05, 0.14)))
        price = o + step + rnd.uniform(-0.04, 0.04)
    lo, hi = min(cd[3] for cd in candles), max(cd[2] for cd in candles)
    y_of = lambda v: h * (0.94 - (v - lo) / max(1e-6, hi - lo) * 0.88)
    layer = Image.new("RGBA", (w, h), tuple(up_c if step >= 0 else down_c) + (0,))
    d = ImageDraw.Draw(layer)
    slot = w / n
    for i, (o, c, high, low) in enumerate(candles):
        x = slot * (i + 0.5)
        col = tuple(up_c if c >= o else down_c) + (int(255 * opacity),)
        d.line([(x, y_of(high)), (x, y_of(low))], fill=col, width=int(slot * 0.11))
        d.rectangle((x - slot * 0.23, min(y_of(o), y_of(c)), x + slot * 0.23, max(y_of(o), y_of(c)) + 6), fill=col)
    return layer.filter(ImageFilter.GaussianBlur(blur))


# ---------------------------------------------------------------------
# YouTube text: thick black outline + gradient fill + soft drop shadow (+ optional colour glow)
# ---------------------------------------------------------------------
def _cap_height(font) -> int:
    box = font.getbbox("H")
    return box[3] - box[1]


def _partitions(words: List[str]) -> List[List[str]]:
    out = [[" ".join(words)]]
    if len(words) >= 2:
        out += [[" ".join(words[:i]), " ".join(words[i:])] for i in range(1, len(words))]
    if len(words) == 3:
        out.append(list(words))
    return out


def fit_headline(text: str, max_w: float, max_h: float, max_size: int = 320, min_size: int = 36):
    """Choose the line break + font size that makes `text` as BIG as possible inside the box.
    Returns (lines, font, stroke_px, line_pitch_px)."""
    best = None
    for lines in _partitions((text or "?").split()):
        for size in range(max_size, min_size - 1, -2):
            font = thumb_font(size)
            stroke = max(3, round(size * 0.085))
            cap = _cap_height(font)
            pitch = cap + stroke * 1.4
            boxes = [font.getbbox(ln, stroke_width=stroke) for ln in lines]
            width = max(bx[2] - bx[0] for bx in boxes)
            height = (len(lines) - 1) * pitch + cap + 2 * stroke
            if (width <= max_w and height <= max_h) or size == min_size:
                if best is None or size > best[0]:
                    best = (size, lines, font, stroke, pitch)
                break
    return best[1], best[2], best[3], best[4]


def render_text_sprite(text: str, font, fill, stroke: int) -> Image.Image:
    """One line of text as an RGBA sprite: solid black outline of `stroke` px + top-to-bottom gradient fill."""
    l, t, r, b = font.getbbox(text, stroke_width=stroke)
    origin = (-l, -t)
    mask = Image.new("L", (r - l, b - t), 0)
    ImageDraw.Draw(mask).text(origin, text, font=font, fill=255)
    face = sprite_from_mask(vertical_gradient(mask.size, fill, _shade(fill, 0.9)), mask)
    if not stroke:
        return face
    sprite = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    ImageDraw.Draw(sprite).text(origin, text, font=font, fill=BLACK + (255,), stroke_width=stroke,
                                stroke_fill=BLACK + (255,))
    sprite.alpha_composite(face)
    return sprite


def draw_youtube_text(canvas: Image.Image, text: str, center: Tuple[float, float], font, fill=WHITE,
                      stroke: Optional[int] = None, shadow: bool = True, glow=None, glow_radius: float = 24,
                      glow_strength: float = 1.6) -> None:
    size = getattr(font, "size", 64)
    stroke = max(3, round(size * 0.085)) if stroke is None else stroke
    off = max(3, round(size * 0.045))
    sprite = decorate_sprite(render_text_sprite(text, font, fill, stroke),
                             shadow=(off, int(off * 1.4)) if shadow else None, shadow_blur=max(2, off),
                             shadow_opacity=0.75, glow=glow, glow_radius=glow_radius, glow_strength=glow_strength)
    thumb_paste_center(canvas, sprite, *center)


def draw_text_block(canvas: Image.Image, lines: List[str], center: Tuple[float, float], font, fills: Sequence,
                    stroke: int, pitch: float, **kw) -> None:
    top = center[1] - (len(lines) - 1) * pitch / 2
    for i, line in enumerate(lines):
        draw_youtube_text(canvas, line, (center[0], top + i * pitch), font, fills[i % len(fills)], stroke, **kw)


# =====================================================================
# 4. LOGOS  (local cache -> FMP -> Logo.dev -> coloured badge with the ticker)
# =====================================================================
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
    return None


def download_logo_dev(ticker: str) -> Optional[str]:
    """Optional online source (needs a free publishable key in LOGO_DEV_TOKEN). Cached in assets/logos/."""
    token = get_secret("LOGO_DEV_TOKEN")
    if not token or not ticker or ticker.startswith("^") or ticker in _LOGO_MISS:
        return None
    dest = LOGO_DIR / f"{ticker}.png"
    try:
        import requests
        r = requests.get(f"https://img.logo.dev/ticker/{ticker}",
                         params={"token": token, "format": "png", "size": 512, "fallback": "404"}, timeout=8)
        if r.ok and len(r.content) > 600:
            LOGO_DIR.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            Image.open(dest).verify()
            return str(dest)
    except Exception as e:
        logger.info("Logo.dev lookup failed for %s: %s", ticker, e)
        dest.unlink(missing_ok=True)
    return None


def fetch_company_logo(ticker: str) -> Optional[Image.Image]:
    """RGBA logo with transparent margins trimmed, or None (caller draws the fallback badge)."""
    ticker = clean_ticker(ticker)
    if not ticker or ticker.startswith("^"):
        return None
    path = ensure_logo(ticker) or download_logo_dev(ticker)
    if not path:
        _LOGO_MISS.add(ticker)
        return None
    try:
        with Image.open(path) as im:
            logo = im.convert("RGBA")
        box = logo.getchannel("A").getbbox()
        return logo.crop(box) if box else None
    except Exception as e:
        logger.warning("Unreadable logo for %s: %s", ticker, e)
        return None


def ticker_label(ticker: str) -> str:
    t = clean_ticker(ticker)
    return INDEX_LABELS.get(t.lstrip("^"), t.lstrip("^")) if t else "?"


def make_logo_badge(ticker: str, diameter: int, ring: int = 8) -> Image.Image:
    """Round badge (black ring + white disc) holding the company logo. If the logo cannot be fetched, a coloured
    disc with the ticker symbol is drawn instead, so a missing logo never breaks the thumbnail."""
    d = diameter
    ring = max(4, int(ring * d / 270)) if d > 270 else ring
    sprite = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    sprite.paste(BLACK + (255,), (0, 0, d, d), aa_disk(d, (d / 2, d / 2), d / 2))
    inner = aa_disk(d, (d / 2, d / 2), d / 2 - ring)
    logo = fetch_company_logo(ticker)
    if logo is not None:
        sprite.paste((255, 255, 255, 255), (0, 0, d, d), inner)
        box = d * 0.64
        k = min(box / logo.width, box / logo.height)
        logo = logo.resize((max(1, int(logo.width * k)), max(1, int(logo.height * k))), RESAMPLE)
        sprite.alpha_composite(logo, ((d - logo.width) // 2, (d - logo.height) // 2))
        return sprite
    color = BADGE_COLORS[zlib.crc32(clean_ticker(ticker).encode()) % len(BADGE_COLORS)]
    sprite.paste(vertical_gradient((d, d), _lighten(color, 0.18), _shade(color, 0.7)), (0, 0), inner)
    label = ticker_label(ticker)
    _, font, stroke, _ = fit_headline(label, d * 0.70, d * 0.36, max_size=int(d * 0.40), min_size=14)
    draw_youtube_text(sprite, label, (d / 2, d / 2), font, WHITE, stroke=max(2, stroke // 2), shadow=False)
    return sprite


# =====================================================================
# 5. AVATAR  (your PNG cut-outs, cropped to the character; falls back to open.png; NO drawn cartoon)
# =====================================================================
@lru_cache(maxsize=16)
def _load_avatar_source(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
    except Exception:
        return None
    box = im.getchannel("A").getbbox()
    return im.crop(box) if box else im


def avatar_source(emotion: str) -> Optional[Image.Image]:
    for path in (AVATAR_PATHS.get(emotion, AVATAR_PATHS["neutral"]), AVATAR_PATHS["neutral"], *AVATAR_LAST_RESORT):
        src = _load_avatar_source(path)
        if src is not None:
            if path != AVATAR_PATHS.get(emotion):
                logger.warning("Thumbnail avatar '%s' missing: using %s instead", emotion, path)
            return src
    logger.warning("No avatar image found in assets/avatar/: the thumbnail is drawn without a face")
    return None


def load_avatar(emotion: str, height: int) -> Optional[Image.Image]:
    """Avatar for the emotion, cropped to the character and scaled so the CHARACTER is `height` px tall."""
    src = avatar_source(emotion)
    if src is None:
        return None
    return src.resize((max(1, round(src.width * height / src.height)), height), RESAMPLE)


def make_block_arrow(direction: str, length: int, width: int, color, outline_px: int = 9) -> Image.Image:
    """Chunky arrow with a thick black outline. `length` = tail-to-tip, `width` = arrow-head width."""
    S, pad = 3, 4 * 3
    L, Wd = length * S, width * S
    m = Image.new("L", (Wd + 2 * pad, L + 2 * pad), 0)
    d, cx, head = ImageDraw.Draw(m), Wd / 2 + pad, L * 0.5
    d.polygon([(cx, pad), (cx + Wd / 2, pad + head), (cx - Wd / 2, pad + head)], fill=255)
    d.rectangle((cx - Wd * 0.23, pad + head - 3, cx + Wd * 0.23, pad + L), fill=255)
    m = m.resize((m.width // S, m.height // S), RESAMPLE)
    arrow = add_outline(sprite_from_mask(vertical_gradient(m.size, _lighten(color, 0.3), color), m), outline_px)
    angle = {"up": 0, "left": 90, "down": 180, "right": -90}[direction]
    return arrow.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def make_hazard_sign(text: str, w: int = 1060, h: int = 340) -> Image.Image:
    """Yellow caution sign (black frame, hazard-stripe bars, screws) with `text` in black inside."""
    S, r, bar = 2, 34, 38
    big = (w * S, h * S)

    def rounded(box, radius):
        m = Image.new("L", big, 0)
        ImageDraw.Draw(m).rounded_rectangle([v * S for v in box], radius=radius * S, fill=255)
        return m.resize((w, h), RESAMPLE)

    sign = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sign.paste(BLACK + (255,), (0, 0, w, h), rounded((0, 0, w, h), r))
    face = vertical_gradient((w, h), (255, 226, 52), (255, 186, 0))
    sign.paste(face, (0, 0), rounded((12, 12, w - 12, h - 12), r - 8))
    stripes = Image.new("RGBA", (w - 48, bar), (255, 204, 0, 255))
    sd = ImageDraw.Draw(stripes)
    for x in range(-bar, w, 60):
        sd.polygon([(x, bar), (x + 30, bar), (x + 30 + bar, 0), (x + bar, 0)], fill=(15, 15, 15, 255))
    sign.alpha_composite(stripes, (24, 24))
    sign.alpha_composite(stripes, (24, h - 24 - bar))
    d = ImageDraw.Draw(sign)
    for sx in (50, w - 50):
        for sy in (24 + bar + 28, h - 24 - bar - 28):
            d.ellipse((sx - 8, sy - 8, sx + 8, sy + 8), fill=(55, 55, 55, 255))
            d.ellipse((sx - 4, sy - 6, sx, sy - 2), fill=(150, 150, 150, 255))
    box_w, box_h = w - 150, h - 2 * (24 + bar) - 36
    lines, font, _, pitch = fit_headline(text, box_w, box_h, max_size=260, min_size=30)
    draw_text_block(sign, lines, (w / 2, h / 2), font, [(18, 18, 18)], 0, pitch, shadow=False)
    return sign


# =====================================================================
# 6. REAL CANDLESTICK CHART + VERIFIED % BADGE  (the new candle_hero building blocks)
# =====================================================================
def candles_from_ohlc(o, h, l, c, max_n: int = 32, min_n: int = 6) -> Optional[List[Tuple[float, float, float, float]]]:
    """Clean OHLC arrays -> at most `max_n` (open, high, low, close) candles. Longer series are merged into
    groups (first open, max high, min low, last close), so a 5-minute intraday download still gives a chunky,
    readable chart. None when there is not enough data."""
    o, h, l, c = (np.asarray(x, dtype=float).flatten() for x in (o, h, l, c))
    ok = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    o, h, l, c = o[ok], h[ok], l[ok], c[ok]
    n = len(c)
    if n < min_n:
        return None
    groups = np.array_split(np.arange(n), max_n) if n > max_n else [np.array([i]) for i in range(n)]
    out = []
    for g in groups:
        op, cl = float(o[g[0]]), float(c[g[-1]])
        out.append((op, max(float(h[g].max()), op, cl), min(float(l[g].min()), op, cl), cl))
    return out


def coerce_candles(data: Any, max_n: int = 32) -> Optional[List[Tuple[float, float, float, float]]]:
    """Accepts a pandas OHLC DataFrame (Open/High/Low/Close), a dict with open/high/low/close lists (what the
    weekly script's fetch_candles returns) or a list of (o, h, l, c) tuples."""
    if data is None:
        return None
    try:
        if hasattr(data, "columns"):
            o, h, l, c = (np.asarray(data[k].values, dtype=float) for k in ("Open", "High", "Low", "Close"))
        elif isinstance(data, dict):
            o, h, l, c = (np.asarray(data[k], dtype=float) for k in ("open", "high", "low", "close"))
        else:
            arr = np.asarray(data, dtype=float)
            o, h, l, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        return candles_from_ohlc(o, h, l, c, max_n)
    except Exception as e:
        logger.warning("Could not read candle data for the thumbnail: %s", e)
        return None


def trend_from_data(pct: Optional[float], candles) -> Optional[str]:
    """bullish / bearish / neutral from the VERIFIED numbers (None when there is nothing to judge from)."""
    if pct is not None and math.isfinite(pct):
        return "bullish" if pct >= 0.05 else "bearish" if pct <= -0.05 else "neutral"
    if candles:
        move = candles[-1][3] / candles[0][0] - 1 if candles[0][0] else 0.0
        return "bullish" if move > 0.0005 else "bearish" if move < -0.0005 else "neutral"
    return None


def make_candle_panel(size: Tuple[int, int], candles, accent, top_frac: float = 0.32, ss: int = 2) -> Image.Image:
    """Crisp candlestick chart in a dark rounded panel: green/red candles with black outlines, a gold 5-period
    average, dotted last-price line, glowing last close, and `top_frac` of the panel left free for the badges."""
    w, h = size
    W, H = w * ss, h * ss
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    radius = int(0.055 * min(W, H))
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=radius, fill=(8, 11, 18, 236))
    ml, mr, mb, mt = int(W * 0.035), int(W * 0.035), int(H * 0.075), int(H * top_frac)
    ax0, ax1, ay0, ay1 = ml, W - mr, mt, H - mb
    arr = np.asarray(candles, dtype=float)
    o, hi, lo, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    n = len(c)
    vmin, vmax = float(lo.min()), float(hi.max())
    pad = (vmax - vmin) * 0.06 or max(abs(vmax) * 0.01, 0.01)
    vmin, vmax = vmin - pad, vmax + pad
    y_of = lambda v: ay1 - (v - vmin) / (vmax - vmin) * (ay1 - ay0)
    for k in range(5):  # faint grid
        y = ay0 + (ay1 - ay0) * k / 4
        d.line([(ax0, y), (ax1, y)], fill=(255, 255, 255, 24), width=max(1, ss))
    slot = (ax1 - ax0) / n
    body_w = max(2.0 * ss, slot * 0.72)
    wick_w = max(2 * ss, int(slot * 0.13))
    outline = slot / ss > 11
    for i in range(n):
        x = ax0 + slot * (i + 0.5)
        col = (UP_RGB if c[i] >= o[i] else DOWN_RGB) + (255,)
        d.line([(x, y_of(hi[i])), (x, y_of(lo[i]))], fill=BLACK + (255,), width=wick_w + 2 * ss)
        d.line([(x, y_of(hi[i])), (x, y_of(lo[i]))], fill=col, width=wick_w)
        top, bot = sorted((y_of(o[i]), y_of(c[i])))
        bot = max(bot, top + 3 * ss)
        d.rectangle((x - body_w / 2, top, x + body_w / 2, bot), fill=col,
                    outline=BLACK + (255,) if outline else None, width=ss)
    last_x, last_y = ax0 + slot * (n - 0.5), y_of(c[-1])
    last_col = UP_RGB if c[-1] >= c[0] else DOWN_RGB
    x = ax0
    while x < ax1:  # dotted last-price line
        d.line([(x, last_y), (min(x + 14 * ss, ax1), last_y)], fill=last_col + (150,), width=2 * ss)
        x += 26 * ss
    if n >= 8:  # gold moving average with a soft glow
        win = max(3, n // 7)
        pts = [(ax0 + slot * (i + 0.5), y_of(float(c[max(0, i - win + 1):i + 1].mean()))) for i in range(n)]
        line = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(line).line(pts, fill=GOLD_RGB + (235,), width=int(3.2 * ss), joint="curve")
        img.alpha_composite(glow_from_alpha(line, GOLD_RGB, 7 * ss, 1.4))
        img.alpha_composite(line)
    d = ImageDraw.Draw(img)
    rr = 12 * ss
    d.ellipse((last_x - rr * 2, last_y - rr * 2, last_x + rr * 2, last_y + rr * 2), fill=last_col + (70,))
    d.ellipse((last_x - rr, last_y - rr, last_x + rr, last_y + rr), fill=WHITE + (255,), outline=last_col + (255,),
              width=3 * ss)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=radius, outline=tuple(accent) + (255,), width=int(6 * ss))
    return img.resize((w, h), RESAMPLE)


def format_pct(pct: float) -> str:
    return f"{pct:+.1f}%" if abs(pct) < 100 else f"{pct:+.0f}%"


def make_pct_badge(pct: float, height: int, label: str = "") -> Image.Image:
    """Pill with a triangle and the VERIFIED % move ('+3.2%'), green or red, optionally a small caption
    underneath ('TODAY', 'THIS WEEK') so the number is always honest about what period it covers."""
    col = UP_RGB if pct >= 0 else DOWN_RGB
    stroke = max(3, round(height * 0.07))
    text = render_text_sprite(format_pct(pct), thumb_font(int(height * 0.80)), WHITE, stroke)
    tri_h = int(height * 0.40)
    tri_w = int(tri_h * 1.15)
    pad, gap = int(height * 0.30), int(height * 0.14)
    w, h = pad + tri_w + gap + text.width + pad, height
    border = max(5, int(height * 0.065))
    pill = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pill.paste(BLACK + (255,), (0, 0, w, h), aa_rounded((w, h), (0, 0, w - 1, h - 1), h / 2))
    inner = aa_rounded((w, h), (border, border, w - 1 - border, h - 1 - border), h / 2 - border)
    pill.paste(vertical_gradient((w, h), _lighten(col, 0.22), _shade(col, 0.72)), (0, 0), inner)
    ty0 = (h - tri_h) / 2
    up = pct >= 0
    pts = ([(0, tri_h), (tri_w, tri_h), (tri_w / 2, 0)] if up else [(0, 0), (tri_w, 0), (tri_w / 2, tri_h)])
    tri_box = (tri_w + 12, tri_h + 12)
    outline_m = aa_polygon(tri_box, [(x + 6 + (x - tri_w / 2) * 0.32, y + 6 + (y - tri_h / 2) * 0.32) for x, y in pts])
    face_m = aa_polygon(tri_box, [(x + 6, y + 6) for x, y in pts])
    tri = Image.new("RGBA", tri_box, (0, 0, 0, 0))
    tri.paste(BLACK + (255,), (0, 0, *tri_box), outline_m)
    tri.paste(WHITE + (255,), (0, 0, *tri_box), face_m)
    pill.alpha_composite(tri, (pad - 6, int(ty0) - 6))
    pill.alpha_composite(text, (pad + tri_w + gap, (h - text.height) // 2))
    if not label:
        return pill
    cap = render_text_sprite(label.upper(), thumb_font(int(height * 0.36)), WHITE, max(2, stroke // 2))
    out = Image.new("RGBA", (max(w, cap.width), h + cap.height - 2), (0, 0, 0, 0))
    out.alpha_composite(pill, ((out.width - w) // 2, 0))
    out.alpha_composite(cap, ((out.width - cap.width) // 2, h - 2))
    return out


# =====================================================================
# 7. THE ENGINE
# =====================================================================
# Where a hand would hold a prop, as a fraction of the CHARACTER's bounding box (x, y) - used by the magnifier.
HAND_FRACTION = {"thinking": (0.70, 0.60), "pointing": (0.86, 0.60), "shocked": (0.80, 0.80), "neutral": (0.80, 0.80)}


class ThumbnailEngine:
    """Renders a thumbnail spec as a PNG (1280x720 weekly, 1080x1920 daily Short).

        ThumbnailEngine(spec, size=LANDSCAPE, candles=..., pct=..., pct_label="THIS WEEK").save("thumb.png")

    Every template follows the same order: background -> big props (logos, candles, lens) -> avatar -> text last,
    so the headline is always the top layer. Landscape: the bottom-right corner (YouTube's duration badge) stays
    free. Portrait: everything important sits inside the central 1080x1350 area, the top / bottom are expendable
    because some surfaces crop Shorts thumbnails."""

    LENS_CENTER, LENS_RADIUS = (860, 395), 212

    def __init__(self, spec: Any, size: Tuple[int, int] = LANDSCAPE, candles: Any = None,
                 pct: Optional[float] = None, pct_label: str = "", default_text: str = "WEEK IN REVIEW"):
        self.W, self.H = size
        self.portrait = self.H > self.W
        self.candles = coerce_candles(candles, 18 if self.H > self.W else 24)  # chunky candles read better small
        self.pct = float(pct) if pct is not None and math.isfinite(float(pct)) else None
        self.pct_label = pct_label
        s = normalize_spec(spec, portrait=self.portrait, have_candles=bool(self.candles), default_text=default_text)
        self.template, self.text = s["template"], s["main_text"]
        self.primary, self.secondary = s["primary_ticker"], s["secondary_ticker"]
        self.emotion = s["avatar_emotion"]
        self.trend = s["trend"]
        if self.template == "candle_hero":  # the mood comes from the verified numbers, never from the LLM
            self.trend = trend_from_data(self.pct, self.candles) or self.trend
        if self.template == "candle_hero" and headline_contradicts(self.text, self.pct):
            safe = shorten(f"{ticker_label(self.primary)} {'SURGES' if self.pct > 0 else 'SLIDES'}?", THUMB_TEXT_MAX_CHARS)
            logger.warning("Headline '%s' contradicts the verified %+.1f%% move; using '%s'", self.text, self.pct, safe)
            self.text = safe
        self.color = TREND_RGB.get(self.trend, TREND_RGB["neutral"])
        self.seed = zlib.crc32(f"{self.template}|{self.text}|{self.primary}".encode())
        self.canvas = Image.new("RGBA", (self.W, self.H), BASE + (255,))

    # ----- public API -----
    def render(self) -> Image.Image:
        prefix = "_ptpl_" if self.portrait else "_tpl_"
        getattr(self, prefix + self.template, getattr(self, prefix + "massive_number"))()
        self._finish()
        return self.canvas.convert("RGB")

    def save(self, path) -> str:
        """Write the PNG. YouTube rejects thumbnails over 2 MB: an oversized file is re-saved with a palette and,
        if that is still too big, as a high-quality JPEG (the returned path then ends in .jpg)."""
        img = self.render()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "PNG", optimize=True)
        if path.stat().st_size > THUMB_MAX_BYTES:
            img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG) \
                .save(path, "PNG", optimize=True)
            logger.info("Thumbnail exceeded %.1f MB; re-saved with a 256-colour palette", THUMB_MAX_BYTES / 1e6)
        if path.stat().st_size > THUMB_MAX_BYTES:
            jpg = path.with_suffix(".jpg")
            img.save(jpg, "JPEG", quality=92, optimize=True, subsampling=0)
            path.unlink(missing_ok=True)
            path = jpg
            logger.info("Thumbnail still too big as PNG; saved as JPEG %s", jpg.name)
        return str(path)

    # ----- shared building blocks -----
    def _headline(self, center, box, max_size: int = 300, fills=None, glow=None, glow_radius: float = 26,
                  glow_strength: float = 1.6, min_size: int = 36) -> None:
        lines, font, stroke, pitch = fit_headline(self.text, box[0], box[1], max_size, min_size)
        fills = fills or ([WHITE] * (len(lines) - 1) + [YELLOW] if len(lines) > 1 else [WHITE])
        draw_text_block(self.canvas, lines, center, font, fills, stroke, pitch, glow=glow, glow_radius=glow_radius,
                        glow_strength=glow_strength)

    def _avatar(self, cx: float, height: int, bottom: Optional[float] = None, glow: bool = True
                ) -> Optional[Tuple[float, float, float, float]]:
        """Paints the character (sticker outline + glow + shadow). `height` is the CHARACTER's height. Returns the
        character's (x, y, w, h) on the canvas, or None when no avatar image exists (nothing is drawn then)."""
        av = load_avatar(self.emotion, height)
        if av is None:
            return None
        sprite = add_outline(av, max(6, round(height / 70)), WHITE)
        deco = decorate_sprite(sprite, shadow=(10, 14), shadow_blur=10, glow=self.color if glow else None,
                               glow_radius=28, glow_strength=1.1)
        bottom = self.H + 6 if bottom is None else bottom
        cy = bottom - sprite.height / 2
        thumb_paste_center(self.canvas, deco, cx, cy)
        return cx - av.width / 2, cy - av.height / 2, av.width, av.height

    def _logo(self, ticker: str, diameter: int, center, glow=None) -> None:
        deco = decorate_sprite(make_logo_badge(ticker, diameter), shadow=(8, 12), shadow_blur=9, glow=glow,
                               glow_radius=30, glow_strength=1.5)
        thumb_paste_center(self.canvas, deco, *center)

    def _prop(self, sprite: Image.Image, center, glow=None, glow_radius: float = 22) -> None:
        thumb_paste_center(self.canvas, decorate_sprite(sprite, glow=glow, glow_radius=glow_radius), *center)

    def _bg(self, **kw) -> None:
        self.canvas = draw_trend_background(self.W, self.H, self.color, **kw)

    def _glowing_line(self, points, width: int, color=WHITE, glow=None) -> None:
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).line(points, fill=tuple(color) + (255,), width=width, joint="curve")
        self.canvas.alpha_composite(glow_from_alpha(layer, glow or self.color, 14, 2.0))
        self.canvas.alpha_composite(layer)

    def _finish(self) -> None:
        """Vignette + a little extra punch, the last thing applied to every template."""
        w, h = self.canvas.size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx / w - 0.5) * 1.2) ** 2 + (yy / h - 0.5) ** 2)
        v = (1.0 - 0.42 * np.clip((r - 0.40) / 0.5, 0, 1) ** 1.4)[..., None]
        rgb = np.asarray(self.canvas.convert("RGB"), np.float32) * v
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
        img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.12)).enhance(1.06)
        self.canvas = img.convert("RGBA")

    def _candle_hero_pieces(self, panel_box, top_frac, logo_d, logo_c, pill_h, pill_c) -> None:
        """Chart panel + logo badge + verified % pill; shared by the landscape and portrait compositions."""
        x, y, w, h = panel_box
        panel = make_candle_panel((w, h), self.candles, self.color, top_frac=top_frac)
        self._prop(panel, (x + w / 2, y + h / 2), glow=self.color, glow_radius=26)
        self._logo(self.primary, logo_d, logo_c, glow=self.color)
        if self.pct is not None:
            self._prop(make_pct_badge(self.pct, pill_h, self.pct_label), pill_c)

    # ================================================================
    # LANDSCAPE (1280x720) templates
    # ================================================================
    def _tpl_candle_hero(self) -> None:
        """Real candles of the featured stock in a panel, its logo + verified % on top of it, a surprised face on
        the left, and a huge headline across the top."""
        self._bg(strength=0.45, spot=(0.62, 0.55))
        self._candle_hero_pieces((470, 222, 770, 432), 0.36, 186, (592, 262), 84, (1052, 276))
        self._avatar(240, 520)
        self._headline((640, 100), (1200, 178), 300, glow=self.color, glow_radius=28)

    def _tpl_divergence(self) -> None:
        """Diagonal red | green split. Primary (falling) left, secondary (rising) right, avatar between."""
        W, H = self.W, self.H
        red, green = TREND_RGB["bearish"], TREND_RGB["bullish"]
        left = draw_trend_background(W, H, red, spot=(0.15, 0.5), strength=0.85)
        right = draw_trend_background(W, H, green, spot=(0.85, 0.5), strength=0.85)
        top_x, bottom_x = 800, 480
        self.canvas = right
        self.canvas.paste(left, (0, 0), aa_polygon((W, H), [(0, 0), (top_x, 0), (bottom_x, H), (0, H)]))
        self._glowing_line([(top_x, -10), (bottom_x, H + 10)], 12, WHITE, glow=WHITE)
        self._logo(self.primary, 270, (205, 405), glow=red)
        self._logo(self.secondary, 270, (1075, 405), glow=green)
        self._avatar(640, 500)
        self._prop(make_block_arrow("down", 130, 118, red), (205, 622))
        self._prop(make_block_arrow("up", 130, 118, green), (1075, 622))
        self._headline((640, 105), (1160, 170), 280)

    def _tpl_chart_alert(self) -> None:
        """Giant blurred candlesticks behind a huge headline; avatar points at the primary ticker's logo."""
        self._bg(strength=0.32)
        self.canvas.alpha_composite(make_candle_layer(self.W, self.H, self.trend, self.seed))
        self._logo(self.primary, 330, (950, 480), glow=self.color)
        self._avatar(285, 410)
        self._prop(make_block_arrow("right", 190, 120, YELLOW), (640, 500))
        self._headline((640, 150), (1200, 250), 330, glow=self.color, glow_radius=30)

    def _tpl_vs_showdown(self) -> None:
        """Vertical (lightning-edged) split: primary left, secondary right, massive VS in the middle."""
        W, H = self.W, self.H
        zig = [(650, 0), (618, 150), (664, 300), (616, 450), (662, 600), (632, H)]
        self.canvas = draw_trend_background(W, H, ORANGE_RGB, spot=(0.85, 0.5), strength=0.75)
        self.canvas.paste(draw_trend_background(W, H, BLUE_RGB, spot=(0.15, 0.5), strength=0.75), (0, 0),
                          aa_polygon((W, H), [(0, 0)] + zig + [(0, H)]))
        self._glowing_line([(650, -10)] + zig[1:-1] + [(632, H + 10)], 12, WHITE, glow=self.color)
        self._logo(self.primary, 340, (235, 410), glow=BLUE_RGB)
        self._logo(self.secondary, 340, (1045, 410), glow=ORANGE_RGB)
        lines, font, stroke, pitch = fit_headline("VS", 440, 340, 420)
        draw_text_block(self.canvas, lines, (640, 395), font, [YELLOW], stroke, pitch, glow=self.color,
                        glow_radius=38, glow_strength=2.0)
        self._headline((640, 85), (1180, 130), 200)

    def _tpl_warning_sign(self) -> None:
        """Big avatar centred; a tilted yellow caution sign across its chest carries the headline in black."""
        self._bg(strength=0.42, spot=(0.5, 0.45))
        self.canvas.alpha_composite(make_sunburst(self.W, self.H, (640, 360), self.color))
        if self.primary:
            self._logo(self.primary, 200, (1140, 150), glow=self.color)
        self._avatar(640, 610, bottom=self.H + 30)
        sign = make_hazard_sign(self.text, 1040, 320).rotate(3, expand=True, resample=Image.Resampling.BICUBIC)
        self._prop(sign, (640, 528))

    def _tpl_magnifying_glass(self) -> None:
        """Blurred market backdrop; avatar (left) holds a lens that shows the primary ticker's logo sharp and huge."""
        W, H = self.W, self.H
        self._bg(strength=0.38)
        self.canvas.alpha_composite(make_candle_layer(W, H, self.trend, self.seed, n=7, blur=3, opacity=0.9))
        sharp = self.canvas.copy()
        self.canvas = ImageEnhance.Brightness(sharp.convert("RGB").filter(ImageFilter.GaussianBlur(11))) \
            .enhance(0.75).convert("RGBA")
        bounds = self._avatar(270, 520, bottom=H + 8, glow=False)
        lens_c, R = self.LENS_CENTER, self.LENS_RADIUS
        if bounds:  # the handle ends at this emotion's hand (or chest) so it looks held
            fx, fy = HAND_FRACTION.get(self.emotion, (0.8, 0.8))
            hand = (bounds[0] + fx * bounds[2], bounds[1] + fy * bounds[3])
        else:
            hand = (330, 560)
        dx, dy = hand[0] - lens_c[0], hand[1] - lens_c[1]
        k = R / math.hypot(dx, dy)
        start = (lens_c[0] + dx * k, lens_c[1] + dy * k)
        handle = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        hd = ImageDraw.Draw(handle)
        mid = (start[0] + (hand[0] - start[0]) * 0.42, start[1] + (hand[1] - start[1]) * 0.42)
        for pts, wd, col in (((start, hand), 58, BLACK), ((start, mid), 40, (205, 210, 220)),
                             ((mid, hand), 40, (45, 48, 56))):
            hd.line(pts, fill=col + (255,), width=wd)
            for p in pts:
                hd.ellipse((p[0] - wd / 2, p[1] - wd / 2, p[0] + wd / 2, p[1] + wd / 2), fill=col + (255,))
        self.canvas.alpha_composite(glow_from_alpha(handle, BLACK, 8, 0.7).transform(
            (W, H), Image.Transform.AFFINE, (1, 0, -8, 0, 1, -12)))
        self.canvas.alpha_composite(handle)
        S = 2 * (R + 12)
        c = (S / 2, S / 2)
        r_in = R - 26
        zoom = sharp.crop((int(lens_c[0] - r_in / 1.25), int(lens_c[1] - r_in / 1.25),
                           int(lens_c[0] + r_in / 1.25), int(lens_c[1] + r_in / 1.25))).resize(
            (int(2 * r_in), int(2 * r_in)), RESAMPLE)
        content = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        content.alpha_composite(ImageEnhance.Brightness(zoom.convert("RGB")).enhance(1.35).convert("RGBA"),
                                (int(c[0] - r_in), int(c[1] - r_in)))
        badge = make_logo_badge(self.primary, int(r_in * 1.5))
        content.alpha_composite(badge, (int(c[0] - badge.width / 2), int(c[1] - badge.height / 2)))
        shine = Image.new("RGBA", (S, S), (255, 255, 255, 0))
        ImageDraw.Draw(shine).arc((c[0] - r_in * 0.82, c[1] - r_in * 0.82, c[0] + r_in * 0.82, c[1] + r_in * 0.82),
                                  200, 262, fill=(255, 255, 255, 140), width=14)
        content.alpha_composite(shine.filter(ImageFilter.GaussianBlur(1.5)))
        lens = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        lens.paste(BLACK + (255,), (0, 0, S, S), aa_disk(S, c, R + 9))
        lens.paste(self.color + (255,), (0, 0, S, S), aa_disk(S, c, R))
        lens.paste(BLACK + (255,), (0, 0, S, S), aa_disk(S, c, r_in + 7))
        lens.paste(content, (0, 0), aa_disk(S, c, r_in))
        self._prop(lens, lens_c, glow=self.color)
        self._headline((640, 92), (1200, 150), 260, glow=self.color, glow_radius=24)

    def _tpl_massive_number(self) -> None:
        """Minimalist dark canvas: shocked avatar left, headline enormous on the right with a strong colour glow."""
        self._bg(grid=False, spot=(0.78, 0.5), strength=0.30)
        self._avatar(235, 560, bottom=self.H + 4)
        top = 40
        if self.primary:
            self._logo(self.primary, 120, (1190, 90), glow=self.color)
            top = 150
        box_h = self.H - 30 - top
        self._headline((895, top + box_h / 2), (760, box_h), 340,
                       fills=[_lighten(self.color, 0.18)], glow=self.color, glow_radius=36, glow_strength=2.4)

    # ================================================================
    # PORTRAIT (1080x1920, 9:16) templates - the Daily Short
    # ================================================================
    def _ptpl_candle_hero(self) -> None:
        """Huge stacked headline, the real candle panel with logo + verified % on its top edge, the surprised face
        bottom-left and the ticker + a block arrow bottom-right."""
        self._bg(strength=0.50, spot=(0.5, 0.55))
        self._headline((540, 372), (980, 440), 440, glow=self.color, glow_radius=34, min_size=90)
        self._candle_hero_pieces((60, 720, 960, 520), 0.34, 270, (215, 725), 138, (770, 740))
        self._avatar(340, 650, bottom=self.H + 8)
        if self.primary:
            label = ticker_label(self.primary)
            _, font, stroke, _ = fit_headline(label, 470, 200, 230, 60)
            draw_youtube_text(self.canvas, label, (790, 1490), font, WHITE, stroke, glow=self.color, glow_radius=26)
        up = self.trend != "bearish"
        self._prop(make_block_arrow("up" if up else "down", 230, 210, self.color), (790, 1730), glow=self.color)

    def _ptpl_massive_number(self) -> None:
        """No chart data: centred giant headline between an optional logo and the face."""
        self._bg(grid=False, strength=0.42, spot=(0.5, 0.45))
        self.canvas.alpha_composite(make_sunburst(self.W, self.H, (540, 760), self.color, alpha=26))
        if self.primary:
            self._logo(self.primary, 300, (540, 300), glow=self.color)
        top = 480 if self.primary else 260
        self._headline((540, top + (1130 - top) / 2), (980, 1130 - top), 520,
                       fills=[_lighten(self.color, 0.18)] if len(self.text.split()) == 1 else None,
                       glow=self.color, glow_radius=38, glow_strength=2.2, min_size=100)
        self._avatar(540, 760, bottom=self.H + 8)


# =====================================================================
# 8. DEMO  (python thumbnail.py --demo out_folder) - renders every template with synthetic candles
# =====================================================================
def demo_candles(trend: str = "bullish", n: int = 40, seed: int = 3) -> List[Tuple[float, float, float, float]]:
    rnd = random.Random(seed)
    drift = {"bullish": 0.5, "bearish": -0.5}.get(trend, 0.0)
    price, out = 100.0, []
    for _ in range(n):
        o = price
        c = o + drift + rnd.uniform(-2.2, 2.2)
        out.append((o, max(o, c) + rnd.uniform(0.2, 1.6), min(o, c) - rnd.uniform(0.2, 1.6), c))
        price = c
    return out


def _demo(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        (LANDSCAPE, dict(template="candle_hero", main_text="NVIDIA EXPLODES?", primary_ticker="NVDA",
                         trend="bullish", avatar_emotion="shocked"), "bullish", 7.9, "THIS WEEK"),
        (LANDSCAPE, dict(template="candle_hero", main_text="TESLA CRASH?", primary_ticker="TSLA",
                         trend="bearish", avatar_emotion="shocked"), "bearish", -9.4, "THIS WEEK"),
        (PORTRAIT, dict(template="candle_hero", main_text="NVIDIA EXPLODES?", primary_ticker="NVDA",
                        trend="bullish", avatar_emotion="shocked"), "bullish", 7.9, "TODAY"),
        (PORTRAIT, dict(template="candle_hero", main_text="TESLA CRASH?", primary_ticker="TSLA",
                        trend="bearish", avatar_emotion="shocked"), "bearish", -9.4, "TODAY"),
        (PORTRAIT, dict(template="massive_number", main_text="MARKET MELTDOWN", trend="bearish",
                        avatar_emotion="shocked"), None, None, ""),
    ]
    for tpl, who, trend in (("divergence", ("TSLA", "NVDA"), "neutral"), ("chart_alert", ("AAPL", ""), "bullish"),
                            ("vs_showdown", ("AMD", "NVDA"), "neutral"), ("warning_sign", ("", ""), "bearish"),
                            ("magnifying_glass", ("PLTR", ""), "bullish"), ("massive_number", ("", ""), "bullish")):
        cases.append((LANDSCAPE, dict(template=tpl, main_text={"divergence": "TECH SPLIT", "chart_alert": "BREAKOUT?",
                                                               "vs_showdown": "WHO WINS?", "warning_sign": "BE CAREFUL",
                                                               "magnifying_glass": "LOOK CLOSER",
                                                               "massive_number": "BEST WEEK"}[tpl],
                                      primary_ticker=who[0], secondary_ticker=who[1], trend=trend,
                                      avatar_emotion="thinking" if tpl == "magnifying_glass" else "shocked"),
                      None, None, ""))
    for size, spec, shape, pct, label in cases:
        eng = ThumbnailEngine(spec, size=size, candles=demo_candles(shape) if shape else None, pct=pct,
                              pct_label=label)
        name = f"{'daily' if size == PORTRAIT else 'weekly'}_{spec['template']}_{re.sub('[^a-z0-9]', '', spec['main_text'].split()[0].lower())}.png"
        print("saved", eng.save(out / name))


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Thumbnail engine demo")
    ap.add_argument("--demo", metavar="OUT_DIR", required=True, help="render every template with synthetic data")
    _demo(ap.parse_args().demo)