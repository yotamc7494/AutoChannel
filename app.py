import streamlit as st
import os
import sys
import base64
import json
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# ייבוא פונקציות קיימות
from myth_buster_shorts import run_myth_buster_pipeline
from stock_recap_shorts import create_stock_short
from streamlit.logger import get_logger
from daily_recap_shorts import run_perfect_pipeline

logger = get_logger(__name__)

# הגדרת העמוד - חובה לקרוא פעם אחת בלבד בראש הקוד!
st.set_page_config(page_title="מחולל סרטוני מניות", page_icon="🚀", layout="centered")

# הדפסה מיידית ללוג השרת עם טעינת הדף
print("🚀 [APP START] Executing app.py", file=sys.stderr, flush=True)

# --- תמיכה ב-Webhook מ-GitHub Actions / Ping ---
query_params = st.query_params
if "WEBHOOK_TOKEN" in query_params:
    if query_params.get("WEBHOOK_TOKEN") == st.secrets.get("WEBHOOK_TOKEN"):
        video_type = query_params.get("video_type")
        print(f"⚡ Webhook triggered: {video_type}", file=sys.stderr, flush=True)

        if video_type == "update_video":
            filename = create_stock_short(upload=True)
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
            st.stop()
        elif video_type == "vs_video":
            filename = run_myth_buster_pipeline(upload=True)
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
            st.stop()
        elif video_type == "daily_summery":
            filename = "perfect_daily_recap.mp4"
            run_perfect_pipeline(output_filename=filename, upload=True)
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
            st.stop()
        elif video_type == "ping":
            logger.warning("🟢 [PING] Keep-Alive Ping Received!")
            print("🟢 [PING] Server is awake and ready!", file=sys.stderr, flush=True)
            st.write("🟢 Keep-Alive Ping Received! Server is awake and ready.")
            st.stop()
    else:
        st.info("Wrong Token!")
        st.stop() # כדאי לעצור את האפליקציה אם הטוקן שגוי

# =========================================================
# פונקציה לייצור ושילוב ה- Live Counter
# =========================================================
def render_live_counter():
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)

    # פונקציית עזר למציאת המועד הקרוב הבא של אירוע
    def get_next_occurrence(hour, minute, weekdays_only=False):
        for day_offset in range(8):
            dt = (now + timedelta(days=day_offset)).replace(hour=hour, minute=minute, second=0, microsecond=0)
            if dt > now:
                if weekdays_only and dt.weekday() >= 5:  # 5=שבת, 6=ראשון
                    continue
                return dt
        return now

    # הגדרת 5 האירועים לחישוב
    targets = [
        {
            "id": "morning_video",
            "title": "🌅 עדכון בוקר",
            "type": "סרטון",
            "ts": get_next_occurrence(8, 14, weekdays_only=False).timestamp()
        },
        {
            "id": "viral_video",
            "title": "🔥 שעה ויראלית",
            "type": "סרטון",
            "ts": get_next_occurrence(13, 25, weekdays_only=False).timestamp()
        },
        {
            "id": "daily_summary",
            "title": "📊 סיכום יומי",
            "type": "סרטון",
            "ts": get_next_occurrence(18, 21, weekdays_only=True).timestamp()
        },
        {
            "id": "market_open",
            "title": "🔔 פתיחת מסחר",
            "type": "בורסה",
            "ts": get_next_occurrence(9, 30, weekdays_only=True).timestamp()
        },
        {
            "id": "market_close",
            "title": "🔕 סגירת מסחר",
            "type": "בורסה",
            "ts": get_next_occurrence(16, 0, weekdays_only=True).timestamp()
        }
    ]

    events_json = json.dumps(targets)

    html_code = f"""
    <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #18191a; padding: 15px; border-radius: 12px; border: 1px solid #333; color: #fff;">

        <div style="margin-bottom: 12px; font-weight: bold; font-size: 1.05rem; color: #64b5f6; display: flex; align-items: center; gap: 8px;">
            🎬 סרטונים מתוזמנים
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 20px;">
            <!-- כרטיסיות סרטונים -->
            <div id="card-morning_video" style="background: #242526; padding: 12px; border-radius: 8px; border-top: 3px solid #ffb74d; text-align: center;">
                <div style="font-size: 0.9rem; font-weight: bold; color: #ffb74d;">🌅 עדכון בוקר</div>
                <div id="time-morning_video" style="font-size: 0.75rem; color: #aaa; margin: 3px 0;">--:--</div>
                <div id="timer-morning_video" style="font-size: 1.25rem; font-weight: bold; color: #fff; font-family: monospace;">--:--:--</div>
            </div>

            <div id="card-viral_video" style="background: #242526; padding: 12px; border-radius: 8px; border-top: 3px solid #ff7043; text-align: center;">
                <div style="font-size: 0.9rem; font-weight: bold; color: #ff7043;">🔥 שעה ויראלית</div>
                <div id="time-viral_video" style="font-size: 0.75rem; color: #aaa; margin: 3px 0;">--:--</div>
                <div id="timer-viral_video" style="font-size: 1.25rem; font-weight: bold; color: #fff; font-family: monospace;">--:--:--</div>
            </div>

            <div id="card-daily_summary" style="background: #242526; padding: 12px; border-radius: 8px; border-top: 3px solid #ab47bc; text-align: center;">
                <div style="font-size: 0.9rem; font-weight: bold; color: #ab47bc;">📊 סיכום יומי</div>
                <div id="time-daily_summary" style="font-size: 0.75rem; color: #aaa; margin: 3px 0;">--:--</div>
                <div id="timer-daily_summary" style="font-size: 1.25rem; font-weight: bold; color: #fff; font-family: monospace;">--:--:--</div>
            </div>
        </div>

        <div style="margin-bottom: 12px; font-weight: bold; font-size: 1.05rem; color: #81c784; display: flex; align-items: center; gap: 8px;">
            📈 שעות שוק (Wall Street)
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <!-- כרטיסיות בורסה -->
            <div id="card-market_open" style="background: #242526; padding: 12px; border-radius: 8px; border-top: 3px solid #66bb6a; text-align: center;">
                <div style="font-size: 0.9rem; font-weight: bold; color: #66bb6a;">🔔 פתיחת מסחר</div>
                <div id="time-market_open" style="font-size: 0.75rem; color: #aaa; margin: 3px 0;">--:--</div>
                <div id="timer-market_open" style="font-size: 1.25rem; font-weight: bold; color: #fff; font-family: monospace;">--:--:--</div>
            </div>

            <div id="card-market_close" style="background: #242526; padding: 12px; border-radius: 8px; border-top: 3px solid #ef5350; text-align: center;">
                <div style="font-size: 0.9rem; font-weight: bold; color: #ef5350;">🔕 סגירת מסחר</div>
                <div id="time-market_close" style="font-size: 0.75rem; color: #aaa; margin: 3px 0;">--:--</div>
                <div id="timer-market_close" style="font-size: 1.25rem; font-weight: bold; color: #fff; font-family: monospace;">--:--:--</div>
            </div>
        </div>

    </div>

    <script>
        const events = {events_json};

        // אתחול תצוגת השעות המקומיות
        events.forEach(item => {{
            const t = new Date(item.ts * 1000);
            const timeStr = t.toLocaleTimeString('he-IL', {{hour: '2-digit', minute:'2-digit'}});
            const dayStr = t.toLocaleDateString('he-IL', {{weekday: 'short'}});
            document.getElementById('time-' + item.id).innerText = dayStr + " " + timeStr;
        }});

        function updateTimers() {{
            const now = new Date().getTime();

            events.forEach(item => {{
                const target = item.ts * 1000;
                const diff = target - now;

                const timerEl = document.getElementById('timer-' + item.id);

                if (diff <= 0) {{
                    timerEl.innerText = "בפעולה! 🚀";
                    timerEl.style.color = "#ff4b4b";
                    return;
                }}

                const h = Math.floor(diff / (1000 * 60 * 60));
                const m = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                const s = Math.floor((diff % (1000 * 60)) / 1000);

                const hStr = h < 10 ? "0" + h : h;
                const mStr = m < 10 ? "0" + m : m;
                const sStr = s < 10 ? "0" + s : s;

                timerEl.innerText = hStr + ":" + mStr + ":" + sStr;
            }});
        }}

        setInterval(updateTimers, 1000);
        updateTimers();
    </script>
    """
    components.html(html_code, height=330)

# =========================================================
# ממשק המשתמש הרגיל של האפליקציה
# =========================================================
st.title("🚀 מחולל סרטוני מניות")

# הרצת הטיימר ממש מתחת לכותרת!
render_live_counter()

# ---------------------------------------------------------
# 3 שדות קלט מרכזיים
# ---------------------------------------------------------
video_type = st.selectbox(
    "🎬 בחר סוג סרטון:",
    [
        "סרטון מנייה",
        "סרטון מיתוסים",
        "סיכום יומי (משודרג)"
    ]
)

col1, col2 = st.columns(2)
with col1:
    is_test_mode = st.checkbox("🧪 מצב ניסיון (טסט בלבד)", value=True)
with col2:
    upload_to_yt = st.checkbox("🚀 העלאה ליוטיוב", value=not is_test_mode)

# אזהרת אינטראקציה בין מצב ניסיון להעלאה
if is_test_mode and upload_to_yt:
    st.warning("⚠️ שים לב: במצב ניסיון הסרטון לא יועלה ליוטיוב בפועל.")


# ---------------------------------------------------------
# פונקציית עזר להצגת הווידאו והורדה אוטומטית / ידנית
# ---------------------------------------------------------
def display_and_download_video(video_path: str):
    if video_path and os.path.exists(video_path):
        with open(video_path, "rb") as file:
            video_bytes = file.read()

        b64_video = base64.b64encode(video_bytes).decode("utf-8")
        file_name = os.path.basename(video_path)

        # סקריפט להורדה אוטומטית בדפדפן
        download_js = f"""
        <script>
            var a = document.createElement('a');
            a.href = 'data:video/mp4;base64,{b64_video}';
            a.download = '{file_name}';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        </script>
        """
        components.html(download_js, height=0, width=0)

        # נגן ווידאו לתצוגה מקדימה
        st.subheader("📺 תצוגה מקדימה:")
        st.video(video_bytes)

        # כפתור הורדה ידני כגיבוי
        st.download_button(
            label="📥 לחץ כאן להורדת הסרטון (MP4)",
            data=video_bytes,
            file_name=file_name,
            mime="video/mp4",
            key="btn_download_video"
        )
    else:
        st.error("❌ הקובץ נוצר אך לא נמצא בדיסק.")


# ---------------------------------------------------------
# הרצת הפייפליין לפי בחירת המשתמש
# ---------------------------------------------------------
st.write("---")
if st.button("🚀 התחל יצירת סרטון", use_container_width=True):
    should_upload = upload_to_yt and not is_test_mode

    with st.spinner(f"⏳ מפיק סרטון {video_type}..."):
        try:
            video_path = ""

            if video_type == "סרטון מנייה":
                video_path = create_stock_short(upload=should_upload)

            elif video_type == "סרטון מיתוסים":
                output_file = "myth_buster_short.mp4"
                video_path = run_myth_buster_pipeline(output_filename=output_file, upload=should_upload)

            elif video_type == "סיכום יומי (משודרג)":
                output_file = "perfect_daily_recap.mp4"
                run_perfect_pipeline(output_filename=output_file, upload=should_upload)
                video_path = output_file

            st.success(f"🎉 סרטון {video_type} נוצר בהצלחה!")

            # הצגה והורדה
            display_and_download_video(video_path)

        except Exception as e:
            st.error(f"❌ שגיאה במהלך יצירת הסרטון: {e}")