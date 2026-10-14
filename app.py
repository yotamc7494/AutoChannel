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
from comparison_video_engine import run_generator
from main import render_final_video, view_final_video
from weekly_summery_video import generate_and_upload_video
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
            filename = render_final_video()
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
            st.stop()
        elif video_type == "vs_video":
            filename = run_generator()
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
            st.stop()
        elif video_type == "weekly_summery":
            filename = generate_and_upload_video(upload=True)
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
    # חישוב זמני ארה"ב (מזרח) לפי הלו"ז שהגדרת
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    events = []

    # בודקים את היום והמחר כדי למצוא את 3 האירועים הקרובים הבאים
    for day_offset in range(2):
        target_date = now + timedelta(days=day_offset)
        is_weekday = target_date.weekday() < 5 # שני עד שישי (0-4)

        # 1. בוקר לפני מסחר (08:14 EST)
        t1 = target_date.replace(hour=8, minute=14, second=0, microsecond=0)
        if t1 > now:
            events.append({"name": "עדכון בוקר (Update / VS)", "timestamp": t1.timestamp()})

        # 2. שעה ויראלית (13:25 EST)
        t2 = target_date.replace(hour=13, minute=25, second=0, microsecond=0)
        if t2 > now:
            events.append({"name": "שעה ויראלית (VS Video)", "timestamp": t2.timestamp()})

        # 3. סיכום יום מסחר (18:21 EST) - רץ רק בימי מסחר!
        if is_weekday:
            t3 = target_date.replace(hour=18, minute=21, second=0, microsecond=0)
            if t3 > now:
                events.append({"name": "סיכום יומי (Daily Summary)", "timestamp": t3.timestamp()})

    # מיון מהקרוב לרחוק ולקיחת 3 הבאים בלבד
    events.sort(key=lambda x: x["timestamp"])
    upcoming_events = events[:3]
    events_json = json.dumps(upcoming_events)

    # HTML ו-JS של טיימר חי (מעוצב יפה)
    html_code = f"""
    <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <div style="text-align: center;">
            <h3 style="margin-top: 0; color: #a5d6a7; font-size: 1.2rem;">⏱️ הסרטון הבא מתחיל בעוד:</h3>
            <div id="countdown" style="font-size: 3rem; font-weight: bold; color: #ff4b4b; text-shadow: 0 0 10px rgba(255,75,75,0.3); margin: 10px 0;">--:--:--</div>
            <div style="font-size: 1.1rem; color: #e0e0e0;">משימה הבאה: <strong id="next-event-name" style="color: #64b5f6;">--</strong></div>
        </div>
        <hr style="border: 0; border-top: 1px solid #444; margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.95rem;">
            <div style="flex: 1;">
                <strong style="color: #ffb74d;">📅 אירועים קרובים:</strong>
                <ul id="upcoming-list" style="margin: 5px 0; padding-right: 20px; color: #bbb;"></ul>
            </div>
            <div style="flex: 1; text-align: left;" dir="ltr">
                <strong style="color: #ffb74d; display: block; text-align: right;" dir="rtl">📈 שעות שוק (ארה"ב):</strong>
                <div style="margin-top: 5px; color: #bbb; text-align: right;" dir="rtl">
                    🟢 פתיחת מסחר: 09:30<br>
                    🔴 סגירת מסחר: 16:00
                </div>
            </div>
        </div>
    </div>
    <script>
        const events = {events_json};
        if(events.length > 0) {{
            const nextEvent = events[0];
            document.getElementById('next-event-name').innerText = nextEvent.name;

            function updateTimer() {{
                const now = new Date().getTime();
                const target = nextEvent.timestamp * 1000;
                const diff = target - now;

                if (diff <= 0) {{
                    document.getElementById('countdown').innerText = "מופעל עכשיו! 🚀";
                    return;
                }}

                const h = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
                const m = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                const s = Math.floor((diff % (1000 * 60)) / 1000);

                document.getElementById('countdown').innerText =
                    (h < 10 ? "0" + h : h) + ":" +
                    (m < 10 ? "0" + m : m) + ":" +
                    (s < 10 ? "0" + s : s);
            }}

            const listEl = document.getElementById('upcoming-list');
            events.forEach(e => {{
                // המרת הזמן לשעה המקומית של הדפדפן של המשתמש! 
                const t = new Date(e.timestamp * 1000);
                const timeString = t.toLocaleTimeString('he-IL', {{hour: '2-digit', minute:'2-digit'}});
                const li = document.createElement('li');
                li.innerText = e.name + " - " + timeString;
                listEl.appendChild(li);
            }});

            setInterval(updateTimer, 1000);
            updateTimer();
        }}
    </script>
    """
    components.html(html_code, height=300)

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
        "סרטון השוואה",
        "סיכום שבועי",
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
            video_path = None

            if video_type == "סרטון מנייה":
                if is_test_mode:
                    video_path = view_final_video()
                else:
                    render_final_video()
                    video_path = "final_stock_video.mp4"  # נתב לפי הקובץ ש-render_final_video מייצרת

            elif video_type == "סרטון השוואה":
                video_path = run_generator(is_test_mode)

            elif video_type == "סיכום שבועי":
                video_path = generate_and_upload_video(upload=should_upload)

            elif video_type == "סיכום יומי (משודרג)":
                output_file = "perfect_daily_recap.mp4"
                run_perfect_pipeline(output_filename=output_file, upload=should_upload)
                video_path = output_file

            st.success(f"🎉 סרטון {video_type} נוצר בהצלחה!")

            # הצגה והורדה
            display_and_download_video(video_path)

        except Exception as e:
            st.error(f"❌ שגיאה במהלך יצירת הסרטון: {e}")