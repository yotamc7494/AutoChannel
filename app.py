import streamlit as st
import os
import sys
import base64
import streamlit.components.v1 as components
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

st.set_page_config(page_title="AI Stock Video Generator", page_icon="🚀")

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

# =========================================================
# ממשק המשתמש הרגיל של האפליקציה
# =========================================================
st.set_page_config(page_title="מחולל סרטוני מניות", page_icon="🚀", layout="centered")

st.title("🚀 מחולל סרטוני מניות")

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

        # כפתור הורדה ידני כגבוי
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