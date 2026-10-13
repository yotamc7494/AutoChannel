import streamlit as st
import os
import sys
import base64
import streamlit.components.v1 as components
import logging
# ייבוא פונקציות קיימות
from comparison_video_engine import run_generator
from main import render_final_video, view_final_video
from daily_recap_generator import build_full_daily_video
from weekly_summery_video import generate_and_upload_video

# ייבוא המנוע היומי המשודרג (החדש)
from perfect_daily_video import run_perfect_pipeline
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="AI Stock Video Generator", page_icon="🚀")

st.title("🚀 מחולל סרטוני מניות - מעקף ידני")
# --- תמיכה ב-Webhook מ-GitHub Actions ---
if "WEBHOOK_TOKEN" in st.query_params:
    if st.query_params["WEBHOOK_TOKEN"] == st.secrets.get("WEBHOOK_TOKEN"):
        video_type = st.query_params["video_type"]
        print(f"⚡ Webhook triggered from GitHub Actions! - {video_type}", flush=True)
        sys.stdout.reconfigure(line_buffering=True)
        
        if video_type == "update_video":
            filename = render_final_video()
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
        elif video_type == "vs_video":
            filename = run_generator()
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
        elif video_type == "weekly_summery":
            filename = generate_and_upload_video(upload=True)
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
        elif video_type == "daily_summery":
            # הפעלת הפייפליין המשודרג
            filename = "perfect_daily_recap.mp4"
            run_perfect_pipeline(output_filename=filename, upload=True)
            st.success(f"✅ הסרטון נוצר בהצלחה: {filename}")
        elif video_type == "ping":
            print("Wakeup", flush=True)
            logging.info("🟢 Wakeup ping executed successfully!")
            st.write("🟢 Keep-Alive Ping Received! Server is awake and ready.")
            st.stop()
    else:
        st.info("Wrong Token!")

# ---------------------------------------------------------
# סקשן 1: סרטוני מנייה
# ---------------------------------------------------------
st.header("סרטוני מנייה")
col1, col2 = st.columns(2)

with col1:
    run_full_stock = st.button("🚀 הרץ והעלה מנייה ליוטיוב", key="btn_stock_full")
with col2:
    run_test_stock = st.button("🧪 הרצת ניסיון מנייה (תצוגה מקדימה)", key="btn_stock_test")

if run_full_stock or run_test_stock:
    if run_test_stock:
        st.warning("🧪 מריץ במצב ניסיון: הסרטון לא יועלה ליוטיוב.")
        filename = view_final_video()
        st.success("✅ הסרטון נוצר בהצלחה!")
        if filename and os.path.exists(filename):
            with open(filename, "rb") as file:
                st.download_button(
                    label="📥 הורד את הסרטון למחשב",
                    data=file,
                    file_name=os.path.basename(filename),
                    mime="video/mp4"
                )
        else:
            st.error("שגיאה: קובץ הוידאו לא נמצא.")
    else:
        st.info("🚀 מריץ תהליך מלא כולל העלאה ליוטיוב...")
        render_final_video()
        st.success("התהליך הסתיים.")

st.write("---")

# ---------------------------------------------------------
# סקשן 2: סרטוני השוואה
# ---------------------------------------------------------
st.header("סרטוני השוואה")
col3, col4 = st.columns(2)

with col3:
    run_full_comp = st.button("🚀 הרץ והעלה השוואה ליוטיוב", key="btn_comp_full")
with col4:
    run_test_comp = st.button("🧪 הרצת ניסיון השוואה (תצוגה מקדימה)", key="btn_comp_test")

if run_full_comp or run_test_comp:
    is_test_mode_c = run_test_comp
    st.info(f"מריץ סרטון השוואה (מצב טסט: {is_test_mode_c})...")
    video_path = run_generator(is_test_mode_c)
    st.success("תהליך יצירת סרטון השוואה הסתיים.")
    if video_path and os.path.exists(video_path):
        with open(video_path, "rb") as file:
            st.download_button(
                label="📥 הורד את ההשוואה למחשב",
                data=file,
                file_name=os.path.basename(video_path),
                mime="video/mp4"
            )

st.write("---")

# ---------------------------------------------------------
# סקשן 3: סרטוני סיכום שבועי
# ---------------------------------------------------------
st.header("סרטוני סיכום שבועי")
col5, col6 = st.columns(2)

with col5:
    run_full_weekly = st.button("🚀 הרץ והעלה סיכום שבועי ליוטיוב", key="btn_weekly_full")
with col6:
    run_test_weekly = st.button("🧪 הרצת ניסיון סיכום שבועי (תצוגה מקדימה)", key="btn_weekly_test")

if run_full_weekly or run_test_weekly:
    is_test_mode_w = run_test_weekly
    st.info(f"מריץ סרטון סיכום שבועי (מצב טסט: {is_test_mode_w})...")
    video_path = generate_and_upload_video(upload=not is_test_mode_w)
    
    if video_path and os.path.exists(video_path):
        with open(video_path, "rb") as file:
            video_bytes = file.read()

        b64_video = base64.b64encode(video_bytes).decode("utf-8")
        file_name = os.path.basename(video_path)

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

        st.download_button(
            label="📥 ההורדה לא התחילה? לחץ כאן להורדה ידנית",
            data=video_bytes,
            file_name=file_name,
            mime="video/mp4",
        )

st.write("---")

# ---------------------------------------------------------
# סקשן 4: סרטוני סיכום יומי (הגרסה המשודרגת 6/6)
# ---------------------------------------------------------
st.header("סרטוני סיכום יומי (משודרג)")
col7, col8 = st.columns(2)

with col7:
    run_full_daily = st.button("🚀 הרץ והעלה סיכום יומי ליוטיוב", key="btn_daily_full")
with col8:
    run_test_daily = st.button("🧪 הרצת ניסיון סיכום יומי (תצוגה מקדימה)", key="btn_daily_test")

if run_full_daily or run_test_daily:
    is_test_mode_d = run_test_daily
    st.info("🔄 מתחיל ביצירת סרטון סיכום יומי (פייפליין 6 השלבים)...")
    
    output_file = "perfect_daily_recap.mp4"
    
    with st.spinner("מוריד תמלול, מאמת נתונים, מפיק TTS ומקליט וידאו..."):
        try:
            # הפעלת הפייפליין החדש
            run_perfect_pipeline(output_filename=output_file, upload=run_full_daily)
            st.success("🎉 סרטון הסיכום היומי נוצר בהצלחה!")
            
            if os.path.exists(output_file) and is_test_mode_d:
                # 1. המרה ל-Base64 והורדה אוטומטית בדפדפן
                with open(output_file, "rb") as file:
                    video_bytes = file.read()

                b64_video = base64.b64encode(video_bytes).decode("utf-8")
                file_name = os.path.basename(output_file)

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

                # 2. נגן וידאו מובנה לתצוגה מקדימה ב-Streamlit
                st.video(video_bytes)

                # 3. לחצן גיבוי להורדה ידנית
                st.download_button(
                    label="📥 לחץ כאן להורדת סרטון הסיכום היומי (MP4)",
                    data=video_bytes,
                    file_name=file_name,
                    mime="video/mp4",
                    key="btn_download_daily_recap"
                )
            else:
                st.error("❌ הקובץ נוצר אך לא נמצא בדיסק.")
        except Exception as e:
            st.error(f"❌ שגיאה במהלך יצירת הסרטון: {e}")
