import json
import os
import tempfile
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from main import evaluate_class_recording, outputSchema

load_dotenv()

st.set_page_config(
    page_title="8x8 Class Quality Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .classification-badge {
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .badge-excellent {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .badge-good {
        background-color: #DBEAFE;
        color: #1D4ED8;
        border: 1px solid #93C5FD;
    }
    .badge-average {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }
    .badge-poor {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def download_video_from_url(url: str, dest_path: Path, progress_bar) -> None:
    """Streams a video from URL to a local destination file."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = min(downloaded / total_size, 1.0)
                    progress_bar.progress(pct, text=f"Downloading: {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")


# ================= Sidebar =================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.success("Gemini API Key detected", icon="✅")
    else:
        st.error("GEMINI_API_KEY not found in .env", icon="⚠️")
        custom_key = st.text_input("Enter Gemini API Key", type="password")
        if custom_key:
            os.environ["GEMINI_API_KEY"] = custom_key

    st.markdown("---")
    st.header("📹 Video Source")
    input_source = st.radio(
        "Choose video source:",
        ["Sample Videos", "Upload Video File", "Video URL"],
        index=0,
    )

    selected_video_path = None
    uploaded_file = None
    video_url = None

    if input_source == "Sample Videos":
        sample_dir = Path("videos")
        sample_files = list(sample_dir.glob("*.mp4")) if sample_dir.exists() else []
        if sample_files:
            file_options = {f.name: f for f in sample_files}
            chosen_name = st.selectbox("Select sample video:", list(file_options.keys()))
            selected_video_path = file_options[chosen_name]
        else:
            st.info("No sample videos found in videos/ directory.")

    elif input_source == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload class recording (.mp4, .mov, .avi, .mkv)",
            type=["mp4", "mov", "avi", "mkv"],
        )

    elif input_source == "Video URL":
        video_url = st.text_input(
            "Direct Video URL (MP4):",
            placeholder="https://example.com/class-recording.mp4",
        )

    st.markdown("---")
    st.markdown("### 🔒 Student Privacy Guarantee")
    st.caption(
        "• Video recordings feature children and are processed securely.\n"
        "• Temporary uploads are deleted immediately after evaluation.\n"
        "• Prompts mandate strict anonymization ('Student A', 'Student B')."
    )

# ================= Main Area =================
st.markdown('<div class="main-title">🎓 8x8 AI Class Quality Evaluator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Automated multimodal assessment of teacher-student interaction, instructional clarity, and visual engagement.</div>',
    unsafe_allow_html=True,
)

# Manage state for evaluation results
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

# Action Button
start_eval = st.button("🚀 Evaluate Recording", type="primary", use_container_width=True)

if start_eval:
    target_path = None
    temp_dir = tempfile.mkdtemp()

    try:
        if input_source == "Sample Videos":
            if not selected_video_path or not Path(selected_video_path).exists():
                st.error("Please select a valid sample video.")
                st.stop()
            target_path = Path(selected_video_path)

        elif input_source == "Upload Video File":
            if not uploaded_file:
                st.error("Please upload a video file first.")
                st.stop()
            target_path = Path(temp_dir) / uploaded_file.name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        elif input_source == "Video URL":
            if not video_url:
                st.error("Please enter a valid video URL.")
                st.stop()
            target_path = Path(temp_dir) / "downloaded_video.mp4"
            dl_progress = st.progress(0, text="Connecting to URL...")
            download_video_from_url(video_url, target_path, dl_progress)
            dl_progress.empty()

        # Run Multi-Agent Workflow
        with st.status("Analyzing class recording with Multi-Agent Pipeline...", expanded=True) as status_box:
            progress_bar = st.progress(0.0)

            def update_ui_progress(message: str, fraction: float):
                progress_bar.progress(fraction)
                status_box.update(label=message)

            result: outputSchema = evaluate_class_recording(
                video_path=target_path,
                progress_callback=update_ui_progress,
            )

            status_box.update(label="Evaluation successfully completed!", state="complete", expanded=False)
            st.session_state.evaluation_result = result
            st.session_state.current_video_path = str(target_path)

    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
    finally:
        pass

# ================= Render Evaluation Results =================
res: outputSchema = st.session_state.evaluation_result

if res:
    # 1. Video Player Preview
    if "current_video_path" in st.session_state and Path(st.session_state.current_video_path).exists():
        with st.expander("📺 Class Video Preview", expanded=False):
            st.video(st.session_state.current_video_path)

    st.markdown("---")

    # 2. Overall Classification Banner
    classification_class = {
        "Excellent": "badge-excellent",
        "Good": "badge-good",
        "Average": "badge-average",
        "Poor": "badge-poor",
    }.get(res.overall_classification, "badge-average")

    col_cls, col_score = st.columns([2, 1])
    with col_cls:
        st.markdown(
            f'<div class="classification-badge {classification_class}">'
            f"Classification: {res.overall_classification}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Overall Pedagogical Score:** `{res.overall_score:.2f} / 4.00`")

    with col_score:
        st.caption("Score Rubric:")
        st.caption("• 3.50 - 4.00: **Excellent** | • 2.75 - 3.49: **Good**")
        st.caption("• 1.75 - 2.74: **Average**   | • 1.00 - 1.74: **Poor**")

    # 3. Executive Summary
    st.subheader("📋 Executive Summary")
    st.info(res.summary, icon="ℹ️")

    # 4. Dimension Scores & Rationales
    st.subheader("📊 Pedagogical Dimension Breakdown")

    dim_cols = st.columns(4)
    dimensions_meta = [
        ("Teacher-Student Interaction", "teacher_student_interaction", "🗣️"),
        ("Verbal Instructional Clarity", "verbal_instructional_clarity", "💡"),
        ("Visual Demonstration", "visual_demonstration", "🖥️"),
        ("Visual Presentation", "visual_presentation", "👤"),
    ]

    for col, (label, key, icon) in zip(dim_cols, dimensions_meta):
        score = res.dimension_scores.get(key, 0)
        with col:
            st.metric(label=f"{icon} {label}", value=f"{score} / 4")
            with st.expander("Rationale", expanded=False):
                st.write(res.rationale.get(key, "No rationale provided."))

    st.markdown("---")

    # 5. Timestamped Observations
    st.subheader(f"⏱️ Timestamped Observations ({len(res.observations)} Events)")

    search_query = st.text_input("🔍 Filter observations by keyword:", "")

    obs_data = []
    for obs in res.observations:
        if not search_query or search_query.lower() in obs.evidence.lower():
            obs_data.append({
                "Start": obs.timestamp_start or "N/A",
                "End": obs.timestamp_end or "N/A",
                "Observable Evidence": obs.evidence,
            })

    if obs_data:
        st.dataframe(obs_data, use_container_width=True, hide_index=True)
    else:
        st.write("No matching observations found.")

    # 6. Export Options
    st.markdown("---")
    st.subheader("💾 Export Report")

    json_data = res.model_dump_json(indent=2)
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_data,
            file_name=f"class_evaluation_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl2:
        markdown_report = f"""# Class Quality Evaluation Report

- **Overall Classification:** {res.overall_classification}
- **Overall Score:** {res.overall_score:.2f} / 4.00

## Executive Summary
{res.summary}

## Dimension Scores & Rationales
- **Teacher-Student Interaction:** {res.dimension_scores.get('teacher_student_interaction', 0)}/4
  - *Rationale:* {res.rationale.get('teacher_student_interaction', '')}
- **Verbal Instructional Clarity:** {res.dimension_scores.get('verbal_instructional_clarity', 0)}/4
  - *Rationale:* {res.rationale.get('verbal_instructional_clarity', '')}
- **Visual Demonstration:** {res.dimension_scores.get('visual_demonstration', 0)}/4
  - *Rationale:* {res.rationale.get('visual_demonstration', '')}
- **Visual Presentation:** {res.dimension_scores.get('visual_presentation', 0)}/4
  - *Rationale:* {res.rationale.get('visual_presentation', '')}

## Timestamped Observations
""" + "\n".join([f"- **[{obs.timestamp_start} - {obs.timestamp_end}]**: {obs.evidence}" for obs in res.observations])

        st.download_button(
            label="📝 Download Markdown Report",
            data=markdown_report,
            file_name=f"class_evaluation_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )
