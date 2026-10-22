import os
import tempfile
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from main import evaluate_class_recording, outputSchema

load_dotenv()

st.set_page_config(
    page_title="Online Class Quality Evaluator",
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
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        st.success("Google Gemini API Key detected", icon="✅")
    else:
        st.error("API Key not found in .env", icon="⚠️")
        custom_key = st.text_input("Enter Gemini API Key", type="password")
        if custom_key:
            os.environ["GOOGLE_API_KEY"] = custom_key
            os.environ["GEMINI_API_KEY"] = custom_key

    st.markdown("---")
    st.header("Upload Video & Transcript Source")
    input_source = st.radio(
        "Choose video source:",
        ["Sample Videos", "Upload Video File", "Video URL"],
        index=0,
    )

    selected_video_path = None
    selected_transcript_path = None
    uploaded_file = None
    uploaded_transcript = None
    video_url = None

    if input_source == "Sample Videos":
        sample_dir = Path("videos")
        sample_files = list(sample_dir.glob("*.mp4")) if sample_dir.exists() else []
        if sample_files:
            file_options = {f.name: f for f in sample_files}
            chosen_name = st.selectbox("Select sample video:", list(file_options.keys()))
            selected_video_path = file_options[chosen_name]

            # Auto-detect or select transcript in videos/
            vtt_files = list(sample_dir.glob("*.vtt")) + list(sample_dir.glob("*.srt")) + list(sample_dir.glob("*.txt"))
            transcript_options = {"None": None}
            transcript_options.update({t.name: t for t in vtt_files})
            
            # Pre-select matching transcript if name resembles video
            default_t_idx = 0
            video_stem = selected_video_path.stem.split(" ")[0].lower()
            for idx, name in enumerate(transcript_options.keys()):
                if video_stem in name.lower():
                    default_t_idx = idx
                    break

            chosen_t = st.selectbox("Matching transcript (optional):", list(transcript_options.keys()), index=default_t_idx)
            selected_transcript_path = transcript_options[chosen_t]
        else:
            st.info("No sample videos found in videos/ directory.")

    elif input_source == "Upload Video File":
        uploaded_file = st.file_uploader(
            "Upload class recording (.mp4, .mov, .avi, .mkv)",
            type=["mp4", "mov", "avi", "mkv"],
        )
        uploaded_transcript = st.file_uploader(
            "Upload transcript (optional - .vtt, .srt, .txt):",
            type=["vtt", "srt", "txt"],
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
st.markdown('<div class="main-title">🎓 Online Class Quality Evaluator</div>', unsafe_allow_html=True)
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
    target_transcript_path = None
    temp_dir = tempfile.mkdtemp()

    try:
        if input_source == "Sample Videos":
            if not selected_video_path or not Path(selected_video_path).exists():
                st.error("Please select a valid sample video.")
                st.stop()
            target_path = Path(selected_video_path)
            target_transcript_path = selected_transcript_path

        elif input_source == "Upload Video File":
            if not uploaded_file:
                st.error("Please upload a video file first.")
                st.stop()
            target_path = Path(temp_dir) / uploaded_file.name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_transcript:
                target_transcript_path = Path(temp_dir) / uploaded_transcript.name
                with open(target_transcript_path, "wb") as f:
                    f.write(uploaded_transcript.getbuffer())

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
                transcript_path=target_transcript_path,
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

    st.subheader("📊 Online Class Quality Evaluation")

    dimensions_meta = [
        ("Visual Compliance & Background", "visual_compliance", "📷"),
        ("Rapport & Onboarding", "rapport_and_onboarding", "🤝"),
        ("Individual Consideration", "individual_consideration", "🎯"),
        ("Homework Assignment", "homework_assignment", "📝"),
        ("Energy & Tone", "energy_and_tone", "⚡"),
        ("Instructional Flow", "instructional_flow", "🔄"),
    ]

    # Display in 2 rows of 3 columns
    for row_start in [0, 3]:
        cols = st.columns(3)
        for col, (label, key, icon) in zip(cols, dimensions_meta[row_start : row_start + 3]):
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
        evidence_text = obs.evidence or ""
        if not search_query or search_query.lower() in evidence_text.lower():
            row = {
                "Start": obs.timestamp_start or "N/A",
                "End": obs.timestamp_end or "N/A",
                "Observable Evidence": evidence_text,
            }
            if getattr(obs, "rationale", None):
                row["Rationale"] = obs.rationale
            obs_data.append(row)

    if obs_data:
        st.dataframe(obs_data, use_container_width=True, hide_index=True)
    else:
        st.write("No matching observations found.")

