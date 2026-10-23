# 🎓 Online Class Quality Evaluator

An AI-powered multimodal application designed to evaluate the quality of online classes for children. The system analyzes full-length class recordings (visual and auditory signals) alongside optional audio transcripts to classify the quality of the online class sessions as **Poor**, **Average**, **Good**, or **Excellent**, delivering concrete, timestamped, evidence-based feedback.

## 🌐 Live Demo

[Try the live application](https://onlineteachingqualityassessmentagent-3wwrpvvs4bqdhwcxc6b8sr.streamlit.app/).

---

## 🌟 Key Features

1. **Multimodal Analysis with Video & Transcript Alignment:**
   - Evaluates direct visual signals (camera status, face framing, posture, lighting).
   - Ingests audio and synchronized VTT/SRT transcripts to evaluate turn-taking, onboarding, audio clarity, and pacing.

2. **6-Dimension Evaluation criteria:**
   - Evaluates sessions across all six core operational quality criteria:
     1. 📷 **Visual Compliance & Background:** Camera kept ON with face visible throughout, good lighting without glitches.
     2. 🤝 **Rapport & Onboarding:** Warm rapport at class start (check-ins/warm-ups), proper introduction/onboarding for new joiners, and structured conclusions.
     3. 🎯 **Individual Consideration:** Calling each student by name, balanced turn-taking, and active engagement of quieter students.
     4. 📝 **Homework Assignment:** Explicit homework assigned before transitioning to tournaments or class conclusion.
     5. ⚡ **Energy & Tone:** Energetic, warm, and pleasant vocal delivery with clear audibility.
     6. 🔄 **Instructional Flow:** Structured progression (concept introduction, interactive demonstration, guided puzzles, and practice games without premature ending).

3. **High-Efficiency Chunked Architecture:**
   - Handles long sessions using lossless FFmpeg video slicing (`imageio-ffmpeg` with `-c copy`) into 30-minute processing windows.
   - Slices and aligns matching VTT transcript cues to each video slice.
   - Optional ability to set overlap to preserve context.

4. **Multi-Agent Pipeline:**
   - **Observation Agent:** Extracts concrete, timestamped observations (`HH:MM:SS`) along with an explanation mentioning which critera the particular chunk is addressing. Uses Gemini 2.5 Pro.
   - **Scoring Agent:** Rates all 6 dimensions on a 1–4 integer rubric backed by rationale citing observed evidence. Uses Gemini 2.5 Flash.
   - **Deterministic Classifier:** Pure Python computation of the arithmetic mean and boundary classification (eliminating LLM arithmetic hallucinations).
   - **Summarising Agent:** Synthesizes an executive pedagogical summary explaining the classification. Uses Gemini 2.5 Flash Lite
   - Use of the model is based on the complexity of the task each agent is assigned.

5. **Strict Privacy by Design:**
   - The individuals identities are strictly anonymized as `"Student A"`, `"Student B"`, or `"Teacher"`.

6. **Interactive Streamlit Web Dashboard:**
   - Supports **File Uploads** (with optional transcript uploader) and **Direct Video URLs**.
   - Live progress indicator showing multi-agent pipeline stages.
   - 2×3 metric grid displaying scores for all 6 dimensions with expandable rationales.
   - Interactive observation table with keyword search and filtering.
   - Embedded video preview for instant verification.

---

## 🏗️ System Architecture

```
                 [ Class Video Recording (.mp4) ] + [ Optional Transcript (.vtt) ]
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │   FFmpeg Chunking Engine │  <-- Lossless stream slicing (30m windows)
                        │  & VTT Transcript Slicer │  <-- Synchronized cue extraction
                        └─────────────┬────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │    Observation Agent     │  <-- Gemini Multimodal + Transcript Cues
                        │   (Gemini Files API)     │  <-- Strict Student Anonymization
                        └─────────────┬────────────┘
                                      │ Timestamped Events & Rationales
                                      ▼
                        ┌──────────────────────────┐
                        │      Scoring Agent       │  <-- Evaluates 6 8x8 Operational Dimensions
                        │  (Gemini + LangChain)    │  <-- Evidence-backed rationales (1-4)
                        └─────────────┬────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │   Deterministic Engine   │  <-- Arithmetic Mean (1.00 - 4.00)
                        │   (Python Code Logic)    │  <-- Exact Threshold Classification
                        └─────────────┬────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │    Summarising Agent     │  <-- Executive Pedagogical Summary
                        └─────────────┬────────────┘
                                      │
                                      ▼
                        [ Streamlit Web Dashboard ]
```

---

## 📋 Evaluation Criteria & Classification Rubric

### 8x8 Operational Dimensions (Scored 1 to 4)
- **1 = Poor:** Significant or repeated problems that interfere with instruction or policy compliance.
- **2 = Average:** Some effective practice with noticeable recurring gaps.
- **3 = Good:** Generally effective practice with only minor gaps.
- **4 = Excellent:** Consistently effective practice with multiple clear examples.

### Classification Thresholds (Arithmetic Mean of 6 Dimensions)
- **3.50 – 4.00:** Excellent
- **2.75 – 3.49:** Good
- **1.75 – 2.74:** Average
- **1.00 – 1.74:** Poor

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key

### 2. Installation
```bash
git clone https://github.com/balagopan/online_teaching_quality_assessment_agent.git
cd online_teaching_quality_assessment_agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Running Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📝 300-Word Project Reflection

> **Approach:**  
> To evaluate 8x8 children's online classes within a 24-hour challenge, we designed a modular multi-agent pipeline anchored on Google Gemini's native multimodal capabilities. Rather than passing unsegmented 60-minute recordings to a monolithic prompt, we built a lossless FFmpeg chunking engine that pairs 30-minute video slices with aligned VTT transcripts while preserving absolute timeline continuity. An Observation Agent first extracts timestamped multimodal evidence across six 8x8 QA criteria (Visual Compliance, Rapport & Onboarding, Individual Consideration, Homework Assignment, Energy & Tone, Instructional Flow). A Scoring Agent rates each dimension (1–4) with rationales, a deterministic Python engine computes arithmetic means and boundary classifications, and a Summarising Agent synthesizes an executive pedagogical overview.
>
> **Successes:**  
> The system excels at detecting nuanced operational non-compliance: identifying when a tutor's camera was turned off during teaching, tracking when the mandatory 8x8 branded virtual background was missing, recognizing audio clarity deficits, flagging skipped new-joiner introductions, and noticing missed homework assignments. Computing overall classifications deterministically in code eliminated LLM arithmetic errors. Crucially, child privacy is strictly protected: student names are automatically anonymized (`Student A`, `Student B`), remote video files are deleted immediately after inference via `finally` blocks, and all recordings are strictly excluded from version control.
>
> **Limitations:**  
> Ingesting long video files via the Files API requires an initial 1–2 minute processing phase on server upload. Furthermore, low-resolution recordings or overlapping student audio can occasionally yield consolidated event intervals rather than speaker-level micro-timings. Future iterations could integrate real-time WebRTC stream analysis and local speaker diarization for granular turn-taking measurement.
