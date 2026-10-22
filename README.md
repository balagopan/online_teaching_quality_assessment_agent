# 🎓 8x8 AI Class Quality Evaluator

An AI-powered multimodal application designed to evaluate the quality of online classes for children (conducted by 8x8). The system analyzes full-length class recordings across visual and auditory signals to categorize teaching sessions as **Poor**, **Average**, **Good**, or **Excellent**, delivering concrete, timestamped, evidence-based pedagogical feedback.

Built for the **Humaniti AI 24-Hour Build Challenge**.

---

## 🌟 Key Features

1. **Multimodal Analysis:** Evaluates direct visual signals (camera framing, posture, screen demonstrations on Lichess/ChessKid) and auditory dialogue (tone, question pacing, checks for understanding).
2. **Decoupled Multi-Agent Architecture:**
   - **Observation Agent (Gemini 2.5 Flash via Files API):** Extracts structured, temporally consolidated (1–3 min) observations without scoring.
   - **Scoring Agent:** Evaluates 4 core pedagogical dimensions on a 1–4 integer rubric and provides evidence-backed rationales.
   - **Deterministic Classifier:** Computes arithmetic mean and assigns threshold-based classifications (`Poor`, `Average`, `Good`, `Excellent`) deterministically in code.
   - **Summarising Agent:** Synthesizes an executive overview explaining the rating.
3. **Strict Child Data Privacy:**
   - Student names are strictly anonymized as `"Student A"`, `"Student B"`, or generic terms.
   - Files uploaded to the Gemini Files API are deleted immediately from remote servers via `finally` blocks upon completion.
   - Video files and `.env` are strictly excluded from version control via `.gitignore`.
4. **Interactive Streamlit Dashboard:**
   - Supports **File Uploads**, **Direct Video URLs**, and **Local Sample Videos**.
   - Live progress indicator showing multi-agent pipeline stages.
   - Interactive observation table with keyword filtering.
   - One-click JSON and Markdown report export.

---

## 🏗️ Architecture & Approach

```
                    [ Class Video Recording ]
                     (Upload / URL / Sample)
                               │
                               ▼
                   ┌───────────────────────┐
                   │   Observation Agent   │  <-- Gemini 2.5 Flash (Multimodal)
                   │  (Files API + Video)  │  <-- Strict Anonymization Prompt
                   └───────────┬───────────┘
                               │ Structured Observations (HH:MM:SS)
                               ▼
                   ┌───────────────────────┐
                   │     Scoring Agent     │  <-- 4 Dimensions (Score 1-4)
                   │ (LangChain + Gemini)  │  <-- Explicit Rationale
                   └───────────┬───────────┘
                               │ Scores & Rationales
                               ▼
                   ┌───────────────────────┐
                   │ Deterministic Engine  │  <-- Arithmetic Mean (1.00 - 4.00)
                   │  (Python Code Logic)  │  <-- Boundary Classification
                   └───────────┬───────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │   Summarising Agent   │  <-- Executive Pedagogical Summary
                   └───────────┬───────────┘
                               │
                               ▼
                [ Streamlit Interactive UI / Report ]
```

---

## 📋 Evaluation Rubric

| Dimension | Focus |
|---|---|
| **1. Teacher-Student Interaction** | Conversational engagement, responsiveness to student questions, check-ins. |
| **2. Verbal Instructional Clarity** | Clarity of conceptual explanation, pacing, breaking down complex rules. |
| **3. Visual Demonstration** | Effective use of visual aids, boards (e.g. lichess.org), and explanatory videos. |
| **4. Visual Presentation** | Teacher camera status, framing/visibility, posture, and learning climate. |

### Classification Thresholds (Arithmetic Mean of 4 Dimensions)
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
git clone <your-repo-url>
cd Humanti_coding_round

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

### 4. Running the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📝 300-Word Project Reflection

> **Approach:**  
> To evaluate children's online classes within a tight 4–5 hour timeframe, we leveraged Gemini 2.5 Flash's native long-context multimodal video processing via the Google GenAI Files API. Rather than relying on a single monolithic prompt, we engineered a modular multi-agent pipeline: an Observation Agent first extracts timestamped multimodal evidence, a Scoring Agent rates four pedagogical dimensions (Interaction, Clarity, Demonstration, Presentation) with explicit rationales, a deterministic Python engine computes arithmetic means and classification boundaries, and a Summarising Agent generates an executive synthesis.
>
> **Successes:**  
> The system excels at detecting subtle visual and auditory signals: camera framing errors (e.g., teacher only partially visible), external media usage (screen-shared videos, interactive chess boards), and student turn-taking. Calculating scores and threshold classifications deterministically in code eliminated LLM arithmetic hallucinations. Most importantly, privacy is preserved through negative prompting that strips student names and an automated lifecycle that deletes video files from remote servers immediately after evaluation.
>
> **Limitations:**  
> Video processing over the Files API introduces a 1–2 minute server-side ingestion delay for long recordings. Additionally, evaluating multi-student sessions with poor audio quality or overlapping dialogue occasionally leads to consolidated timestamps rather than speaker-level micro-timings. Future iterations could incorporate real-time WebRTC stream analysis and local whisper-based diarization for granular speaker separation.
