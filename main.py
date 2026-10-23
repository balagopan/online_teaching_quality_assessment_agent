import os
import time
from pathlib import Path
from typing import Callable, Literal
import imageio_ffmpeg
import re
import subprocess
import tempfile



from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# class OperationalCriteria(BaseModel):
#     """Detailed explanations for each operational criteria observed or missing."""
#     visual_compliance: str | None = Field(default=None , description="Whether the tutor used the mandatory '8x8' branded virtual background, kept the camera on, and maintained a well-lit environment without severe virtual background glitching/clarity issues.")
#     rapport_and_onboarding: str | None = Field(default=None, description="Whether the tutor built warm personal rapport at the start of class, provided a clear structured conclusion, and properly onboarded new students.")
#     individual_consideration: str | None = Field(default=None, description="Whether the tutor gave adequate attention to each student's needs and provided personalized feedback.")
#     homework_assignment: str | None = Field(default=None, description="Whether the tutor assigned meaningful homework and provided constructive feedback.")
#     energy_and_tone: str | None = Field(default=None, description="Whether the tutor maintained an engaging and positive energy level throughout the session.")
#     instructional_flow: str | None = Field(default=None, description="Whether the tutor followed a logical and effective instructional sequence.")

class ObservationItem(BaseModel):
    """A single observation item with timestamps and evidence."""
    timestamp_start: str | None = Field(
        description="Start timestamp in HH:MM:SS format ie. hours, minutes, seconds (e.g., '00:01:45' for 1 min 45 sec). Do not include frames.",
        default=None,
    )
    timestamp_end: str | None = Field(
        description="End timestamp in HH:MM:SS format ie. hours, minutes, seconds (e.g., '00:01:45' for 1 min 45 sec). Do not include frames.",
        default=None,
    )
    evidence: str | None = Field(
        description="Evidence describing what was observed or heard. While creating the evidences make sure the names of the individuals are not mentioned, use generic terms instead.",
        default=None,
    )
    rationale: str | None = Field(
        description="Rationale for the observation, explaining why it was noted and how it relates to the operational criteria.",
        default=None,
    )
    # operational_positive_criteria: OperationalCriteria | None = Field(
    #     description="The operational criteria(s) that was observed through out the video (e.g., rapport_and_onboarding, visual_compliance, individual_consideration etc).",
    #     default=None,
    # )   
    # operational_negative_criteria: OperationalCriteria | None = Field(
    #     description="The operational criteria(s) that was not observed through out the video (e.g., rapport_and_onboarding, visual_compliance, individual_consideration etc).",
    #     default=None,
    # )
    # observed_operational_criteria: dict[
    #     Literal["visual_compliance", "rapport_and_onboarding", "individual_consideration", "homework_assignment", "energy_and_tone", "instructional_flow"],str] | None = Field(
    #     description="The operational criteria(s) that was observed through out the video (e.g., rapport_and_onboarding, visual_compliance, individual_consideration etc).",
    #     default=None,
    # )
    # unobserved_operational_criteria: dict[
    #     Literal["visual_compliance", "rapport_and_onboarding", "individual_consideration", "homework_assignment", "energy_and_tone", "instructional_flow"],str] | None = Field(
    #     description="The operational criteria(s) that was not observed through out the video (e.g., rapport_and_onboarding, visual_compliance, individual_consideration etc).",
    #     default=None,
    # )

class Observation_agent_schema(BaseModel):
    observations: list[ObservationItem] = Field(description="List of concrete observations.")

class DimensionScores_agent_schema(BaseModel):
    """Scores for each of the four dimensions, each an integer from 1 to 4."""
    visual_compliance: int = Field(
        ge=1, le=4,
        description="Whether the tutor used the mandatory '8x8' branded virtual background, kept the camera on, and maintained a well-lit environment without severe virtual background glitching/clarity issues."
    )
    rapport_and_onboarding: int = Field(
        ge=1, le=4,
        description="Whether the tutor built warm personal rapport at the start of class, provided a clear structured conclusion, and properly onboarded new students."
    )
    individual_consideration: int = Field(
        ge=1, le=4,
        description="Whether the tutor gave adequate attention to each student's needs and provided personalized feedback."
    )   
    homework_assignment: int = Field(
        ge=1, le=4,
        description="Whether the tutor assigned meaningful homework and provided constructive feedback."
    )
    energy_and_tone: int = Field(
        ge=1, le=4,
        description="Whether the tutor maintained an engaging and positive energy level throughout the session, and used an appropriate tone of voice."
    )
    instructional_flow: int = Field(
        ge=1, le=4,
        description="Whether the tutor maintained a logical and coherent instructional flow, effectively transitioning between topics and activities."
    )


class ObservationScoringOutput_agent_schema(BaseModel):
    dimension_scores: DimensionScores_agent_schema
    rationale: dict[
        Literal[
            "visual_compliance",
            "rapport_and_onboarding",
            "individual_consideration",
            "homework_assignment",
            "energy_and_tone",
            "instructional_flow"
        ],
        str,
    ] = Field(
        description="A rationale for each of the scores, explaining the reasoning behind the assigned score for each of the operational criteria."
    )


class summarising_agent_schema(BaseModel):
    summary: str


class outputSchema(BaseModel):
    overall_classification: Literal["Poor", "Average", "Good", "Excellent"] = Field(
        description="Overall classification of the teaching session based on the dimensions as Poor, Average, Good, Excellent."
    )
    overall_score: float = Field(
        description="Arithmetic mean of the dimension scores, rounded to two decimal places."
    )
    summary: str = Field(
        description="A concise evidence-based explanation of the classification."
    )
    dimension_scores: dict[
        Literal[
            "visual_compliance",
            "rapport_and_onboarding",
            "individual_consideration",
            "homework_assignment",
            "energy_and_tone",
            "instructional_flow"
        ],
        int,
    ] = Field(description="Scores for each of the dimensions, each an integer from 1 to 4.")
    rationale: dict[
        Literal[
            "visual_compliance",
            "rapport_and_onboarding",
            "individual_consideration",
            "homework_assignment",
            "energy_and_tone",
            "instructional_flow"
        ],
        str,
    ] = Field(description="A rationale for each dimension score based on the observations.")
    observations: list[ObservationItem] = Field(
        description="A list of observations, each containing timestamps and evidence."
    )


observation_agent_prompt = """
Analyze the video and transcript against these exact operational criteria:

1. **visual_compliance (Inspect Video Frames Carefully):**
   - Did the tutor use the mandatory "8x8" branded virtual background throughout the session?
   - Was the tutor's camera kept ON the entire time while teaching, with their face clearly visible and properly framed? (Turning the camera off while teaching or having the face hidden is a critical failure).
   - Was the tutor sitting in a well-lit environment without severe virtual background glitching/clarity issues?

2. **rapport_and_onboarding (Inspect First 10-15 Mins & Last 5 Mins of Transcript/Video):**
   - Did the tutor build warm personal rapport at the start of class (asking kids about their day, hobbies, holidays, or running an energetic warm-up activity like Puzzle Rush)?
   - Or did they jump coldly and directly into revising the previous class/solving puzzles?
   - **New Joiner Check:** If a new student joined (or a student mentioned it was their first day), did the tutor give them proper time for a warm self-introduction at the beginning of class?
   - Did the tutor provide a clear, structured conclusion at the end of the session?

3. **individual_consideration (Inspect Transcript Turn-Taking):**
   - Did the tutor actively include everyone in the class by calling on each student individually by name during concept checks and puzzle-solving?
   - Did they manage dominant students (e.g., using private chat or taking turns) so quieter students got a fair chance to answer?

4. **homework_assignment (Inspect Transcript/Video Near End of Teaching Phase):**
   - Did the tutor explicitly assign Homework (HW) before transitioning to the final tournament or ending the class (e.g., asking students to take a screenshot of a puzzle/position to solve for next class, or stating that HW will be shared)?

5. **energy_and_tone (Inspect Audio Track & Facial Expressions):**
   - Was the tutor's communication energetic, happy, warm, and pleasant?
   - Or was the tutor monotone, low-energy, robotic, or too quiet/inaudible?

6. **instructional_flow (Inspect Overall Lesson Structure):**
   - Did the tutor clearly introduce the chess topic, play a topic-related instructional video (pausing to ask questions interactively), guide students through practice puzzles, and facilitate practice games/tournaments without ending the class prematurely?

"""

observation_agent_prompt_template = f"""
You are the Observation Agent.

{observation_agent_prompt}

Review the supplied class recording and the transcriptand produce concrete observations.

Each observation must contain:
- timestamp_start in HH:MM:SS format ie. hours:minutes:seconds (e.g., '00:01:45' for 1 min 45 sec) when determinable.
- timestamp_end in HH:MM:SS format ie. hours:minutes:seconds (e.g., '00:01:45' for 1 min 45 sec) when determinable.
- Evidence describing what is directly observable or audible. Evidence should NOT include names of individuals; use generic terms like 'Teacher', 'Student A', 'Student B'.
- The corresponding operational critera(s) that the observation relates to (e.g., rapport_and_onboarding, visual_compliance, individual_consideration etc).

<important>
- Consolidate related conversational exchanges into unified events (e.g. 1-7 minute segments) rather than generating an entry for every individual sentence.
- Never use the names of individuals in the observations; always use generic terms like 'Teacher', 'Student A', 'Student B'.
</important>

CRITICAL CONSOLIDATION RULE:
- NEVER output repetitive micro-events (e.g. logging every 10-second chess move or back-and-forth exchange individually).
- When a teacher and student work through a puzzle, game, or rapid sequence of moves/questions:
  BAD: 10 separate entries like "Teacher asks move, Student suggests H3... Teacher asks move, Student suggests H2..."
  GOOD: 1 unified entry: "00:58:00 - 00:59:59: The teacher guides Student B through a continuous endgame puzzle sequence, prompting for successive moves while demonstrating Student B's suggestions (King to H3, H2, H1, G1, etc.) on the board."
"""

scoring_agent_prompt_template = """
Use these strict decision rules derived from 8x8's QA standards:

Dimensions (Score 1 to 4):
1. visual_compliance: Mandatory 8x8 background used, camera kept ON with face visible throughout, good lighting without glitches.
2. rapport_and_onboarding: Warm rapport at start (asking kids about day/hobbies or warm-up), proper onboarding for new joiners, clear conclusion.
3. individual_consideration: Calling each student by name, giving everyone fair turn-taking, managing dominant students.
4. homework_assignment: Explicit homework assigned before end of teaching phase or tournament.
5. energy_and_tone: Energetic, happy, warm, pleasant tone throughout (not monotone or low energy).
6. instructional_flow: Clear topic introduction, interactive video demonstration, guided practice puzzles, and games without ending prematurely.

Anchors:
1 = Poor: Significant or repeated problems.
2 = Average: Some effective practice with recurring gaps.
3 = Good: Generally effective with minor gaps.
4 = Excellent: Consistently effective with multiple clear examples.

Provide a score (1-4) and a concrete, evidence-based rationale citing the observations for each dimension.
"""

def parse_vtt_timestamp(ts_str: str) -> float:
    """Converts HH:MM:SS.mmm to seconds."""
    parts = ts_str.strip().split(":")
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)

def slice_vtt_transcript(vtt_content: str, start_sec: float, end_sec: float) -> str:
    """Extracts transcript blocks that fall within [start_sec, end_sec]."""
    cues = re.findall(
        r"(\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*-->\s*(\d{2}:\d{2}(?::\d{2})?\.\d{3})\s*\n(.*?)(?=\n\n|\Z)",
        vtt_content,
        re.DOTALL,
    )
    extracted = []
    for cue_start, cue_end, text in cues:
        t_start = parse_vtt_timestamp(cue_start)
        t_end = parse_vtt_timestamp(cue_end)
        # Check for overlap
        if t_end >= start_sec and t_start <= end_sec:
            clean_text = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
            extracted.append(f"[{cue_start[:8]} - {cue_end[:8]}] {clean_text}")

    return "\n".join(extracted)

def get_video_duration(video_path: Path) -> float:
    """Returns video duration in seconds using ffprobe/ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-i", str(video_path)]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError("Could not determine video duration.")

def slice_video_chunk(
    input_path: Path, output_path: Path, start_sec: float, duration_sec: float
):
    """Slices a video chunk quickly without re-encoding (-c copy)."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-t", str(duration_sec),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def deduplicate_observations(observations: list) -> list:
    """Removes duplicate or near-duplicate observations from overlapping windows."""
    unique = []
    seen = set()
    for obs in observations:
        evidence_snippet = (obs.evidence or "")[:40].lower()
        key = (obs.timestamp_start, evidence_snippet)
        if key not in seen:
            seen.add(key)
            unique.append(obs)
    return unique

def format_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def evaluate_class_recording(
    video_path: str | Path,
    transcript_path: str | Path | None = None,
    chunk_minutes: int = 30,
    overlap_seconds: int = 0,
    progress_callback: Callable[[str, float], None] | None = None,
) -> outputSchema:
    video_path = Path(video_path)
    total_duration = get_video_duration(video_path)
    
    transcript_content = ""
    if transcript_path and Path(transcript_path).exists():
        transcript_content = Path(transcript_path).read_text(encoding="utf-8")

    chunk_duration = chunk_minutes * 60
    step_duration = chunk_duration - overlap_seconds
    
    chunks = []
    start = 0.0
    while start < total_duration:
        dur = min(chunk_duration, total_duration - start)
        chunks.append((start, dur))
        if start + dur >= total_duration:
            break
        start += step_duration

    def update_progress(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)
        else:
            print(f"[{int(frac * 100)}%] {msg}")

    client = genai.Client()
    scoring_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    all_observations = []

    update_progress(f"Starting evaluation ({len(chunks)} segments)...", 0.05)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        for i, (chunk_start, chunk_len) in enumerate(chunks):
            chunk_file = temp_dir / f"chunk_{i}.mp4"
            video_ref = None
            try:     
                chunk_base_pct = 0.05 + (i / len(chunks)) * 0.60
                update_progress(f"Uploading and processing Chunk {i+1}/{len(chunks)}...", chunk_base_pct)
       
                slice_video_chunk(video_path, chunk_file, chunk_start, chunk_len)

                chunk_transcript = slice_vtt_transcript(transcript_content, chunk_start, chunk_start + chunk_len)

                video_ref = client.files.upload(file=str(chunk_file))

                max_wait = 180
                waited = 0
                while video_ref.state.name == "PROCESSING":
                    if waited >= max_wait:
                        raise TimeoutError(f"Processing chunk {i+1} timed out.")
                    time.sleep(2)
                    waited += 2
                    video_ref = client.files.get(name=video_ref.name)

                if video_ref.state.name == "FAILED":
                    raise ValueError(f"Video processing failed: {video_ref.error}")

                chunk_context_prompt = f"""
IMPORTANT TIMING NOTE:
You are provided a slice from a full video to analyze.This video slice covers the time: {format_seconds_to_hhmmss(chunk_start)} to {format_seconds_to_hhmmss(chunk_start + chunk_len)} of a video.
All timestamps in your observations MUST use absolute class timeline format (starting from {format_seconds_to_hhmmss(chunk_start)} onwards), matching the provided transcript cues.
"""

                contents = [video_ref]
                if chunk_transcript:
                    contents.append(f"### OFFICIAL CLASS AUDIO TRANSCRIPT ###\n\n{chunk_transcript}\n\n### END OF TRANSCRIPT ###")
                contents.append("Review the provided video recording alongside the audio transcript to generate concrete observations.")
                
                chunk_analyze_pct = chunk_base_pct + (0.60 / len(chunks)) * 0.5

                update_progress(f"Analyzing Chunk {i+1}/{len(chunks)}...", chunk_analyze_pct)


                
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=observation_agent_prompt_template + "\n" + chunk_context_prompt,
                        response_mime_type="application/json",
                        response_json_schema=Observation_agent_schema.model_json_schema(),
                        temperature=0,
                    ),
                )

                parsed = response.parsed
                if isinstance(parsed, dict):
                    parsed = Observation_agent_schema.model_validate(parsed)
                
                if parsed and parsed.observations:
                    all_observations.extend(parsed.observations)

            finally:
                if video_ref:
                    try:
                        client.files.delete(name=video_ref.name)
                    except Exception:
                        pass
                if chunk_file.exists():
                    chunk_file.unlink()

    final_observations = deduplicate_observations(all_observations)
    obs_result = Observation_agent_schema(observations=final_observations)


    if not obs_result or not obs_result.observations:
        raise ValueError("Failed to analyze the video.")


    observations_formatted = "\n".join([
        f"[{obs.timestamp_start} - {obs.timestamp_end}]: {obs.evidence} (observed factor: {obs.rationale})"
        for obs in obs_result.observations
    ])

    update_progress("Scoring the video...", 0.70)
    scoring_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert teaching evaluation agent.\n\n{scoring_agent_prompt_template}"),
        ("human", "Here are the observations recorded from the class:\n\n{observations_text}\n\nEvaluate and score each dimension."),
    ])
    observation_scoring_agent = scoring_llm.with_structured_output(ObservationScoringOutput_agent_schema)
    scoring_chain = scoring_prompt | observation_scoring_agent
    scoring_result: ObservationScoringOutput_agent_schema = scoring_chain.invoke(
        {"observations_text": observations_formatted}
    )

    dimension_scores = scoring_result.dimension_scores.model_dump()
    calculated_overall_score = round(sum(dimension_scores.values()) / len(dimension_scores), 2)

    if calculated_overall_score >= 3.50:
        classification = "Excellent"
    elif calculated_overall_score >= 2.75:
        classification = "Good"
    elif calculated_overall_score >= 1.75:
        classification = "Average"
    else:
        classification = "Poor"

    update_progress("Summarizing the result...", 0.85)
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an educational evaluator. Write a concise, evidence-based executive summary explaining the reason for providing the final rating.",
        ),
        (
            "human","""
    Classification: {classification} 
    calculated_overall_score: {calculated_overall_score}
    Dimension Scores: {dimension_scores}
    Rationales: {rationales}

    Key Observations:
    {observations_formatted}

    Provide a concise summary paragraph.
    """,
        ),
    ])
    summarising_agent = llm.with_structured_output(summarising_agent_schema)
    summarising_chain = summary_prompt | summarising_agent
    summary_result = summarising_chain.invoke({
        "classification": classification,
        "calculated_overall_score": calculated_overall_score,
        "dimension_scores": str(dimension_scores),
        "rationales": str(scoring_result.rationale),
        "observations_formatted": observations_formatted,
    })

    update_progress("Finalizing the result...", 1.0)
    final_output = outputSchema(
        overall_classification=classification,
        overall_score=calculated_overall_score,
        summary=summary_result.summary,
        dimension_scores=dimension_scores,
        rationale=scoring_result.rationale,
        observations=obs_result.observations,
    )

    return final_output


if __name__ == "__main__":
    sample_video = "videos/B482 Intermediate Part  - 2026-09-02.mp4"
    sample_transcript = "videos/B482 Intermediate -transcript-1.1.vtt"
    evaluation = evaluate_class_recording(sample_video, sample_transcript)
    print("\n" + evaluation.model_dump_json(indent=2))
