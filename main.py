import os
import time
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


class ObservationItem(BaseModel):
    """A single observation item with timestamps and evidence."""
    timestamp_start: str | None = Field(
        description="Start timestamp in HH:MM:SS format (e.g., '00:01:45' for 1 min 45 sec). Do not include frames.",
        default=None,
    )
    timestamp_end: str | None = Field(
        description="End timestamp in HH:MM:SS format (e.g., '00:01:45' for 1 min 45 sec). Do not include frames.",
        default=None,
    )
    evidence: str = Field(
        description="Evidence describing what was observed or heard. While creating the evidences make sure the names of the individuals are not mentioned, use generic terms instead.",
        default=None,
    )


class Observation_agent_schema(BaseModel):
    observations: list[ObservationItem] = Field(description="List of concrete observations.")


class DimensionScores_agent_schema(BaseModel):
    """Scores for each of the four dimensions, each an integer from 1 to 4."""
    teacher_student_interaction: int = Field(
        ge=1, le=4,
        description="How effectively the teacher engaged in conversations and discussions with students, and how well they responded",
    )
    verbal_instructional_clarity: int = Field(
        ge=1, le=4,
        description="How clearly the teacher explained a concept?",
    )
    visual_demonstration: int = Field(
        ge=1, le=4,
        description="How effectively the teacher used visual aids or demonstrations to support instruction?",
    )
    visual_presentation: int = Field(
        ge=1, le=4,
        description="How well the lecturer presented themselves visually? Was the camera on? Were they visible? Posture? Smile?",
    )


class ObservationScoringOutput_agent_schema(BaseModel):
    dimension_scores: DimensionScores_agent_schema
    rationale: dict[
        Literal[
            "teacher_student_interaction",
            "verbal_instructional_clarity",
            "visual_demonstration",
            "visual_presentation",
        ],
        str,
    ] = Field(
        description="A rationale for each dimension score, explaining the reasoning behind the assigned score based on the observations provided."
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
            "teacher_student_interaction",
            "verbal_instructional_clarity",
            "visual_demonstration",
            "visual_presentation",
        ],
        int,
    ] = Field(description="Scores for each of the dimensions, each an integer from 1 to 4.")
    rationale: dict[
        Literal[
            "teacher_student_interaction",
            "verbal_instructional_clarity",
            "visual_demonstration",
            "visual_presentation",
        ],
        str,
    ] = Field(description="A rationale for each dimension score based on the observations.")
    observations: list[ObservationItem] = Field(
        description="A list of observations, each containing timestamps and evidence."
    )


RUBRIC_OF_OBSERVATION_AGENT = """
<evidences to note>
1. teacher_student_interaction: How effectively the teacher engaged in conversations and discussions with students, and how well they responded
2. verbal_instructional_clarity: How clearly the teacher explained a concept?
3. visual_demonstration: How effectively the teacher used visual aids or demonstrations to support instruction?
4. visual_presentation: How well the lecturer presented themselves visually (camera on, visible, posture, smile)
</evidences to note>

Observations have to be made based on the evidences to note.
"""

OBSERVATION_SYSTEM_INSTRUCTION = f"""
You are the Observation Agent.

{RUBRIC_OF_OBSERVATION_AGENT}

Review the supplied class recording and produce concrete observations only.

Each observation must contain:
- timestamp_start in HH:MM:SS format (e.g., '00:01:45' for 1 min 45 sec) when determinable.
- timestamp_end in HH:MM:SS format (e.g., '00:01:45' for 1 min 45 sec) when determinable.
- evidence describing what is directly observable or audible. Evidence should NOT include names of individuals; use generic terms like 'Teacher', 'Student A', 'Student B'.
- Consolidate related conversational exchanges into unified events (e.g. 1-3 minute segments) rather than generating an entry for every individual sentence.
- Focus specifically on teaching practices, explanations, demonstrations, and student-teacher discussions as outlined in the rubric.

Use null timestamps when time cannot be determined. Do not score the class.
If evidence is unavailable, return an empty observations list.
"""

SCORING_PROMPT_TEMPLATE = """
Evaluate the class on these four dimensions using only the provided observations:

Dimensions (Score 1 to 4):
1. teacher_student_interaction: How effectively the teacher engaged in conversations and discussions with students, and how well they responded.
2. verbal_instructional_clarity: How clearly the teacher explained a concept.
3. visual_demonstration: How effectively the teacher used visual aids or demonstrations to support instruction.
4. visual_presentation: How well the lecturer presented themselves visually (camera on, visible, posture, facial expression).

Anchors:
1 = Poor: Significant or repeated problems.
2 = Average: Some effective practice with recurring gaps.
3 = Good: Generally effective with minor gaps.
4 = Excellent: Consistently effective with multiple clear examples.

Provide a score (1-4) and a concrete, evidence-based rationale citing the observations for each dimension.
"""


def evaluate_class_recording(
    video_path: str | Path,
    progress_callback: Callable[[str, float], None] | None = None,
    ) -> outputSchema:
    """Evaluates an online class recording using a multi-agent AI pipeline.

    Args:
        video_path: Path to the video file on local disk.
        progress_callback: Optional callable(stage_text, fraction_float) for UI progress updates.

    Returns:
        outputSchema: Validated evaluation output containing classification, scores, rationales, summary, and observations.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    def update_progress(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)
        else:
            print(f"[{int(frac * 100)}%] {msg}")

    client = genai.Client()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    video_file = None
    try:
        update_progress("Uploading video...", 0.10)
        video_file = client.files.upload(file=str(video_path))

        update_progress("Processing the video...", 0.20)
        max_wait = 300
        waited = 0
        while video_file.state.name == "PROCESSING":
            if waited >= max_wait:
                raise TimeoutError("Video processing timed out.")
            time.sleep(3)
            waited += 3
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.error}")

        update_progress("Analyzing the video...", 0.40)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[video_file, "Review this class recording."],
            config=types.GenerateContentConfig(
                system_instruction=OBSERVATION_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=Observation_agent_schema,
                temperature=0,
            ),
        )

        obs_result: Observation_agent_schema = response.parsed
        if not obs_result or not obs_result.observations:
            raise ValueError("Failed to analyze the video.")

    finally:
        if video_file:
            try:
                client.files.delete(name=video_file.name)
            except Exception as del_err:
                print(f"Warning: Failed to delete remote file: {del_err}")

    observations_formatted = "\n".join([
        f"[{obs.timestamp_start} - {obs.timestamp_end}]: {obs.evidence}"
        for obs in obs_result.observations
    ])

    update_progress("Scoring the video...", 0.70)
    scoring_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert teaching evaluation agent.\n\n{SCORING_PROMPT_TEMPLATE}"),
        ("human", "Here are the observations recorded from the class:\n\n{observations_text}\n\nEvaluate and score each dimension."),
    ])
    observation_scoring_agent = llm.with_structured_output(ObservationScoringOutput_agent_schema)
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
    sample_video = "videos/B435 Intermediate Part  - 2026-08-28.mp4"
    print(f"Evaluating {sample_video}...")
    evaluation = evaluate_class_recording(sample_video)
    print("\n" + evaluation.model_dump_json(indent=2))
