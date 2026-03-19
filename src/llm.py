# src/llm.py
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"


def build_prompt(patient: dict, trials: list[dict]) -> str:
    """Build a prompt asking the LLM to assess each trial for the patient."""

    patient_summary = f"""
PATIENT PROFILE:
- Condition: {patient.get('condition', 'N/A')}
- Age: {patient.get('age', 'N/A')}
- Gender: {patient.get('gender', 'N/A')}
- Medical history: {patient.get('history', 'N/A')}
- Current medications: {patient.get('medications', 'N/A')}
"""

    trials_text = ""
    for i, trial in enumerate(trials, 1):
        trials_text += f"""
TRIAL {i}:
- NCT ID: {trial.get('nct_id')}
- Title: {trial.get('title')}
- Phase: {trial.get('phase', 'N/A')}
- Status: {trial.get('status')}
- Min age: {trial.get('min_age', 'N/A')}
- Max age: {trial.get('max_age', 'N/A')}
- Gender: {trial.get('gender', 'N/A')}
- Conditions: {', '.join(trial.get('conditions', []))}
- Interventions: {', '.join(trial.get('interventions', []))}
- Relevant text: {trial.get('text', '')[:600]}
"""

    prompt = f"""
You are a clinical trial matching assistant. Assess whether each trial suits the patient.

{patient_summary}

Below are {len(trials)} candidate trials. For each trial assess:
1. Is it suitable for this patient?
2. Which eligibility criteria are MET?
3. Which criteria are NOT MET or concerning?
4. A match score from 0-10
5. A one-sentence recommendation

Return ONLY a JSON array — no explanation, no markdown, no code blocks:
[
  {{
    "nct_id": "NCT...",
    "title": "...",
    "suitable": true or false,
    "met_criteria": ["criterion 1", "criterion 2"],
    "unmet_criteria": ["criterion 1"],
    "match_score": 8,
    "recommendation": "one sentence summary",
    "cited_evidence": "exact quote from trial text supporting your assessment"
  }}
]

{trials_text}
"""
    return prompt


def assess_trials(patient: dict, trials: list[dict]) -> list[dict]:
    """Send trials to Groq LLM and get structured match assessments."""

    top_trials = trials[:5]
    prompt     = build_prompt(patient, top_trials)

    print("Sending to Groq (llama-3.3-70b) for reasoning...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a clinical trial matching expert. "
                           "Always respond with valid JSON only. "
                           "No markdown, no explanation, just the JSON array."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case model adds markdown fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        assessments = json.loads(raw)
        print(f"Groq assessed {len(assessments)} trials.")
        return assessments
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {raw[:500]}")
        return []


def print_results(assessments: list[dict]):
    """Pretty print the match report."""
    print("\n" + "="*60)
    print("CLINICAL TRIAL MATCH REPORT")
    print("="*60)

    suitable = [a for a in assessments if a.get("suitable")]
    print(f"\nSuitable trials: {len(suitable)} / {len(assessments)}\n")

    for a in assessments:
        status = "MATCH"    if a.get("suitable") else "NO MATCH"
        score  = a.get("match_score", 0)
        bar    = "█" * score + "░" * (10 - score)

        print(f"[{status}] {a.get('title', '')[:60]}")
        print(f"  NCT ID   : {a.get('nct_id')}")
        print(f"  Score    : {bar} {score}/10")
        print(f"  Met      : {', '.join(a.get('met_criteria', []))}")
        print(f"  Unmet    : {', '.join(a.get('unmet_criteria', []))}")
        print(f"  Evidence : {a.get('cited_evidence', '')[:150]}")
        print(f"  Verdict  : {a.get('recommendation', '')}")
        print()


if __name__ == "__main__":
    from retrieve import retrieve_trials, deduplicate

    sample_patient = {
        "condition": "non-small cell lung cancer",
        "age": "58",
        "gender": "Male",
        "history": "Stage III, previously treated with chemotherapy",
        "medications": "carboplatin, paclitaxel",
    }

    print("Step 1: Retrieving trials...")
    chunks      = retrieve_trials(sample_patient, top_k=20)
    trials      = deduplicate(chunks)

    print("\nStep 2: Assessing with Groq...")
    assessments = assess_trials(sample_patient, trials)

    print("\nStep 3: Results")
    print_results(assessments)