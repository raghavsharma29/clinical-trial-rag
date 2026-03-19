# src/ingest.py
import requests, json, os
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def fetch_trials(condition: str = "cancer", max_trials: int = 500) -> list[dict]:
    """Fetch trials from ClinicalTrials.gov API and parse into clean chunks."""
    trials = []
    next_token = None
    page_size = 100

    print(f"Fetching up to {max_trials} trials for condition: {condition}")

    with tqdm(total=max_trials) as pbar:
        while len(trials) < max_trials:
            params = {
                "query.cond": condition,
                "pageSize": min(page_size, max_trials - len(trials)),
                "format": "json",
                "fields": "NCTId,BriefTitle,OfficialTitle,BriefSummary,"
                          "DetailedDescription,EligibilityCriteria,"
                          "OverallStatus,Phase,Condition,InterventionName,"
                          "MinimumAge,MaximumAge,Gender,StudyType"
            }
            if next_token:
                params["pageToken"] = next_token

            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            studies = data.get("studies", [])
            if not studies:
                break

            for study in studies:
                parsed = parse_study(study)
                if parsed:
                    trials.append(parsed)
                    pbar.update(1)

            next_token = data.get("nextPageToken")
            if not next_token:
                break

    print(f"Fetched {len(trials)} trials")
    return trials


def parse_study(study: dict) -> dict | None:
    """Extract and flatten relevant fields from a raw study object."""
    try:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        desc_mod = proto.get("descriptionModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        cond_mod = proto.get("conditionsModule", {})
        intervention_mod = proto.get("armsInterventionsModule", {})

        nct_id = id_mod.get("nctId", "")
        if not nct_id:
            return None

        # Extract interventions
        interventions = [
            i.get("name", "") for i in
            intervention_mod.get("interventions", [])
        ]

        return {
            "nct_id": nct_id,
            "title": id_mod.get("briefTitle", ""),
            "official_title": id_mod.get("officialTitle", ""),
            "summary": desc_mod.get("briefSummary", ""),
            "description": desc_mod.get("detailedDescription", ""),
            "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
            "min_age": elig_mod.get("minimumAge", ""),
            "max_age": elig_mod.get("maximumAge", ""),
            "gender": elig_mod.get("sex", "ALL"),
            "status": status_mod.get("overallStatus", ""),
            "phase": ", ".join(design_mod.get("phases", [])),
            "conditions": cond_mod.get("conditions", []),
            "interventions": interventions,
            "study_type": design_mod.get("studyType", ""),
        }
    except Exception as e:
        print(f"Error parsing study: {e}")
        return None


def chunk_trial(trial: dict) -> list[dict]:
    """Split one trial into multiple chunks, each with full metadata."""
    chunks = []
    base_meta = {
        "nct_id": trial["nct_id"],
        "title": trial["title"],
        "status": trial["status"],
        "phase": trial["phase"],
        "conditions": trial["conditions"],
        "interventions": trial["interventions"],
        "min_age": trial["min_age"],
        "max_age": trial["max_age"],
        "gender": trial["gender"],
    }

    # Chunk 1: summary
    if trial["summary"].strip():
        chunks.append({**base_meta, "section": "summary",
                        "text": trial["summary"].strip()})

    # Chunk 2: eligibility criteria (most important for matching)
    if trial["eligibility_criteria"].strip():
        chunks.append({**base_meta, "section": "eligibility",
                        "text": trial["eligibility_criteria"].strip()})

    # Chunk 3: detailed description (split if long)
    desc = trial["description"].strip()
    if desc:
        # Split into 800-char chunks with 100-char overlap
        chunk_size, overlap = 800, 100
        for i in range(0, len(desc), chunk_size - overlap):
            chunk_text = desc[i:i + chunk_size]
            if len(chunk_text) > 100:
                chunks.append({**base_meta, "section": "description",
                                "text": chunk_text})

    return chunks


def ingest_and_save(condition: str = "cancer", max_trials: int = 500):
    trials = fetch_trials(condition, max_trials)
    all_chunks = []
    for trial in trials:
        all_chunks.extend(chunk_trial(trial))

    out_path = PROCESSED_DIR / f"{condition}_chunks.json"
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {out_path}")
    return all_chunks


if __name__ == "__main__":
    ingest_and_save(condition="cancer", max_trials=200)