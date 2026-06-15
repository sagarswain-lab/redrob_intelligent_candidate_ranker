
"""
rank.py

Ranks candidates from candidates.jsonl against the job description and
writes a top-100 CSV (candidate_id, rank, score, reasoning).

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Also accepts a gzipped .jsonl.gz or the original challenge .zip - it'll
find candidates.jsonl inside either one.

The pipeline: drop the structurally-impossible "honeypot" profiles, score
everything else on skills/title/company/location/availability using the
tables in reference_data.py, add a TF-IDF similarity score against the job
description, sort, take the top 100, and write out a short reasoning for
each. See README.md for why the scoring is set up the way it is.
"""

import argparse
import csv
import gzip
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_data as rd


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

REF_DATE = date(2026, 6, 11)  # "today" for recency calculations

PROFICIENCY_MULT = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.5, "beginner": 0.25}


# ---------------------------------------------------------------------------
# I/O: read candidates from .jsonl / .jsonl.gz / .zip
# ---------------------------------------------------------------------------

def iter_candidates(path_str: str):
    """Yield parsed candidate dicts from a .jsonl, .jsonl.gz, or .zip file.

    Also prints a "Source: ..." line to stderr identifying which file was
    actually read - for a .zip this is the candidates.jsonl found inside it,
    which is otherwise invisible to anything calling this from outside.
    """
    path = Path(path_str)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            target = None
            for name in z.namelist():
                if name.endswith("candidates.jsonl") and "__MACOSX" not in name:
                    target = name
                    break
            if target is None:
                raise FileNotFoundError(f"No candidates.jsonl found inside {path}")
            print(f"Source: {Path(target).name} (from {path.name})", file=sys.stderr)
            with z.open(target) as raw:
                for line in raw:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return

    print(f"Source: {path.name}", file=sys.stderr)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# HONEYPOT DETECTION
# ---------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> bool:
    """
    Flag "subtly impossible" profiles via three structural checks, each
    validated against the full 100K pool to fire on a clean, designed
    subset (~21-24 candidates each, ~65 total with near-zero overlap):

      A. >=3 skills claim "expert" proficiency with duration_months == 0.
         (The per-candidate distribution of this count is {0: 99979,
         3: 8, 4: 5, 5: 8} - i.e. literally nobody has 1 or 2, only a clean
         designed subset has 3+.)

      B. total career_history duration is < 40% of years_of_experience*12
         (e.g. "14.1 years of experience" backed by a single 56-month job).

      C. total career_history duration is > 160% of years_of_experience*12
         (e.g. "3.0 years of experience" but three jobs summing to 61
         months - more months worked than years claimed).

    B and C catch exactly the JD's example ("8 years of experience at a
    company founded 3 years ago") in spirit: the claimed tenure doesn't
    reconcile with the timeline. A few of these honeypots even carry
    *great* titles/companies on the surface (e.g. "Machine Learning Engineer
    @ Dream11" or "Applied ML Engineer @ Haptik") - the timeline check is
    what catches them, not the skill/title score.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    # A. expert + zero-duration skills
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    if expert_zero >= 3:
        return True

    # B / C. years_of_experience vs. total career_history duration
    total_months = sum(j.get("duration_months", 0) or 0 for j in career)
    if yoe > 0:
        ratio = total_months / (yoe * 12)
        if ratio < 0.4 or ratio > 1.6:
            return True

    return False


# ---------------------------------------------------------------------------
# STRUCTURED FEATURE SCORING
# ---------------------------------------------------------------------------

def score_skills(skills: list) -> tuple:
    """
    Sum of (skill_weight * proficiency_mult * duration_mult) across all
    skills. duration_mult ramps from 0.7 (0 months) to 1.0 (>=24 months),
    so a freshly-claimed skill counts less than a well-established one.

    Returns (total_score, list of (name, weight, proficiency, duration)
    for the top 4 weighted skills) for use in reasoning generation.
    """
    contributions = []
    for s in skills:
        name = s.get("name", "")
        weight = rd.SKILL_WEIGHTS.get(name, rd.DEFAULT_SKILL_WEIGHT)
        if weight <= 0:
            continue
        prof = s.get("proficiency", "beginner")
        prof_mult = PROFICIENCY_MULT.get(prof, 0.25)
        dur = s.get("duration_months", 0) or 0
        dur_mult = 0.7 + 0.3 * min(1.0, dur / 24.0)
        contribution = weight * prof_mult * dur_mult
        contributions.append((contribution, name, weight, prof, dur))

    total = sum(c[0] for c in contributions)
    contributions.sort(key=lambda x: -x[0])
    top_skills = [(name, weight, prof, dur) for _, name, weight, prof, dur in contributions[:4]]
    return total, top_skills


def score_title(current_title: str) -> float:
    return rd.TITLE_TIER_SCORE.get(current_title, rd.DEFAULT_TITLE_SCORE)


def score_career(profile: dict, career: list) -> dict:
    """
    Returns a dict with:
      - product_exposure_score : max company-tier score across current +
                                  all career_history companies
      - consulting_penalty     : -8 if EVERY company the candidate has ever
                                  worked at (current + history) is in the
                                  JD's named consulting-firm list (the JD's
                                  explicit "only worked at consulting firms"
                                  disqualifier). 0 otherwise - note this is
                                  *not* penalized if the candidate is
                                  *currently* at a consulting firm but has
                                  product-company history, per the JD's
                                  stated exception.
      - experience_fit_score   : trapezoidal score peaking at the JD's
                                  "6-8 years, ideally" sweet spot, tapering
                                  to 0 outside roughly 3-12 years.
      - best_company           : name of the highest-tier company found
                                  (for reasoning), or None.
      - best_company_tier      : "ai_native" / "faang" / "indian_product" /
                                  None - for reasoning.
    """
    current_company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0) or 0

    companies = {current_company}
    companies.update(j.get("company", "") for j in career)

    best_score = -1
    best_company = None
    for c in companies:
        s = rd.COMPANY_PRODUCT_SCORE.get(c, rd.DEFAULT_COMPANY_SCORE)
        if s > best_score:
            best_score = s
            best_company = c
    if best_score <= 0:
        best_company = None

    best_company_tier = None
    if best_company in rd.AI_NATIVE_COMPANIES:
        best_company_tier = "ai_native"
    elif best_company in rd.FAANG_COMPANIES:
        best_company_tier = "faang"
    elif best_company in rd.INDIAN_PRODUCT_COMPANIES:
        best_company_tier = "indian_product"

    consulting_penalty = 0.0
    if companies and companies.issubset(rd.CONSULTING_COMPANIES):
        consulting_penalty = -8.0

    # Experience-fit: trapezoid, peak 6-8, tapering out by ~3 and ~12
    if 6 <= yoe <= 8:
        exp_fit = 10.0
    elif 5 <= yoe < 6 or 8 < yoe <= 9:
        exp_fit = 8.0
    elif 4 <= yoe < 5 or 9 < yoe <= 10.5:
        exp_fit = 5.0
    elif 3 <= yoe < 4 or 10.5 < yoe <= 12:
        exp_fit = 2.0
    else:
        exp_fit = 0.0

    return {
        "product_exposure_score": max(best_score, 0),
        "consulting_penalty": consulting_penalty,
        "experience_fit_score": exp_fit,
        "best_company": best_company,
        "best_company_tier": best_company_tier,
    }


def score_location(profile: dict, signals: dict) -> float:
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate", False))

    if any(city in loc for city in rd.LOCATION_TOP):
        return 10.0
    if any(city in loc for city in rd.LOCATION_WELCOME):
        return 8.0
    if "india" in country:
        return 5.0 if relocate else 3.0
    return 1.0 if relocate else 0.0


def score_availability(signals: dict) -> dict:
    """
    Returns dict with the total availability_score plus the individual
    component values (used in reasoning to call out specific concerns,
    e.g. "5% recruiter response rate" per the JD's own example phrasing).
    """
    # Recency
    last_active_str = signals.get("last_active_date", "")
    days_inactive = None
    if last_active_str:
        try:
            days_inactive = (REF_DATE - date.fromisoformat(last_active_str)).days
        except ValueError:
            pass

    if days_inactive is None:
        recency_score = 0.0
    elif days_inactive <= 7:
        recency_score = 10.0
    elif days_inactive <= 30:
        recency_score = 8.0
    elif days_inactive <= 90:
        recency_score = 5.0
    elif days_inactive <= 180:
        recency_score = 2.0
    else:
        recency_score = -2.0

    # Open to work
    otw = bool(signals.get("open_to_work_flag", False))
    otw_score = 2.0 if otw else 0.0

    # Recruiter response rate - JD explicitly calls out a "5% response rate"
    # example as a strong down-weight signal.
    rr = signals.get("recruiter_response_rate", 0.0) or 0.0
    if rr < 0.10:
        rr_score = -5.0
    elif rr < 0.25:
        rr_score = -2.0
    elif rr < 0.50:
        rr_score = 0.0
    elif rr < 0.75:
        rr_score = 2.0
    else:
        rr_score = 4.0

    # Notice period - JD: sub-30 ideal, 30+ "still in scope but bar is higher"
    notice = signals.get("notice_period_days", 90)
    if notice is None:
        notice = 90
    if notice <= 15:
        notice_score = 3.0
    elif notice <= 30:
        notice_score = 2.0
    elif notice <= 60:
        notice_score = 0.0
    elif notice <= 90:
        notice_score = -1.0
    else:
        notice_score = -2.0

    icr = signals.get("interview_completion_rate", 0.5) or 0.5
    icr_score = icr * 2.0

    total = recency_score + otw_score + rr_score + notice_score + icr_score

    return {
        "total": total,
        "days_inactive": days_inactive,
        "otw": otw,
        "response_rate": rr,
        "notice_period": notice,
        "interview_completion_rate": icr,
    }


# ---------------------------------------------------------------------------
# TEXT FOR HYBRID (TF-IDF) SIGNAL
# ---------------------------------------------------------------------------

def build_candidate_text(candidate: dict) -> str:
    profile = candidate.get("profile", {})
    parts = [profile.get("summary", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
    for s in candidate.get("skills", []):
        # repeat skill name so it carries weight proportional to count,
        # and so high-signal skill names appear in the TF-IDF vocabulary
        parts.append(s.get("name", ""))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# REASONING GENERATION
# ---------------------------------------------------------------------------

def _fmt_years(months):
    if not months:
        return "0mo"
    if months < 12:
        return f"{months}mo"
    yrs = months / 12.0
    if abs(yrs - round(yrs)) < 0.05:
        return f"{round(yrs)}y"
    return f"{yrs:.1f}y"


def _h(cid: str, salt: str, n: int) -> int:
    """Deterministic per-candidate, per-slot index in [0, n)."""
    return (hash((cid, salt)) % n + n) % n


def generate_reasoning(candidate: dict, rank: int, feats: dict) -> str:
    """
    Build a 1-2 sentence, fact-grounded reasoning string. Every fact used
    here comes directly from `candidate` or `feats` (computed purely from
    `candidate`) - nothing is invented. Phrasing is selected per-candidate
    via a hash of candidate_id so that the 100 reasoning strings use varied
    sentence structures rather than one fill-in-the-blank template.
    """
    profile = candidate["profile"]
    sig = candidate["redrob_signals"]
    career = candidate.get("career_history", [])
    cid = candidate["candidate_id"]

    title = profile.get("current_title", "?")
    company = profile.get("current_company", "?")
    yoe = profile.get("years_of_experience", 0) or 0
    location = profile.get("location", "?")

    top_skills = feats["top_skills"]  # list of (name, weight, prof, dur)
    best_company = feats["career"]["best_company"]
    best_company_tier = feats["career"]["best_company_tier"]
    avail = feats["availability"]

    # ---- Build a pool of POSITIVE fact phrases (only if true) -----------
    positives = []

    if top_skills:
        n1, w1, p1, d1 = top_skills[0]
        if w1 >= 8:
            opts = [
                f"{p1} {n1} ({_fmt_years(d1)})",
                f"{p1}-level {n1} experience ({_fmt_years(d1)})",
                f"{_fmt_years(d1)} of {p1} {n1}",
            ]
            positives.append(("skill1", opts[_h(cid, "skill1", len(opts))]))
        if len(top_skills) > 1:
            n2, w2, p2, d2 = top_skills[1]
            if w2 >= 6:
                opts = [
                    f"{p2} {n2} ({_fmt_years(d2)})",
                    f"{n2} at {p2} level ({_fmt_years(d2)})",
                ]
                positives.append(("skill2", opts[_h(cid, "skill2", len(opts))]))

    if best_company_tier == "ai_native":
        opts = [
            f"AI-native company experience at {best_company}",
            f"has worked at {best_company}, an AI-focused product company",
        ]
        positives.append(("company", opts[_h(cid, "company", len(opts))]))
    elif best_company_tier == "faang":
        opts = [
            f"large-scale product experience at {best_company}",
            f"has worked at {best_company}",
        ]
        positives.append(("company", opts[_h(cid, "company", len(opts))]))
    elif best_company_tier == "indian_product":
        opts = [
            f"product-company experience at {best_company}",
            f"has worked at {best_company}, a consumer product company",
        ]
        positives.append(("company", opts[_h(cid, "company", len(opts))]))

    if 5 <= yoe <= 9:
        opts = [
            f"{yoe:.1f} years of experience, within the JD's 5-9 year band",
            f"{yoe:.1f} yrs experience fits the target range",
        ]
        positives.append(("exp", opts[_h(cid, "exp", len(opts))]))

    if avail["response_rate"] >= 0.5 and avail["otw"]:
        opts = [
            f"open to work with a {avail['response_rate']:.0%} recruiter response rate",
            f"actively engaged ({avail['response_rate']:.0%} response rate, open to work)",
        ]
        positives.append(("avail", opts[_h(cid, "avail", len(opts))]))

    if avail["notice_period"] <= 30:
        opts = [
            f"a {avail['notice_period']}-day notice period",
            f"can start within {avail['notice_period']} days",
        ]
        positives.append(("notice", opts[_h(cid, "notice", len(opts))]))

    # career history note - cite a specific past role if it carries a
    # relevant title (search/ranking/recommendation/ML)
    for job in career:
        jt = (job.get("title") or "").lower()
        if any(k in jt for k in ["search", "ranking", "recommend", "machine learning",
                                   "applied scientist", "nlp", "ai engineer", "data scientist"]):
            opts = [
                f"previously {job.get('title')} at {job.get('company')} ({_fmt_years(job.get('duration_months'))})",
                f"earlier held a {job.get('title')} role at {job.get('company')}",
            ]
            positives.append(("history", opts[_h(cid, "history", len(opts))]))
            break

    # ---- Build a pool of CONCERN phrases (only if true) ------------------
    concerns = []

    title_score = feats["title_score"]
    if title_score < 0:
        opts = [
            f"current title ({title}) is outside the AI/ML track",
            f"title is {title}, not an AI/ML role",
        ]
        concerns.append(("title", opts[_h(cid, "ctitle", len(opts))]))
    elif title_score == 0:
        opts = [
            f"title ({title}) is general software engineering, not ML-focused",
        ]
        concerns.append(("title", opts[_h(cid, "ctitle", len(opts))]))

    if not (5 <= yoe <= 9):
        if yoe < 5:
            opts = [
                f"{yoe:.1f} years of experience is below the JD's 5-9 year band",
                f"only {yoe:.1f} yrs experience, under the target range",
            ]
        else:
            opts = [
                f"{yoe:.1f} years of experience runs above the JD's 5-9 year band",
                f"{yoe:.1f} yrs experience is more senior than the target range",
            ]
        concerns.append(("exp", opts[_h(cid, "cexp", len(opts))]))

    loc_score = feats["location_score"]
    if loc_score <= 3:
        country = profile.get("country", "")
        if "india" not in country.lower():
            opts = [
                f"based outside India ({location}, {country}), and the role doesn't sponsor visas",
                f"located in {location}, {country} - outside the JD's India-based hiring pool",
            ]
        else:
            relocate = sig.get("willing_to_relocate", False)
            if relocate:
                opts = [
                    f"based in {location}, willing to relocate but outside Noida/Pune/NCR",
                ]
            else:
                opts = [
                    f"based in {location} and not flagged as willing to relocate to Noida/Pune",
                    f"{location}-based with no relocation flag set",
                ]
        concerns.append(("loc", opts[_h(cid, "cloc", len(opts))]))

    if avail["response_rate"] < 0.25:
        opts = [
            f"a low {avail['response_rate']:.0%} recruiter response rate suggests limited engagement",
            f"recruiter response rate is only {avail['response_rate']:.0%}",
        ]
        concerns.append(("resp", opts[_h(cid, "cresp", len(opts))]))

    if avail["days_inactive"] is not None and avail["days_inactive"] > 90:
        opts = [
            f"inactive on the platform for {avail['days_inactive']} days",
            f"last active {avail['days_inactive']} days ago",
        ]
        concerns.append(("active", opts[_h(cid, "cactive", len(opts))]))

    if avail["notice_period"] > 90:
        opts = [
            f"a long {avail['notice_period']}-day notice period",
            f"notice period of {avail['notice_period']} days is well above the JD's preference",
        ]
        concerns.append(("notice", opts[_h(cid, "cnotice", len(opts))]))

    if feats["career"]["consulting_penalty"] < 0:
        opts = [
            "entire career has been at consulting firms with no product-company experience",
            "career history is consulting-only (no product company exposure)",
        ]
        concerns.append(("consulting", opts[_h(cid, "ccons", len(opts))]))

    has_retrieval_skill = any(
        s.get("name") in rd.RETRIEVAL_RELATED_SKILLS for s in candidate.get("skills", [])
    )
    if not has_retrieval_skill:
        opts = [
            "no embeddings, vector-DB, or retrieval/ranking skills listed",
            "skill profile lacks any embeddings/retrieval/ranking-system entries the JD asks for",
        ]
        concerns.append(("skills", opts[_h(cid, "cskills", len(opts))]))

    # ---- JD-connection closer, used whenever there's no flagged concern --
    # Maps the candidate's top-weighted skill to a short clause naming
    # *which* JD requirement it satisfies, so even "clean" profiles get an
    # explicit tie-back to the JD instead of generic filler.
    top_skill_name = top_skills[0][0] if top_skills else None
    jd_group = rd.SKILL_GROUP.get(top_skill_name)
    jd_phrase_options = rd.GROUP_PHRASES.get(jd_group, rd.DEFAULT_JD_PHRASES)
    jd_phrase = jd_phrase_options[_h(cid, "jdphrase", len(jd_phrase_options))]
    jd_sentence = f"This {jd_phrase}."

    # ---- Assemble final string -------------------------------------------
    pos_texts = [p[1] for p in positives]
    con_texts = [c[1] for c in concerns]

    openers = [
        f"{title} at {company}",
        f"{title} ({company})",
        f"Currently a {title} at {company}",
        f"{title} ({company}, {yoe:.1f} yrs)",
    ]
    opener = openers[_h(cid, "opener", len(openers))]

    if rank <= 10:
        body = "; ".join(pos_texts[:3]) if pos_texts else f"{yoe:.1f} years of experience"
        sentence = f"{opener}: {body}."
        if con_texts:
            sentence += f" Minor watch-point: {con_texts[0]}."
        else:
            sentence += f" {jd_sentence}"
    elif rank <= 40:
        body = "; ".join(pos_texts[:2]) if pos_texts else f"{yoe:.1f} years of experience, {title}"
        sentence = f"{opener}: {body}."
        if con_texts:
            sentence += f" Concern: {con_texts[0]}."
        else:
            sentence += f" {jd_sentence}"
    else:
        # rank 41-100: vary SENTENCE STRUCTURE (not just word choice) across
        # 4 templates, chosen per-candidate, so the bottom of the shortlist
        # doesn't read as one fill-in-the-blank template repeated 60 times.
        strength = pos_texts[0] if pos_texts else f"{yoe:.1f} years of experience"
        strength2 = pos_texts[1] if len(pos_texts) > 1 else None
        concern = con_texts[0] if con_texts else None
        concern2 = con_texts[1] if len(con_texts) > 1 else None

        structure = _h(cid, "structure", 4)
        if structure == 0:
            sentence = f"{opener}: {strength}."
            sentence += f" On the downside, {concern}." if concern else f" {jd_sentence}"
        elif structure == 1:
            if concern:
                sentence = f"{opener} brings {strength}, but {concern}."
            else:
                extra = f"; also {strength2}" if strength2 else ""
                sentence = f"{opener} brings {strength}{extra}. {jd_sentence}"
        elif structure == 2:
            if concern:
                cap_concern = concern[0].upper() + concern[1:]
                sentence = f"{cap_concern}; still, {opener} brings {strength}."
            else:
                sentence = f"{opener} offers {strength}. {jd_sentence}"
        else:
            if strength2:
                sentence = f"{opener}: {strength}; {strength2}."
            else:
                sentence = f"{opener}: {strength}."
            if concern:
                sentence += f" Worth noting: {concern}."
            elif not strength2:
                sentence += f" {jd_sentence}"

        # Bottom of the shortlist: surface a second concern too, if any,
        # so the lowest-ranked picks are explicit about multiple gaps.
        if rank > 70 and concern2:
            sentence += f" Also: {concern2}."

    return sentence[:400]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True,
                         help="Path to candidates.jsonl / .jsonl.gz / .zip")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    print(f"[1/5] Reading candidates from {args.candidates} ...", file=sys.stderr)

    records = []          # one dict per non-honeypot candidate
    candidate_texts = []  # parallel list, for TF-IDF
    n_total = 0
    n_honeypot = 0

    for c in iter_candidates(args.candidates):
        n_total += 1
        if detect_honeypot(c):
            n_honeypot += 1
            continue

        profile = c["profile"]
        sig = c["redrob_signals"]
        career = c.get("career_history", [])
        skills = c.get("skills", [])

        skill_total, top_skills = score_skills(skills)
        title_sc = score_title(profile.get("current_title", ""))
        career_feats = score_career(profile, career)
        loc_sc = score_location(profile, sig)
        avail = score_availability(sig)

        structured_score = (
            skill_total
            + title_sc
            + career_feats["product_exposure_score"]
            + career_feats["consulting_penalty"]
            + career_feats["experience_fit_score"]
            + loc_sc
            + avail["total"]
        )

        feats = {
            "top_skills": top_skills,
            "title_score": title_sc,
            "career": career_feats,
            "location_score": loc_sc,
            "availability": avail,
            "structured_score": structured_score,
        }

        records.append({"candidate": c, "feats": feats})
        candidate_texts.append(build_candidate_text(c))

        if n_total % 20000 == 0:
            print(f"    ... {n_total:,} read", file=sys.stderr)

    print(f"    {n_total:,} total candidates, {n_honeypot} honeypots excluded, "
          f"{len(records):,} remain.", file=sys.stderr)

    # -------------------------------------------------------------------
    print("[2/5] Computing TF-IDF hybrid lexical/semantic signal ...", file=sys.stderr)
    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    all_texts = [rd.JD_TEXT] + candidate_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = tfidf_matrix[0:1]
    cand_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vec, cand_vecs).ravel()  # shape (n_records,)

    # Scale TF-IDF similarity into a bonus comparable in magnitude to the
    # structured-score modifiers (roughly 0-10 points).
    sim_min, sim_max = float(sims.min()), float(sims.max())
    sim_range = max(sim_max - sim_min, 1e-9)
    tfidf_bonus = (sims - sim_min) / sim_range * 10.0

    # -------------------------------------------------------------------
    print("[3/5] Combining scores ...", file=sys.stderr)
    for rec, bonus in zip(records, tfidf_bonus):
        rec["feats"]["tfidf_bonus"] = float(bonus)
        rec["final_score"] = rec["feats"]["structured_score"] + float(bonus)

    records.sort(key=lambda r: (-r["final_score"], r["candidate"]["candidate_id"]))
    top = records[: args.top_k]

    print(f"    Top-{args.top_k} score range: "
          f"{top[0]['final_score']:.2f} .. {top[-1]['final_score']:.2f}", file=sys.stderr)

    # -------------------------------------------------------------------
    print("[4/5] Generating reasoning ...", file=sys.stderr)
    rows = []
    raw_scores = [r["final_score"] for r in top]
    smin, smax = min(raw_scores), max(raw_scores)
    srange = max(smax - smin, 1e-9)

    for i, rec in enumerate(top, start=1):
        c = rec["candidate"]
        feats = rec["feats"]
        reasoning = generate_reasoning(c, i, feats)
        # Map raw_score -> (0, 1], strictly decreasing with rank, ties
        # broken by a tiny epsilon so the score column is non-increasing
        # even when raw scores tie (the spec allows ties, but a strictly
        # decreasing sequence is simplest to validate).
        norm = 0.40 + 0.59 * (rec["final_score"] - smin) / srange
        norm -= (i - 1) * 1e-4
        norm = max(norm, 0.0001)
        rows.append((c["candidate_id"], i, round(norm, 4), reasoning))

    # -------------------------------------------------------------------
    print(f"[5/5] Writing {args.out} ...", file=sys.stderr)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        writer.writerows(rows)

    print("\nTop 10:", file=sys.stderr)
    for cid, i, sc, reasoning in rows[:10]:
        print(f"  {i:3d}. {cid}  score={sc:.4f}", file=sys.stderr)
        print(f"       {reasoning}", file=sys.stderr)

    print(f"\nDone. Wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()