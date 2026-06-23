"""
reranker.py
===========
Module 2 of the Candidate Ranking System – Intelligent LLM Reranker (Groq Edition).

Why Groq?
---------
Groq's LPU inference hardware runs llama3-8b-8192 at ~500 tokens/second — roughly
10-20× faster than a local CPU Ollama setup. No GPU required on your machine.

Pipeline
--------
1. Load FAISS index + metadata (via Module 1).
2. Retrieve Top-K candidates semantically (via Module 1).
3. Decompose the Job Description into a structured rubric via Groq LLM.
4. Rerank the top-K candidates: (rich_profile + rubric) → Groq → score 0-10.
5. Sort by LLM score and display the final ranked table.

API Key Setup
-------------
Get a FREE Groq API key (no credit card needed) at: https://console.groq.com/keys

Option A – .env file (recommended):
    Create d:\\dinchak code\\CV checker\\.env with:
        GROQ_API_KEY=gsk_...

Option B – PowerShell environment variable:
    $env:GROQ_API_KEY = "gsk_..."

Usage
-----
    python reranker.py                              # default: top-50, llama3-8b-8192
    python reranker.py --top-k 30                  # retrieve 30 candidates
    python reranker.py --model mixtral-8x7b-32768  # use Mixtral instead
    python reranker.py --concurrency 10            # 10 parallel calls (check your RPM)
    python reranker.py --output-csv final.csv      # save results to CSV

Rate Limits (free tier as of 2025)
------------------------------------
    llama3-8b-8192   : 30 RPM, 14 400 RPD, 6000 TPM
    llama3-70b-8192  : 30 RPM, 14 400 RPD, 6000 TPM
    mixtral-8x7b-32768: 30 RPM, 14 400 RPD, 5000 TPM

Dependencies
------------
    pip install groq python-dotenv pandas python-docx tqdm
    (faiss-cpu, sentence-transformers already required by Module 1)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Reconfigure stdout/stderr to use UTF-8 on Windows to prevent UnicodeEncodeErrors
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from dotenv import load_dotenv
from groq import AsyncGroq, APIStatusError, APIConnectionError, RateLimitError
from tqdm.asyncio import tqdm as async_tqdm

# ── Module 1 imports ──────────────────────────────────────────────────────────
from candidate_ranker import (
    INDEX_PATH,
    JD_PATH,
    METADATA_PATH,
    RetrievalResult,
    extract_jd_text,
    get_top_k_candidates,
)
from skill_graph import SkillGraph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent

# Load .env from the project directory (silently ignored if file doesn't exist)
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"  # Groq's fastest model (replaces decommissioned llama3-8b-8192)

# Other available Groq models (as of 2025):
#   llama-3.3-70b-versatile   — highest quality, same free-tier RPM
#   llama-3.1-70b-versatile   — good balance
#   gemma2-9b-it              — Google's Gemma 2, very fast
#   mixtral-8x7b-32768        — large context window (32K)

# Groq free-tier: 6000 TPM limit on llama-3.1-8b-instant.
# Run concurrent requests to speed up the process while our retry logic handles rate limiting.
DEFAULT_CONCURRENCY = 3

# Retry config — Groq occasionally returns 503 or 429 under load.
# With concurrency=1, we will wait for x-ratelimit-reset-tokens.
MAX_RETRIES      = 12
INITIAL_BACKOFF  = 5.0    # seconds
BACKOFF_MULT     = 1.5    # exponential multiplier


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JDRubric:
    """Structured decomposition of a Job Description."""
    must_have_skills:         list[str] = field(default_factory=list)
    nice_to_have_skills:      list[str] = field(default_factory=list)
    minimum_experience_years: float     = 0.0
    key_responsibilities:     list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)
    expanded_must_have_skills: dict[str, list[str]] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Render the rubric as a compact string for inclusion in LLM prompts."""
        must = ", ".join(self.must_have_skills)       or "Not specified"
        nice = ", ".join(self.nice_to_have_skills)    or "Not specified"
        resp = "; ".join(self.key_responsibilities)   or "Not specified"
        
        expanded_text = ""
        if self.expanded_must_have_skills:
            parts = []
            for skill, related in self.expanded_must_have_skills.items():
                parts.append(f"{skill} (Related: {', '.join(related)})")
            expanded_text = f"EXPANDED MUST-HAVE SKILLS: {'; '.join(parts)}\n"

        return (
            f"MUST-HAVE SKILLS: {must}\n"
            f"{expanded_text}"
            f"NICE-TO-HAVE SKILLS: {nice}\n"
            f"MINIMUM EXPERIENCE: {self.minimum_experience_years} years\n"
            f"KEY RESPONSIBILITIES: {resp}"
        )


@dataclass
class RankedCandidate:
    """A candidate after LLM reranking."""
    final_rank:     int
    candidate_id:   str
    llm_score:      float   # 0–10 from Groq LLM
    semantic_score: float   # cosine similarity from FAISS (Module 1)
    reason:         str     # 1-sentence justification
    rich_profile:   str
    graph_match:    bool    = False


# ---------------------------------------------------------------------------
# Groq client factory
# ---------------------------------------------------------------------------

def _make_groq_client() -> AsyncGroq:
    """
    Create an authenticated async Groq client.

    Reads GROQ_API_KEY from the environment (or .env file, already loaded).
    Raises RuntimeError with clear setup instructions if the key is missing.

    Returns
    -------
    AsyncGroq
        Ready-to-use async Groq client.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set.\n"
            "  Get a FREE key at: https://console.groq.com/keys\n\n"
            "  Option A — .env file (recommended):\n"
            "    Create .env in the project folder with:\n"
            "      GROQ_API_KEY=gsk_...\n\n"
            "  Option B — PowerShell:\n"
            "    $env:GROQ_API_KEY = 'gsk_...'\n"
        )
    return AsyncGroq(api_key=api_key)


# ---------------------------------------------------------------------------
# JSON repair / extraction
# ---------------------------------------------------------------------------

def _repair_and_parse_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.

    LLMs sometimes wrap JSON in markdown fences, add preambles, use single
    quotes, or omit quotes around keys. This function applies a cascade of
    repair strategies before giving up.

    Strategies (in order)
    ---------------------
    1. Strip markdown code fences (```json … ``` or ``` … ```).
    2. Direct ``json.loads`` on the cleaned string.
    3. Regex: extract the FIRST greedy ``{ … }`` block.
    4. Regex: extract the first non-greedy ``{…}`` block (handles nesting).
    5. Single-quote → double-quote repair and retry.
    6. Bare-key repair  {score: 7} → {"score": 7}  and retry.

    Parameters
    ----------
    raw : str
        Raw text returned by the LLM.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If every repair strategy fails.
    """
    # 1. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.strip().rstrip("`").strip()

    # 2. Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Greedy {…} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Non-greedy {…} (handles nested objects)
    m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 5. Single-quote → double-quote repair
    sq = cleaned.replace("'", '"')
    try:
        return json.loads(sq)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", sq, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # 6. Bare-key repair  {score: 7} → {"score": 7}
    bk = re.sub(r'(\w+)\s*:', r'"\1":', cleaned)
    bk = re.sub(r'""(\w+)""', r'"\1"', bk)   # de-dup double-quotes
    try:
        return json.loads(bk)
    except json.JSONDecodeError:
        pass

    raise ValueError(
        f"JSON repair failed after 6 strategies.\n"
        f"First 400 chars of raw response:\n{raw[:400]}"
    )


# ---------------------------------------------------------------------------
# Rate limit parser
# ---------------------------------------------------------------------------

def _parse_tpm_retry_after(exc: Exception) -> float | None:
    """
    Parse the retry-after or reset time from a Groq RateLimitError.
    Looks at headers and parses the exception message as fallback.
    """
    # 1. Try to read from headers
    if hasattr(exc, "response") and exc.response is not None:
        headers = exc.response.headers
        
        # Check standard retry-after header
        if "retry-after" in headers:
            try:
                return float(headers["retry-after"])
            except (ValueError, TypeError):
                pass
                
        # Check Groq-specific rate limit reset headers
        for h in ["x-ratelimit-reset-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset"]:
            if h in headers:
                val = str(headers[h]).strip()
                # e.g., "459ms" or "459 ms"
                m_ms = re.search(r"(\d+(?:\.\d+)?)\s*ms", val)
                if m_ms:
                    return float(m_ms.group(1)) / 1000.0
                
                # e.g., "6s" or "6 s"
                m_s = re.search(r"(\d+(?:\.\d+)?)\s*s", val)
                if m_s:
                    return float(m_s.group(1))
                
                # e.g., "1m12s"
                m_min = re.search(r"(\d+)\s*m", val)
                m_sec = re.search(r"(\d+)\s*s", val)
                total = 0.0
                if m_min:
                    total += float(m_min.group(1)) * 60
                if m_sec:
                    total += float(m_sec.group(1))
                if total > 0:
                    return total
                    
    # 2. Try to parse from the exception message as fallback
    msg = str(exc)
    m = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
        
    return None


# ---------------------------------------------------------------------------
# Core async LLM call (Groq)
# ---------------------------------------------------------------------------

async def _groq_call(
    client:  AsyncGroq,
    model:   str,
    prompt:  str,
    max_retries: int   = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
) -> str:
    """
    Call the Groq API asynchronously with exponential backoff on rate limits
    and transient server errors.

    Parameters
    ----------
    client : AsyncGroq
        Authenticated async Groq client.
    model : str
        Groq model identifier (e.g. "llama3-8b-8192").
    prompt : str
        Full prompt string to send as the user message.
    max_retries : int
        Number of retries on 429 / 503 errors.
    initial_backoff : float
        Initial sleep time in seconds before the first retry.

    Returns
    -------
    str
        The model's text response.
    """
    backoff = initial_backoff
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):
        try:
            chat = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            return chat.choices[0].message.content or ""

        except RateLimitError as exc:
            last_exc = exc
            err_msg = str(exc).lower()
            
            # Check if this is a daily limit (TPD/RPD) error
            is_daily_limit = "tokens per day" in err_msg or "tpd" in err_msg or "requests per day" in err_msg or "rpd" in err_msg
            
            if is_daily_limit and model != "llama-3.1-8b-instant":
                fallback_model = "llama-3.1-8b-instant"
                logger.warning(
                    "Daily Token/Request Limit (TPD/RPD) reached for model '%s'. "
                    "Automatically falling back to '%s' to continue pipeline without interruption.",
                    model, fallback_model
                )
                model = fallback_model
                continue

            parsed_wait = _parse_tpm_retry_after(exc)
            # Add a small buffer of 1.5s to ensure the rate limit window has fully cleared
            wait = (parsed_wait + 1.5) if parsed_wait is not None else backoff
            if attempt <= max_retries:
                logger.warning(
                    "Groq rate limit (attempt %d/%d) → waiting %.2fs…",
                    attempt, max_retries + 1, wait,
                )
                await asyncio.sleep(wait)
                if parsed_wait is None:
                    backoff *= BACKOFF_MULT
            else:
                logger.error("Groq rate limit — all %d attempts exhausted.", max_retries + 1)

        except APIStatusError as exc:
            last_exc = exc
            # 400 errors (model_decommissioned, invalid_request, etc.) are
            # permanent — retrying will never fix them. Fail fast with a
            # clear, actionable message.
            if exc.status_code == 400:
                code = ""
                try:
                    code = exc.response.json().get("error", {}).get("code", "")
                except Exception:
                    pass
                if code == "model_decommissioned":
                    raise RuntimeError(
                        f"Model '{model}' has been decommissioned by Groq.\n"
                        "  Fix: use one of these current models:\n"
                        "    --model llama-3.1-8b-instant      (fastest)\n"
                        "    --model llama-3.3-70b-versatile   (highest quality)\n"
                        "    --model gemma2-9b-it              (Google, very fast)\n"
                        "  Full list: https://console.groq.com/docs/models"
                    ) from exc
                raise RuntimeError(
                    f"Groq rejected the request (400): {exc}\n"
                    "  This is a permanent error — check your prompt or model name."
                ) from exc
            # 5xx and other transient errors — retry with backoff
            if attempt <= max_retries:
                logger.warning(
                    "Groq API error (attempt %d/%d): %s → waiting %.1fs…",
                    attempt, max_retries + 1, str(exc)[:80], backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= BACKOFF_MULT
            else:
                logger.error("Groq API error — all %d attempts exhausted.", max_retries + 1)

        except APIConnectionError as exc:
            last_exc = exc
            if attempt <= max_retries:
                logger.warning(
                    "Groq connection error (attempt %d/%d): %s → waiting %.1fs…",
                    attempt, max_retries + 1, str(exc)[:80], backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= BACKOFF_MULT
            else:
                logger.error("Groq connection error — all %d attempts exhausted.", max_retries + 1)

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.error("Unexpected error in Groq call: %s", exc)
            raise

    raise RuntimeError(
        f"All {max_retries + 1} Groq API attempts failed.\n"
        f"  Model: {model}\n"
        f"  Last error: {last_exc}\n"
        "  Tips:\n"
        "    - Check quota at https://console.groq.com/usage\n"
        "    - Try a lower --concurrency value\n"
        "    - Try --model llama3-8b-8192 (highest free-tier RPM)"
    ) from last_exc


# ---------------------------------------------------------------------------
# JD Decomposition
# ---------------------------------------------------------------------------

_JD_DECOMPOSE_PROMPT = """\
You are a senior technical recruiter. Analyse the following Job Description and \
extract a structured hiring rubric.

Return ONLY a valid JSON object with EXACTLY these keys (no extra text, no markdown):
{{
  "must_have_skills": ["skill1", "skill2", ...],
  "nice_to_have_skills": ["skill1", "skill2", ...],
  "minimum_experience_years": <number>,
  "key_responsibilities": ["responsibility1", "responsibility2", ...]
}}

Rules:
- must_have_skills: skills/technologies explicitly required or mandatory.
- nice_to_have_skills: skills mentioned as preferred, bonus, or advantageous.
- minimum_experience_years: minimum years of professional experience (0 if not stated).
- key_responsibilities: up to 6 concise responsibilities from the JD.
- Output ONLY the JSON object. No preamble, no explanation, no markdown.

JOB DESCRIPTION:
{jd_text}
"""


async def analyze_jd(
    jd_text: str,
    client:  AsyncGroq,
    model:   str = DEFAULT_MODEL,
) -> JDRubric:
    """
    Decompose a raw Job Description into a structured JDRubric using Groq.

    Parameters
    ----------
    jd_text : str
        Raw JD text extracted from the docx file.
    client : AsyncGroq
        Authenticated Groq async client.
    model : str
        Groq model identifier.

    Returns
    -------
    JDRubric
        Structured rubric with must_have_skills, nice_to_have_skills,
        minimum_experience_years, and key_responsibilities.
    """
    logger.info("Decomposing JD into structured rubric via Groq (%s)…", model)
    prompt = _JD_DECOMPOSE_PROMPT.format(jd_text=jd_text)
    raw = await _groq_call(client, model, prompt)

    try:
        data = _repair_and_parse_json(raw)
    except ValueError as exc:
        logger.error("JD decomposition — JSON repair failed: %s\nRaw:\n%s", exc, raw[:500])
        data = {}

    rubric = JDRubric(
        must_have_skills=data.get("must_have_skills", []),
        nice_to_have_skills=data.get("nice_to_have_skills", []),
        minimum_experience_years=float(data.get("minimum_experience_years", 0)),
        key_responsibilities=data.get("key_responsibilities", []),
        raw=data,
    )

    # Module 3: Knowledge Graph Skill Expansion
    sg = SkillGraph()
    rubric.expanded_must_have_skills = sg.expand_skills(rubric.must_have_skills)
    
    logger.info(
        "Rubric extracted — must-have: %d | nice-to-have: %d | min YoE: %.1f",
        len(rubric.must_have_skills),
        len(rubric.nice_to_have_skills),
        rubric.minimum_experience_years,
    )
    return rubric


# ---------------------------------------------------------------------------
# Per-candidate scoring
# ---------------------------------------------------------------------------

_SCORE_PROMPT = """\
You are a world-class Technical Recruiter. Do not penalize candidates for terminology differences. \
If a candidate lists a tool that is a known equivalent to a requirement (e.g., listing 'Pinecone' \
satisfies 'Vector Database' or 'Embeddings-based retrieval'), give them FULL credit for that skill. \
Stop being a literal keyword matcher and act as a competency judge.

HIRING RUBRIC:
{rubric_text}

CANDIDATE PROFILE:
{profile}

Scoring instructions:
- Give an integer or decimal score from 0 to 10.
- A score of 9-10 represents a perfect match in capability (e.g. they meet all must-have requirements conceptually, regardless of exact wording).
- 0 = completely unqualified for this role.
- If the candidate lacks a must-have skill but possesses a skill listed in EXPANDED MUST-HAVE SKILLS, consider this a match and give them FULL credit.
- Write ONE concise sentence (max 25 words) justifying the score.
- Respond with ONLY this JSON (no markdown, no extra text):
  {{"score": <number 0-10>, "reason": "<one sentence>", "graph_match": <boolean true/false>}}
  Set "graph_match" to true if you used an EXPANDED MUST-HAVE SKILL (or equivalent tool) to satisfy a must-have skill requirement that the candidate otherwise lacked.
"""

_SCORE_BATCH_PROMPT = """\
You are a world-class Technical Recruiter. Score the following {num_candidates} candidate profiles \
against the hiring rubric provided.

Do not penalize candidates for terminology differences. If a candidate lists a tool that is a known \
equivalent to a requirement (e.g., listing 'Pinecone' satisfies 'Vector Database' or 'Embeddings-based retrieval'), \
give them FULL credit for that skill. Stop being a literal keyword matcher and act as a competency judge.

HIRING RUBRIC:
{rubric_text}

CANDIDATES:
{candidates_profiles}

Scoring instructions:
- For each candidate, give an integer or decimal score from 0 to 10.
- A score of 9-10 represents a perfect match in capability conceptually, regardless of exact wording.
- 0 = completely unqualified for this role.
- If the candidate lacks a must-have skill but possesses a skill listed in EXPANDED MUST-HAVE SKILLS, consider this a match and give them FULL credit.
- Write ONE concise sentence (max 25 words) justifying the score.
- Set "graph_match" to true if you used an EXPANDED MUST-HAVE SKILL (or equivalent tool) to satisfy a must-have skill requirement that the candidate otherwise lacked.

Respond with ONLY a valid JSON list containing exactly {num_candidates} objects, in the same order as listed above, matching this schema (no markdown, no extra text):
[
  {{"candidate_id": "<candidate_id_1>", "score": <number 0-10>, "reason": "<one sentence>", "graph_match": <boolean true/false>}},
  ...
]
"""


def _compress_profile(profile: str) -> str:
    """
    Compress a rich profile to save tokens without compromising accuracy.
    It keeps the introductory fields, Skills, Certifications, and Platform Signals intact.
    In Career History, it keeps only the 3 most recent jobs, and truncates their descriptions to the first 60 characters.
    This preserves the critical core context (like exact responsibilities and project verbs) while keeping token count small.
    """
    career_match = re.search(r"(Career History:.*?)(Education:.*)", profile)
    if not career_match:
        return profile[:2000]
        
    career_part = career_match.group(1)
    rest_part = career_match.group(2)
    intro_part = profile[:career_match.start(1)]
    
    jobs = career_part.replace("Career History:", "").split("|")
    compressed_jobs = []
    
    # Keep only the 3 most recent jobs
    for job in jobs[:3]:
        job = job.strip()
        if not job:
            continue
        
        # Match duration e.g. "27 months (current)." or "55 months." and separate description
        desc_match = re.search(r"(.*?\b\d+\s*months(?:\s*\(\s*current\s*\))?\.)\s*(.*)", job, re.IGNORECASE)
        if desc_match:
            header = desc_match.group(1)
            desc = desc_match.group(2)
            if len(desc) > 60:
                desc = desc[:60].strip() + "..."
            compressed_jobs.append(f"{header} {desc}")
        else:
            if len(job) > 150:
                compressed_jobs.append(job[:150] + "...")
            else:
                compressed_jobs.append(job)
                
    compressed_career = "Career History: " + " | ".join(compressed_jobs) + " "
    return intro_part + compressed_career + rest_part


def compute_graph_match(profile_text: str, must_have_skills: list[str], sg: SkillGraph) -> bool:
    """
    Ensure the graph_match boolean is set to True if any skill in the candidate's profile
    maps to a 'Must-Have' requirement via the SkillGraph, even if the exact requirement word is missing.
    """
    profile_lower = profile_text.lower()
    
    for m in must_have_skills:
        m_clean = m.strip()
        m_lower = m_clean.lower()
        
        # Check if the exact requirement word/phrase is missing from the profile
        pattern_must = rf"\b{re.escape(m_lower)}\b"
        if not re.search(pattern_must, profile_lower):
            # Get equivalents (like "Pinecone", "Milvus", etc.)
            equivalents = sg.expand_skill(m_clean)
            for eq in equivalents:
                eq_lower = eq.strip().lower()
                pattern_eq = rf"\b{re.escape(eq_lower)}\b"
                if re.search(pattern_eq, profile_lower):
                    return True
    return False


async def _score_candidate(
    candidate:  RetrievalResult,
    rubric:     JDRubric,
    client:     AsyncGroq,
    model:      str,
    semaphore:  asyncio.Semaphore,
) -> RankedCandidate:
    """
    Score a single candidate against the JD rubric via Groq.

    The semaphore bounds concurrent API calls to stay within Groq's RPM limit.

    Parameters
    ----------
    candidate : RetrievalResult
        Candidate from Module 1 FAISS retrieval.
    rubric : JDRubric
        Structured JD rubric from analyze_jd().
    client : AsyncGroq
        Authenticated Groq async client.
    model : str
        Groq model identifier.
    semaphore : asyncio.Semaphore
        Rate-limit guard — limits simultaneous Groq calls.

    Returns
    -------
    RankedCandidate
        Candidate enriched with LLM score and reason.
    """
    compressed_profile = _compress_profile(candidate.rich_profile)
    prompt = _SCORE_PROMPT.format(
        rubric_text=rubric.to_prompt_text(),
        profile=compressed_profile,
    )

    async with semaphore:
        raw = await _groq_call(client, model, prompt)

    # Programmatic graph match check
    sg = SkillGraph()
    prog_graph_match = compute_graph_match(candidate.rich_profile, rubric.must_have_skills, sg)

    score  = 0.0
    reason = "Scoring failed — could not parse LLM response."
    graph_match = prog_graph_match
    try:
        data   = _repair_and_parse_json(raw)
        score  = float(data.get("score", 0))
        
        # Combine programmatic check with LLM's output
        llm_graph_match = bool(data.get("graph_match", False))
        graph_match = prog_graph_match or llm_graph_match
        
        if graph_match:
            score += 0.5
            
        score  = max(0.0, min(10.0, score))   # clamp to [0, 10]
        reason = str(data.get("reason", reason)).strip()
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not parse score for %s: %s | Raw: %.200s",
            candidate.candidate_id, exc, raw,
        )

    return RankedCandidate(
        final_rank=0,                       # assigned after full sort
        candidate_id=candidate.candidate_id,
        llm_score=score,
        semantic_score=candidate.similarity_score,
        reason=reason,
        rich_profile=candidate.rich_profile,
        graph_match=graph_match,
    )


# ---------------------------------------------------------------------------
# Reranking engine
# ---------------------------------------------------------------------------

async def _score_candidate_batch(
    candidates: list[RetrievalResult],
    rubric:     JDRubric,
    client:     AsyncGroq,
    model:      str,
    semaphore:  asyncio.Semaphore,
) -> list[RankedCandidate]:
    """
    Score a batch of candidates against the JD rubric via Groq.
    """
    profiles_text_list = []
    for idx, cand in enumerate(candidates, start=1):
        compressed = _compress_profile(cand.rich_profile)
        profiles_text_list.append(
            f"Candidate #{idx}:\n"
            f"Candidate ID: {cand.candidate_id}\n"
            f"{compressed}\n"
            f"---"
        )
    
    candidates_profiles = "\n\n".join(profiles_text_list)
    prompt = _SCORE_BATCH_PROMPT.format(
        num_candidates=len(candidates),
        rubric_text=rubric.to_prompt_text(),
        candidates_profiles=candidates_profiles,
    )
    
    async with semaphore:
        raw = await _groq_call(client, model, prompt)
        
    ranked_results = []
    try:
        data = _repair_and_parse_json(raw)
        if not isinstance(data, list):
            if isinstance(data, dict) and "candidates" in data:
                data = data["candidates"]
            elif isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        data = val
                        break
        
        scores_map = {}
        if isinstance(data, list):
            for item in data:
                cid = item.get("candidate_id")
                if cid:
                    scores_map[str(cid).strip()] = item
                    
        sg = SkillGraph()
        for cand in candidates:
            item = scores_map.get(cand.candidate_id)
            if item:
                score = float(item.get("score", 0.0))
                reason = str(item.get("reason", "Scoring succeeded.")).strip()
                llm_graph_match = bool(item.get("graph_match", False))
            else:
                score = 0.0
                reason = "LLM missed this candidate in batch response."
                llm_graph_match = False
                
            prog_graph_match = compute_graph_match(cand.rich_profile, rubric.must_have_skills, sg)
            graph_match = prog_graph_match or llm_graph_match
            if graph_match:
                score += 0.5
            score = max(0.0, min(10.0, score))
            
            ranked_results.append(RankedCandidate(
                final_rank=0,
                candidate_id=cand.candidate_id,
                llm_score=score,
                semantic_score=cand.similarity_score,
                reason=reason,
                rich_profile=cand.rich_profile,
                graph_match=graph_match,
            ))
            
    except Exception as exc:
        logger.warning("Batch scoring parse failed: %s. Falling back to individual scoring.", exc)
        for cand in candidates:
            res = await _score_candidate(cand, rubric, client, model, semaphore)
            ranked_results.append(res)
            
    return ranked_results


async def _score_candidate_batch_with_delay(
    candidates: list[RetrievalResult],
    rubric:     JDRubric,
    client:     AsyncGroq,
    model:      str,
    semaphore:  asyncio.Semaphore,
    delay:      float,
) -> list[RankedCandidate]:
    if delay > 0:
        await asyncio.sleep(delay)
    return await _score_candidate_batch(candidates, rubric, client, model, semaphore)


async def rerank_candidates(
    top_candidates: list[RetrievalResult],
    jd_rubric:      JDRubric,
    client:         AsyncGroq,
    model:          str = DEFAULT_MODEL,
    concurrency:    int = DEFAULT_CONCURRENCY,
) -> list[RankedCandidate]:
    """
    Rerank a list of pre-retrieved candidates using Groq LLM scoring.

    All candidate scoring calls are paced out with a delay to stay within Groq's RPM limits,
    while running concurrently bounded by a semaphore.

    Parameters
    ----------
    top_candidates : list[RetrievalResult]
        Candidates from Module 1 FAISS retrieval.
    jd_rubric : JDRubric
        Structured rubric from analyze_jd().
    client : AsyncGroq
        Authenticated Groq async client.
    model : str
        Groq model identifier.
    concurrency : int
        Maximum simultaneous Groq API calls.

    Returns
    -------
    list[RankedCandidate]
        Candidates sorted descending by LLM score (semantic score as tiebreaker).
    """
    semaphore = asyncio.Semaphore(concurrency)

    logger.info(
        "Reranking %d candidates via Groq '%s' in batches of 5 (concurrency=%d, paced by 1.5s)…",
        len(top_candidates), model, concurrency,
    )
    t0 = time.time()

    # Divide candidates into batches of 5
    batch_size = 5
    batches = [
        top_candidates[i:i + batch_size]
        for i in range(0, len(top_candidates), batch_size)
    ]

    tasks = [
        _score_candidate_batch_with_delay(batch, jd_rubric, client, model, semaphore, i * 1.5)
        for i, batch in enumerate(batches)
    ]

    batch_results = await async_tqdm.gather(
        *tasks,
        desc="Scoring candidate batches",
        unit=" batch",
    )

    results = [cand for batch in batch_results for cand in batch]

    # Primary sort: LLM score ↓, tiebreaker: semantic score ↓
    results.sort(key=lambda r: (r.llm_score, r.semantic_score), reverse=True)
    for rank, candidate in enumerate(results, start=1):
        candidate.final_rank = rank

    logger.info("Reranking complete in %.1fs.", time.time() - t0)
    return results


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def reranked_to_dataframe(results: list[RankedCandidate]) -> pd.DataFrame:
    """Convert reranked results to a pandas DataFrame."""
    return pd.DataFrame([
        {
            "rank":           r.final_rank,
            "candidate_id":   r.candidate_id,
            "llm_score":      round(r.llm_score, 2),
            "semantic_score": round(r.semantic_score, 6),
            "graph_match":    "Yes" if r.graph_match else "No",
            "reason":         r.reason,
        }
        for r in results
    ])


def print_reranked_table(results: list[RankedCandidate], n: int = 50) -> None:
    """Pretty-print the final reranked table to stdout."""
    col_w = 125
    print(f"\n{'=' * col_w}")
    print(f"  {'RANK':<6} {'CANDIDATE ID':<16} {'LLM':>5} {'SEM':>8} {'GRAPH MATCH':<11} REASON")
    print(f"{'-' * col_w}")
    for r in results[:n]:
        snippet = r.reason[:62] + "..." if len(r.reason) > 62 else r.reason
        g_match = "Yes" if r.graph_match else "No"
        try:
            print(
                f"  #{r.final_rank:<5} {r.candidate_id:<16} "
                f"{r.llm_score:>4.1f}  {r.semantic_score:>8.4f}  {g_match:<11} {snippet}"
            )
        except UnicodeEncodeError:
            clean_snippet = snippet.encode("ascii", "replace").decode("ascii")
            print(
                f"  #{r.final_rank:<5} {r.candidate_id:<16} "
                f"{r.llm_score:>4.1f}  {r.semantic_score:>8.4f}  {g_match:<11} {clean_snippet}"
            )
    print(f"{'=' * col_w}\n")


# ---------------------------------------------------------------------------
# Full pipeline (async)
# ---------------------------------------------------------------------------

async def run_pipeline(
    jd_path:       Path,
    top_k:         int  = 50,
    model:         str  = DEFAULT_MODEL,
    concurrency:   int  = DEFAULT_CONCURRENCY,
    index_path:    Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
    output_csv:    Optional[Path] = None,
) -> list[RankedCandidate]:
    """
    End-to-end async pipeline:
      extract JD → retrieve top-K → decompose JD → rerank → display.

    Parameters
    ----------
    jd_path : Path
        Path to job_description.docx.
    top_k : int
        Number of candidates to retrieve from FAISS.
    model : str
        Groq model identifier.
    concurrency : int
        Max simultaneous Groq API calls.
    index_path : Path
        Path to candidates_index.faiss.
    metadata_path : Path
        Path to metadata.json.
    output_csv : Path or None
        If set, save the final table to this CSV path.

    Returns
    -------
    list[RankedCandidate]
        Final reranked candidates.
    """
    # ── Step 1: Extract JD text ──────────────────────────────────────────────
    jd_text = extract_jd_text(jd_path)

    # ── Step 2: Retrieve top-K candidates from FAISS (Module 1) ─────────────
    logger.info("Retrieving top-%d candidates from FAISS index…", top_k)
    top_candidates = get_top_k_candidates(
        job_description_text=jd_text,
        k=top_k,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    logger.info("Retrieved %d candidates from FAISS.", len(top_candidates))

    # ── Step 3: Initialise Groq client ───────────────────────────────────────
    client = _make_groq_client()
    logger.info("Groq client ready — model: %s", model)

    # ── Step 4: Decompose JD into structured rubric ──────────────────────────
    rubric = await analyze_jd(jd_text, client, model)
    logger.info("\n%s\n", rubric.to_prompt_text())

    # ── Step 5: LLM Reranking ────────────────────────────────────────────────
    ranked = await rerank_candidates(
        top_candidates=top_candidates,
        jd_rubric=rubric,
        client=client,
        model=model,
        concurrency=concurrency,
    )

    # ── Step 6: Display + save ───────────────────────────────────────────────
    print_reranked_table(ranked)

    if output_csv:
        df = reranked_to_dataframe(ranked)
        df.to_csv(output_csv, index=False)
        logger.info("Results saved → %s", output_csv)

    return ranked


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate Ranking System – Module 2 (Groq LLM Reranker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "API Key Setup:\n"
            "  Get a FREE key at: https://console.groq.com/keys\n\n"
            "  Option A — .env file in project folder:\n"
            "    GROQ_API_KEY=gsk_...\n\n"
            "  Option B — PowerShell:\n"
            "    $env:GROQ_API_KEY = 'gsk_...'\n\n"
            "Current Groq models (2025):\n"
            "  llama-3.1-8b-instant     (default — fastest, free tier)\n"
            "  llama-3.3-70b-versatile  (best quality, free tier)\n"
            "  gemma2-9b-it             (Google Gemma 2, very fast)\n"
            "  mixtral-8x7b-32768       (32K context window)\n"
            "  Full list: https://console.groq.com/docs/models\n\n"
            "Examples:\n"
            "  python reranker.py\n"
            "  python reranker.py --model llama-3.3-70b-versatile --top-k 30\n"
            "  python reranker.py --concurrency 8 --output-csv final.csv\n"
        ),
    )
    parser.add_argument(
        "--jd", type=Path, default=JD_PATH,
        help="Path to job_description.docx (default: %(default)s)",
    )
    parser.add_argument(
        "--index", type=Path, default=INDEX_PATH,
        help="Path to FAISS index (default: %(default)s)",
    )
    parser.add_argument(
        "--metadata", type=Path, default=METADATA_PATH,
        help="Path to metadata.json (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Candidates to retrieve from FAISS (default: %(default)s)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Groq model name (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="Max simultaneous Groq API calls (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="If set, save final ranked table to this CSV path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_pipeline(
            jd_path=args.jd,
            top_k=args.top_k,
            model=args.model,
            concurrency=args.concurrency,
            index_path=args.index,
            metadata_path=args.metadata,
            output_csv=args.output_csv,
        )
    )
