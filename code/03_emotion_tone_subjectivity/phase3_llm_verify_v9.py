#!/usr/bin/env python3
"""
PHASE 3 LLM VERIFICATION SAMPLER — v9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Critical changes from v8:
  1. FRESH SAMPLER — globs *_Phase3_v2.xlsx directly (seed=137 now has real effect)
  2. RETRY LOGIC — 3-tier cascade: full → simplified → ultra-minimal (kills parse errors)
  3. TRUST FIX — redefined to prevent firing on conviction/loyalty language
  4. ACHIEVEMENT FIX — reduced from 3× to 2× few-shots; added 2nd ISSUE example
  5. PROMISE/ACHIEVEMENT TENSE CONTRAST — new few-shot showing identical structure,
     different tense → opposite labels (the key teaching moment for phi4-mini)
  6. DEFENSIVE FIX — stronger rebuttal-dominant example (no leader-praise at end)
  7. SUBJECTIVITY LABEL added to schema (fact_based | opinion_based)
"""

import glob, json, os, re, time, textwrap
import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SOURCE_PATTERN   = "*_Phase3_v2.xlsx"   # glob 24 source files in cwd
SAMPLE_N         = 5                    # rows per file × 24 files = 120 rows
RANDOM_SEED      = 137                  # NOW has real effect on which 5 rows are drawn
FALLBACK_FILE    = "verification_sample_Phase3_v5.xlsx"  # used if source files missing
CAPTION_COL      = "caption_en"
SAMPLE_OUT       = f"verification_sample_Phase3_v9_draw{RANDOM_SEED}.xlsx"
OUTPUT_FILE      = "verification_sample_Phase3_v9_llm.xlsx"

OLLAMA_BASE      = "http://localhost:11434"
MODEL            = "phi4-mini"
TEMPERATURE      = 0.05

# ─────────────────────────────────────────
# PASS / FAIL (recalibrated v8 thresholds — keep same)
# ─────────────────────────────────────────
PASS_CRITERIA = [
    ("issue framing",       "issue",       None, 0.70,  "≤70%"),
    ("achievement framing", "achievement", 0.10, None,  "≥10%"),
    ("promise framing",     "promise",     0.05, None,  "≥5%"),
    ("attack framing",      "attack",      0.15, None,  "≥15%"),
    ("pride emotion",       "pride",       None, 0.75,  "≤75%"),
    ("anger emotion",       "anger",       0.05, None,  "≥5%"),
    ("hope emotion",        "hope",        0.05, None,  "≥5%"),
    ("inspirational tone",  "inspirational",None,0.70,  "≤70%"),
    ("aggressive tone",     "aggressive",  0.10, None,  "≥10%"),
    ("emotion low-conf",    "low_conf_e",  None, 0.50,  "≤50%"),
    ("tone low-conf",       "low_conf_t",  None, 0.40,  "≤40%"),
]

# ─────────────────────────────────────────
# SPOT CHECKS (10 — same as v8)
# ─────────────────────────────────────────
SPOT_CHECKS = [
    ("framing", "Had the Chintalapudi lift project been completed,",          "attack"),
    ("framing", "20 lakh jobs for youth in five years, Rs.3000 per month",   "promise"),
    ("framing", "Telugu Desam - Janasena stands by the youth. We pr",        "promise"),
    ("tone",    "Neither the individual, nor the system, nor the go",        "aggressive"),
    ("framing", "Gudiwada town is looking new with completed Tidco",         "achievement"),
    ("framing", "CM YS Jagan has literally spent Rs.1,61,236",              "achievement"),
    ("framing", "As soon as TDP comes to power, we will allocate funds",     "promise"),
    ("framing", "Visakha court sentenced YCP MLC",                          "issue"),
    ("framing", "The joy that the days of liberation from a Rakshasapida",  "issue"),
    ("framing", "CM Shri YS Jagan has given exgratia of 30 lakhs",         "achievement"),
]

# ─────────────────────────────────────────
# SYSTEM PROMPT — v9 (~380 tokens)
# Changes vs v8: trust redefined; subjectivity_label added to schema
# ─────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a political social-media classifier for Andhra Pradesh, India.
Return ONLY valid JSON — no markdown, no prose.

SCHEMA:
{
  "emotion":             "pride|anger|hope|fear|trust",
  "emotion_confidence":  0.0–1.0,
  "tone":                "inspirational|aggressive|persuasive|defensive|victim_narrative",
  "tone_confidence":     0.0–1.0,
  "framing":             "achievement|attack|promise|victim|issue",
  "framing_confidence":  0.0–1.0,
  "subjectivity":        0.0–1.0,
  "subjectivity_label":  "fact_based|opinion_based"
}

EMOTION:
• pride   — celebrating own party, leaders, governance, cultural identity
• anger   — condemning crimes, atrocities, opponent failures, corruption, violence
• hope    — aspiring for better future, faith in promises or electoral change
• fear    — danger, threats, violence, accidents, persecution of own side
• trust   — explicit confidence in a NAMED leader's competence or verified record;
            NOT for general slogans, loyalty pledges, or rhetorical convictions

FRAMING (primary purpose):
• achievement — COMPLETED governance action WITH MEASURABLE EVIDENCE.
  Must contain: specific Rs. amount, beneficiary count, or named project DELIVERED.
  TENSE RULE: past tense + specific number/name = achievement.
  Meetings, events, rallies, padayatras, tributes, launches, judicial rulings → issue.
• promise  — EXPLICIT FUTURE COMMITMENT: future tense + "will", "promise to",
  "guarantee", "as soon as X comes to power". Slogans without a pledge → issue.
• attack   — damages or dismisses a NAMED opponent; sarcasm, failure accusations,
  violence BY opponent are all attack framing.
• victim   — own party/leader portrayed as innocent victim of persecution.
• issue    — DEFAULT for everything else: events, mobilisation, tributes,
  historical legacy, thin slogans, judicial outcomes, policy debates.

TONE (apply in PRIORITY ORDER — stop at first match):
1. aggressive      — primary goal is to attack or mock a named opponent
2. defensive       — primary goal is to rebut a specific false accusation using
                     counter-evidence or counter-questions
3. victim_narrative — own side portrayed as suffering persecution; not fighting back
4. persuasive      — uses logic, data, or moral argument to convince
5. inspirational   — rallies supporters with pride or hope

SUBJECTIVITY: 0.0 = pure fact/statistics, 1.0 = pure opinion/rhetoric.
subjectivity_label: "fact_based" if ≤0.40, else "opinion_based".
""").strip()

# ─────────────────────────────────────────
# FEW-SHOT EXAMPLES — v9
# Key changes vs v8:
#   - achievement 3× → 2× (removed Gudiwada)
#   - Added TENSE CONTRAST promise example (same structure as achievement, future tense)
#   - Coalition rally moved from spot-check to few-shot as 2nd issue example
#   - Defensive example uses counter-question structure (no leader praise at end)
# Balance: achievement 2×, promise 3×, attack 3×, issue 2×, victim 1×, defensive 1×
# ─────────────────────────────────────────
FEW_SHOT = [
    # ── ACHIEVEMENT ×2: past tense + specific Rs. + specific count ────────
    # 1. YSRCP delivery — Rs amount + driver count
    (
        "We have deposited Rs.276 crores in the accounts of 2,75,931 brothers and sisters "
        "of auto and taxi drivers for the fifth year in a row. Our government has given "
        "a total of Rs.1,300 crore through this scheme till now. #YSRVahaMithra",
        '{"emotion":"pride","emotion_confidence":0.90,"tone":"inspirational","tone_confidence":0.87,'
        '"framing":"achievement","framing_confidence":0.95,"subjectivity":0.62,'
        '"subjectivity_label":"opinion_based"}'
    ),
    # 2. YSRCP delivery — Rs amount + student count (clear past tense "have deposited")
    (
        "We have deposited Rs.11,317 crores in the accounts of 24,53,389 students mothers "
        "under the Jaganna Vidyadevena scheme in these four years. #JagannaVidyadevena",
        '{"emotion":"pride","emotion_confidence":0.92,"tone":"inspirational","tone_confidence":0.88,'
        '"framing":"achievement","framing_confidence":0.95,"subjectivity":0.58,'
        '"subjectivity_label":"opinion_based"}'
    ),

    # ── PROMISE ×3: future tense + explicit pledge ──────────────────────
    # 3. TENSE CONTRAST — same structure as achievement but future tense → promise NOT achievement
    #    This is the key teaching example: numbers + party content but FUTURE = promise
    (
        "When TDP forms government, we will deposit Rs.3,000 every month into the accounts "
        "of 20 lakh unemployed youth as a guaranteed stipend until they find work. "
        "#BabuSuper6 #YouthGuarantee",
        '{"emotion":"hope","emotion_confidence":0.88,"tone":"inspirational","tone_confidence":0.84,'
        '"framing":"promise","framing_confidence":0.93,"subjectivity":0.72,'
        '"subjectivity_label":"opinion_based"}'
    ),
    # 4. Conditional future "As soon as X comes to power, we will"
    (
        "As soon as TDP comes to power, we will allocate funds proportionately through "
        "the corporation and support the fishermen. #TDPForFisherman #YuvaGalamPadayatra",
        '{"emotion":"hope","emotion_confidence":0.83,"tone":"inspirational","tone_confidence":0.81,'
        '"framing":"promise","framing_confidence":0.92,"subjectivity":0.70,'
        '"subjectivity_label":"opinion_based"}'
    ),
    # 5. "We promise to provide" explicit pledge
    (
        "Telugu Desam - Janasena stands by the youth. We promise to provide employment "
        "and secure your future. #tdpjsptogether",
        '{"emotion":"hope","emotion_confidence":0.86,"tone":"inspirational","tone_confidence":0.83,'
        '"framing":"promise","framing_confidence":0.91,"subjectivity":0.78,'
        '"subjectivity_label":"opinion_based"}'
    ),

    # ── ATTACK ×3: direct naming, sarcasm, violence-by-opponent ──────────
    # 6. Counterfactual implying opponent failure
    (
        "Had the Chintalapudi lift project been completed, 36 mandals would have received "
        "water. #JaganFailsIrrigation",
        '{"emotion":"anger","emotion_confidence":0.80,"tone":"aggressive","tone_confidence":0.83,'
        '"framing":"attack","framing_confidence":0.89,"subjectivity":0.78,'
        '"subjectivity_label":"opinion_based"}'
    ),
    # 7. Direct naming + Dalit atrocity by opponent
    (
        "Janjala Krishnaiah, a Dalit, was beaten to death with sticks by the Uttamareddy "
        "family in Pulivendula. The SC/ST Atrocities Act was not used for Dalit protection. "
        "#DalitDhrohiJagan #DalitLivesDontMatterInAP",
        '{"emotion":"anger","emotion_confidence":0.92,"tone":"aggressive","tone_confidence":0.90,'
        '"framing":"attack","framing_confidence":0.94,"subjectivity":0.85,'
        '"subjectivity_label":"opinion_based"}'
    ),
    # 8. Sarcastic mockery — attack even when indirect, no explicit "Jagan is bad"
    (
        "The poor man who got one chance, that foot. That foot woke up a terrible drought "
        "of 122 years. #WhyAPHatesJagan #IronLegJagan #JaganFailedFarmers",
        '{"emotion":"anger","emotion_confidence":0.83,"tone":"aggressive","tone_confidence":0.86,'
        '"framing":"attack","framing_confidence":0.90,"subjectivity":0.82,'
        '"subjectivity_label":"opinion_based"}'
    ),

    # ── ISSUE ×2: positive party content WITHOUT delivery numbers ─────────
    # 9. Coalition rally — positive, pride-filled, zero delivery evidence → issue NOT achievement
    (
        "Yellow Dal, Jana Sena... the great confluence of both forces. The final hours of "
        "the peoples rule. #TDPJSPTogether #YuvaGalamJoshBegins #AndhraPradesh",
        '{"emotion":"pride","emotion_confidence":0.84,"tone":"inspirational","tone_confidence":0.86,'
        '"framing":"issue","framing_confidence":0.82,"subjectivity":0.65,'
        '"subjectivity_label":"opinion_based"}'
    ),
    # 10. Padayatra daily diary — campaign event report, no delivery → issue NOT achievement
    (
        "I started from Chollangipet on the 214th day of Yuvagalam Padayatra. Christian "
        "leaders offered blessings at Kakinada. Fishermen, Dwakra women, disabled people "
        "submitted petitions. #YuvaGalamPadayatra",
        '{"emotion":"pride","emotion_confidence":0.78,"tone":"persuasive","tone_confidence":0.75,'
        '"framing":"issue","framing_confidence":0.85,"subjectivity":0.52,'
        '"subjectivity_label":"fact_based"}'
    ),

    # ── VICTIM ×1: own side persecuted ───────────────────────────────────
    # 11. Fear for leader's safety — victim_narrative tone, victim framing
    (
        "After seeing him, a part seems to have been left behind. I fear for Chandrababu's "
        "safety. #IAmWithBabu #WeWillGiveItBack #PsychoJagan #CBNWillBeBackWithABang",
        '{"emotion":"fear","emotion_confidence":0.85,"tone":"victim_narrative","tone_confidence":0.82,'
        '"framing":"victim","framing_confidence":0.88,"subjectivity":0.88,'
        '"subjectivity_label":"opinion_based"}'
    ),

    # ── DEFENSIVE ×1: counter-question rebuttal, no leader-praise at end ─
    # 12. Rebutting specific false accusation using counter-questions (no Rs./counts → not achievement)
    (
        "The opposition claims our government tampered with Polavaram specifications. This "
        "is a politically motivated lie. All technical records and engineer certifications "
        "are intact. We challenge Chandrababu: show one specific modification with proof. "
        "These baseless accusations will not distract us from delivering the project.",
        '{"emotion":"trust","emotion_confidence":0.78,"tone":"defensive","tone_confidence":0.85,'
        '"framing":"issue","framing_confidence":0.74,"subjectivity":0.80,'
        '"subjectivity_label":"opinion_based"}'
    ),
]

# ── SIMPLIFIED PROMPT (Attempt 2 on parse error) ──────────────────────────
SIMPLIFIED_PROMPT = textwrap.dedent("""
You are a political post classifier. Return ONLY valid JSON, nothing else.
{"emotion":"pride|anger|hope|fear|trust","emotion_confidence":0.0-1.0,
 "tone":"inspirational|aggressive|persuasive|defensive|victim_narrative","tone_confidence":0.0-1.0,
 "framing":"achievement|attack|promise|victim|issue","framing_confidence":0.0-1.0,
 "subjectivity":0.0-1.0,"subjectivity_label":"fact_based|opinion_based"}
Rules: achievement=past delivery+Rs./count. promise=future pledge+"will". attack=names opponent.
victim=own side persecuted. issue=default. Classify this post:
""").strip()

# ── ULTRA-MINIMAL PROMPT (Attempt 3) ──────────────────────────────────────
ULTRA_PROMPT = textwrap.dedent("""
Output ONLY this JSON for the political post, filling in the values:
{"emotion":"FILL_pride_anger_hope_fear_trust","tone":"FILL_inspirational_aggressive_persuasive_defensive_victim_narrative","framing":"FILL_achievement_attack_promise_victim_issue","subjectivity":0.5,"subjectivity_label":"FILL_fact_based_opinion_based","emotion_confidence":0.7,"tone_confidence":0.7,"framing_confidence":0.7}
""").strip()

# ─────────────────────────────────────────
# VALID LABEL SETS
# ─────────────────────────────────────────
VALID_EMOTION   = {"pride", "anger", "hope", "fear", "trust"}
VALID_TONE      = {"inspirational","aggressive","persuasive","defensive","victim_narrative"}
VALID_FRAMING   = {"achievement","attack","promise","victim","issue"}
VALID_SUBJ_LBL  = {"fact_based","opinion_based"}
FALLBACK_CONF   = 0.50

def normalize_output(out: dict) -> dict:
    if out.get("framing") == "victim_narrative":
        out["framing"] = "victim"
    if out.get("framing") not in VALID_FRAMING:
        out["framing"] = "issue"
    if out.get("tone") not in VALID_TONE:
        out["tone"] = "inspirational"
    if out.get("emotion") not in VALID_EMOTION:
        out["emotion"] = "pride"
    # derive subjectivity_label if missing or wrong
    subj = float(out.get("subjectivity", 0.5))
    if out.get("subjectivity_label") not in VALID_SUBJ_LBL:
        out["subjectivity_label"] = "fact_based" if subj <= 0.40 else "opinion_based"
    for k in ["emotion_confidence","tone_confidence","framing_confidence","subjectivity"]:
        out[k] = max(0.0, min(1.0, float(out.get(k, FALLBACK_CONF))))
    return out

def safe_parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(raw2)
        except json.JSONDecodeError:
            return None

# ─────────────────────────────────────────
# CLASSIFY WITH 3-TIER RETRY
# ─────────────────────────────────────────
def build_full_messages(caption: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for cap, resp in FEW_SHOT:
        msgs.append({"role": "user",      "content": cap})
        msgs.append({"role": "assistant", "content": resp})
    msgs.append({"role": "user", "content": caption})
    return msgs

def call_ollama(messages: list[dict], temp: float = TEMPERATURE) -> str | None:
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": MODEL, "messages": messages,
                  "options": {"temperature": temp, "num_predict": 200},
                  "stream": False},
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception:
        return None

def classify_with_retry(caption: str) -> tuple[dict | None, int]:
    """
    Returns (result_dict, attempt_number_used).
    attempt 1: full few-shot  (temp=0.05)
    attempt 2: simplified 2-example prompt (temp=0.0)
    attempt 3: ultra-minimal, no examples (temp=0.0)
    """
    # Attempt 1 — full prompt
    raw = call_ollama(build_full_messages(caption), temp=TEMPERATURE)
    if raw:
        out = safe_parse_json(raw)
        if out and "framing" in out:
            return normalize_output(out), 1

    # Attempt 2 — simplified prompt
    simp_msgs = [
        {"role": "system", "content": SIMPLIFIED_PROMPT},
        # 2 grounding examples only
        {"role": "user",      "content": "We have deposited Rs.276 crores to 2,75,931 drivers. #YSRVahaMithra"},
        {"role": "assistant", "content": '{"emotion":"pride","emotion_confidence":0.88,"tone":"inspirational","tone_confidence":0.85,"framing":"achievement","framing_confidence":0.93,"subjectivity":0.60,"subjectivity_label":"opinion_based"}'},
        {"role": "user",      "content": "Had the Chintalapudi lift project been completed, 36 mandals would have received water. #JaganFailsIrrigation"},
        {"role": "assistant", "content": '{"emotion":"anger","emotion_confidence":0.80,"tone":"aggressive","tone_confidence":0.82,"framing":"attack","framing_confidence":0.88,"subjectivity":0.76,"subjectivity_label":"opinion_based"}'},
        {"role": "user",      "content": caption},
    ]
    raw2 = call_ollama(simp_msgs, temp=0.0)
    if raw2:
        out2 = safe_parse_json(raw2)
        if out2 and "framing" in out2:
            return normalize_output(out2), 2

    # Attempt 3 — ultra-minimal (no examples, fill-in-blank format)
    ultra_msgs = [
        {"role": "system", "content": ULTRA_PROMPT},
        {"role": "user",   "content": caption},
    ]
    raw3 = call_ollama(ultra_msgs, temp=0.0)
    if raw3:
        out3 = safe_parse_json(raw3)
        if out3:
            return normalize_output(out3), 3

    return None, -1  # all attempts failed

# ─────────────────────────────────────────
# CLASSIFIABILITY
# ─────────────────────────────────────────
STOP_WORDS = {
    "the","a","an","is","it","in","on","at","to","and","or","of","for",
    "with","that","this","was","are","be","by","from","as","we","i","he",
    "she","they","our","you","not","but","if","so","up","has","have",
    "his","her","its","all","my","your","their","will","been","had",
    "can","more","also","who","what","how","when","were","did","do",
}

def is_classifiable(text: str) -> bool:
    words = [w for w in re.sub(r"[^a-zA-Z\s]", " ", str(text)).split()
             if len(w) > 2 and w.lower() not in STOP_WORDS]
    return len(words) >= 4

# ─────────────────────────────────────────
# FRESH SAMPLER
# ─────────────────────────────────────────
def load_fresh_sample() -> pd.DataFrame:
    """
    Globs *_Phase3_v2.xlsx in cwd. Draws SAMPLE_N classifiable rows
    per file (stratified by file). Saves the raw sample before classification.
    Falls back to FALLBACK_FILE if no source files found.
    """
    source_files = sorted(glob.glob(SOURCE_PATTERN))
    if not source_files:
        print(f"  [WARN] No *_Phase3_v2.xlsx files found. Falling back to {FALLBACK_FILE}")
        df = pd.read_excel(FALLBACK_FILE)
        df["source_file"] = FALLBACK_FILE
        return df

    print(f"  [SAMPLER] Found {len(source_files)} source files. Drawing {SAMPLE_N}/file...")
    frames = []
    for fpath in tqdm(source_files, desc="  Sampling"):
        parts = os.path.basename(fpath).replace("_Phase3_v2.xlsx", "").split("_", 2)
        entity   = parts[0] if len(parts) > 0 else "unknown"
        platform = parts[1] if len(parts) > 1 else "unknown"
        period   = parts[2] if len(parts) > 2 else "unknown"
        try:
            tmp = pd.read_excel(fpath)
            tmp["entity"]      = entity
            tmp["platform"]    = platform
            tmp["period"]      = period
            tmp["source_file"] = os.path.basename(fpath)
            # ensure caption col exists
            if CAPTION_COL not in tmp.columns:
                alt = [c for c in tmp.columns if "caption" in c.lower() and "en" in c.lower()]
                if alt:
                    tmp[CAPTION_COL] = tmp[alt[0]]
            # recompute classifiable if not present
            if "classifiable" not in tmp.columns:
                tmp["classifiable"] = tmp[CAPTION_COL].apply(is_classifiable)
            classifiable = tmp[tmp["classifiable"] == True]
            n = min(SAMPLE_N, len(classifiable))
            drawn = classifiable.sample(n=n, random_state=RANDOM_SEED)
            frames.append(drawn)
        except Exception as e:
            print(f"  [WARN] Could not load {fpath}: {e}")

    if not frames:
        print(f"  [WARN] All source files failed. Falling back to {FALLBACK_FILE}")
        df = pd.read_excel(FALLBACK_FILE)
        df["source_file"] = FALLBACK_FILE
        return df

    df = pd.concat(frames, ignore_index=True)
    df.to_excel(SAMPLE_OUT, index=False)
    print(f"  [SAVED] Fresh sample: {SAMPLE_OUT} ({len(df)} rows from {len(frames)} files)")
    return df

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 LLM VERIFICATION SAMPLER — v9")
    print(f"Model : {MODEL}  |  Temp: {TEMPERATURE}  |  Seed: {RANDOM_SEED}")
    print("=" * 60)

    # ── Ollama check ──────────────────────────────────────────
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        found = [m["name"] for m in resp.json().get("models", []) if "phi4-mini" in m["name"]]
        if not found:
            raise RuntimeError("phi4-mini not found — run: ollama pull phi4-mini")
        print(f"[OK] Ollama running. Model: {found[0]}\n")
    except requests.exceptions.ConnectionError:
        raise SystemExit("[ERROR] Ollama not running — start with: ollama serve")

    # ── Load (fresh or fallback) ──────────────────────────────
    print("[SAMPLE] Drawing fresh stratified sample...")
    df = load_fresh_sample()
    df["classifiable"] = df[CAPTION_COL].apply(is_classifiable)
    print(f"\n[LOAD] Rows: {len(df)}  |  Classifiable: {df['classifiable'].sum()}\n")

    # ── Classify ──────────────────────────────────────────────
    print(f"[CLASSIFY] Processing {len(df)} posts with {MODEL} (3-tier retry)...")
    print(f"           (~15–20 min on RTX 1650)\n")

    results = []
    parse_errors  = 0
    retry2_count  = 0
    retry3_count  = 0
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Classifying"):
        cap = str(row[CAPTION_COL])
        if not is_classifiable(cap):
            results.append({
                "llm_emotion": "INSUFFICIENTTEXT", "llm_emotion_confidence": 0,
                "llm_tone":    "INSUFFICIENTTEXT", "llm_tone_confidence":    0,
                "llm_framing": "INSUFFICIENTTEXT", "llm_framing_confidence": 0,
                "llm_subjectivity": 0, "llm_subjectivity_label": "INSUFFICIENTTEXT",
                "llm_emotion_low_conf": False, "llm_tone_low_conf": False,
                "llm_framing_display": "INSUFFICIENT TEXT", "llm_retry_tier": 0,
            })
            continue

        out, tier = classify_with_retry(cap)

        if out is None:
            parse_errors += 1
            out = {
                "emotion": "pride", "emotion_confidence": FALLBACK_CONF,
                "tone":    "inspirational", "tone_confidence": FALLBACK_CONF,
                "framing": "issue", "framing_confidence": FALLBACK_CONF,
                "subjectivity": FALLBACK_CONF, "subjectivity_label": "opinion_based",
            }
            tier = -1
        elif tier == 2:
            retry2_count += 1
        elif tier == 3:
            retry3_count += 1

        results.append({
            "llm_emotion":             out["emotion"],
            "llm_emotion_confidence":  out["emotion_confidence"],
            "llm_tone":                out["tone"],
            "llm_tone_confidence":     out["tone_confidence"],
            "llm_framing":             out["framing"],
            "llm_framing_confidence":  out["framing_confidence"],
            "llm_subjectivity":        out["subjectivity"],
            "llm_subjectivity_label":  out.get("subjectivity_label", "opinion_based"),
            "llm_emotion_low_conf":    out["emotion_confidence"] < 0.60,
            "llm_tone_low_conf":       out["tone_confidence"] < 0.60,
            "llm_framing_display":     out["framing"],
            "llm_retry_tier":          tier,
        })

    elapsed = time.time() - t0
    print(f"\n  Done. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Parse errors (all 3 tiers failed): {parse_errors}")
    print(f"  Tier-2 retries: {retry2_count}  |  Tier-3 retries: {retry3_count}")

    res_df = pd.DataFrame(results)
    for k in res_df.columns:
        df[k] = res_df[k].values
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n[SAVED] {OUTPUT_FILE}")

    # ── Distribution ──────────────────────────────────────────
    cl = df[df["classifiable"] == True]
    total_cl = len(cl)
    BAR = "█"
    def bar(pct, w=40): return BAR * int(pct / 100 * w)

    print("\n" + "=" * 60)
    print("SAMPLE DISTRIBUTION SUMMARY")
    print("=" * 60)
    for dim, col in [("EMOTION_LABEL","llm_emotion"),("TONE_LABEL","llm_tone"),("FRAMING_LABEL","llm_framing")]:
        print(f"\n  {dim}:")
        for label, cnt in cl[col].value_counts().items():
            pct = cnt / total_cl * 100
            print(f"    {label:32s} {cnt:3d} ({pct:5.1f}%)  {bar(pct)}")

    # Subjectivity
    print(f"\n  SUBJECTIVITY_LABEL:")
    for label, cnt in cl["llm_subjectivity_label"].value_counts().items():
        pct = cnt / total_cl * 100
        print(f"    {label:32s} {cnt:3d} ({pct:5.1f}%)  {bar(pct)}")

    lce = cl["llm_emotion_low_conf"].sum()
    lct = cl["llm_tone_low_conf"].sum()
    print(f"\n  emotion_low_confidence : {lce} ({lce/total_cl*100:.1f}%)")
    print(f"  tone_low_confidence    : {lct} ({lct/total_cl*100:.1f}%)")

    # ── Pass/Fail ─────────────────────────────────────────────
    fr = cl["llm_framing"].value_counts(normalize=True) * 100
    em = cl["llm_emotion"].value_counts(normalize=True) * 100
    tn = cl["llm_tone"].value_counts(normalize=True) * 100
    print("\n" + "=" * 60)
    print("v9 LLM PASS/FAIL CRITERIA")
    print("=" * 60)
    passes = 0
    for name, label, lo, hi, target in PASS_CRITERIA:
        if label == "low_conf_e":   pct = lce / total_cl * 100
        elif label == "low_conf_t": pct = lct / total_cl * 100
        elif "emotion" in name:     pct = em.get(label, 0)
        elif "tone" in name:        pct = tn.get(label, 0)
        else:                       pct = fr.get(label, 0)
        ok = (lo is None or pct >= lo * 100) and (hi is None or pct <= hi * 100)
        flag = "✅ PASS" if ok else "❌ FAIL"
        if ok: passes += 1
        print(f"  {flag}  {name:25s}: {pct:5.1f}%  (target {target})")
    rate = passes / len(PASS_CRITERIA) * 100
    print(f"\n  Result: {passes} PASS / {len(PASS_CRITERIA)-passes} FAIL  ({rate:.0f}%)")
    gate = "✅ ACCURACY SUFFICIENT — proceed to phase3b" if passes >= 9 else "⚠️  REVIEW REQUIRED"
    print(f"  → {gate}")

    # ── Spot checks ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TARGETED ROW SPOT-CHECKS")
    print("=" * 60)
    sc_pass = 0
    for dim, prefix, expected in SPOT_CHECKS:
        col   = "llm_framing" if dim == "framing" else "llm_tone"
        match = df[df[CAPTION_COL].str.startswith(prefix, na=False)]
        if match.empty:
            print(f"  ⚠️  [{dim:7s}] '{prefix[:52]}...'  — ROW NOT IN FRESH SAMPLE")
            continue
        got  = str(match.iloc[0][col]).strip().lower()
        ok   = got == expected.lower()
        if ok: sc_pass += 1
        conf = match.iloc[0].get(f"llm_{dim}_confidence", 0)
        tier = match.iloc[0].get("llm_retry_tier", 1)
        flag = "✅" if ok else "❌"
        tier_label = f"  [T{tier}]" if tier > 1 else ""
        print(f"  {flag} [{dim:7s}] '{prefix[:52]}...'")
        print(f"       Expected: {expected:20s}|  Got: {got}  (conf: {conf:.2f}){tier_label}")
    print(f"\n  Spot-check: {sc_pass} correct / {len(SPOT_CHECKS)} total")

    # ── Retry summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RETRY TIER BREAKDOWN")
    print("=" * 60)
    tier_counts = df["llm_retry_tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        label = {1:"Tier 1 (full prompt)", 2:"Tier 2 (simplified)", 3:"Tier 3 (ultra-minimal)", -1:"Failed (all tiers)"}.get(tier, str(tier))
        print(f"  {label:35s}: {cnt:3d} rows ({cnt/len(df)*100:.1f}%)")
    print(f"\n  Target: Tier-1 ≥95%, Failed = 0")

    # ── Source diversity ──────────────────────────────────────
    if "source_file" in df.columns:
        print("\n" + "=" * 60)
        print("SAMPLE COMPOSITION")
        print("=" * 60)
        if "entity" in df.columns:
            print("\n  By entity:")
            for ent, cnt in df["entity"].value_counts().items():
                print(f"    {ent:20s}: {cnt}")
        if "platform" in df.columns:
            print("\n  By platform:")
            for plt, cnt in df["platform"].value_counts().items():
                print(f"    {plt:20s}: {cnt}")

    print(f"\n[DONE] Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"       Sample  : {SAMPLE_OUT}")
    print(f"       Output  : {OUTPUT_FILE}")
