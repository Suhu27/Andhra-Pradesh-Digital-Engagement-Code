#!/usr/bin/env python3
"""
phase3_hybrid_classifier_v2.py
=======================================================
Phase 3 Hybrid Classifier v2 — All fixes applied
See METHODOLOGY section below and the output .txt for full documentation

TAXONOMY CHANGES vs v1:
  EMOTION: Added 'grief' as 6th label (condolence/tribute posts)
  TONE:    5 labels unchanged; hypotheses revised; cross-dim overrides added
  FRAMING: 5 labels unchanged; promise + achievement rules extended
  SUBJECTIVITY: Multi-signal (TextBlob + Phase1 + evaluative keywords)
=======================================================
"""

import os, re, json, time, glob, random
import pandas as pd
import torch
from transformers import pipeline
from textblob import TextBlob
from tqdm import tqdm
from collections import Counter
from datetime import datetime

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATTERN   = os.path.join(SCRIPT_DIR, "*_FullAnalysis.xlsx")
OUTPUT_FILE      = os.path.join(SCRIPT_DIR, "phase3_hybrid_v2_600sample.xlsx")
METHODOLOGY_FILE = os.path.join(SCRIPT_DIR, "phase3_hybrid_v2_METHODOLOGY.txt")

SAMPLE_N      = 25
RANDOM_SEED   = 42
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
NLI_MODEL     = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
BATCH_SIZE    = 4
MAX_LENGTH    = 512
MIN_WORDS     = 5
MIN_CW        = 4

STOP_WORDS = {
    'the','and','for','that','this','with','are','was','were','has','have','had',
    'not','but','from','they','their','our','his','her','its','who','what','how',
    'why','when','where','will','can','could','would','should','may','might',
    'also','just','very','too','so','at','in','on','to','of','a','an','is',
    'it','we','you','i','he','she'
}

# ─────────────────────────────────────────────────────────────
# TONE NLI HYPOTHESES v2
# Changes: inspirational made more restrictive; aggressive broadened
# ─────────────────────────────────────────────────────────────
TONE_LABELS = ["persuasive","aggressive","defensive","inspirational","victim narrative"]

TONE_HYPOTHESES = {
    "persuasive": (
        "This text is primarily making a logical case or presenting reasoned arguments "
        "to convince voters or supporters to back a candidate, party, or policy. "
        "The dominant mode is argument and evidence, not attack."
    ),
    "aggressive": (
        "This text is primarily attacking, condemning, or exposing a political opponent, "
        "ruling government, or opposing party — using accusatory, condemnatory, or "
        "humiliating language, citing their crimes, failures, corruption, or "
        "misgovernance, with the intent of damaging their credibility."
    ),
    "defensive": (
        "This text is primarily responding to accusations, allegations, or attacks "
        "that have been directed against the writer's own party or leader, aiming to "
        "rebut or refute those charges."
    ),
    "inspirational": (
        "This text is primarily rallying supporters with an uplifting, forward-looking "
        "message — it motivates through a positive vision, shared values, or a call to "
        "action centered on hope and aspiration. The text does NOT primarily describe "
        "government failures, crimes, atrocities, or political corruption. There is no "
        "dominant condemnation of an opponent; the register is overwhelmingly positive."
    ),
    "victim narrative": (
        "This text is primarily portraying the writer's own party, leader, or community "
        "as an innocent victim of unjust political persecution, targeted harassment, "
        "deliberate suppression, or violent attack by those in power or opponents."
    ),
}

# ─────────────────────────────────────────────────────────────
# GRIEF PRE-FILTER (new in v2)
# ─────────────────────────────────────────────────────────────
CONDOLENCE_PATTERNS = [
    r'\bpassing of\b', r'\bdemise of\b', r'\bpassed away\b', r'\brest in peace\b',
    r'\bdeeply saddened\b', r'\bsaddened to (learn|hear|know)\b',
    r'\bpained to (learn|hear|know)\b', r'\bcondolences?\b', r'\bheavy heart\b',
    r'\btragic (death|loss|passing|accident|demise)\b', r'\bmourn(ing|ed|s)?\b',
    r'\bgrief\b', r'\bgone too soon\b', r'\bpremature (loss|death)\b',
    r'\bfuneral\b', r'\blast rites\b', r'\bdeath anniversary\b',
    r'\bin loving memory\b', r'\bdeepest condolences?\b',
]

def is_condolence(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in CONDOLENCE_PATTERNS)

# ─────────────────────────────────────────────────────────────
# EMOTION MAPPING
# v2 change: j-hartmann sadness → grief (was: fear in v1)
# ─────────────────────────────────────────────────────────────
HARTMANN_MAP = {
    'anger':   'anger',
    'disgust': 'anger',
    'fear':    'fear',
    'sadness': 'grief',   # v2: was 'fear' in v1
    'joy':     'JOY',
    'neutral': 'NEUTRAL',
    'surprise':'NEUTRAL',
}

HOPE_RE  = [r'\bwill\b',r'\bpromise\b',r'\bguarantee\b',r'\bfuture\b',
            r'\bafter\b.{0,30}\bpower\b',r'will (provide|give|build|create)',
            r'\bsoon\b',r'\bhope\b',r'\bdream\b',r'\bvision\b']
PRIDE_RE = [r'\bwe have (provided|built|completed|achieved|delivered|won|done|spent)\b',
            r'\bour government has?\b.{0,30}(provided|built|delivered)',
            r'\bproud\b',r'\bpride\b',r'\bcelebrat\b',r'\bvictory\b',r'\blegacy\b']

def split_joy(t: str, sent: str) -> str:
    h = sum(1 for p in HOPE_RE  if re.search(p, t))
    p = sum(1 for p in PRIDE_RE if re.search(p, t))
    if h > p: return 'hope'
    if p >= h: return 'pride'
    return 'pride' if sent == 'Positive' else 'hope'

def resolve_neutral(t: str, sent: str, sneg: float, spos: float) -> str:
    if sneg > 0.55: return 'anger'
    if spos > 0.55: return 'pride'
    if sum(1 for p in HOPE_RE if re.search(p, t)) >= 2: return 'hope'
    return 'pride'

# ─────────────────────────────────────────────────────────────
# FRAMING RULES v2
# ─────────────────────────────────────────────────────────────
OPPONENTS = [
    'jagan','jagan reddy','jaganmohan','ysrcp','ysr congress','ycp',
    'chandrababu','naidu','cbn','tdp','telugu desam',
    'pawan kalyan','janasena','jana sena','nara',
    'bjp','bandaru','balakrishna','ramoji',
    'the chief minister','cm jagan','this government','the ruling party',
]

ATTACK_WORDS = {
    'failed','failure','fail','destroy','destroyed','ruin','ruined','loot','looted',
    'rob','robbed','steal','stole','stolen','corrupt','corruption','crime','criminal',
    'lie','lied','lies','lying','false','fake','unfit','incompetent','negligent',
    'neglect','anarchy','anarchic','psycho','betrayal','betrayed','backstab',
    'atrocity','atrocities','illegal','murder','murdered','killed','scam','cheated',
    'cheat','fraud','mismanage','inhumane','unacceptable','shameful','disgrace',
    'mistakes','hurt','harmed','harm','damage','damaged','broken','broke',
    'violated','violation','worst','terrible','horrible','outrag',
    'enemy','enemies','rowdy','goon','thug','anti','wrong',
    'obstruct','block','wicke','collapse','disastrous','pathetic','hopeless',
    'misrule','misgovern','exploited','suppressed','detained','arrested',
}

ATTACK_HASH = [
    'jaganfail','byebyejagan','whyaphatesjagan','endoftdp','ycpantham',
    '2024jagannomore','jaganfailedcm','psychojagan','nalugellanarakam',
    'ycpcriminalpolitics','corruptbabunaidu','cbNbreakscourtorders',
    'mosagadubabu','tdpantibc','tdpattacks','jarugujagan','dalitdrohijagan',
    'rashtramaaravanakaashtam','seemadrohijagan','rythudrohijagan',
    'jaganfailsirrigation','idhemkarmamanarashtramki',
]

ACH_RE = [
    r'\bwe have (provided|distributed|given|spent|built|constructed|launched|implemented'
    r'|introduced|completed|started|opened|delivered|released|allocated|deposited|funded|achieved)\b',
    r'\bour government has? (provided|distributed|given|spent|built|launched|implemented'
    r'|introduced|completed|delivered|released|allocated|funded|created|established)\b',
    r'\btoday\b.{0,50}(providing|distributing|conducting|launching|releasing|giving)',
    # v2 NEW: opposition-era TDP references to 2014-19
    r'\bduring (our|tdp|the) (government|rule|tenure|term|regime)\b',
    r'\bduring (2014|2015|2016|2017|2018|2019).{0,30}(built|developed|created|launched)',
    r'\bwhen we were in power\b',
    r'\b(tdp|our) government\b.{0,40}(built|developed|created|launched|completed|gave|provided)',
    r'\b(built|constructed|developed|completed|launched|inaugurated|created)\b.{0,30}during\b',
    r'\bevidence of.{0,30}(progress|development|achievement|success)\b',
    r'\brs\.?\s*\d[\d,\.]*\s*(crore|lakh|thousand)',
    r'₹\s*\d[\d,\.]*\s*(crore|lakh)',
    r'\b\d[\d,]*\s*(famil|beneficiar|student|farmer|women|poor|worker|household)',
    r'\b(scheme|programme|program|yojana|initiative|project).{0,40}(launch|implement|distribut|complet|deliver)',
    r'\b(built|constructed|completed|inaugurated|opened).{0,40}(hospital|school|road|bridge|dam|college)',
]

PROMISE_RE = [
    r'\bwe (will|shall|promise to|commit to|guarantee)\b',
    r'\bwe promise\b', r'\bwe guarantee\b', r'\bwe commit\b',
    r'\bI promise\b', r'\bI guarantee\b', r'\bI commit\b',
    r'\bmy assurance\b', r'\bmy guarantee\b', r'\bmy promise\b',  # v2 NEW
    r'\bthis is my assurance\b', r'\bI assure\b', r'\bI give my word\b',
    r'\bwill (provide|give|ensure|implement|build|allocate|support|protect|create'
    r'|develop|deliver|solve|fix|address|resolve|remove|eliminate|eradicate'
    r'|stop|end|finish|complete|establish|launch|open|make)\b',
    r'\bin (exactly |just )?\d+ (days?|months?|years?)\b.{0,60}(will|shall|promise|guarantee)',  # v2 NEW
    r'\bwithin \d+ (days?|months?|years?)\b',
    r'\b\d+ (days?|months?|years?)\b.{0,40}(will|shall|promise|guarantee|ensure|deliver)',
    r'\b(mafia|criminals?|goons?|corruption|drugs?|ganja)\b.{0,40}(will flee|will leave|will go|will end|will stop)\b',  # v2 NEW
    r'\b(will flee|will be driven out|will leave|will be removed)\b',
    r'\bafter (we come|tdp comes|janasena comes|alliance comes|we win|we form).{0,30}(power|government)\b',
    r'\bas soon as.{0,40}(comes? to power|wins?|forms? government)\b',
    r'\bonce (we|tdp|alliance).{0,30}(comes? to power|wins?|forms?)\b',
    r'\bour (guarantee|manifesto|pledge|commitment)\b',
    r'\bcommitment to (the people|you|ap|andhra)\b',
]

VICTIM_RE = [
    r'\bfalse cases?\b', r'\bfalse allegations?\b',
    r'\bpolitical (persecution|vendetta|suppression|targeting|revenge|arrest)\b',
    r'\bunjust (arrest|imprisonment|case|charges?|treatment)\b',
    r'\b(targeting|harassing|arresting|persecuting).{0,30}(our|the) (leader|cadre|workers?|party)\b',
    r'\b(jailed|arrested|imprisoned|in jail|in prison|behind bars)\b',
    r'\b(our|my) (leader|party|cadre).{0,50}(attacked|targeted|harassed|persecuted|suppressed)\b',
    r'\bpolitical (prisoner|prosecution|witch hunt)\b',
    r'\binnocent (arrest|victim)\b', r'\bgoons? attacked\b',
]


def classify_framing(text, raw, sent, sneg, spos):
    t   = text.lower()
    raw = (raw or text).lower()
    sc  = {'attack framing':0.0,'achievement framing':0.0,
           'promise framing':0.0,'victim framing':0.0,'issue framing':0.0}

    opp = any(op in t for op in OPPONENTS)
    neg = any(w in t for w in ATTACK_WORDS)
    hsh = sum(1 for h in ATTACK_HASH if h in raw.replace(' ',''))

    if opp and neg:   sc['attack framing'] += 0.55
    elif opp and sneg > 0.5: sc['attack framing'] += 0.35
    if hsh: sc['attack framing'] += min(0.35, 0.15 * hsh)
    if sent == 'Negative' and opp: sc['attack framing'] += 0.10

    ah = sum(1 for p in ACH_RE if re.search(p, t))
    if ah: sc['achievement framing'] += min(0.70, 0.28 * ah)
    if sent == 'Positive' and ah: sc['achievement framing'] += 0.10

    ph = sum(1 for p in PROMISE_RE if re.search(p, t))
    if ph: sc['promise framing'] += min(0.75, 0.32 * ph)
    if re.search(r'\bmy assurance\b', t): sc['promise framing'] += 0.30
    if re.search(r'\bin \d+ days?\b', t): sc['promise framing'] += 0.25

    vh = sum(1 for p in VICTIM_RE if re.search(p, t))
    if vh: sc['victim framing'] += min(0.65, 0.28 * vh)

    # Phase 1 constraint
    if sneg > 0.65:
        sc['attack framing']      *= 1.30
        sc['achievement framing'] *= 0.60
        sc['promise framing']     *= 0.70
    if spos > 0.65:
        sc['achievement framing'] *= 1.25
        sc['promise framing']     *= 1.15
        sc['attack framing']      *= 0.60
        sc['victim framing']      *= 0.70
    if spos > 0.50 and sneg < 0.25:
        sc['attack framing'] *= 0.65

    mx = max(sc['attack framing'], sc['achievement framing'],
             sc['promise framing'], sc['victim framing'])
    sc['issue framing'] = 0.50 if mx < 0.25 else (0.20 if mx < 0.40 else 0.08)

    tot = sum(sc.values())
    if tot > 0: sc = {k: round(v/tot,4) for k,v in sc.items()}
    top = max(sc, key=sc.get)
    return top, sc[top], sc


# ─────────────────────────────────────────────────────────────
# SUBJECTIVITY: Multi-signal v2
# ─────────────────────────────────────────────────────────────
EVAL_KW = {
    'wicked','corrupt','criminal','evil','shameful','disgraceful','outrageous',
    'terrible','horrible','pathetic','despicable','incompetent','traitor','thug',
    'goon','psycho','liar','cheater','betrayed','betrayal','selfishness','greed',
    'arrogance','brilliant','visionary','outstanding','legendary','hero',
    'inspiration','glory','honour','pride',
}

def subjectivity(text, tb_score, sent_conf, sneg, spos):
    sigs = 0
    if tb_score > 0.15: sigs += 1
    if max(sneg, spos) > 0.65: sigs += 1
    if set(re.findall(r'\b\w+\b', text.lower())) & EVAL_KW: sigs += 1
    return ('opinion-based' if sigs >= 2 else 'fact-based'), sigs


# ─────────────────────────────────────────────────────────────
# CROSS-DIM TONE OVERRIDE (new in v2)
# ─────────────────────────────────────────────────────────────
def tone_override(emo, ec, tone, tc, frame, fc):
    if emo == 'grief':
        return 'victim narrative', True, 'Rule C: grief→victim narrative'
    if (tone == 'inspirational' and emo == 'anger' and ec > 0.80
            and frame == 'attack framing' and tc < (ec - 0.10)):
        return 'aggressive', True, f'Rule A: anger({ec:.2f})+attack overrides inspirational({tc:.2f})'
    if (tone == 'inspirational' and frame == 'victim framing' and fc > 0.50):
        return 'victim narrative', True, f'Rule B: victim_framing({fc:.2f}) overrides inspirational'
    return tone, False, ''


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def classifiable(text):
    if not text or not str(text).strip(): return False
    words = str(text).split()
    cw = [w for w in words if len(w) > 2 and w.lower() not in STOP_WORDS]
    return len(words) >= MIN_WORDS and len(cw) >= MIN_CW

def stratify(df, n, seed=RANDOM_SEED):
    rng = random.Random(seed)
    idx = set()
    sc  = 'sentiment_label' if 'sentiment_label' in df.columns else None
    if sc:
        for lbl in ['Negative','Positive','Neutral']:
            pool = df[df[sc]==lbl].index.tolist()
            rng.shuffle(pool)
            for i in pool:
                if len(idx) >= n: break
                idx.add(i)
    rem = [i for i in df.index if i not in idx]
    rng.shuffle(rem)
    while len(idx) < n and rem: idx.add(rem.pop())
    return df.loc[sorted(idx)].copy()

def run_tone_nli(texts, clf, flags):
    hyps = [TONE_HYPOTHESES[l] for l in TONE_LABELS]
    res  = {}
    oi   = [i for i,ok in enumerate(flags) if ok]
    ot   = [texts[i] for i in oi]
    for s in tqdm(range(0, len(ot), BATCH_SIZE), desc="  [TONE]"):
        bt, bi = ot[s:s+BATCH_SIZE], oi[s:s+BATCH_SIZE]
        try:
            outs = clf(bt, candidate_labels=hyps, multi_label=False,
                       hypothesis_template="{}", truncation=True, max_length=MAX_LENGTH)
            if isinstance(outs, dict): outs = [outs]
            h2l = {v:k for k,v in TONE_HYPOTHESES.items()}
            for idx, o in zip(bi, outs):
                res[idx] = {h2l.get(h,h): round(float(sc),4) for h,sc in zip(o['labels'],o['scores'])}
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                for idx,txt in zip(bi,bt):
                    try:
                        o = clf([txt], candidate_labels=hyps, multi_label=False,
                                hypothesis_template="{}", truncation=True, max_length=MAX_LENGTH)
                        if isinstance(o, list): o = o[0]
                        h2l = {v:k for k,v in TONE_HYPOTHESES.items()}
                        res[idx] = {h2l.get(h,h): round(float(sc),4) for h,sc in zip(o['labels'],o['scores'])}
                    except: res[idx] = {l:0.0 for l in TONE_LABELS}
            else:
                for idx in bi: res[idx] = {l:0.0 for l in TONE_LABELS}
    return [res.get(i, {l:0.0 for l in TONE_LABELS}) for i in range(len(texts))]

def run_emotion(texts, clf, flags, cond_flags, sents, snegs, sposs):
    results = [None]*len(texts)
    oi = [i for i,ok in enumerate(flags) if ok and not cond_flags[i]]
    ot = [texts[i] for i in oi]
    all_out = []
    print(f"  j-hartmann on {len(ot)} texts...")
    for s in tqdm(range(0, len(ot), BATCH_SIZE*2), desc="  [EMOTION]"):
        bt = ot[s:s+BATCH_SIZE*2]
        try:
            out = clf(bt, truncation=True, max_length=MAX_LENGTH)
            if isinstance(out[0], dict): out = [[o] for o in out]
            all_out.extend(out)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                for text in bt:
                    try:
                        o = clf([text], truncation=True, max_length=MAX_LENGTH)
                        all_out.append(o[0] if isinstance(o[0],list) else [o[0]])
                    except: all_out.append([{'label':'neutral','score':1.0}])
            else:
                for _ in bt: all_out.append([{'label':'neutral','score':1.0}])

    for ri, (orig, out) in enumerate(zip(oi, all_out)):
        t    = str(texts[orig]).lower()
        sent = str(sents[orig])
        sn   = float(snegs[orig]) if snegs[orig]==snegs[orig] else 0.33
        sp   = float(sposs[orig]) if sposs[orig]==sposs[orig] else 0.33
        raw  = {}
        for item in (out if isinstance(out,list) else [out]):
            raw[item['label'].lower()] = round(float(item['score']),4)
        mp = {'anger':0.0,'fear':0.0,'grief':0.0,'hope':0.0,'pride':0.0,'trust':0.0}
        for h,sc in raw.items():
            bkt = HARTMANN_MAP.get(h,'NEUTRAL')
            if bkt == 'anger':   mp['anger'] += sc
            elif bkt == 'fear':  mp['fear']  += sc
            elif bkt == 'grief': mp['grief'] += sc
            elif bkt == 'JOY':
                mp[split_joy(t,sent)] += sc
            else:
                mp[resolve_neutral(t,sent,sn,sp)] += sc*0.5
        tot = sum(mp.values())
        if tot > 0: mp = {k:round(v/tot,4) for k,v in mp.items()}
        top = max(mp, key=mp.get)
        results[orig] = {'label':top,'confidence':round(mp[top],4),
                         'scores_json':json.dumps(mp),'is_condolence':False}

    for i in range(len(texts)):
        if cond_flags[i] and flags[i]:
            gs = {'anger':0.0,'fear':0.05,'grief':0.90,'hope':0.02,'pride':0.01,'trust':0.02}
            results[i] = {'label':'grief','confidence':0.90,
                          'scores_json':json.dumps(gs),'is_condolence':True}
        elif results[i] is None:
            zs = {'anger':0.0,'fear':0.0,'grief':0.0,'hope':0.0,'pride':0.0,'trust':0.0}
            results[i] = {'label':'INSUFFICIENT_TEXT','confidence':0.0,
                          'scores_json':json.dumps(zs),'is_condolence':False}
    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
t0 = time.time()
print("="*65)
print("PHASE 3 HYBRID CLASSIFIER v2")
print(f"Seed: {RANDOM_SEED} | {SAMPLE_N} rows/file")
print("="*65)

DEV = 0 if torch.cuda.is_available() else -1
print(f"[INFO] {'CUDA: '+torch.cuda.get_device_name(0) if DEV==0 else 'CPU'}")

# STEP 1: Find files
files = sorted(glob.glob(SOURCE_PATTERN))
print(f"\n[1] {len(files)} FullAnalysis files found")
if not files: raise SystemExit("[ERROR] No *_FullAnalysis.xlsx found")

# STEP 2: Sample
print(f"\n[2] Sampling {SAMPLE_N}/file...")
samples = []
for fp in files:
    fn  = os.path.basename(fp)
    pts = fn.replace("_FullAnalysis.xlsx","").split("_")
    try: df = pd.read_excel(fp)
    except Exception as e: print(f"  SKIP {fn}: {e}"); continue
    cc = ("caption_for_sentiment" if "caption_for_sentiment" in df.columns
          else "caption_en" if "caption_en" in df.columns else None)
    if not cc: print(f"  SKIP {fn}: no caption col"); continue
    s = stratify(df, SAMPLE_N)
    s["_src"] = fn
    s["_ent"] = pts[0] if pts else "?"
    s["_plt"] = pts[1] if len(pts)>1 else "?"
    s["_per"] = "_".join(pts[2:]) if len(pts)>2 else "?"
    samples.append(s)
    print(f"  ✓ {fn:52s} {len(s)} rows")

comb = pd.concat(samples, ignore_index=True)
print(f"[INFO] Total: {len(comb)} rows")

# Extract arrays
tc_arr = [str(r.get("caption_for_sentiment","") or r.get("caption_en","") or "").strip()
          for _,r in comb.iterrows()]
tr_arr = [str(r.get("caption_en","") or tc_arr[i]).strip()
          for i,(_, r) in enumerate(comb.iterrows())]

cf   = [classifiable(t) for t in tc_arr]
cond = [is_condolence(t) for t in tc_arr]
comb["classifiable"]   = cf
comb["is_condolence_post"] = cond

sents  = comb.get("sentiment_label",pd.Series(["Neutral"]*len(comb))).fillna("Neutral").tolist()
sconfs = comb.get("sentiment_confidence",pd.Series([0.5]*len(comb))).fillna(0.5).tolist()
snegs  = comb.get("score_negative",pd.Series([0.33]*len(comb))).fillna(0.33).tolist()
sposs  = comb.get("score_positive",pd.Series([0.33]*len(comb))).fillna(0.33).tolist()

print(f"  Classifiable: {sum(cf)} | Condolence: {sum(cond)}")

# STEP 3: Emotion
print(f"\n[3] Loading emotion model: {EMOTION_MODEL}")
ep = pipeline("text-classification", model=EMOTION_MODEL, device=DEV, top_k=None)
er = run_emotion(tc_arr, ep, cf, cond, sents, snegs, sposs)
comb["emotion_label"]       = [r["label"]       for r in er]
comb["emotion_confidence"]  = [r["confidence"]  for r in er]
comb["emotion_scores_json"] = [r["scores_json"] for r in er]
comb["emotion_low_conf"]    = ((comb["emotion_label"]!="INSUFFICIENT_TEXT") &
                               (comb["emotion_confidence"]<0.45))
del ep
if DEV==0: torch.cuda.empty_cache()
print("  Emotion model unloaded")

# STEP 4: Tone NLI
print(f"\n[4] Loading tone NLI: {NLI_MODEL}")
np_ = pipeline("zero-shot-classification", model=NLI_MODEL,
               device=DEV, truncation=True, max_length=MAX_LENGTH)
tone_nli_flags = [ok and not c for ok,c in zip(cf,cond)]
print("\n[5] Classifying TONE...")
tsl = run_tone_nli(tc_arr, np_, tone_nli_flags)

tl_raw, tc_raw, tj = [], [], []
for i,sc in enumerate(tsl):
    if not cf[i]:
        tl_raw.append("INSUFFICIENT_TEXT"); tc_raw.append(0.0)
        tj.append(json.dumps({l:0.0 for l in TONE_LABELS}))
    elif cond[i]:
        cs = {l:0.0 for l in TONE_LABELS}; cs["victim narrative"]=0.85
        tl_raw.append("victim narrative"); tc_raw.append(0.85)
        tj.append(json.dumps(cs))
    else:
        top = max(sc, key=sc.get)
        tl_raw.append(top); tc_raw.append(sc[top])
        tj.append(json.dumps({l:sc.get(l,0.0) for l in TONE_LABELS}))

comb["tone_label_raw"] = tl_raw
comb["tone_confidence"] = tc_raw
comb["tone_scores_json"] = tj
del np_
if DEV==0: torch.cuda.empty_cache()
print("  NLI model unloaded")

# STEP 6: Framing
print("\n[6] Classifying FRAMING (rule-based)...")
fl, fc_arr, fj = [], [], []
for i,(ct,rt) in enumerate(zip(tc_arr,tr_arr)):
    if not cf[i]:
        fl.append("INSUFFICIENT_TEXT"); fc_arr.append(0.0); fj.append(json.dumps({}))
    elif cond[i]:
        fs={'attack framing':0.0,'achievement framing':0.0,
            'promise framing':0.0,'victim framing':0.0,'issue framing':1.0}
        fl.append("issue framing"); fc_arr.append(1.0); fj.append(json.dumps(fs))
    else:
        lb,cn,sd = classify_framing(ct,rt,sents[i],float(snegs[i]),float(sposs[i]))
        fl.append(lb); fc_arr.append(cn); fj.append(json.dumps(sd))

comb["framing_label"]      = fl
comb["framing_confidence"] = fc_arr
comb["framing_low_conf"]   = ((comb["framing_label"]!="INSUFFICIENT_TEXT") &
                               (comb["framing_confidence"]<0.40))
comb["framing_scores_json"]= fj

# STEP 7: Cross-dim tone override
print("\n[7] Applying cross-dim tone override...")
ftl, ov_app, ov_rsn = [], [], []
for i in range(len(comb)):
    el=comb["emotion_label"].iloc[i]; ec=float(comb["emotion_confidence"].iloc[i])
    tl=comb["tone_label_raw"].iloc[i]; tc_=float(comb["tone_confidence"].iloc[i])
    fm=comb["framing_label"].iloc[i]; fcv=float(comb["framing_confidence"].iloc[i])
    nt,ov,rs = tone_override(el,ec,tl,tc_,fm,fcv)
    ftl.append(nt); ov_app.append(ov); ov_rsn.append(rs)

comb["tone_label"]            = ftl
comb["tone_override_applied"] = ov_app
comb["tone_override_reason"]  = ov_rsn
comb["tone_low_conf"]         = ((comb["tone_label"]!="INSUFFICIENT_TEXT") &
                                  (~comb["tone_override_applied"]) &
                                  (comb["tone_confidence"]<0.40))

n_ov = sum(ov_app)
print(f"  Overrides: {n_ov}")
for r,c in Counter([x for x in ov_rsn if x]).most_common():
    print(f"    {c}× {r[:80]}")

# STEP 8: Subjectivity
print("\n[8] Subjectivity (multi-signal)...")
tb = [round(TextBlob(str(t)).sentiment.subjectivity,4) for t in tc_arr]
sl, ss = [], []
for i in range(len(comb)):
    if not cf[i]:
        sl.append("INSUFFICIENT_TEXT"); ss.append(0)
    else:
        lb,sg = subjectivity(tc_arr[i],tb[i],float(sconfs[i]),float(snegs[i]),float(sposs[i]))
        sl.append(lb); ss.append(sg)

comb["subjectivity_textblob"]  = tb
comb["subjectivity_label"]     = sl
comb["subjectivity_signals"]   = ss

# Rename meta cols
comb.rename(columns={"_src":"source_file","_ent":"entity","_plt":"platform","_per":"period"},inplace=True)

# Save
KEEP = [
    "entity","platform","period","source_file",
    "Hashtag_Theme","sentiment_label","sentiment_confidence",
    "score_negative","score_neutral","score_positive",
    "caption_en","caption_for_sentiment",
    "emotion_label","emotion_confidence","emotion_low_conf",
    "emotion_scores_json","is_condolence_post",
    "tone_label","tone_label_raw","tone_confidence","tone_low_conf",
    "tone_override_applied","tone_override_reason","tone_scores_json",
    "framing_label","framing_confidence","framing_low_conf","framing_scores_json",
    "subjectivity_textblob","subjectivity_label","subjectivity_signals",
    "classifiable",
]
out = comb[[c for c in KEEP if c in comb.columns]].copy()
out.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"\n[SAVED] {OUTPUT_FILE} — {len(out)} rows × {len(out.columns)} cols")

# Summary
cl = out[out["classifiable"]==True]
N  = len(cl)
print(f"\n{'='*65}\nDISTRIBUTION SUMMARY (n={N} classifiable)\n{'='*65}")
for dim,col in [("EMOTION",  "emotion_label"),
                ("TONE",     "tone_label"),
                ("FRAMING",  "framing_label"),
                ("SUBJECTIVITY","subjectivity_label")]:
    print(f"\n  {dim}:")
    for lb,ct in cl[col].value_counts().items():
        print(f"    {lb:32s} {ct:4d} ({ct/N*100:5.1f}%)  {'█'*int(ct/N*40)}")

print(f"\n  Cross-dim overrides:  {cl['tone_override_applied'].sum()}")
print(f"  Condolence posts:     {cl['is_condolence_post'].sum()}")
print(f"  tone_low_conf:        {cl['tone_low_conf'].sum()}")
print(f"  framing_low_conf:     {cl['framing_low_conf'].sum()}")

print("\n  TONE by entity:")
print(cl.groupby(["entity","tone_label"]).size().unstack(fill_value=0).to_string())

# Methodology file
rt = time.time() - t0
try:
    mlines = [
        "PHASE 3 HYBRID CLASSIFIER v2 — METHODOLOGY",
        "="*65,
        f"Timestamp : {datetime.now().isoformat()}",
        f"Runtime   : {rt:.1f}s ({rt/60:.1f} min) | Device: {'CUDA' if DEV==0 else 'CPU'}",
        f"Seed      : {RANDOM_SEED} | Rows/file: {SAMPLE_N} | Total: {len(out)}",
        "",
        "TAXONOMY CHANGES VS v1",
        "-"*40,
        "EMOTION: Added 'grief' as 6th label",
        "  v1: anger, fear, hope, pride, trust (5)",
        "  v2: anger, fear, GRIEF, hope, pride, trust (6)",
        "  grief = condolence/tribute/bereavement posts",
        "  Source: keyword pre-filter + j-hartmann sadness score",
        "  v1 bug: j-hartmann sadness was mapped to fear; condolence posts",
        "          got fear+inspirational — severe systematic misclassification",
        "",
        "TONE: Labels unchanged (5). Changes:",
        "  1. inspirational hypothesis made more restrictive:",
        "     Added: 'The text does NOT primarily describe government failures,",
        "     crimes, atrocities, or political corruption.'",
        "  2. aggressive hypothesis broadened to catch implicit attacks",
        "  3. Cross-dimension override rules (new in v2):",
        "     Rule A: anger(>0.80)+attack_framing → aggressive",
        "     Rule B: victim_framing(>0.50) → victim narrative",
        "     Rule C: grief → victim narrative",
        "     Column 'tone_override_applied' = True where fired (transparent)",
        "",
        "FRAMING: Labels unchanged (5). Rule changes:",
        "  Promise: Added 'my assurance', time-bound pledges ('in X days'),",
        "           expanded verb list, 'will flee/end' patterns",
        "  Achievement: Added opposition-era TDP 2014-19 reference patterns",
        "",
        "SUBJECTIVITY: Multi-signal approach (was TextBlob-only)",
        "  3 signals: TextBlob > 0.15 | Phase1 max_sentiment > 0.65 | eval keywords",
        "  Rule: 2+ signals → opinion-based | 0-1 → fact-based",
        "  Column 'subjectivity_signals' records vote count (0-3)",
        "  v1 bug: TextBlob scored 0.0 on 72/587 rhetorical posts",
        "",
        "MODELS",
        "-"*40,
        f"Emotion  : {EMOTION_MODEL} (fine-tuned 7-class)",
        f"Tone NLI : {NLI_MODEL} (zero-shot, hypothesis_template='{{}}')",
        "Framing  : Rule-based (deterministic)",
        "Subjectivity: Multi-signal heuristic",
        "",
        "TONE NLI HYPOTHESES v2",
        "-"*40,
    ]
    for lb,hyp in TONE_HYPOTHESES.items():
        mlines += [f"  {lb}:", f"    {hyp}", ""]
    mlines += [
        "DISTRIBUTIONS","-"*40,
        f"Emotion  : {cl['emotion_label'].value_counts().to_dict()}",
        f"Tone     : {cl['tone_label'].value_counts().to_dict()}",
        f"Framing  : {cl['framing_label'].value_counts().to_dict()}",
        f"Subjectivity: {cl['subjectivity_label'].value_counts().to_dict()}",
        f"Overrides: {n_ov}",
        f"Condolence: {sum(cond)}",
    ]
    with open(METHODOLOGY_FILE,"w",encoding="utf-8") as f:
        f.write("\n".join(mlines))
    print(f"\n[SAVED] {METHODOLOGY_FILE}")
except Exception as e:
    print(f"[WARN] Methodology write failed: {e}")

print(f"\n[DONE] {rt:.1f}s | Upload {os.path.basename(OUTPUT_FILE)}")
