import pandas as pd

df = pd.read_csv("Master_Hashtag_Dictionary.csv")

# Step 1: Identify all HTTP error rows by their fixed confidence score
HTTP_ERROR_CONF = 0.853813
http_mask = df['sentiment_confidence'].round(6) == round(HTTP_ERROR_CONF, 6)
print(f"HTTP error rows: {http_mask.sum()}")

# Step 2: Manual corrections for HTTP error hashtags
manual_fixes = {
    '#ncbn':                    ('NCBN (N. Chandrababu Naidu)',         'Neutral',  0.85, 0.08, 0.07),
    '#polavarampaaye':          ('Polavaram Paaye (project fell/failed)','Negative', 0.70, 0.20, 0.10),
    '#pulivendulasiddham':      ('Pulivendula Siddham (Ready from Pulivendula)', 'Positive', 0.05, 0.25, 0.70),
    '#papalapeddireddy':        ('Papala Peddireddy (person name)',      'Neutral',  0.05, 0.88, 0.07),
    '#jaganannaillapattalu':    ('Jagananna Illa Pattalu (housing scheme)', 'Positive', 0.03, 0.30, 0.67),
    '#jagananenenu':            ('Jagan Ane Nenu (I am Jagan)',          'Positive', 0.04, 0.22, 0.74),
    '#bhuvanammadeeksha':       ('Bhuvanamma Deeksha (protest fast)',    'Neutral',  0.15, 0.70, 0.15),
    '#whynotpulivendula':       ('Why Not Pulivendula',                  'Negative', 0.60, 0.30, 0.10),
    '#pawankalyanantikapu':     ('Pawan Kalyan Anti Kapu',               'Negative', 0.65, 0.28, 0.07),
    '#raajadhanifiles':         ('Raajadhani Files (capital city issue)','Negative', 0.60, 0.32, 0.08),
    '#illapattalu':             ('Illapattalu (housing plots)',           'Positive', 0.03, 0.27, 0.70),
    '#yuvathaudyoga':           ('Yuva Udyoga (youth employment)',       'Neutral',  0.05, 0.75, 0.20),
    '#zillamarindhi':           ('Zilla Marindhi (district changed)',    'Neutral',  0.10, 0.75, 0.15),
    '#zillamarindi':            ('Zilla Marindi (district changed)',     'Neutral',  0.10, 0.75, 0.15),
    '#siddhamonfeb18':          ('Siddham On Feb 18 (rally date)',       'Positive', 0.04, 0.25, 0.71),
    '#cbninkavali':             ('CBN In Kavali (location visit)',       'Neutral',  0.05, 0.85, 0.10),
    '#kondapisiddham':          ('Kondapi Siddham (Kondapi Ready)',      'Positive', 0.04, 0.26, 0.70),
    '#jaganannaagenenda':       ('Jagananna Agenda',                    'Neutral',  0.06, 0.80, 0.14),
    '#cheppadantechestadante':  ('Cheppadante Chesthadanthe (as promised, delivered)', 'Positive', 0.04, 0.22, 0.74),
}

for hashtag, (translated, label, neg, neu, pos) in manual_fixes.items():
    mask = df['hashtag_raw'] == hashtag
    if mask.any():
        df.loc[mask, 'hashtag_translated'] = translated
        df.loc[mask, 'sentiment_label']    = label
        df.loc[mask, 'score_negative']     = neg
        df.loc[mask, 'score_neutral']      = neu
        df.loc[mask, 'score_positive']     = pos
        df.loc[mask, 'sentiment_confidence'] = max(neg, neu, pos)
        df.loc[mask, 'translation_changed'] = True
        df.loc[mask, 'manual_override']    = True

# Step 3: Political context overrides
political_overrides = {
    # Anti-Jagan (used by NCBN/TDP) — clearly Negative
    '#nalugellanarakam':              ('Nalugella Narakam (Four Years of Hell)',        'Negative', 0.88, 0.09, 0.03),
    '#byebyejaganin2024':             ('Bye Bye Jagan In 2024',                         'Negative', 0.82, 0.14, 0.04),
    '#idhemkarmamanarashtraniki':     ('Idhem Karma Mana Rashtraniki (What fate for our state)', 'Negative', 0.78, 0.18, 0.04),
    '#rashtramaaravanakaashtamaa':    ('Rashtramaa Ravanakaashtamaa (Will state ever recover)', 'Negative', 0.80, 0.16, 0.04),
    '#2024jagannomore':               ('2024 Jagan No More',                            'Negative', 0.79, 0.17, 0.04),
    '#psychojagan':                   ('Psycho Jagan (insult hashtag)',                 'Negative', 0.85, 0.12, 0.03),
    '#ycpantham':                     ('YCP Antham (YCP is finished)',                  'Negative', 0.83, 0.13, 0.04),
    '#psychopovalicycleravali':       ('Psycho Povali Cycle Ravali (Psycho go, Cycle come)', 'Negative', 0.80, 0.15, 0.05),
    '#jaganpaniayipoyindhi':          ('Jagan Pani Ayipoyindhi (Jagan is done)',        'Negative', 0.78, 0.17, 0.05),
    '#dalitdrohijagan':               ('Dalit Drohi Jagan (Jagan betrayed Dalits)',     'Negative', 0.82, 0.14, 0.04),
    '#bcdrohijagan':                  ('BC Drohi Jagan (Jagan betrayed BCs)',           'Negative', 0.82, 0.14, 0.04),
    '#seemadrohijagan':               ('Seema Drohi Jagan (Jagan betrayed Seema)',      'Negative', 0.80, 0.15, 0.05),
    '#endoftdp':                      ('End Of TDP',                                    'Negative', 0.75, 0.20, 0.05),
    '#falsecasesagainstnaidu':        ('False Cases Against Naidu',                     'Negative', 0.78, 0.17, 0.05),
    '#gajadongachandrababu':          ('Gajadonga Chandrababu (Elephant Thief CBN)',    'Negative', 0.80, 0.15, 0.05),
    '#richestcmjagan':                ('Richest CM Jagan (corruption attack)',          'Negative', 0.78, 0.16, 0.06),
    # Pro-Jagan / Pro-YSRCP — clearly Positive
    '#memanthasiddham':               ('Memantha Siddham (We are ready — YSRCP slogan)','Positive', 0.04, 0.17, 0.79),
    '#voteforfan':                    ('Vote For Fan (Vote for YSRCP symbol)',          'Positive', 0.04, 0.18, 0.78),
    '#ysjaganagain':                  ('YS Jagan Again (re-election push)',             'Positive', 0.05, 0.19, 0.76),
    '#pulivendulasiddham':            ('Pulivendula Siddham (Ready — YSRCP base)',      'Positive', 0.04, 0.23, 0.73),
    # Pro-TDP — clearly Positive
    '#yuvagalampadayatra':            ('Yuva Galam Padayatra (Lokesh 4000km march)',    'Positive', 0.04, 0.20, 0.76),
    '#tdpjspbjpwinning':              ('TDPJSPBJP Winning',                             'Positive', 0.03, 0.18, 0.79),
}

if 'manual_override' not in df.columns:
    df['manual_override'] = False

for hashtag, (translated, label, neg, neu, pos) in political_overrides.items():
    mask = df['hashtag_raw'] == hashtag
    if mask.any():
        df.loc[mask, 'hashtag_translated']   = translated
        df.loc[mask, 'sentiment_label']      = label
        df.loc[mask, 'score_negative']       = neg
        df.loc[mask, 'score_neutral']        = neu
        df.loc[mask, 'score_positive']       = pos
        df.loc[mask, 'sentiment_confidence'] = max(neg, neu, pos)
        df.loc[mask, 'manual_override']      = True

df.to_csv("Master_Hashtag_Dictionary_Patched.csv", index=False)

overridden = df['manual_override'].sum() if 'manual_override' in df.columns else 0
print(f"\nTotal manual overrides applied : {overridden}")
print(f"Saved → Master_Hashtag_Dictionary_Patched.csv")

# New distribution
dist = df['sentiment_label'].value_counts()
total = len(df)
print("\nNew distribution after patch:")
for l, c in dist.items():
    print(f"  {l:<10}: {c:4d} ({c/total*100:.1f}%)")