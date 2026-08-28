import json

# The 34 standard Amharic base consonants
base_chars = [
    0x1200, # ሀ
    0x1208, # ለ
    0x1210, # ሐ
    0x1218, # መ
    0x1220, # ሠ
    0x1228, # ረ
    0x1230, # ሰ
    0x1238, # ሸ
    0x1240, # ቀ
    0x1260, # በ
    0x1268, # ቨ
    0x1270, # ተ
    0x1278, # ቸ
    0x1280, # ኀ
    0x1290, # ነ
    0x1298, # ኘ
    0x12A0, # አ
    0x12A8, # ከ
    0x12B8, # ኸ
    0x12C8, # ወ
    0x12D0, # ዐ
    0x12D8, # ዘ
    0x12E0, # ዠ
    0x12E8, # የ
    0x12F0, # ደ
    0x1300, # ጀ
    0x1308, # ገ
    0x1320, # ጠ
    0x1328, # ጨ
    0x1330, # ጰ
    0x1338, # ጸ
    0x1340, # ፀ
    0x1348, # ፈ
    0x1350, # ፐ
]

chars = []
for base in base_chars:
    for order in range(7):
        chars.append(chr(base + order))

# Add the 5 labiovelars (q, h, k, g) * 5 orders
labiovelars = [
    [0x1248, 0x124A, 0x124B, 0x124C, 0x124D], # ቈ...
    [0x1288, 0x128A, 0x128B, 0x128C, 0x128D], # ኈ...
    [0x12B0, 0x12B2, 0x12B3, 0x12B4, 0x12B5], # ኰ... (KWA)
    [0x1310, 0x1312, 0x1313, 0x1314, 0x1315], # ጐ... (GWA)
]
for family in labiovelars:
    for cp in family:
        chars.append(chr(cp))

# Add the standard "wa" labialized variants
wa_variants = [
    0x120F, # ሏ (LWA)
    0x1217, # ሗ (HHWA)
    0x121F, # ሟ (MWA)
    0x1227, # ሧ (SZWA)
    0x122F, # ሯ (RWA)
    0x1237, # ሷ (SWA)
    0x123F, # ሿ (SHWA)
    0x1267, # ቧ (BWA)
    0x126F, # ቏ (VWA)
    0x1277, # ቷ (TWA)
    0x127F, # ቿ (CWA)
    0x1297, # ኗ (NWA)
    0x129F, # ኟ (NYWA)
    0x12DF, # ዟ (ZWA)
    0x12E7, # ዧ (ZHWA)
    0x12F7, # ዷ (DWA)
    0x1307, # ጇ (JWA)
    0x1327, # ጧ (THWA)
    0x132F, # ጯ (CHWA)
    0x1337, # ጿ (PHWA) - wait, PHA is 1330. PHWA is 1337.
    0x133F, # ጿ (TSWA) - wait, TSA is 1338. TSWA is 133F. Let's check Unicode names.
    0x1347, # ፇ (TZWA)
    0x134F, # ፏ (FWA)
    0x1357, # ፗ (PWA)
]

# Let's dynamically add the 8th order 'wa' if they exist in Ethiopic block.
import unicodedata
for base in base_chars:
    wa_char = chr(base + 7)
    if unicodedata.name(wa_char, "").startswith("ETHIOPIC SYLLABLE"):
        if wa_char not in chars:
            chars.append(wa_char)

# Deduplicate and sort by code point
chars = sorted(list(set(chars)))

# Write to characters.json in src/
with open("src/characters.json", "w", encoding="utf-8") as f:
    json.dump(chars, f, ensure_ascii=False, indent=4)

print(f"Generated {len(chars)} characters.")
