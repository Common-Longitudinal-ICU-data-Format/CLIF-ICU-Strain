"""Threshold constants and criteria prose for the ICU strain ARF analysis.

Both the analysis (04) and the HTML report (05) import from here so the
documented criteria cannot drift from the implemented ones.

Source: references/main_text_12272025.docx, Methods, "Patient Population and
ARF Clinical Definition".
"""

# --- ARF physiologic thresholds ---
SPO2_HYPOXEMIA = 90          # %, strict less-than, room air only
PAO2_HYPOXEMIA = 60          # mmHg, less-than-or-equal, room air only
PF_RATIO_THRESHOLD = 300     # PaO2/FiO2, less-than-or-equal, any FiO2
SF_RATIO_THRESHOLD = 315     # SpO2/FiO2, less-than-or-equal, no-ABG fallback
PACO2_HYPERCAPNIA = 45       # mmHg, greater-than-or-equal
PH_HYPERCAPNIA = 7.35        # strict less-than

# --- Windows and tolerances ---
ARF_WINDOW_HOURS = 24        # hours after ICU admission
ICU_BLOCK_GAP_HOURS = 1      # ADT contiguity threshold
ABG_PAIR_TOLERANCE_MIN = 5   # minutes, pairing pco2 with ph

# --- Cohort ---
ANALYSIS_YEAR = 2024
MIN_AGE = 18

# --- Reporting ---
MIN_CELL_SIZE = 10           # suppress any reported cell below this

# --- CLIF category values (exact matches; venous/capillary deliberately absent) ---
ABG_PO2 = "po2_arterial"
ABG_PCO2 = "pco2_arterial"
ABG_PH = "ph_arterial"
SPO2_VITAL = "spo2"
ICU_LOCATION = "icu"
IMV_DEVICE = "IMV"
ROOM_AIR_DEVICE = "Room Air"

CRITERIA_TEXT = {
    "hypoxemic": (
        f"SpO2 < {SPO2_HYPOXEMIA}% or PaO2 <= {PAO2_HYPOXEMIA} mmHg while on room air, "
        f"OR PaO2/FiO2 <= {PF_RATIO_THRESHOLD} at any FiO2, "
        f"OR SpO2/FiO2 <= {SF_RATIO_THRESHOLD} as a fallback when the hospitalization "
        "has no arterial blood gas anywhere in the window."
    ),
    "hypercapnic": (
        f"PaCO2 >= {PACO2_HYPERCAPNIA} mmHg AND arterial pH < {PH_HYPERCAPNIA}, "
        f"measured on the same arterial draw (paired within "
        f"{ABG_PAIR_TOLERANCE_MIN} minutes). Venous and capillary gases are excluded."
    ),
    "mixed": (
        "Meets hypoxemic criteria at some point in the window and hypercapnic criteria "
        "at some point in the window. The two need not be simultaneous."
    ),
    "not_ards": (
        "These are physiologic acute respiratory failure subtypes, NOT ARDS. The New "
        "Global Definition of ARDS requires bilateral opacities on chest imaging and "
        "exclusion of cardiogenic pulmonary edema; CLIF contains no imaging table, so "
        "ARDS cannot be ascertained here. Do not read these counts as ARDS cases."
    ),
}

SF_FALLBACK_CAVEAT = (
    f"The SpO2/FiO2 <= {SF_RATIO_THRESHOLD} fallback is a local construction, not part "
    "of the manuscript's ARF definition. It carries no SpO2 <= 97% ceiling, so above "
    "97% saturation the ratio loses meaning and hypoxemia may be over-classified."
)
