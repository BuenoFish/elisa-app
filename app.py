import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO
import base64

st.set_page_config(page_title="ELISA 4PL Auswertung", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stApp { background-color: #f5f4f0; color: #1a1a1a; }
    .header-box { background:#1a1a1a; color:#f5f4f0; padding:2rem 2.5rem; border-radius:4px; margin-bottom:2rem; }
    .header-box h1 { font-size:1.8rem; margin:0 0 0.3rem 0; color:#f5f4f0; }
    .header-box p  { margin:0; color:#aaa; font-size:0.9rem; font-family:'IBM Plex Mono',monospace; }
    .param-card  { background:white; border:1px solid #e0e0e0; border-radius:4px; padding:1rem 1.2rem; margin-bottom:0.5rem; }
    .param-label { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px; }
    .param-value { font-family:'IBM Plex Mono',monospace; font-size:1.3rem; font-weight:600; color:#1a1a1a; }
    .info-box { background:#eef6ff; border-left:3px solid #2563eb; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .warn-box { background:#fff8e6; border-left:3px solid #f59e0b; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .err-box  { background:#fff0f0; border-left:3px solid #ef4444; padding:0.8rem 1rem; border-radius:0 4px 4px 0; font-size:0.88rem; margin-bottom:1rem; }
    .stButton>button { background:#1a1a1a; color:white; border:none; border-radius:4px;
        font-family:'IBM Plex Mono',monospace; font-size:0.85rem; padding:0.5rem 1.5rem; }
    div[data-testid="stExpander"] { background:white; border:1px solid #e0e0e0; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_b64 = img_to_base64("frauenkirche.jpg")

st.markdown(f"""
<div class="header-box" style="display:flex; align-items:stretch; justify-content:space-between; padding:0; overflow:hidden; border-radius:4px; margin:0;">
    <div style="padding:2rem 2.5rem; flex:1;">
        <h1 style="font-size:1.8rem; margin:0 0 0.3rem 0; color:#f5f4f0;">🧪 ELISA 4PL Auswertung</h1>
        <p style="margin:0; color:#aaa; font-size:0.9rem; font-family:'IBM Plex Mono',monospace;">
            Gebaut an der Elbe · Gefittet mit 4 Parametern · Kein Medizinprodukt
        </p>
    </div>
    <div style="width:350px; flex-shrink:0; position:relative; line-height:0;">
        <img src="data:image/jpeg;base64,{img_b64}"
             style="width:100%; height:100%; object-fit:cover; object-position:center; display:block;">
        <div style="position:absolute; top:0; left:0; width:100%; height:100%;
            background: linear-gradient(to right, #1a1a1a 0%, transparent 30%);"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def read_file(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        content = uploaded.read().decode("utf-8", errors="replace")
        uploaded.seek(0)
        sep = ";" if content.count(";") > content.count(",") else ","
        dec = "," if sep == ";" else "."
        return pd.read_csv(uploaded, sep=sep, decimal=dec, header=None)
    else:
        return pd.read_excel(uploaded, header=None)

def to_float(val):
    try:
        return float(str(val).replace(",", ".").strip())
    except Exception:
        return np.nan

def parse_plate(df):
    arr = np.full((8, 12), np.nan)
    for r in range(min(len(df), 8)):
        for c in range(min(len(df.columns), 12)):
            arr[r, c] = to_float(df.iloc[r, c])
    return arr

def mean_cv(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals: return np.nan, np.nan
    m = np.mean(vals)
    if len(vals) < 2 or m == 0: return m, np.nan
    return m, np.std(vals, ddof=1) / m * 100

def four_pl(x, A, B, C, D):
    return D + (A - D) / (1.0 + (x / C) ** B)

def inv_4pl(y, A, B, C, D):
    try:
        inner = (A - D) / (y - D) - 1.0
        return C * inner ** (1.0 / B) if inner > 0 else np.nan
    except Exception:
        return np.nan

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def fmt_cv(cv):
    return f"{cv:.1f}" if (cv is not None and not np.isnan(cv)) else "–"

ROW_LABELS = list("ABCDEFGH")
COL_LABELS  = [str(i) for i in range(1, 13)]

VALID_TYPES = ["leer", "blank", "S1","S2","S3","S4","S5","S6","S7","S8", "Probe"]

def default_plate_df():
    """Erstellt Standard-Layout als DataFrame mit Well-Bezeichnungen."""
    data = {}
    for c_idx, col in enumerate(COL_LABELS):
        col_vals = []
        for r_idx in range(8):
            if r_idx == 0 and c_idx < 2:
                col_vals.append("blank")
            elif 1 <= r_idx <= 7 and c_idx < 2:
                col_vals.append(f"S{r_idx}")
            elif c_idx >= 2:
                col_vals.append("Probe")
            else:
                col_vals.append("leer")
        data[col] = col_vals
    return pd.DataFrame(data, index=ROW_LABELS)

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 1 – Platteneditor
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · Plattendesign festlegen")

st.markdown("""
<div class="info-box">
Klicke auf eine Zelle und wähle aus dem Dropdown: <b>blank</b>, <b>S1–S8</b>, <b>Probe</b> oder <b>leer</b>.
Ziehen über mehrere Zellen funktioniert ebenfalls.
</div>
""", unsafe_allow_html=True)

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    if st.button("📋 Voreinstellung laden"):
        st.session_state.plate_df = default_plate_df()
        st.rerun()
with col_btn2:
    if st.button("🗑️ Alles leeren"):
        st.session_state.plate_df = pd.DataFrame(
            "leer", index=ROW_LABELS, columns=COL_LABELS)
        st.rerun()

if "plate_df" not in st.session_state:
    st.session_state.plate_df = default_plate_df()

# Farbkodierung für die Vorschau
def colorize(val):
    if val == "blank":
        return "background-color:#c0bdb4; color:#333;"
    if val and val.startswith("S"):
        greens = {
            "S1":"#064e3b","S2":"#065f46","S3":"#047857","S4":"#059669",
            "S5":"#10b981","S6":"#34d399","S7":"#6ee7b7","S8":"#a7f3d0",
        }
        bg = greens.get(val, "#059669")
        fg = "white" if val in ("S1","S2","S3","S4") else "#064e3b"
        return f"background-color:{bg}; color:{fg};"
    if val == "Probe":
        return "background-color:#85B7EB; color:#0C447C;"
    return "background-color:#f0f0ee; color:#aaa;"

# Dropdown-Optionen für jede Spalte
column_config = {
    col: st.column_config.SelectboxColumn(
        col,
        options=VALID_TYPES,
        required=True,
        width="small",
    )
    for col in COL_LABELS
}

edited_df = st.data_editor(
    st.session_state.plate_df,
    column_config=column_config,
    use_container_width=True,
    hide_index=False,
    key="plate_editor",
)
st.session_state.plate_df = edited_df

# Farbige Vorschau
st.markdown("**Vorschau:**")
st.dataframe(
    edited_df.style.applymap(colorize),
    use_container_width=True,
    hide_index=False,
)

# Layout auslesen
plate_map = {}   # well -> typ  ("blank" | "S1".."S8" | "Probe" | "leer")
for r_idx, row in enumerate(ROW_LABELS):
    for c_idx, col in enumerate(COL_LABELS):
        well = f"{row}{c_idx+1}"
        plate_map[well] = edited_df.at[row, col]

blanks    = [w for w, t in plate_map.items() if t == "blank"]
probe_wells = [w for w, t in plate_map.items() if t == "Probe"]
std_map   = {}   # "S1" -> [wells]
for w, t in plate_map.items():
    if t.startswith("S") and t[1:].isdigit():
        std_map.setdefault(t, []).append(w)
std_keys_sorted = sorted(std_map.keys(), key=lambda s: int(s[1:]))

# Zusammenfassung
s1, s2, s3 = st.columns(3)
s1.markdown(f"**Blank:** {len(blanks)} Wells – {', '.join(blanks) if blanks else '–'}")
s2.markdown(f"**Standards:** {', '.join([f'{k}: {len(v)} Wells' for k,v in sorted(std_map.items(), key=lambda x: int(x[0][1:]))])  if std_map else '–'}")
s3.markdown(f"**Proben:** {len(probe_wells)} Wells")

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 2 – Konzentrationen
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 2 · Konzentrationen der Standards")

if not std_map:
    st.markdown('<div class="warn-box">⚠️ Noch keine Standards auf der Platte markiert.</div>',
                unsafe_allow_html=True)
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    conc_unit = st.text_input("Einheit", value="ng/mL")
with c2:
    conc_top = st.number_input("Höchste Konzentration (S1)", value=25.0, min_value=0.0001, format="%.4f")
with c3:
    use_serial = st.radio("Eingabe", ["Serielle Verdünnung", "Manuell"], horizontal=True)
with c4:
    if use_serial == "Serielle Verdünnung":
        dil_factor = st.number_input("Verdünnungsfaktor", value=2.0, min_value=1.001, format="%.2f")
    else:
        dil_factor = None

n_s = len(std_keys_sorted)
defaults = [25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.195]

if use_serial == "Serielle Verdünnung":
    conc_dict = {sk: conc_top / (dil_factor ** i) for i, sk in enumerate(std_keys_sorted)}
else:
    conc_dict = {}
    man_cols = st.columns(n_s)
    for i, sk in enumerate(std_keys_sorted):
        with man_cols[i]:
            conc_dict[sk] = st.number_input(
                sk, value=defaults[i] if i < len(defaults) else 1.0,
                format="%.5f", key=f"conc_{sk}")

preview = "  →  ".join([f"{sk}: {conc_dict[sk]:.4g} {conc_unit}" for sk in std_keys_sorted])
st.markdown(
    f"<div style='font-family:IBM Plex Mono,monospace;font-size:13px;"
    f"background:white;border:1px solid #e0e0e0;border-radius:4px;"
    f"padding:8px 12px;margin-bottom:1rem;'>📐 {preview}</div>",
    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 3 – OD-Datei hochladen
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 3 · OD-Rohdaten hochladen")
st.markdown("""
<div class="info-box">
Lade die Rohdaten deines Plattenlesers: eine <b>8 Zeilen × 12 Spalten</b> Tabelle.
Keine Beschriftungen – nur die 96 OD-Zahlenwerte.
</div>
""", unsafe_allow_html=True)

ex = np.array([
    [0.082,0.085,0.41, 0.38,0.72,0.69,1.05,1.08,0.55,0.52,0.88,0.91],
    [1.78, 1.81, 0.33, 0.31,0.63,0.61,0.95,0.98,0.44,0.47,0.77,0.74],
    [1.55, 1.52, 0.29, 0.27,0.58,0.55,0.88,0.85,0.38,0.41,0.68,0.71],
    [1.10, 1.13, 0.25, 0.23,0.51,0.49,0.79,0.76,0.31,0.33,0.59,0.62],
    [0.70, 0.73, 0.21, 0.19,0.44,0.42,0.70,0.68,0.22,0.24,0.48,0.51],
    [0.40, 0.42, 0.17, 0.16,0.37,0.35,0.61,0.59,0.15,0.17,0.38,0.40],
    [0.22, 0.21, 0.14, 0.13,0.30,0.28,0.52,0.50,0.11,0.12,0.29,0.31],
    [0.12, 0.11, 0.11, 0.10,0.23,0.22,0.43,0.41,0.08,0.09,0.20,0.22],
])
buf_ex = BytesIO()
pd.DataFrame(ex).to_excel(buf_ex, index=False, header=False, engine="openpyxl")
st.download_button("📥 Beispiel-Excel herunterladen", buf_ex.getvalue(),
                   "elisa_beispiel.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

uploaded = st.file_uploader("CSV oder Excel (8×12)", type=["csv","xlsx"])
if uploaded is None:
    st.stop()

raw_df = read_file(uploaded)
if raw_df is None or raw_df.shape[0] < 8 or raw_df.shape[1] < 12:
    st.markdown(
        f'<div class="err-box">❌ Datei muss 8 Zeilen × 12 Spalten haben. '
        f'Gefunden: {getattr(raw_df,"shape",("?","?"))[0]} × '
        f'{getattr(raw_df,"shape",("?","?"))[1]}.</div>', unsafe_allow_html=True)
    st.stop()

plate_od = parse_plate(raw_df)

with st.expander("🔍 Plattenvorschau (Rohdaten)"):
    prev = pd.DataFrame(plate_od, index=ROW_LABELS, columns=COL_LABELS)
    st.dataframe(prev.style.format("{:.4f}").background_gradient(cmap="YlOrRd", axis=None),
                 use_container_width=True)

def well_od(well):
    r = ROW_LABELS.index(well[0])
    c = int(well[1:]) - 1
    return plate_od[r, c]

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 4 – 4PL Fit
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 4 · Standardkurve & 4PL-Fit")

blank_ods = [well_od(w) for w in blanks]
blank_mean, blank_cv = mean_cv(blank_ods)

if not np.isnan(blank_cv) and blank_cv > 15:
    st.markdown(f'<div class="warn-box">⚠️ Blank CV = {blank_cv:.1f}% – Wells: {", ".join(blanks)}</div>',
                unsafe_allow_html=True)
else:
    vals_str = ", ".join([f"{v:.4f}" for v in blank_ods])
    cv_str = f"  |  CV: {fmt_cv(blank_cv)}%" if len(blank_ods) > 1 else ""
    st.markdown(f'<div class="info-box">Blank Ø = {blank_mean:.4f}  ({vals_str}){cv_str}</div>',
                unsafe_allow_html=True)

std_records = []
for sk in std_keys_sorted:
    wells = sorted(std_map[sk])
    ods   = [well_od(w) for w in wells]
    raw_m, cv = mean_cv(ods)
    od_corr = raw_m - blank_mean
    std_records.append({
        "Standard":                     sk,
        "Wells":                        ", ".join(wells),
        f"Konzentration [{conc_unit}]": conc_dict[sk],
        "OD Werte":                     ", ".join([f"{v:.4f}" for v in ods]),
        "OD roh (Ø)":                   round(raw_m, 4),
        "CV [%]":                       fmt_cv(cv),
        "OD (blank-subt.)":             round(od_corr, 4),
    })

df_std = pd.DataFrame(std_records)
x_all = df_std[f"Konzentration [{conc_unit}]"].values.astype(float)
y_all = df_std["OD (blank-subt.)"].values.astype(float)
mask  = y_all > 0

if mask.sum() < 4:
    st.markdown('<div class="err-box">❌ Weniger als 4 positive OD-Werte nach Blank-Subtraktion.</div>',
                unsafe_allow_html=True)
    st.stop()

x_fit, y_fit = x_all[mask], y_all[mask]

try:
    popt, _ = curve_fit(
        four_pl, x_fit, y_fit,
        p0=[y_fit.min(), 1.5, np.median(x_fit), y_fit.max()],
        bounds=([0, 0.1, x_fit.min()*0.001, 0],
                [y_fit.max()*3, 10, x_fit.max()*1000, y_fit.max()*3]),
        maxfev=30000)
    A, B, C, D = popt
    y_pred_fit = four_pl(x_fit, *popt)
    r2 = r2_score(y_fit, y_pred_fit)
except RuntimeError:
    st.markdown('<div class="err-box">❌ 4PL-Fit konvergiert nicht.</div>', unsafe_allow_html=True)
    st.stop()

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#f5f4f0"); ax.set_facecolor("white")
x_plot = np.logspace(np.log10(x_fit.min()*0.4), np.log10(x_fit.max()*2.0), 500)
y_plot = four_pl(x_plot, *popt)
ax.plot(x_plot, y_plot, color="#1a1a1a", lw=2, label="4PL-Fit", zorder=3)
ax.scatter(x_fit, y_fit, color="#059669", s=70, zorder=4,
           edgecolors="white", lw=0.8, label="Standards (blank-subt.)")
for xi, yi, yp in zip(x_fit, y_fit, y_pred_fit):
    ax.plot([xi, xi], [yi, yp], color="#ef4444", lw=0.8, alpha=0.5, zorder=2)
ax.set_xscale("log")
ax.set_xlabel(f"Konzentration [{conc_unit}]", fontsize=11, fontfamily="monospace")
ax.set_ylabel("OD (blank-subtrahiert)", fontsize=11, fontfamily="monospace")
ax.set_title(f"4PL-Standardkurve   R² = {r2:.5f}", fontsize=12,
             fontfamily="monospace", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, which="both", ls="--", alpha=0.3)
for sp in ax.spines.values(): sp.set_color("#e0e0e0")
st.pyplot(fig); plt.close(fig)

st.markdown("### 4PL-Parameter")
pc = st.columns(5)
for col, lbl, val in zip(pc,
    ["A – Unteres Plateau","B – Hill-Koeff.",f"C – EC50 [{conc_unit}]","D – Oberes Plateau","R²"],
    [f"{A:.4f}",f"{B:.4f}",f"{C:.4f}",f"{D:.4f}",f"{r2:.6f}"]):
    col.markdown(f'<div class="param-card"><div class="param-label">{lbl}</div>'
                 f'<div class="param-value">{val}</div></div>', unsafe_allow_html=True)

if r2 < 0.99:
    st.markdown(f'<div class="warn-box">⚠️ R² = {r2:.4f} unter 0.99 – Ausreißer prüfen.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="info-box">✅ R² = {r2:.5f} – guter Fit.</div>', unsafe_allow_html=True)

with st.expander("📋 Standardkurve Details"):
    st.dataframe(df_std, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 5 – Proben
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 5 · Proben")
st.markdown(f'<div class="info-box">{len(probe_wells)} Proben-Wells aus dem Plattendesign.</div>',
            unsafe_allow_html=True)

probe_records = []
for well in sorted(probe_wells, key=lambda w: (w[0], int(w[1:]))):
    od_raw  = well_od(well)
    od_corr = od_raw - blank_mean
    in_range = A < od_corr < D
    conc = inv_4pl(od_corr, A, B, C, D) if in_range else np.nan
    flag = ("⚠️ unter Plateau" if od_corr <= A else
            "⚠️ über Plateau"  if od_corr >= D else "")
    probe_records.append({
        "Well":                         well,
        "OD roh":                       round(od_raw, 4),
        "OD (blank-subt.)":             round(od_corr, 4),
        f"Konzentration [{conc_unit}]": round(conc, 4) if not np.isnan(conc) else "–",
        "Hinweis":                      flag,
    })

df_probes = pd.DataFrame(probe_records)

out_range = df_probes[df_probes["Hinweis"] != ""]
if len(out_range) > 0:
    st.markdown(f'<div class="warn-box">⚠️ {len(out_range)} Well(s) außerhalb Kurvenbereich: '
                f'{", ".join(out_range["Well"].tolist())}</div>', unsafe_allow_html=True)

st.dataframe(df_probes, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor("#f5f4f0"); ax2.set_facecolor("white")
ax2.plot(x_plot, y_plot, color="#1a1a1a", lw=2, zorder=3, label="4PL-Fit")
ax2.scatter(x_fit, y_fit, color="#059669", s=50, zorder=4, edgecolors="white", lw=0.8, label="Standards")
first = True
for _, row in df_probes.iterrows():
    cv = row[f"Konzentration [{conc_unit}]"]
    if cv != "–":
        c_val, o_val = float(cv), float(row["OD (blank-subt.)"])
        ax2.plot([c_val],[o_val], marker="D", color="#2563eb", ms=5, zorder=5,
                 label="Probe" if first else "")
        ax2.plot([c_val,c_val],[o_val,0], color="#2563eb", lw=0.5, ls=":", alpha=0.35)
        first = False
ax2.set_xscale("log")
ax2.set_xlabel(f"Konzentration [{conc_unit}]", fontsize=11, fontfamily="monospace")
ax2.set_ylabel("OD (blank-subtrahiert)", fontsize=11, fontfamily="monospace")
ax2.set_title("Rückrechnung der Proben", fontsize=12, fontfamily="monospace", fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, which="both", ls="--", alpha=0.3)
for sp in ax2.spines.values(): sp.set_color("#e0e0e0")
st.pyplot(fig2); plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# SCHRITT 6 – Export
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 6 · Export")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    df_probes.to_excel(writer, sheet_name="Proben", index=False)
    df_std.to_excel(writer, sheet_name="Standardkurve", index=False)
    pd.DataFrame([
        {"Parameter":"A (Unteres Plateau)",     "Wert":round(A,6)},
        {"Parameter":"B (Hill-Koeffizient)",     "Wert":round(B,6)},
        {"Parameter":f"C (EC50 [{conc_unit}])", "Wert":round(C,6)},
        {"Parameter":"D (Oberes Plateau)",       "Wert":round(D,6)},
        {"Parameter":"R²",                       "Wert":round(r2,8)},
        {"Parameter":"Blank Ø OD",               "Wert":round(blank_mean,4)},
        {"Parameter":"Blank Wells",              "Wert":", ".join(blanks)},
    ]).to_excel(writer, sheet_name="Parameter", index=False)

st.download_button("📥 Ergebnisse als Excel herunterladen",
                   buf.getvalue(), "elisa_auswertung.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#aaa;'>"
            "ELISA 4PL Tool · Nur für Forschungszwecke · Kein Medizinprodukt</p>",
            unsafe_allow_html=True)
