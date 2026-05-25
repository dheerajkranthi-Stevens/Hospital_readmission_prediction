"""
Hospital Readmission Prediction (Diabetes 130-US Hospitals 1999-2008)
Author: Dheeraj Kranthi
GitHub Project 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "/mnt/user-data/uploads/1779752866227_diabetic_data.csv"
OUTPUT_DIR = "/mnt/user-data/outputs"

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#16A34A",
    "accent":    "#DC2626",
    "neutral":   "#6B7280",
    "bg":        "#F8FAFC",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("HOSPITAL READMISSION PREDICTION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────
print("\n[1/5] Cleaning data...")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop high-missing & low-value columns
df.drop(columns=['weight', 'payer_code', 'medical_specialty',
                  'encounter_id', 'patient_nbr'], inplace=True)

# Drop rows with missing race/gender
df.dropna(subset=['race', 'gender'], inplace=True)

# Fill max_glu_serum and A1Cresult with 'None' (clinical meaning: not tested)
df['max_glu_serum'].fillna('None', inplace=True)
df['A1Cresult'].fillna('None', inplace=True)

# Remove invalid gender
df = df[df['gender'] != 'Unknown/Invalid']

# Remove expired / hospice discharge (not relevant for readmission)
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

# ─── Target: binary readmission (readmitted within 30 days = 1)
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)

print(f"   After cleaning: {df.shape[0]:,} rows")
print(f"   Readmitted <30 days: {df['readmitted_binary'].mean()*100:.1f}%")

# ─────────────────────────────────────────────
# 3. EDA VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[2/5] Generating EDA charts...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Hospital Readmission — Exploratory Data Analysis",
             fontsize=18, fontweight='bold', y=0.98)
fig.patch.set_facecolor(PALETTE["bg"])

# ── Chart 1: Readmission distribution
ax = axes[0, 0]
counts = df['readmitted'].value_counts()
colors = [PALETTE["accent"], PALETTE["neutral"], PALETTE["secondary"]]
bars = ax.bar(counts.index, counts.values, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
ax.set_title("Readmission Distribution", fontweight='bold', fontsize=13)
ax.set_xlabel("Readmitted")
ax.set_ylabel("Patient Count")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10)

# ── Chart 2: Age group vs readmission rate
ax = axes[0, 1]
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
age_data = df.groupby('age')['readmitted_binary'].mean().reindex(age_order) * 100
ax.bar(range(len(age_data)), age_data.values, color=PALETTE["primary"],
       alpha=0.85, edgecolor='white', linewidth=1.2)
ax.set_xticks(range(len(age_data)))
ax.set_xticklabels(age_order, rotation=45, ha='right', fontsize=9)
ax.set_title("Readmission Rate by Age Group", fontweight='bold', fontsize=13)
ax.set_ylabel("Readmission Rate (%)")
ax.axhline(df['readmitted_binary'].mean()*100, color=PALETTE["accent"],
           linestyle='--', linewidth=1.5, label='Overall avg')
ax.legend(fontsize=9)

# ── Chart 3: Time in hospital distribution
ax = axes[0, 2]
ax.hist(df['time_in_hospital'], bins=14, color=PALETTE["secondary"],
        alpha=0.85, edgecolor='white', linewidth=1.2)
ax.set_title("Length of Hospital Stay", fontweight='bold', fontsize=13)
ax.set_xlabel("Days in Hospital")
ax.set_ylabel("Count")
ax.axvline(df['time_in_hospital'].mean(), color=PALETTE["accent"],
           linestyle='--', linewidth=2, label=f'Mean: {df["time_in_hospital"].mean():.1f} days')
ax.legend(fontsize=10)

# ── Chart 4: Num medications vs readmission
ax = axes[1, 0]
readmitted_meds = df[df['readmitted_binary'] == 1]['num_medications']
not_readmitted_meds = df[df['readmitted_binary'] == 0]['num_medications']
ax.hist(not_readmitted_meds, bins=30, alpha=0.6, color=PALETTE["secondary"],
        label='Not Readmitted', density=True)
ax.hist(readmitted_meds, bins=30, alpha=0.6, color=PALETTE["accent"],
        label='Readmitted <30d', density=True)
ax.set_title("Medications: Readmitted vs Not", fontweight='bold', fontsize=13)
ax.set_xlabel("Number of Medications")
ax.set_ylabel("Density")
ax.legend(fontsize=10)

# ── Chart 5: Insulin usage breakdown
ax = axes[1, 1]
insulin_read = df.groupby('insulin')['readmitted_binary'].mean() * 100
insulin_read = insulin_read.sort_values(ascending=False)
bars = ax.bar(insulin_read.index, insulin_read.values,
              color=[PALETTE["primary"], PALETTE["secondary"],
                     PALETTE["accent"], PALETTE["neutral"]][:len(insulin_read)],
              edgecolor='white', linewidth=1.5)
ax.set_title("Readmission Rate by Insulin Usage", fontweight='bold', fontsize=13)
ax.set_xlabel("Insulin Dosage Change")
ax.set_ylabel("Readmission Rate (%)")
for bar, val in zip(bars, insulin_read.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# ── Chart 6: Number of inpatient visits
ax = axes[1, 2]
inpatient_rate = df.groupby('number_inpatient')['readmitted_binary'].mean() * 100
inpatient_rate = inpatient_rate[inpatient_rate.index <= 10]
ax.plot(inpatient_rate.index, inpatient_rate.values, 'o-',
        color=PALETTE["primary"], linewidth=2.5, markersize=8, markerfacecolor='white',
        markeredgewidth=2)
ax.fill_between(inpatient_rate.index, inpatient_rate.values,
                alpha=0.15, color=PALETTE["primary"])
ax.set_title("Prior Inpatient Visits → Readmission Risk", fontweight='bold', fontsize=13)
ax.set_xlabel("Number of Prior Inpatient Visits")
ax.set_ylabel("Readmission Rate (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ EDA dashboard saved")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING + MODEL
# ─────────────────────────────────────────────
print("\n[3/5] Preparing features & training model...")

# Encode age as ordinal
age_map = {a: i for i, a in enumerate(
    ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])}
df['age_num'] = df['age'].map(age_map)

# Select features
FEATURES = [
    'age_num', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses',
    'race', 'gender', 'insulin', 'change', 'diabetesMed',
    'max_glu_serum', 'A1Cresult', 'metformin', 'glipizide', 'glyburide',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
]

X = df[FEATURES].copy()
y = df['readmitted_binary']

# Label encode categoricals
le = LabelEncoder()
cat_cols = X.select_dtypes(include='object').columns
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n   ── Model Performance ──")
print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted <30d']))
auc = roc_auc_score(y_test, y_proba)
print(f"   ROC-AUC Score: {auc:.4f}")

# ─────────────────────────────────────────────
# 5. MODEL PERFORMANCE CHARTS
# ─────────────────────────────────────────────
print("\n[4/5] Generating model performance charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Random Forest — Model Performance Dashboard",
             fontsize=17, fontweight='bold', y=1.02)
fig.patch.set_facecolor(PALETTE["bg"])

# ── Confusion Matrix
ax = axes[0]
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Not Readmitted', 'Readmitted <30d'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix", fontweight='bold', fontsize=13)

# ── ROC Curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color=PALETTE["primary"], linewidth=2.5,
        label=f'ROC Curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["primary"])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight='bold', fontsize=13)
ax.legend(fontsize=10)

# ── Feature Importance
ax = axes[2]
importances = pd.Series(model.feature_importances_, index=FEATURES)
top10 = importances.nlargest(10).sort_values()
colors_fi = [PALETTE["primary"] if v > top10.median() else PALETTE["neutral"]
             for v in top10.values]
ax.barh(top10.index, top10.values, color=colors_fi,
        edgecolor='white', linewidth=1.2)
ax.set_title("Top 10 Feature Importances", fontweight='bold', fontsize=13)
ax.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_performance.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Model performance dashboard saved")

# ─────────────────────────────────────────────
# 6. SUMMARY STATS FOR README
# ─────────────────────────────────────────────
print("\n[5/5] Key numbers for your README:")
print(f"   Dataset size:         {df.shape[0]:,} patient encounters")
print(f"   Features used:        {len(FEATURES)}")
print(f"   Readmission rate:     {df['readmitted_binary'].mean()*100:.1f}%")
print(f"   Model:                Random Forest (200 trees)")
print(f"   ROC-AUC:              {auc:.4f}")
from sklearn.metrics import accuracy_score, f1_score
print(f"   Accuracy:             {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"   F1 Score (minority):  {f1_score(y_test, y_pred):.4f}")
print(f"\n✅ All outputs saved to {OUTPUT_DIR}/")
print("=" * 60)
