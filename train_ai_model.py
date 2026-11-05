import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

np.random.seed(42)
N_SAMPLES = 800

print("1. Generating Ground-Truth Multimodal Agricultural Dataset...")
data = []
for _ in range(N_SAMPLES):
    condition = np.random.choice(["Healthy", "Water_Stress", "Nutrient_Stress", "Foliar_Stress"])
    temp = np.random.uniform(22, 34)
    ph   = np.random.uniform(6.0, 7.5)

    if condition == "Healthy":
        moist  = np.random.uniform(45, 65)
        ec     = np.random.uniform(300, 700)
        n_val  = np.random.uniform(40, 70)
        p_val  = np.random.uniform(30, 50)
        k_val  = np.random.uniform(60, 90)
        canopy = np.random.uniform(60, 85)
        exg    = np.random.uniform(0.18, 0.35)
        yellow = np.random.uniform(0, 5)
        label  = 0
    elif condition == "Water_Stress":
        moist  = np.random.uniform(10, 24) # low moisture
        ec     = np.random.uniform(650, 1100)
        n_val  = np.random.uniform(35, 60)
        p_val  = np.random.uniform(25, 45)
        k_val  = np.random.uniform(50, 80)
        canopy = np.random.uniform(35, 55) # wilting reduces canopy area
        exg    = np.random.uniform(0.08, 0.16)
        yellow = np.random.uniform(5, 15)
        label  = 1
    elif condition == "Nutrient_Stress":
        moist  = np.random.uniform(40, 60) # moisture normal
        ec     = np.random.uniform(100, 250)
        n_val  = np.random.uniform(5, 20)  # NPK severely depleted
        p_val  = np.random.uniform(5, 15)
        k_val  = np.random.uniform(10, 30)
        canopy = np.random.uniform(45, 65)
        exg    = np.random.uniform(0.05, 0.14) # chlorosis/pale leaves
        yellow = np.random.uniform(15, 35)
        label  = 2
    else: # Foliar_Stress (Heat / Leaf Disease - soil is fine, but crop looks damaged)
        moist  = np.random.uniform(45, 65)
        ec     = np.random.uniform(350, 600)
        n_val  = np.random.uniform(40, 65)
        p_val  = np.random.uniform(30, 48)
        k_val  = np.random.uniform(55, 85)
        canopy = np.random.uniform(35, 55)
        exg    = np.random.uniform(0.02, 0.10)
        yellow = np.random.uniform(30, 65) # high yellowing/necrosis
        label  = 3

    data.append([moist, temp, ec, ph, n_val, p_val, k_val, canopy, exg, yellow, label])

cols = ["moisture", "temp", "ec", "ph", "N", "P", "K", "canopy", "exg", "yellow", "label"]
df = pd.DataFrame(data, columns=cols)
df.to_csv("/home/deek/Projects/smart_agriculture/dataset.csv", index=False)
print(f"✅ Dataset generated and saved (Shape: {df.shape})")

# Feature Splits
soil_cols = ["moisture", "temp", "ec", "ph", "N", "P", "K"]
crop_cols = ["canopy", "exg", "yellow"]
all_cols  = soil_cols + crop_cols

X_soil = df[soil_cols]
X_crop = df[crop_cols]
X_all  = df[all_cols]
y      = df["label"]

X_tr_s, X_te_s, y_train, y_test = train_test_split(X_soil, y, test_size=0.25, random_state=42)
X_tr_c, X_te_c, _, _ = train_test_split(X_crop, y, test_size=0.25, random_state=42)
X_tr_a, X_te_a, _, _ = train_test_split(X_all,  y, test_size=0.25, random_state=42)

print("\n2. Training the 4 Models...")
# Model 1: Soil-Only
m_soil = RandomForestClassifier(n_estimators=60, random_state=42).fit(X_tr_s, y_train)
acc_s = accuracy_score(y_test, m_soil.predict(X_te_s))
f1_s  = f1_score(y_test, m_soil.predict(X_te_s), average="weighted")

# Model 2: Crop-Only
m_crop = RandomForestClassifier(n_estimators=60, random_state=42).fit(X_tr_c, y_train)
acc_c = accuracy_score(y_test, m_crop.predict(X_te_c))
f1_c  = f1_score(y_test, m_crop.predict(X_te_c), average="weighted")

# Model 3: Simple Concatenation Fusion
m_all = RandomForestClassifier(n_estimators=60, random_state=42).fit(X_tr_a, y_train)
acc_a = accuracy_score(y_test, m_all.predict(X_te_a))
f1_a  = f1_score(y_test, m_all.predict(X_te_a), average="weighted")

# Model 4: Proposed Intelligent Weighted Decision Fusion
# Soil is authoritative on Moisture & Nutrients; Crop is authoritative on Foliar status
p_soil = m_soil.predict_proba(X_te_s)
p_crop = m_crop.predict_proba(X_te_c)
p_fused = 0.55 * p_soil + 0.45 * p_crop
y_pred_fused = np.argmax(p_fused, axis=1)
acc_f = accuracy_score(y_test, y_pred_fused)
f1_f  = f1_score(y_test, y_pred_fused, average="weighted")

print("\n" + "="*60)
print("📈 EXPERIMENTAL RESULTS (FOR FINAL REPORT & PPT)")
print("="*60)
res = pd.DataFrame({
    "Model Architecture": ["1. Soil-Only", "2. Crop-Only", "3. Simple Concatenation", "4. Proposed Intelligent Fusion"],
    "Accuracy (%)": [acc_s*100, acc_c*100, acc_a*100, acc_f*100],
    "Weighted F1-Score": [f1_s, f1_c, f1_a, f1_f]
})
print(res.to_string(index=False))

# Save publication comparison plot
plt.figure(figsize=(8.5, 4.8))
colors = ['#4A90E2', '#F5A623', '#7ED321', '#9013FE']
bars = plt.bar(res["Model Architecture"], res["Accuracy (%)"], color=colors, edgecolor='black', width=0.55)
plt.ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold')
plt.title("Comparative Performance Across Sensing Modalities & Fusion", fontsize=12, fontweight='bold')
plt.ylim(60, 102)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2.0, h + 1, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig("/home/deek/Projects/smart_agriculture/ai_model_comparison.png", dpi=300)
print("\n✅ High-Resolution Plot saved to: /home/deek/Projects/smart_agriculture/ai_model_comparison.png")
