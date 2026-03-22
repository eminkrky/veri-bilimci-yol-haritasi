# Katman B — Klasik Makine Öğrenmesi

> Bu katmanda ne öğrenilir: Problem framing'den model deploy'a kadar klasik ML sürecinin tamamı. Doğrusal modeller, ağaçlar, boosting, model değerlendirme, hyperparameter tuning, feature engineering ve yorumlanabilirlik.
>
> Süre: 2–4 hafta. Bu katman iş görüşmelerinin %70'ini kapsar.


<div class="prereq-box">
<strong>Önkoşul:</strong> <strong>Katman A (Temeller)</strong> tamamlanmış olmalı — Python/Pandas, SQL ve istatistik temel bilgisi gereklidir.
</div>

---

## B.1 Problem Framing (Çerçeveleme)

### Sezgisel Açıklama

ML kurulmadan önce sorulması gereken sorular: Bu iş sorunu mu, veri sorunu mu, operasyon sorunu mu? Model gerekiyor mu?

**Yanlış çerçeveleme örneği:** "Kullanıcılar neden ayrılıyor?" → Churn tahmini kur.
**Doğru soru:** "Churn tahmini sonucu ne yapacağız? Kimse aksiyon almayacaksa model işe yaramaz."

### Framing Checklist

```
1. İş sorusu nedir? (sayısal, ölçülebilir)
2. ML mi gerekiyor? (basit kural/sorgu yeterli mi?)
3. Tahmin sonucu hangi aksiyonu tetikleyecek?
4. Yanlış pozitif/negatif maliyeti ne?
5. Başarı kriteri nedir? (business metric)
6. Data dönemi ve kapsam?
7. Prediction latency gereksinimi? (batch? realtime?)
8. Model yorumlanabilir olmalı mı? (düzenlemeli sektör?)
```

### Kod Örneği — Maliyet Matrisi

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Churn modeli maliyet matrisi
# TP: Churner'ı yakaladık, kampanya gönder → +100 TL değer
# FP: Kalacak müşteriye kampanya gönderdik → -15 TL maliyet
# FN: Churner'ı kaçırdık → -80 TL kayıp
# TN: Doğru "kalacak" tahmini → 0 TL

cost_matrix = {
    'TP': 100, 'FP': -15,
    'FN': -80, 'TN': 0
}

def expected_profit(y_true, y_prob, threshold, cost_matrix):
    """Verilen threshold için beklenen kâr/zarar hesapla."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    profit = (tp * cost_matrix['TP'] + fp * cost_matrix['FP'] +
              fn * cost_matrix['FN'] + tn * cost_matrix['TN'])
    return profit

# Farklı threshold'lar için kâr analizi
np.random.seed(42)
n = 1000
y_true = np.random.binomial(1, 0.1, n)  # %10 churn
y_prob = np.where(y_true == 1,
                  np.random.beta(5, 2, n),   # Churners: yüksek olasılık
                  np.random.beta(1, 5, n))    # Non-churners: düşük olasılık

thresholds = np.linspace(0.05, 0.95, 50)
profits = [expected_profit(y_true, y_prob, t, cost_matrix) for t in thresholds]

optimal_idx = np.argmax(profits)
print(f"Optimal threshold: {thresholds[optimal_idx]:.2f}")
print(f"Maksimum kâr: {profits[optimal_idx]:,.0f} TL")

plt.plot(thresholds, profits)
plt.axvline(thresholds[optimal_idx], color='red', linestyle='--', label='Optimal')
plt.xlabel("Threshold"), plt.ylabel("Beklenen Kâr (TL)")
plt.title("Maliyet Matrisine Göre Optimal Threshold")
plt.legend()
```

> **Senior Notu:** Threshold seçimini AUC'ye değil iş etkisine göre yap. Aynı AUC'ye sahip iki model farklı maliyet matrislerinde çok farklı iş değeri üretir.

---

## B.2 Veri Bölme Stratejileri

### Sezgisel Açıklama

Bölme sırası çok önemli. Yanlış bölme = leakage = gerçekçi olmayan performans = prod'da hayal kırıklığı.

**Kural:** Preprocessing (scaler, encoder) sadece train'e fit, val/test'e sadece transform.

```python
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GroupKFold
import pandas as pd

# 1. Random split (zaman bağımsız veriler için)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify: sınıf oranı koru
)

# 2. Zaman bazlı split (zaman serisi, churn, gelir tahmini)
df = df.sort_values("date")
train_cutoff = pd.Timestamp("2023-12-31")
test_cutoff = pd.Timestamp("2024-03-31")

df_train = df[df["date"] <= train_cutoff]
df_val = df[(df["date"] > train_cutoff) & (df["date"] <= test_cutoff)]

# 3. Group K-Fold (aynı kullanıcı train ve testte olmasın)
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=user_ids)):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]

# 4. Walk-forward validation (zaman serisi için doğru CV)
# Expanding window: her adımda train penceresi büyür (tüm geçmiş kullanılır)
# Sliding window: train penceresi sabit boyutta kayar (eski veri düşer)

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# --- Expanding Window (varsayılan TimeSeriesSplit davranışı) ---
tss = TimeSeriesSplit(n_splits=5)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(tss.split(X)):
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    model_cv = LogisticRegression(max_iter=1000)
    model_cv.fit(X_tr, y_tr)
    y_prob_cv = model_cv.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, y_prob_cv)
    fold_metrics.append({"fold": fold, "train_size": len(train_idx),
                         "val_size": len(val_idx), "auc": auc})
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}, AUC={auc:.4f}")

print(f"\nOrtalama AUC: {np.mean([m['auc'] for m in fold_metrics]):.4f}")

# --- Sliding Window (sabit pencere boyutu) ---
window_size = len(X) // 3   # sabit train penceresi
step_size = len(X) // 10    # her adımda ilerleme miktarı
sliding_metrics = []

for start in range(0, len(X) - window_size - step_size, step_size):
    tr_idx = range(start, start + window_size)
    va_idx = range(start + window_size, min(start + window_size + step_size, len(X)))
    X_tr, X_va = X[list(tr_idx)], X[list(va_idx)]
    y_tr, y_va = y[list(tr_idx)], y[list(va_idx)]

    model_sw = LogisticRegression(max_iter=1000)
    model_sw.fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, model_sw.predict_proba(X_va)[:, 1])
    sliding_metrics.append(auc)

# Görselleştirme: fold bazlı performans trendi
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot([m["fold"] for m in fold_metrics], [m["auc"] for m in fold_metrics], "o-")
axes[0].set_xlabel("Fold"), axes[0].set_ylabel("AUC")
axes[0].set_title("Expanding Window — Walk-Forward CV")
axes[0].axhline(np.mean([m["auc"] for m in fold_metrics]), ls="--", color="red", label="Ortalama")
axes[0].legend()

axes[1].plot(range(len(sliding_metrics)), sliding_metrics, "s-", color="green")
axes[1].set_xlabel("Adım"), axes[1].set_ylabel("AUC")
axes[1].set_title("Sliding Window — Sabit Pencere CV")
axes[1].axhline(np.mean(sliding_metrics), ls="--", color="red", label="Ortalama")
axes[1].legend()
plt.tight_layout()
```

> **Expanding vs Sliding Window:** Expanding window tüm geçmişi kullanır — veri az olduğunda tercih edilir. Sliding window eski veriyi düşürür — veri dağılımı zamanla değişiyorsa (concept drift) daha uygundur.

---

## B.3 Model Ailesi ve Seçim

### Doğrusal Modeller

```python
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Logistic Regression — binary sınıflandırma baseline
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])
pipe_lr.fit(X_train, y_train)

# Ridge (L2) — regresyon için düzenleme
# Lasso (L1) — feature seçimi yapar (bazı katsayıları sıfırlar)
# ElasticNet — L1+L2 karışımı, yüksek boyutlu veri için
```

### Gradient Boosting — Derinlemesine

### Sezgisel Açıklama

Boosting = sıralı modeller zinciri. Her model, öncekinin hatalarını öğrenir. XGBoost ve LightGBM bu fikrin optimize edilmiş versiyonları.

**LightGBM vs XGBoost:**
- LightGBM: leaf-wise büyüme → aynı ağaç derinliğinde daha iyi kayıp azaltımı → büyük veriyle daha hızlı
- XGBoost: level-wise büyüme → daha stabil, küçük-orta veri için iyi
- CatBoost: kategorik feature encoding otomatik, leakage riski düşük, az parametre ayarı

```python
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np

# LightGBM — temel kullanım
lgb_params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,          # 2^max_depth - 1 idealdir
    "min_child_samples": 20,   # Overfitting önler
    "subsample": 0.8,          # Her ağaç için veri alt örnekleme
    "colsample_bytree": 0.8,   # Her ağaç için feature alt örnekleme
    "reg_alpha": 0.1,          # L1 düzenleme
    "reg_lambda": 1.0,         # L2 düzenleme
    "class_weight": "balanced",  # Dengesiz veri için
    "random_state": 42,
    "n_jobs": -1,
}

model = lgb.LGBMClassifier(**lgb_params)

# Early stopping ile fit
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print(f"Best iteration: {model.best_iteration_}")
```

### Hyperparameter Tuning — Optuna

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5,
                            scoring="roc_auc", n_jobs=-1).mean()
    return score

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

> **Senior Notu:** Grid search çok yavaş. Random search genelde Grid'den iyi. Optuna (Bayesian optimization) en iyi. n_trials=50 çoğu zaman yeterli. Daha da önemli: hyperparameter tuning'den önce feature engineering yap — yanlış feature ile model hiç iyi olmaz.

---

## B.4 Değerlendirme Metrikleri

### Sınıflandırma Metrikleri

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, calibration_curve
)
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_prob):.4f}")
print("\n" + classification_report(y_test, y_pred))

# ROC vs PR karşılaştırması — imbalanced veri için PR daha bilgilendirici
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
ax1.plot([0,1],[0,1], "k--")
ax1.set_xlabel("FPR"), ax1.set_ylabel("TPR")
ax1.set_title("ROC Curve"), ax1.legend()

# PR Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax2.plot(rec, prec, label=f"AP={average_precision_score(y_test, y_prob):.3f}")
ax2.axhline(y_test.mean(), color="k", linestyle="--", label="Baseline")
ax2.set_xlabel("Recall"), ax2.set_ylabel("Precision")
ax2.set_title("PR Curve"), ax2.legend()

plt.tight_layout()
```

### Kalibrasyon (Calibration)

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Kalibrasyon grafiği
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--", label="Mükemmel kalibrasyon")
plt.plot(prob_pred, prob_true, "o-", label="Model")
plt.xlabel("Tahmin edilen olasılık")
plt.ylabel("Gerçek oran")
plt.title("Kalibrasyon Grafiği (Reliability Diagram)")
plt.legend()

# Kalibrasyon düzeltme (gerekirse)
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)
y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
```

> **Senior Notu:** Model skorunu "olasılık" olarak kullanıyorsan kalibrasyon şart. LightGBM/XGBoost genelde iyi kalibre değil. Platt scaling (sigmoid) veya isotonic regression ile düzelt. Özellikle fraud ve churn modellerinde skorun gerçek olasılık anlamı taşıması kritik.

---

## B.5 Dengesiz Veri ile Sınıflandırma (Imbalanced Learning)

### Sezgisel Açıklama

1000 hastadan 10'u gerçekten hasta olsun. Bir model herkese "sağlam" derse %99 doğruluk (accuracy) elde eder — ama hiçbir hastayı yakalayamaz. Accuracy'nin yanıltıcı olduğu bu tür problemlerde özel teknikler gerekir.

**Dengesiz veri nerede karşımıza çıkar?**
- Fraud detection (%0.1–1 fraud)
- Churn prediction (%5–15 churn)
- Hastalık tanısı (%1–5 pozitif)
- Anomali tespiti (%0.01–0.5 anomali)

### 1. class_weight Ayarlama — İlk Adım

```python
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# scikit-learn: class_weight="balanced" otomatik ağırlık hesaplar
# ağırlık = n_samples / (n_classes * np.bincount(y))
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(X_train, y_train)

# LightGBM: is_unbalance veya scale_pos_weight
lgb_model = lgb.LGBMClassifier(
    is_unbalance=True,        # Otomatik ağırlıklama
    # veya manuel: scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    n_estimators=500,
    learning_rate=0.05,
    random_state=42
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)
```

### 2. SMOTE vs ADASYN — Oversampling

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# SMOTE: azınlık sınıfından yapay örnekler üretir (k-NN interpolasyonu)
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"SMOTE sonrası: {np.bincount(y_res)}")

# ADASYN: öğrenmesi zor örneklere yakın daha fazla yapay örnek üretir
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
print(f"ADASYN sonrası: {np.bincount(y_ada)}")

# En iyi pratik: imblearn Pipeline ile SMOTE + model birlikte
pipe_smote = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])
pipe_smote.fit(X_train, y_train)
y_pred_smote = pipe_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote))

# Karışım: SMOTE + Undersampling (dengeli yaklaşım)
pipe_combo = ImbPipeline([
    ("over", SMOTE(sampling_strategy=0.3, random_state=42)),
    ("under", RandomUnderSampler(sampling_strategy=0.6, random_state=42)),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])
pipe_combo.fit(X_train, y_train)
```

### 3. Random Undersampling Ne Zaman Tercih Edilir?

- Veri çok büyükse (milyonlarca satır) — eğitim süresini kısaltır
- Çoğunluk sınıfı gürültülüyse — gürültüyü temizleme etkisi
- Model zaten güçlüyse (ensemble) ve veri kaybı tolere edilebiliyorsa
- **Dikkat:** Küçük veri setlerinde undersampling bilgi kaybına yol açar

### 4. Focal Loss — Deep Learning'de Dengesiz Veri

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss: kolay örneklerin ağırlığını düşürür,
    zor örneklere odaklanır. Dengesiz veri için CrossEntropy'den üstün.

    alpha: sınıf dengeleme ağırlığı (azınlık sınıfı için yüksek)
    gamma: odaklanma parametresi (gamma=0 → standart CE)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)  # doğru sınıfın olasılığı
    loss = alpha * (1 - p_t) ** gamma * bce
    return loss.mean()
```

### Karşılaştırma Tablosu

| Yöntem | Avantaj | Dezavantaj | Ne Zaman Kullan |
|--------|---------|------------|-----------------|
| **class_weight** | Basit, ek veri üretmez, her modelde var | İnce ayar sınırlı | **İlk denenmesi gereken yöntem** |
| **SMOTE** | Yapay azınlık örnekleri üretir, bilgi artırır | Gürültüyü de çoğaltabilir, yüksek boyutta sorunlu | Tabular veri, orta boy veri setleri |
| **ADASYN** | Zor örneklere odaklanır | SMOTE'tan daha fazla gürültü riski | Karar sınırı karmaşıksa |
| **Random Undersampling** | Hızlı, büyük veri için uygun | Bilgi kaybı | Çok büyük veri setleri |
| **Focal Loss** | Zor örneklere otomatik odaklanır | Sadece DL modelleri için | Deep learning + dengesiz veri |
| **Ensemble (Balanced)** | Birden fazla dengeli alt küme kullanır | Eğitim süresi artar | Prod-grade sistemler |

> **Senior Notu:** Önce `class_weight="balanced"` dene — çoğu durumda yeterli. SMOTE son çare olarak düşün; çok büyük veri setlerinde gereksiz karmaşıklık ekler. **Zaman serisi veride SMOTE KULLANMA** — yapay örnekler temporal yapıyı bozar, leakage yaratır. Zaman serisinde class_weight veya cost-sensitive learning tercih et.

---

## B.6 Feature Engineering

### Leakage (Veri Sızıntısı) — En Kritik Konu

### Sezgisel Açıklama

Leakage = modelin eğitimde test zamanında bilmemesi gereken bilgiyi kullanması. Eğitimde harika performans, prod'da sıfır değer.

```
Churn probleminde doğru pencereler:

Gözlem zamanı = t
Feature window: [t-60, t]     ← bu süre içindeki geçmiş davranış
Label window: (t, t+30]       ← bu sürede churn mu?

YANLIŞ: last_login_date = t - 1 gün (target'ı neredeyse açıklıyor)
DOĞRU: t-7 günden önceki son login kullanılmalı
```

```python
# Leakage kontrol fonksiyonu
def check_for_leakage(df: pd.DataFrame, target_col: str,
                       observation_date_col: str = "obs_date"):
    """Basit leakage kontrolü — feature-target korelasyonu."""
    import pandas as pd
    num_cols = df.select_dtypes(include="number").columns
    corrs = {}
    for col in num_cols:
        if col != target_col:
            corr = df[[col, target_col]].corr().iloc[0, 1]
            corrs[col] = abs(corr)

    high_corr = {k: v for k, v in corrs.items() if v > 0.7}
    if high_corr:
        print("⚠️  Şüpheli yüksek korelasyon (olası leakage):")
        for col, c in sorted(high_corr.items(), key=lambda x: -x[1]):
            print(f"  {col}: {c:.3f}")
    return high_corr
```

### Temel Feature Engineering Teknikleri

```python
import pandas as pd
import numpy as np

# 1. Agregasyon featureları
user_features = df.groupby("user_id").agg(
    n_orders=("order_id", "count"),
    total_spend=("amount", "sum"),
    avg_order=("amount", "mean"),
    std_order=("amount", "std"),
    max_order=("amount", "max"),
    days_since_first=("order_date", lambda x: (x.max() - x.min()).days),
).reset_index()

# 2. Oran featureları
user_features["return_rate"] = user_features["n_returns"] / user_features["n_orders"]
user_features["premium_ratio"] = user_features["premium_orders"] / user_features["n_orders"]

# 3. Etkileşim featureları
user_features["spend_per_day"] = user_features["total_spend"] / (user_features["days_active"] + 1)

# 4. Lag ve rolling featureları (zaman serisinde kritik)
df = df.sort_values(["user_id", "order_date"])
df["prev_amount"] = df.groupby("user_id")["amount"].shift(1)
df["diff_from_prev"] = df["amount"] - df["prev_amount"]
df["rolling_7d_mean"] = df.groupby("user_id")["amount"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# 5. Tarih featureları
df["day_of_week"] = df["order_date"].dt.dayofweek
df["month"] = df["order_date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["quarter"] = df["order_date"].dt.quarter
df["days_since_signup"] = (df["order_date"] - df["signup_date"]).dt.days

# 6. Eksik değer göstergesi (MNAR durumunda bilgi taşır)
df["income_is_null"] = df["income"].isna().astype(int)
df["income_filled"] = df["income"].fillna(df["income"].median())
```

### Encoding Stratejileri

```python
# One-hot: düşük kardinalite (<20)
df_ohe = pd.get_dummies(df, columns=["country", "device"])

# Target encoding: yüksek kardinalite + K-fold ile leakage önleme
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold

def target_encode_kfold(df: pd.DataFrame, col: str, target: str,
                         n_splits: int = 5) -> pd.Series:
    """Target encoding — K-fold ile leakage önlenir."""
    result = pd.Series(index=df.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        encoding = train.groupby(col)[target].mean()
        result.iloc[val_idx] = val[col].map(encoding)

    # Eksik (görmediğimiz kategoriler) → global ortalama
    global_mean = df[target].mean()
    result = result.fillna(global_mean)
    return result
```

---

## B.7 Model Yorumlanabilirliği (SHAP, LIME ve Counterfactual)

### Sezgisel Açıklama

SHAP (SHapley Additive exPlanations): Her feature'ın tahmine katkısını oyun teorisi ile hesaplar. "Model neden bu kararı verdi?" sorusunu yanıtlar — ama "bu feature gerçekten churn'e yol açıyor mu?" sorusunu yanıtlamaz (nedensellik değil).

```python
import shap
import lightgbm as lgb

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)

# SHAP değerleri hesapla
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. Global önem (summary plot)
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names,
                   plot_type="bar", max_display=15)

# 2. Beeswarm (dağılım + yön)
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names,
                   max_display=15)

# 3. Tek tahmin açıklaması (waterfall)
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][0],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[0],
    feature_names=feature_names
))

# 4. Feature bağımlılığı
shap.dependence_plot("days_since_last_order", shap_values[1], X_test,
                      interaction_index="total_spend")
```

> **Senior Notu:** SHAP "model bu feature'a ne kadar dikkat etti" der. "Bu feature gerçekten churn'e sebep oluyor mu?" demez. Nedensellik için causal inference gerekir (bkz. `katman-C`). SHAP ile Permutation Importance genelde tutarlı. Tutarsızlık varsa multicollinearity şüphesi.

### LIME — Yerel Açıklanabilirlik

LIME (Local Interpretable Model-agnostic Explanations), tek bir tahmini açıklamak için o noktanın çevresinde basit bir doğrusal model kurar. Model-agnostic: herhangi bir sınıflandırıcı/regresör ile çalışır.

```python
import lime
import lime.lime_tabular
import numpy as np

# LIME TabularExplainer oluştur
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values if hasattr(X_train, 'values') else X_train,
    feature_names=feature_names,
    class_names=["Kalacak", "Churn"],
    mode="classification",
    discretize_continuous=True
)

# Tek bir örnek için açıklama
idx = 0  # açıklamak istediğimiz örneğin indexi
explanation = lime_explainer.explain_instance(
    data_row=X_test.iloc[idx].values if hasattr(X_test, 'iloc') else X_test[idx],
    predict_fn=model.predict_proba,
    num_features=10
)

# Görselleştirme
explanation.show_in_notebook()  # Jupyter'da
# veya
fig = explanation.as_pyplot_figure()
plt.tight_layout()

# Açıklama listesi olarak
print("En etkili feature'lar:")
for feature, weight in explanation.as_list():
    direction = "↑ Churn" if weight > 0 else "↓ Kalacak"
    print(f"  {feature}: {weight:+.4f} ({direction})")
```

### SHAP vs LIME — Ne Zaman Hangisi?

| Kriter | SHAP | LIME |
|--------|------|------|
| **Teori** | Oyun teorisi (Shapley değerleri) — matematiksel garantili | Yerel doğrusal yaklaşım — sezgisel |
| **Kapsam** | Global + lokal açıklama | Sadece lokal açıklama |
| **Hız** | Yavaş (özellikle KernelSHAP) | Hızlı (tek örnek için) |
| **Tree modeller** | TreeSHAP ile çok hızlı | Genel yöntem, özel optimizasyon yok |
| **Tutarlılık** | Aynı girdi → aynı çıktı (deterministik) | Pertürbasyona bağlı, farklı çalıştırmalarda varyans olabilir |
| **Korelasyonlu feature'lar** | Sorunlu ama farkında (interaction değerleri var) | Daha sorunlu, yerel yaklaşım aldanabilir |
| **Kullanım alanı** | Model geliştirme, raporlama, audit | Hızlı prototipleme, müşteriye tek tahmin açıklaması |
| **Görselleştirme** | Zengin (waterfall, beeswarm, dependence) | Basit bar chart |

> **Pratik kural:** Tree-based modellerde (LightGBM, XGBoost) → **SHAP** (TreeSHAP hızlı). Herhangi bir modelde tek tahmin hızlı açıklama → **LIME**. İkisini birlikte kullanmak en iyisi: SHAP global resim, LIME tek vakanın hızlı açıklaması.

### Counterfactual Explanations — "Ne Değişmeli?"

Counterfactual açıklamalar, modelin kararını değiştirmek için minimum feature değişikliğini gösterir. SHAP ve LIME "neden bu karar?" sorusunu yanıtlarken, counterfactual "bu kararı değiştirmek için ne yapmalı?" sorusunu yanıtlar.

**Örnek:** "Bu müşterinin churn'den kurtulması için ne değişmeli?"

```python
import dice_ml
from dice_ml import Dice

# Veriyi ve modeli DiCE formatında tanımla
data_dice = dice_ml.Data(
    dataframe=df_train,      # eğitim verisi (features + target)
    continuous_features=["age", "monthly_spend", "tenure_months",
                         "support_calls", "days_since_last_order"],
    outcome_name="churn"     # hedef değişken
)

model_dice = dice_ml.Model(model=model, backend="sklearn")

# DiCE Explainer oluştur
dice_exp = Dice(data_dice, model_dice, method="random")

# Churn olarak tahmin edilen bir müşteri için counterfactual üret
query_instance = X_test.iloc[[42]]  # churn tahmini alan müşteri

counterfactuals = dice_exp.generate_counterfactuals(
    query_instance,
    total_CFs=3,             # 3 farklı counterfactual
    desired_class="opposite" # kararı tersine çevir (churn → kalacak)
)

# Sonuçları göster
counterfactuals.visualize_as_dataframe(show_only_changes=True)

# Çıktı örneği:
# "Müşterinin churn'den kurtulması için:
#   - monthly_spend: 45 → 78 TL (artırılmalı)
#   - days_since_last_order: 32 → 12 gün (daha sık alışveriş)
#   - support_calls: 5 → 2 (sorun çözülmeli)"
```

> **Senior Notu:** Counterfactual explanations aksiyon odaklıdır — müşteriye veya operasyon ekibine "bunu yap" diyebilirsin. Ama dikkat: counterfactual korelasyona dayanır, nedenselliğe değil. "monthly_spend artırılmalı" demek "zorla para harcatırsan churn durur" anlamına gelmez. Nedensel çıkarım için `katman-C`'ye bak.

---

## B.8 Hata Analizi (Error Analysis)

### Sezgisel Açıklama

Model sonuçları yetersizse neden? Hata analizi olmadan "daha büyük model dene" tuzağına düşersin. Hata analizi "nerede yanlış" sorusunu yanıtlar, "nasıl düzeltilir" yönünü gösterir.

```python
import pandas as pd
import numpy as np

def error_analysis(X_test, y_test, y_pred, y_prob, feature_names, segment_cols):
    """Segment bazlı hata analizi."""
    results = pd.DataFrame(X_test, columns=feature_names)
    results["y_true"] = y_test
    results["y_pred"] = y_pred
    results["y_prob"] = y_prob
    results["correct"] = (results["y_true"] == results["y_pred"]).astype(int)

    # Genel metrikler
    print(f"Genel doğruluk: {results['correct'].mean():.1%}")
    print(f"FP sayısı: {((results['y_true']==0) & (results['y_pred']==1)).sum()}")
    print(f"FN sayısı: {((results['y_true']==1) & (results['y_pred']==0)).sum()}")

    # Segment analizi
    for col in segment_cols:
        if col in results.columns:
            seg_acc = results.groupby(col)["correct"].mean().sort_values()
            print(f"\n{col} bazlı doğruluk:")
            print(seg_acc)

    # En kötü tahminler (büyük hata)
    fn_worst = results[
        (results["y_true"] == 1) & (results["y_pred"] == 0)
    ].nsmallest(10, "y_prob")

    fp_worst = results[
        (results["y_true"] == 0) & (results["y_pred"] == 1)
    ].nlargest(10, "y_prob")

    return fn_worst, fp_worst
```

---

## B.9 Kümeleme (Unsupervised)

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# K-Means — elbow yöntemi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Optimal k: sil_score maksimum
optimal_k = k_range[np.argmax(sil_scores)]
print(f"Optimal k: {optimal_k}")

# DBSCAN — yoğunluk bazlı (farklı şekil kümeler)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"DBSCAN: {n_clusters} küme, {n_noise} gürültü noktası")
```

---

## B.10 Senior Modelleme Süreci

Bu süreç sırası sabittir:

```
1. Veri anlama ve leakage kontrolü (asla atla!)
2. Baseline kur → logistic regression veya naïve tahmin
3. Güçlü model → LightGBM (tabular için standart)
4. Hyperparameter tuning → Optuna
5. Ablation → hangi feature grubu ne kadar katkı sağlıyor?
6. Calibration → skor olasılık gibi mi davranıyor?
7. Error analysis → hangi segmentte kötü, neden?
8. Threshold seçimi → maliyet matrisine göre
9. SHAP → iş paydaşı için açıklama
10. Prod hazırlığı → src/ kodu, testler, model card
```

### Sektör Notu — 2026 Tabular ML

2026 itibarıyla tabular veri için:
- **LightGBM** hâlâ dominant: üretim ML pipeline'larının %60+'ında LightGBM var (Microsoft, Alibaba, Booking.com)
- **XGBoost** stabil, enterprise tercih: Airbnb, Uber production sistemleri
- **CatBoost** kategorik ağırlıklı veri için (e-ticaret, öneri)
- **AutoML** (AutoGluon, FLAML) rapid prototyping için ama prod'a kör gönderme — anlayarak kullan
- **TabPFN, TabNet** gibi deep tabular modeller akademide ilgi görüyor ama üretimde LightGBM/XGBoost hâlâ kazanıyor

> Araştırma bulgusu: 2025 benchmark çalışmalarında LightGBM, büyük tabular veri setlerinde deep learning modellerini %80 vakada geçiyor.

---

## B.11 AutoML — Otomatik Model Seçimi

### Sezgisel Açıklama

AutoML, model seçimi, hyperparameter tuning ve ensembling adımlarını otomatikleştirir. Hızlı baseline kurmak için mükemmel; ama prod'da kör kullanmak riskli.

### AutoGluon — Temel Kullanım

```python
from autogluon.tabular import TabularDataset, TabularPredictor

# Veriyi yükle
train_data = TabularDataset("train.csv")
test_data = TabularDataset("test.csv")

# 3 satırda model eğitimi — AutoGluon gerisini halleder
predictor = TabularPredictor(label="churn", eval_metric="roc_auc").fit(
    train_data,
    time_limit=600,          # 10 dakika süre sınırı
    presets="best_quality"   # veya "medium_quality", "high_quality"
)

# Tahmin ve değerlendirme
predictions = predictor.predict(test_data)
leaderboard = predictor.leaderboard(test_data, silent=True)
print(leaderboard)
```

### Ne Zaman AutoML, Ne Zaman Manuel?

| Durum | AutoML | Manuel (LightGBM + Optuna) |
|-------|--------|---------------------------|
| **Hızlı baseline/POC** | Kesinlikle | Gereksiz zaman kaybı |
| **Kaggle yarışması** | İlk gün baseline | Sonra manuel ince ayar |
| **Prod pipeline** | Dikkatli kullan | Tercih edilen yol |
| **Feature engineering gerekli** | Sınırlı kontrol | Tam kontrol |
| **Yorumlanabilirlik gerekli** | Zor (stacking modelleri) | Tek model → SHAP kolay |
| **Reproducibility kritik** | Versiyon pinleme gerekir | Tam kontrol |
| **Veri < 1000 satır** | Overfitting riski | Basit model yeterli |
| **Veri > 100K satır** | AutoGluon iyi çalışır | Manuel da iyi çalışır |

> **Senior Notu:** AutoML baseline için harika — 10 dakikada "bu problemde ne kadar AUC mümkün?" sorusunu yanıtlar. Ama prod'da dikkat: **reproducibility** (aynı sonucu tekrar üretme) ve **interpretability** (modelin ne yaptığını açıklama) sorunları var. AutoGluon stacking yapıyorsa, o modeli SHAP ile açıklamak zor. Prod'da tek LightGBM modeli genelde daha iyi bir denge sunar.

---

## B.12 Fairness — Model Adaleti

### Sezgisel Açıklama

Bir kredi modeli erkeklere %20, kadınlara %10 onay veriyorsa — model doğru bile olsa adil mi? Fairness metrikleri, modelin farklı demografik gruplara eşit davranıp davranmadığını ölçer. Düzenlemeli sektörlerde (finans, sağlık, işe alım) yasal zorunluluk olabilir.

### Demographic Parity (Demografik Eşitlik)

**Formül:** P(Y_hat = 1 | A = 0) = P(Y_hat = 1 | A = 1)

Yani: modelin pozitif tahmin oranı, korunan gruba (A) bağımsız olmalı.

```python
import numpy as np

def demographic_parity(y_pred, sensitive_attr):
    """
    Demographic parity farkını hesapla.
    0'a yakın → adil, büyük fark → bias var.
    """
    groups = np.unique(sensitive_attr)
    rates = {}
    for g in groups:
        mask = sensitive_attr == g
        rates[g] = y_pred[mask].mean()

    dp_difference = max(rates.values()) - min(rates.values())
    print(f"Grup bazlı pozitif tahmin oranları: {rates}")
    print(f"Demographic Parity farkı: {dp_difference:.4f}")
    return dp_difference

# Kullanım
dp = demographic_parity(y_pred, df_test["gender"].values)
```

### Equalized Odds (Eşitlenmiş Olasılıklar)

**Formül:** P(Y_hat = 1 | A = a, Y = y) aynı olmalı her a ve y için.

Yani: True Positive Rate ve False Positive Rate her grupta eşit olmalı.

```python
def equalized_odds(y_true, y_pred, sensitive_attr):
    """
    Equalized odds farkını hesapla (TPR ve FPR bazlı).
    """
    groups = np.unique(sensitive_attr)
    tpr_dict, fpr_dict = {}, {}

    for g in groups:
        mask = sensitive_attr == g
        y_t, y_p = y_true[mask], y_pred[mask]

        # True Positive Rate
        pos_mask = y_t == 1
        tpr_dict[g] = y_p[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

        # False Positive Rate
        neg_mask = y_t == 0
        fpr_dict[g] = y_p[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

    tpr_diff = max(tpr_dict.values()) - min(tpr_dict.values())
    fpr_diff = max(fpr_dict.values()) - min(fpr_dict.values())

    print(f"TPR farkı: {tpr_diff:.4f} | FPR farkı: {fpr_diff:.4f}")
    print(f"Grup TPR: {tpr_dict}")
    print(f"Grup FPR: {fpr_dict}")
    return tpr_diff, fpr_diff

# Kullanım
tpr_d, fpr_d = equalized_odds(y_test, y_pred, df_test["gender"].values)
```

### Fairlearn ile Bias Tespiti ve Azaltma

```python
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score, recall_score, precision_score

# MetricFrame: tüm metrikleri grup bazlı hesapla
metrics = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test["gender"]
)

print("Grup bazlı metrikler:")
print(metric_frame.by_group)
print(f"\nDemographic Parity farkı: "
      f"{demographic_parity_difference(y_test, y_pred, sensitive_features=df_test['gender']):.4f}")
print(f"Equalized Odds farkı: "
      f"{equalized_odds_difference(y_test, y_pred, sensitive_features=df_test['gender']):.4f}")

# Görselleştirme
metric_frame.by_group.plot.bar(subplots=True, layout=(1, 3), figsize=(14, 4),
                                legend=False, rot=0)
plt.suptitle("Fairness — Grup Bazlı Metrikler")
plt.tight_layout()
```

> **Senior Notu:** Fairness metrikleri arasında trade-off var — **demographic parity ve equalized odds aynı anda tam sağlanamaz** (Impossibility Theorem). Hangi metriğin önemli olduğu iş bağlamına göre değişir: kredi başvurusunda equalized odds daha uygun (aynı profile sahip insanlar aynı sonucu almalı), işe alımda demographic parity daha uygun olabilir. Her zaman domain expert ile birlikte karar ver. Ayrıca bkz. `katman-C` — adalet soruları genelde nedensel (causal) sorulardır.

---

## B.13 Alıştırma Soruları — ML Case Studies

Aşağıdaki sorularda gerçek dünya senaryolarını ele alarak bu katmandaki kavramları pekiştirin.

**Soru 1 — Fraud Detection Pipeline**
Bir e-ticaret şirketi günlük 500K işlem yapıyor, bunların %0.05'i fraud. Bir ML pipeline tasarlayın:
- Hangi veri bölme stratejisi? (random split mi, time-based mi?)
- class_weight mi, SMOTE mi? Neden?
- Evaluation metriği olarak accuracy kullanır mısınız? Neden / neden değil?
- Threshold seçimini nasıl yaparsınız?

**Soru 2 — Churn Model Yorumlama**
Bir telekom şirketi için churn modeli kurdunuz, AUC=0.82. Pazarlama müdürü "hangi müşterilere kampanya gönderelim?" ve "neden bu müşteriler churn riski taşıyor?" diye soruyor.
- SHAP mı LIME mı kullanırsınız? Her ikisi için kullanım senaryosu verin.
- "Kampanya gönderilmezse bu müşteri churn eder" cümlesi korelasyon mu nedensellik mi?
- Counterfactual explanation bu senaryoda nasıl yardımcı olur?

**Soru 3 — Fairness Audit**
Bir banka kredi skoru modeli geliştirdi. Düzenleyici kurum cinsiyet bazlı adalet raporu istiyor.
- Demographic parity ve equalized odds arasında hangisini tercih edersiniz? Neden?
- Model adil değilse ne yaparsınız? (post-processing, re-training, feature removal?)
- "Cinsiyet feature'ını kaldırırsam model adil olur" doğru mu?

**Soru 4 — Feature Engineering ve Leakage**
Bir sigorta şirketi hasar tahmin modeli kuruyor. Feature listesinde şunlar var:
- `claim_amount` (hasar tutarı), `policy_start_date`, `customer_age`, `total_past_claims`, `claim_status`
Hangisi leakage riski taşır? Neden? Nasıl düzeltirsiniz?

**Soru 5 — AutoML vs Manuel Karar**
Yeni bir startup'ta tek data scientist sizsiniz. CEO "2 haftada churn modeli istiyorum" diyor.
- AutoML (AutoGluon) ile başlar mısınız? Neden?
- Prod'a AutoGluon modeli mi koyarsınız yoksa manuel LightGBM mi? Trade-off'ları tartışın.
- Reproducibility sorununu nasıl çözersiniz?

**Soru 6 — Zaman Serisi CV**
Aylık satış tahmini modeli kuruyorsunuz. 3 yıllık veri var.
- Neden random K-fold kullanmamalısınız?
- Expanding window vs sliding window: bu senaryoda hangisi daha uygun?
- Concept drift varsa ne yaparsınız?

**Soru 7 — Kalibrasyon Problemi**
Bir hastalık tanı modeliniz var. Model "bu hastanın %80 olasılıkla hastalığı var" dediğinde, gerçekte %60'ında hastalık çıkıyor.
- Bu ne tür bir kalibrasyon sorunu? (over-confident mu, under-confident mu?)
- Nasıl düzeltirsiniz? (Platt scaling, isotonic regression?)
- Kalibrasyon neden özellikle sağlık alanında kritik?

---

## Çapraz Referanslar

Bu katmandaki kavramların diğer katmanlarla ilişkisi:

| Konu | İlgili Katman | Bağlantı |
|------|--------------|----------|
| SHAP "neden" sorusuna yanıt vermez → nedensellik | **Katman C** | Korelasyon vs nedensellik farkı, causal inference teknikleri |
| Tabular veri için DL mı klasik ML mı? | **Katman D** | TabPFN, TabNet vs LightGBM benchmark karşılaştırması |
| Model prod'a nasıl gider? | **Katman E** | Model serving, API tasarımı, monitoring, A/B test |
| Fairness → causal fairness | **Katman C** | Adalet soruları temelde nedensel sorulardır |
| Feature engineering → embedding | **Katman D** | Kategorik feature'lar için learned embeddings |
| Kalibrasyon → karar destek sistemleri | **Katman E** | Prod'da olasılık tahmini ve threshold yönetimi |

---

## Katman B Kontrol Listesi

- [ ] Problem framing checklist'i bir gerçek problemde uyguladım
- [ ] Zaman bazlı split + group k-fold biliyorum
- [ ] Walk-forward validation (expanding + sliding window) uyguladım
- [ ] LightGBM'i baştan sona kurdum (early stopping dahil)
- [ ] Optuna ile hyperparameter tuning yaptım
- [ ] ROC-AUC vs PR-AUC farkını açıklayabilirim
- [ ] Kalibrasyon grafiği çizdim, neden gerektiğini biliyorum
- [ ] Dengesiz veri tekniklerini biliyorum (class_weight, SMOTE, focal loss)
- [ ] SHAP summary + waterfall plot oluşturdum
- [ ] LIME ile tek tahmin açıklayabildim
- [ ] SHAP vs LIME farkını açıklayabilirim
- [ ] Counterfactual explanation (DiCE) ile "ne değişmeli?" sorusunu yanıtladım
- [ ] Hata analizi yaptım: en kötü segment hangisi?
- [ ] Maliyet matrisine göre threshold seçtim
- [ ] AutoML (AutoGluon) baseline kurdum, manuel modelle karşılaştırdım
- [ ] Fairness metrikleri hesapladım (demographic parity, equalized odds)
- [ ] Alıştırma sorularından en az 3 tanesini çözdüm
- [ ] Proje-1 (Churn Tahmini) tamamlandı

---

<div class="nav-footer">
  <span><a href="#file_katman_A_temeller">← Önceki: Katman A — Temeller</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_C_deney_nedensellik">Sonraki: Katman C — Deney/Nedensellik →</a></span>
</div>
