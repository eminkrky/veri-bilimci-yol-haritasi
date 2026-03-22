# Portföy — Senior'a Götüren Proje Seti (Proje 0–7)

> Her proje için beklenen teslimatlar belirtilmiştir. Projeleri sırayla yap — her biri bir sonrakini besler.
>
> **Portföy kuralı:** 3 farklı alan kapsayan proje > 12 benzer proje. Çeşitlilik önemli.

---

## Genel Teslimat Standartları

Her proje için minimum:

```
proje-N-isim/
├── README.md             # Amaç, veri, yöntem, bulgular, sınırlılıklar
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── train.py
├── tests/
│   └── test_features.py
└── reports/
    ├── executive_summary.md
    └── figures/
```

**README şablonu:**
```markdown
## [Proje Adı]

### Problem
1 cümle: Ne soruyoruz?

### Veri
Kaynak, dönem, boyut, hedef değişken.

### Yöntem
3 bullet: Yaklaşım özeti.

### Bulgular
- Ana bulgu 1 (sayısal)
- Ana bulgu 2
- Ana bulgu 3

### Sınırlılıklar
Neler eksik, neler varsayıldı?

### Sonraki Adım
Deploy, A/B test, daha derin analiz?
```

---

## Proje-0: Analitik Paket (Python + SQL)

> **Çapraz referans:** Bu proje `katman-1` (Python temelleri) ve `katman-2` (SQL & veri tabanları) bilgisini gerektirir.

**Amaç:** 1 hafta içinde iş görüşmesinde gösterebileceğin mini portföy başlangıcı.

**Veri önerisi:**
- [Brazilian E-Commerce (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle, gerçek sipariş verisi
- [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — Taksi verisi
- Kendi seçimin: iş alanına yakın bir veri seti bul (fintech, sağlık, e-ticaret)

**Çıktılar:**

```
proje-0-analitik/
├── analysis.ipynb    # EDA + grafikler
├── queries.sql       # SQL analizleri
└── README.md         # 1 sayfa özet
```

**İçerik standartları:**

`analysis.ipynb`:
- Veri kalitesi raporu (eksik değer, veri tipleri, kardinalite)
- En az 5 anlamlı grafik (dağılım, trend, segment)
- En az 1 non-obvious bulgu ("beklenmedik bir şey buldum")

`queries.sql` — zorunlu sorgular:
```sql
-- ═══════════════════════════════════════
-- 1. Aylık Gelir Trendi + MoM Büyüme
-- ═══════════════════════════════════════
WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date)     AS month,
        SUM(order_amount)                   AS revenue
    FROM orders
    WHERE order_status = 'completed'
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month)     AS prev_month_revenue,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
              / NULLIF(LAG(revenue) OVER (ORDER BY month), 0),
    2)                                      AS mom_growth_pct
FROM monthly_revenue
ORDER BY month;

-- ═══════════════════════════════════════
-- 2. Cohort Retention Analizi
-- ═══════════════════════════════════════
WITH cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    GROUP BY user_id
),
user_activity AS (
    SELECT
        o.user_id,
        c.cohort_month,
        DATE_TRUNC('month', o.order_date)    AS activity_month,
        DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', o.order_date)) AS months_since_cohort
    FROM orders o
    JOIN cohorts c ON o.user_id = c.user_id
)
SELECT
    cohort_month,
    months_since_cohort,
    COUNT(DISTINCT user_id)                   AS active_users,
    FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
        PARTITION BY cohort_month ORDER BY months_since_cohort
    )                                          AS cohort_size,
    ROUND(
        100.0 * COUNT(DISTINCT user_id)
              / FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
                    PARTITION BY cohort_month ORDER BY months_since_cohort
                ),
    1)                                         AS retention_rate
FROM user_activity
GROUP BY cohort_month, months_since_cohort
ORDER BY cohort_month, months_since_cohort;

-- ═══════════════════════════════════════
-- 3. Satın Alma Funnel Analizi
-- ═══════════════════════════════════════
WITH funnel_steps AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_type = 'page_view'       THEN 1 ELSE 0 END) AS step1_view,
        MAX(CASE WHEN event_type = 'add_to_cart'     THEN 1 ELSE 0 END) AS step2_cart,
        MAX(CASE WHEN event_type = 'checkout_start'  THEN 1 ELSE 0 END) AS step3_checkout,
        MAX(CASE WHEN event_type = 'purchase'        THEN 1 ELSE 0 END) AS step4_purchase
    FROM events
    WHERE event_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT
    SUM(step1_view)                                      AS visitors,
    SUM(step2_cart)                                      AS added_to_cart,
    SUM(step3_checkout)                                  AS started_checkout,
    SUM(step4_purchase)                                  AS purchased,
    ROUND(100.0 * SUM(step2_cart)     / NULLIF(SUM(step1_view),    0), 1) AS view_to_cart_pct,
    ROUND(100.0 * SUM(step3_checkout) / NULLIF(SUM(step2_cart),    0), 1) AS cart_to_checkout_pct,
    ROUND(100.0 * SUM(step4_purchase) / NULLIF(SUM(step3_checkout), 0), 1) AS checkout_to_purchase_pct,
    ROUND(100.0 * SUM(step4_purchase) / NULLIF(SUM(step1_view),    0), 1) AS overall_conversion_pct
FROM funnel_steps;
```

### EDA Kontrol Listesi

- [ ] Veri boyutu ve sütun tipleri doğrulandı (`df.info()`, `df.describe()`)
- [ ] Eksik değer oranları hesaplandı ve imputation stratejisi belirlendi
- [ ] Numerik değişkenlerde outlier analizi yapıldı (IQR veya z-score)
- [ ] Kategorik değişkenlerde kardinalite kontrolü yapıldı (high-cardinality flag)
- [ ] Hedef değişken dağılımı incelendi (imbalance oranı raporlandı)
- [ ] En az 3 bivariate analiz yapıldı (hedef vs önemli feature)
- [ ] Korelasyon matrisi çıkarıldı ve multicollinearity kontrol edildi
- [ ] Zaman bazlı trend veya mevsimsellik kontrolü yapıldı (tarih sütunu varsa)

**Süre hedefi:** 1 hafta

---

## Proje-1: Uçtan Uca Churn Tahmini

> **Çapraz referans:** Bu proje `katman-3` (istatistik & olasılık), `katman-4` (makine öğrenmesi) ve `katman-5` (feature engineering) bilgisini gerektirir.

**Amaç:** Tam ML pipeline — framing'den error analysis'e.

**Veri önerisi:**
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — başlangıç
- [Olist Churn Derivation](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — daha gerçekçi (kendin tanımla)
- Kendi şirket verisi (en iyi seçenek)

**Zorunlu bileşenler:**

| Bileşen | Açıklama |
|---------|---------|
| Zaman bazlı split | Gelecek bilgisi train'de yok |
| Leakage kontrolü | Feature pencereleri doğru |
| Baseline | Logistic regression veya naïve |
| Güçlü model | LightGBM + erken durdurma |
| Optuna tuning | 50+ deneme |
| Kalibrasyon | Reliability diagram |
| SHAP analizi | Global + lokal |
| Hata analizi | Segment bazlı FP/FN inceleme |
| Maliyet matrisi | Threshold optimizasyonu |
| Model card | Standart şablon |

**Kod — `src/features.py`:**

```python
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, observation_date: pd.Timestamp) -> pd.DataFrame:
    """
    Leakage'siz feature oluştur.
    Feature window: [obs_date - 90 gün, obs_date]
    Label window:   (obs_date, obs_date + 30 gün]
    """
    feat_start = observation_date - pd.Timedelta(days=90)
    mask = (df["event_date"] >= feat_start) & (df["event_date"] <= observation_date)
    df_window = df[mask].copy()

    agg = df_window.groupby("customer_id").agg(
        total_orders=("order_id", "nunique"),
        total_revenue=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
        distinct_categories=("category", "nunique"),
        last_order_date=("event_date", "max"),
        first_order_date=("event_date", "min"),
    ).reset_index()

    # --- Temporal features ---
    agg["recency_days"] = (observation_date - agg["last_order_date"]).dt.days
    agg["tenure_days"] = (agg["last_order_date"] - agg["first_order_date"]).dt.days
    agg["order_frequency"] = agg["total_orders"] / (agg["tenure_days"].replace(0, 1) / 30)

    # --- Lag features: son 30 gün vs önceki 60 gün ---
    last30_start = observation_date - pd.Timedelta(days=30)
    prev60_start = observation_date - pd.Timedelta(days=90)

    last30 = df[(df["event_date"] > last30_start) & (df["event_date"] <= observation_date)]
    prev60 = df[(df["event_date"] >= prev60_start) & (df["event_date"] <= last30_start)]

    rev_last30 = last30.groupby("customer_id")["revenue"].sum().rename("rev_last30")
    rev_prev60 = prev60.groupby("customer_id")["revenue"].sum().rename("rev_prev60")
    agg = agg.merge(rev_last30, on="customer_id", how="left")
    agg = agg.merge(rev_prev60, on="customer_id", how="left")
    agg[["rev_last30", "rev_prev60"]] = agg[["rev_last30", "rev_prev60"]].fillna(0)

    # --- Ratio features ---
    agg["rev_trend_ratio"] = agg["rev_last30"] / (agg["rev_prev60"].replace(0, 1) / 2)
    agg["avg_to_total_ratio"] = agg["avg_order_value"] / (agg["total_revenue"].replace(0, 1))

    # --- Rolling mean (haftalık sipariş sayısı ortalaması) ---
    weekly = df_window.set_index("event_date").groupby("customer_id").resample("W")["order_id"].nunique()
    rolling_mean = weekly.groupby("customer_id").transform(lambda x: x.rolling(4, min_periods=1).mean())
    last_rolling = rolling_mean.groupby("customer_id").last().rename("weekly_order_rolling4")
    agg = agg.merge(last_rolling, on="customer_id", how="left").fillna(0)

    # --- Label oluşturma ---
    label_start = observation_date
    label_end = observation_date + pd.Timedelta(days=30)
    future = df[(df["event_date"] > label_start) & (df["event_date"] <= label_end)]
    churned_ids = set(agg["customer_id"]) - set(future["customer_id"].unique())
    agg["churn"] = agg["customer_id"].isin(churned_ids).astype(int)

    drop_cols = ["last_order_date", "first_order_date"]
    agg.drop(columns=[c for c in drop_cols if c in agg.columns], inplace=True)
    return agg
```

**Kod — `src/train.py`:**

```python
import lightgbm as lgb
import optuna
import mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                n_trials: int = 60) -> lgb.Booster:
    """Optuna + LightGBM + early stopping ile 5-fold CV tabanlı model eğitimi."""

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_vl)
            cv_scores.append(roc_auc_score(y_vl, preds))

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # --- Final model: en iyi parametrelerle tüm train verisinde eğit ---
    best_params = study.best_params
    best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})

    dtrain_full = lgb.Dataset(X_train, label=y_train)
    dval_full = lgb.Dataset(X_val, label=y_val, reference=dtrain_full)

    final_model = lgb.train(
        best_params, dtrain_full,
        num_boost_round=1000,
        valid_sets=[dval_full],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_auc", study.best_value)
    return final_model
```

**Kod — `src/evaluate.py`:**

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, average_precision_score,
    calibration_curve,
)
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, cost_matrix: dict = None):
    """
    AUC, precision, recall, F1, PR-AUC, kalibrasyon eğrisi hesapla.
    cost_matrix örneği: {"TP": 100, "FP": -30, "FN": -200, "TN": 0}
    """
    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # --- Kalibrasyon eğrisi ---
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(prob_pred, prob_true, marker="o", label="Model")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("True probability")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()

    # --- Maliyet bazlı threshold optimizasyonu ---
    if cost_matrix:
        thresholds = np.linspace(0.05, 0.95, 50)
        profits = []
        for t in thresholds:
            yp = (y_prob >= t).astype(int)
            tp = ((yp == 1) & (y_test == 1)).sum()
            fp = ((yp == 1) & (y_test == 0)).sum()
            fn = ((yp == 0) & (y_test == 1)).sum()
            tn = ((yp == 0) & (y_test == 0)).sum()
            profit = (tp * cost_matrix["TP"] + fp * cost_matrix["FP"]
                      + fn * cost_matrix["FN"] + tn * cost_matrix["TN"])
            profits.append(profit)

        best_idx = np.argmax(profits)
        best_threshold = thresholds[best_idx]
        metrics["optimal_threshold"] = best_threshold
        metrics["max_profit"] = profits[best_idx]

        axes[1].plot(thresholds, profits, color="green")
        axes[1].axvline(best_threshold, color="red", linestyle="--",
                        label=f"Best threshold={best_threshold:.2f}")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Profit")
        axes[1].set_title("Cost-Based Threshold Optimization")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/figures/evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()

    return metrics
```

**Süre hedefi:** 2–4 hafta

---

## Proje-2: A/B Test Analizi Paketi

> **Çapraz referans:** Bu proje `katman-3` (istatistik & olasılık) ve `katman-6` (deney tasarımı / causal inference) bilgisini gerektirir.

**Amaç:** Deney tasarımı ve analizi — frequentist + Bayesian + CUPED.

**Senaryo:** "E-ticaret sitesinde ürün sayfasında önerilen ürün widgetının değiştirilmesi."

**Çıktılar:**

```
proje-2-ab-test/
├── power_analysis.ipynb         # MDE + sample size hesabı
├── ab_analysis_frequentist.ipynb # t-test + bootstrap CI + CUPED
├── ab_analysis_bayesian.ipynb   # Beta-Binomial + P(treatment > control)
├── peeking_simulation.ipynb     # Peeking etkisi simülasyonu
├── experiment_design.md         # Tasarım dokümanı
└── README.md
```

**`power_analysis.ipynb` içeriği:**
- MDE analizi: %1, %2, %5 için gerekli n
- Power grafiği (n vs güç)
- α ve power tradeoff

**`ab_analysis_frequentist.ipynb` içeriği:**
- SRM kontrolü
- t-test + p-value + effect size
- Bootstrap CI
- CUPED düzeltmesi (CI daralmasını göster)
- Guardrail ihlal senaryosu

**`peeking_simulation.ipynb`:**
```python
# Simülasyon: H₀ doğruyken her gün bakmanın etkisi
# Hedef: 14 günlük deneyde günlük peeking → %25–30 yanlış pozitif
# Sabit horizon → %5 yanlış pozitif
# Farkı görsel olarak göster
```

### A/B Test Kontrol Listesi

- [ ] Hipotez ve birincil metrik net tanımlandı (tek cümle)
- [ ] Power analizi yapıldı: MDE, α, β, gerekli n hesaplandı
- [ ] Randomizasyon birimi seçildi (kullanıcı / oturum / cihaz) ve gerekçelendirildi
- [ ] SRM (Sample Ratio Mismatch) kontrolü eklendi
- [ ] Guardrail metrikleri belirlendi (latency, bounce rate, revenue vb.)
- [ ] Peeking problemi adreslendi (sabit horizon veya sequential testing)
- [ ] CUPED / variance reduction tekniği uygulandı ve CI daralması gösterildi
- [ ] Bayesian analiz ile P(treatment > control) hesaplandı
- [ ] Segment bazlı heterogeneous treatment effect incelendi
- [ ] Sonuç raporu: effect size, CI, practical significance yorumlandı

**Süre hedefi:** 1–2 hafta

---

## Proje-3: Zaman Serisi Tahmin + Backtesting

> **Çapraz referans:** Bu proje `katman-3` (istatistik), `katman-4` (makine öğrenmesi) ve `katman-5` (feature engineering — lag/rolling) bilgisini gerektirir.

**Amaç:** Leakage'siz walk-forward backtesting, çoklu model karşılaştırması.

**Veri önerisi:**
- [Store Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) — Kaggle
- [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- Kendi veri: aylık gelir, web trafiği, sipariş sayısı

**Zorunlu bileşenler:**
- Walk-forward validation (leakage yok)
- Baseline: naïve (son değer), sezonsal ortalama
- Orta: Prophet, SARIMA
- İleri: LightGBM (tabular zaman serisi — lag + rolling features)
- Metrikler: MAPE, SMAPE, MAE, RMSE
- Görselleştirme: gerçek vs tahmin + güven aralığı

```python
# Walk-forward validation iskeliti
def walk_forward_validation(df, model_fn, train_window=365, test_window=30, step=30):
    """
    Her adımda: son train_window günü eğit, sonraki test_window günü tahmin.
    Hiçbir zaman geleceği görme.
    """
    results = []
    dates = df["date"].unique()

    for i in range(0, len(dates) - train_window - test_window, step):
        train_end = dates[i + train_window]
        test_end = dates[i + train_window + test_window]

        df_train = df[df["date"] <= train_end]
        df_test = df[(df["date"] > train_end) & (df["date"] <= test_end)]

        model = model_fn(df_train)
        preds = model.predict(df_test)

        results.append({
            "period_start": train_end,
            "mae": mean_absolute_error(df_test["target"], preds),
            "mape": mean_absolute_percentage_error(df_test["target"], preds),
        })

    return pd.DataFrame(results)
```

### Walk-Forward Validation Kontrol Listesi

- [ ] Train/test split kesinlikle kronolojik sırada yapıldı
- [ ] Walk-forward pencere boyutu belirlendi (train_window, test_window, step)
- [ ] Hiçbir feature gelecekten bilgi sızdırmıyor (lag kontrolü)
- [ ] Baseline model sonuçları raporlandı (naïve + sezonsal ortalama)
- [ ] En az 3 farklı model karşılaştırıldı (baseline, statistical, ML)
- [ ] Her fold için metrikler ayrı ayrı kaydedildi (stabilite analizi)
- [ ] Hata analizi: hangi dönemlerde / segmentlerde tahmin kötü?
- [ ] Güven aralığı (prediction interval) görselleştirildi

**Süre hedefi:** 2–3 hafta

---

## Proje-4: NLP Şikayet Sınıflandırma

> **Çapraz referans:** Bu proje `katman-4` (makine öğrenmesi), `katman-7` (NLP temelleri) ve `katman-8` (derin öğrenme / transformer) bilgisini gerektirir.

**Amaç:** Baseline'dan fine-tuned BERT'e NLP pipeline.

**Veri önerisi:**
- [Turkish Sentiment Dataset](https://huggingface.co/datasets/Turkish-NLP-Collection) — Türkçe
- [Customer Complaints](https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints) — İngilizce
- Synthetically generated Turkish complaints (GPT-4 ile oluştur)

**Üç katmanlı yaklaşım:**

```
Seviye 1 — Baseline (1 gün):
  TF-IDF + Logistic Regression
  → Hızlı, yorumlanabilir, karşılaştırma noktası

Seviye 2 — Orta (3–5 gün):
  Fine-tuned Turkish BERT (dbmdz/bert-base-turkish-cased)
  → Hugging Face Trainer API
  → Epoch başı F1-macro ölçümü

Seviye 3 — İleri (1 hafta):
  Few-shot GPT-4 ile zero/few-shot sınıflandırma VEYA
  LoRA fine-tuning (Llama 3.2 3B)
  → PEFT kütüphanesi ile
```

**Değerlendirme:**
- F1-macro (dengesiz sınıflar)
- Per-class F1 (hangi kategori zor?)
- Confusion matrix + hata analizi
- Model karşılaştırma tablosu: TF-IDF vs BERT vs GPT

### NLP Kontrol Listesi

- [ ] Metin ön işleme standardı belirlendi (lowercase, stopword, stemming/lemma)
- [ ] Sınıf dağılımı incelendi ve dengesizlik stratejisi seçildi (oversampling / class weight)
- [ ] Train/val/test split yapıldı (stratified, data leakage yok)
- [ ] Baseline model (TF-IDF + LogReg) sonuçları raporlandı
- [ ] Fine-tuned BERT eğitim eğrileri (loss, F1) izlendi — overfitting kontrolü
- [ ] Per-class F1 raporu oluşturuldu — hangi sınıf zayıf?
- [ ] Hata analizi: yanlış sınıflandırılan örnekler incelendi (pattern var mı?)
- [ ] Model karşılaştırma tablosu oluşturuldu (TF-IDF vs BERT vs GPT)
- [ ] Inference süresi (latency) karşılaştırması yapıldı
- [ ] En iyi modelin prediction API'si veya demo notebook'u hazırlandı

**Süre hedefi:** 2–4 hafta

---

## Proje-5: RecSys — Two-Stage Öneri Sistemi

> **Çapraz referans:** Bu proje `katman-4` (makine öğrenmesi), `katman-5` (feature engineering), `katman-8` (derin öğrenme — embedding) ve `katman-9` (sistem tasarımı) bilgisini gerektirir.

**Amaç:** Retrieval + ranking iki aşamalı öneri sistemi.

**Veri önerisi:**
- [MovieLens 25M](https://grouplens.org/datasets/movielens/) — büyük, ücretsiz
- [Amazon Product Reviews](https://amazon-reviews-2023.github.io/) — e-ticaret
- [Spotify Tracks](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — müzik

**Zorunlu bileşenler:**

```
Aşama 1 — Retrieval:
  Item2Vec veya Two-Tower modeli
  FAISS ile ANN arama
  Top-100 aday üretimi

Aşama 2 — Ranking:
  LightGBM ile pointwise ranking
  User × Item interaction features
  Top-10 final önerisi

Değerlendirme:
  Offline: NDCG@10, MAP@10, Coverage, Diversity
  Cold start stratejisi (yeni kullanıcı + yeni ürün)
  A/B test tasarımı (offline simülasyon)
```

**Kod — `src/recsys_evaluator.py`:**

```python
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import cosine

class RecSysEvaluator:
    def __init__(self, test_interactions: pd.DataFrame):
        """
        test_interactions: DataFrame with columns [user_id, item_id, rating/relevance]
        """
        self.test = test_interactions
        self._build_ground_truth()

    def _build_ground_truth(self):
        """Her kullanıcı için ground truth relevant item listesi oluştur."""
        self.ground_truth = (
            self.test[self.test["rating"] >= 4.0]
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

    def ndcg_at_k(self, predictions: dict, k: int = 10) -> float:
        """
        predictions: {user_id: [item_id, ...] sıralı liste}
        Normalized Discounted Cumulative Gain @ k hesaplar.
        """
        ndcg_scores = []
        for user_id, pred_items in predictions.items():
            if user_id not in self.ground_truth:
                continue
            relevant = self.ground_truth[user_id]
            pred_items_k = pred_items[:k]

            # DCG
            dcg = 0.0
            for i, item in enumerate(pred_items_k):
                if item in relevant:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 çünkü log2(1)=0

            # Ideal DCG
            ideal_hits = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    def diversity_at_k(self, predictions: dict, item_features: pd.DataFrame,
                        k: int = 10) -> float:
        """
        Ortalama intra-list diversity (ILD).
        item_features: DataFrame, index=item_id, columns=feature vektörleri.
        Her kullanıcının top-k listesindeki item çiftleri arasındaki
        ortalama cosine distance hesaplanır.
        """
        diversities = []
        feature_matrix = item_features.values
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_features.index)}

        for user_id, pred_items in predictions.items():
            top_k = [it for it in pred_items[:k] if it in item_id_to_idx]
            if len(top_k) < 2:
                continue

            pair_distances = []
            for it_a, it_b in combinations(top_k, 2):
                vec_a = feature_matrix[item_id_to_idx[it_a]]
                vec_b = feature_matrix[item_id_to_idx[it_b]]
                pair_distances.append(cosine(vec_a, vec_b))  # 0=identical, 1=orthogonal

            diversities.append(np.mean(pair_distances))

        return float(np.mean(diversities)) if diversities else 0.0

    def coverage(self, predictions: dict, all_items: set) -> float:
        """
        Önerilen item'ların tüm katalogdaki oranı.
        Yüksek coverage = daha az popularity bias.
        """
        recommended_items = set()
        for pred_items in predictions.values():
            recommended_items.update(pred_items)

        return len(recommended_items & all_items) / len(all_items) if all_items else 0.0
```

### Cold Start Stratejisi

Cold start, öneri sistemlerinin en kritik production zorluklarından biridir. Aşağıda yeni kullanıcı ve yeni ürün senaryoları için çözüm kodları yer alır.

#### Yeni Kullanıcı: Popularity-Based + Demographic Fallback

```python
import pandas as pd
import numpy as np

class ColdStartUserHandler:
    """
    Yeni kullanıcılar için kademeli fallback stratejisi:
    1. Demografik segmente göre popüler ürünler
    2. Global popülerlik (son fallback)

    RecSys 2025 araştırmalarına göre LLM tabanlı enrichment ile
    kullanıcı metadata'sından yapay etkileşim geçmişi oluşturulabilir,
    ancak bu basit ve kanıtlanmış yaklaşım üretimde güvenilir bir başlangıç sağlar.
    """

    def __init__(self, interactions: pd.DataFrame, user_meta: pd.DataFrame,
                 item_meta: pd.DataFrame):
        self.interactions = interactions
        self.user_meta = user_meta   # columns: [user_id, age_group, gender, country]
        self.item_meta = item_meta

        self._build_popularity_tables()

    def _build_popularity_tables(self):
        """Segment bazlı ve global popülerlik tabloları oluştur."""
        merged = self.interactions.merge(self.user_meta, on="user_id", how="left")

        # Global popülerlik: Bayesian average ile smooth
        item_counts = self.interactions.groupby("item_id").agg(
            n_ratings=("rating", "count"),
            mean_rating=("rating", "mean"),
        )
        C = item_counts["mean_rating"].mean()  # global ortalama
        m = item_counts["n_ratings"].quantile(0.25)  # minimum oy eşiği
        item_counts["bayesian_score"] = (
            (item_counts["n_ratings"] * item_counts["mean_rating"] + m * C)
            / (item_counts["n_ratings"] + m)
        )
        self.global_popular = (
            item_counts.sort_values("bayesian_score", ascending=False).index.tolist()
        )

        # Demografik segment popülerliği
        self.segment_popular = {}
        for segment_col in ["age_group", "gender", "country"]:
            if segment_col not in merged.columns:
                continue
            seg_pop = (
                merged.groupby([segment_col, "item_id"])["rating"]
                .agg(["count", "mean"])
                .reset_index()
            )
            seg_pop["score"] = seg_pop["count"] * seg_pop["mean"]
            for seg_val, group in seg_pop.groupby(segment_col):
                key = (segment_col, seg_val)
                self.segment_popular[key] = (
                    group.sort_values("score", ascending=False)["item_id"].tolist()
                )

    def recommend(self, user_id: str, n: int = 10) -> list:
        """
        Yeni kullanıcı için öneri üret.
        Öncelik: demografik segment → global popülerlik.
        """
        user_info = self.user_meta[self.user_meta["user_id"] == user_id]

        if user_info.empty:
            return self.global_popular[:n]

        user_row = user_info.iloc[0]
        recommendations = []
        seen = set()

        # Demografik segmentlerden sırayla öneriler topla
        for seg_col in ["age_group", "gender", "country"]:
            if seg_col not in user_row.index:
                continue
            key = (seg_col, user_row[seg_col])
            if key in self.segment_popular:
                for item in self.segment_popular[key]:
                    if item not in seen:
                        recommendations.append(item)
                        seen.add(item)
                    if len(recommendations) >= n:
                        return recommendations[:n]

        # Fallback: global popülerlik
        for item in self.global_popular:
            if item not in seen:
                recommendations.append(item)
                seen.add(item)
            if len(recommendations) >= n:
                break

        return recommendations[:n]
```

#### Yeni Ürün: Content-Based Feature ile Warm Start

```python
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

class ColdStartItemHandler:
    """
    Yeni ürün cold start: content-based feature ile warm start.

    RecSys 2025'te öne çıkan yaklaşım: Semantic ID tokenization ile
    yeni item'ların kalitesini etkileşim verisi olmadan çıkarmak.
    Bu implementasyon, item content feature'larından benzer warm item'ların
    embedding'lerini ödünç alarak (borrow) yeni item'ı sisteme entegre eder.
    """

    def __init__(self, item_features: np.ndarray, item_ids: list,
                 item_embeddings: np.ndarray):
        """
        item_features:   (n_items, n_content_features) — kategori, fiyat, metin TF-IDF vs.
        item_ids:        item ID listesi (aynı sırada)
        item_embeddings: (n_items, embedding_dim) — collaborative filtering embeddings
        """
        self.item_features = normalize(item_features)
        self.item_ids = item_ids
        self.item_embeddings = item_embeddings
        self.id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    def warm_start_embedding(self, new_item_features: np.ndarray,
                              top_k_neighbors: int = 10) -> np.ndarray:
        """
        Yeni ürün için embedding oluştur:
        1. Content feature'larına göre en benzer top_k warm item'ı bul
        2. Bu item'ların collaborative embedding'lerinin ağırlıklı ortalamasını al
        3. Ağırlık = content similarity score
        """
        new_feat = normalize(new_item_features.reshape(1, -1))
        similarities = cosine_similarity(new_feat, self.item_features)[0]

        # En benzer k item
        top_k_idx = np.argsort(similarities)[-top_k_neighbors:][::-1]
        top_k_sims = similarities[top_k_idx]

        # Similarity-weighted average embedding
        weights = top_k_sims / top_k_sims.sum()
        warm_embedding = np.average(
            self.item_embeddings[top_k_idx], axis=0, weights=weights
        )

        return warm_embedding

    def recommend_similar_to_new(self, new_item_features: np.ndarray,
                                  n: int = 10) -> list:
        """
        Yeni ürün content feature'ına göre 'buna benzer ürünler' listesi döndür.
        (Yeni ürünü tanıtmak için complementary recommendations.)
        """
        new_feat = normalize(new_item_features.reshape(1, -1))
        similarities = cosine_similarity(new_feat, self.item_features)[0]
        top_idx = np.argsort(similarities)[-n:][::-1]
        return [self.item_ids[i] for i in top_idx]
```

### RecSys Kontrol Listesi

- [ ] Etkileşim verisinde zaman bazlı train/test split yapıldı (geleceğe bakma yok)
- [ ] Retrieval aşaması çalışıyor: top-100 aday üretimi (ANN / FAISS)
- [ ] Ranking aşaması çalışıyor: top-10 final sıralama
- [ ] NDCG@10, MAP@10 metrikleri hesaplandı
- [ ] Coverage ve Diversity metrikleri raporlandı (popularity bias kontrolü)
- [ ] Cold start — yeni kullanıcı: popularity + demographic fallback test edildi
- [ ] Cold start — yeni ürün: content-based warm start embedding test edildi
- [ ] Offline A/B test simülasyonu yapıldı (replay evaluation)
- [ ] Serving latency ölçüldü (retrieval + ranking toplam < 100ms hedefi)
- [ ] Item ve user embedding'ler periyodik güncelleme planı belirlendi

**Süre hedefi:** 3–6 hafta

---

## Proje-6: MLOps Mini Platform

> **Çapraz referans:** Bu proje `katman-9` (sistem tasarımı), `katman-10` (MLOps & deployment) ve `katman-4` (makine öğrenmesi — Proje-1 modeli) bilgisini gerektirir.

**Amaç:** Proje-1'i tam MLOps pipeline'ına dönüştür.

**Zorunlu bileşenler:**

| Bileşen | Teknoloji | Açıklama |
|---------|-----------|---------|
| Kod paketleme | src/ yapısı | Notebook → üretilebilir kod |
| Servis | FastAPI | /health + /predict + /predict/batch |
| Konteynerleşme | Docker + docker-compose | Build + run |
| Deney takibi | MLflow | Params, metrics, artifacts |
| Model registry | MLflow Registry | Staging → Production |
| Drift monitoring | Evidently | PSI + KS + prediction drift |
| CI/CD | GitHub Actions | Test + build + evaluate |
| Dashboard | Streamlit | Model performance + drift |

**GitHub Actions pipeline:**
```yaml
# Test → Eğit → Değerlendir → Docker Build → Test → (Deploy)
```

**Streamlit dashboard:**
```python
import streamlit as st

st.title("Churn Model — MLOps Dashboard")

# Drift raporu
col1, col2, col3 = st.columns(3)
col1.metric("Current AUC", "0.84", "-0.03")
col2.metric("Max PSI", "0.12", "+0.02", delta_color="inverse")
col3.metric("Predictions/day", "15,240", "+240")

# Drift visualization
# ... Evidently çıktıları
```

**Süre hedefi:** 2–4 hafta

---

## Proje-7: Sistem Tasarım Dokümanı

> **Çapraz referans:** Bu proje `katman-9` (sistem tasarımı), `katman-10` (MLOps) ve `katman-6` (deney tasarımı — A/B test planlaması) bilgisini gerektirir.

**Amaç:** Senior DS'in yapabileceği şeyi göster: teknik kararları yazıya dökmek.

**Senaryo:** "Real-time fraud tespiti için ML sistemi tasarla."

**Çıktı:** Teknik doküman (5–10 sayfa)

**Zorunlu bölümler:**

```markdown
## 1. Problem Tanımı
- Gereksinimler: latency, throughput, uptime
- Kısıtlar: bütçe, ekip boyutu, mevcut altyapı

## 2. Yüksek Seviye Mimari
- Bileşenler ve veri akışı diyagramı
- Online vs offline ayrımı

## 3. Feature Store Tasarımı
- Online store: hangi feature'lar, hangi teknoloji
- Offline store: training için
- Feature güncelleme sıklığı

## 4. Model Tasarımı
- Model seçimi gerekçesi (LightGBM, neden?)
- İlk sürüm vs gelecek sürüm
- Retrieval stage gerekiyor mu?

## 5. Deployment Stratejisi
- Shadow mode → A/B test → tam rollout
- Rollback planı
- Version yönetimi

## 6. İzleme ve Alarm
- Servis metrikleri (latency, error rate)
- Data drift (PSI, KS)
- Business metric (fraud yakalama oranı)
- Alert threshold ve sahipleri

## 7. Retraining Stratejisi
- Zaman bazlı vs trigger bazlı
- Label delay yönetimi
- Champion-challenger yaklaşımı

## 8. Maliyet Tahmini
- Compute: serving + training
- Storage: online + offline
- Toplam aylık tahmini

## 9. Riskler ve Kısıtlar
- Teknik riskler
- İş riskleri (yanlış pozitif maliyeti)
- Düzenleyici riskler (KVKK, PCI-DSS)

## 10. İlk 3 Ay Yol Haritası
- Ay 1: Temel pipeline
- Ay 2: Monitoring + A/B test
- Ay 3: İyileştirmeler + ölçekleme
```

#### Proje-7 Sistem Tasarımı Doküman Şablonu

````markdown
# [Sistem Adı] — Sistem Tasarımı Dokümanı
**Tarih:** YYYY-MM-DD | **Yazar:** [Ad] | **Durum:** Taslak / İncelemede / Onaylandı

## 1. Problem Tanımı
[Ne inşa ediyoruz? Neden gerekli? Hangi iş metriğini etkiliyor?]

## 2. Gereksinimler
### Fonksiyonel
- [ ] [Temel özellik 1]
- [ ] [Temel özellik 2]

### Non-Fonksiyonel (SLA'lar)
| Metrik | Hedef |
|--------|-------|
| Latency (p99) | < 200ms |
| Throughput | 1000 req/s |
| Erişilebilirlik | %99.9 |

## 3. Yüksek Seviye Mimari
[ASCII diyagram veya metin açıklama]

## 4. Veri Akışı
1. Kullanıcı → [bileşen A]
2. [Bileşen A] → [bileşen B]
3. [Bileşen B] → Yanıt

## 5. Bileşen Detayları
### [Bileşen Adı]
- **Sorumluluk:**
- **Teknoloji seçimi ve gerekçe:**
- **Ölçekleme stratejisi:**

## 6. Latency Bütçesi
| Bileşen | Bütçe | Gerçek |
|---------|-------|--------|
| Network | 10ms | - |
| Feature fetch | 20ms | - |
| Model inference | 30ms | - |
| **Toplam** | **60ms** | - |

## 7. Trade-off Analizi
| Seçenek | Artılar | Eksiler | Neden Seçilmedi/Seçildi |
|---------|---------|---------|------------------------|

## 8. Açık Sorular
- [ ] [Yanıtlanması gereken soru 1]

## 9. Sonraki Adımlar
- [ ] [Aksiyon 1] — Sorumlu: [Ad] — Tarih: YYYY-MM-DD
````

### Sistem Tasarım Dokümanı Kontrol Listesi

- [ ] Problem tanımı: SLA'lar net (latency < X ms, uptime > Y%)
- [ ] Mimari diyagram çizildi (online/offline path ayrı gösterildi)
- [ ] Feature store tasarımı: online vs offline ayrımı yapıldı
- [ ] Model seçimi gerekçelendirildi (neden bu model, neden bu mimari?)
- [ ] Deployment stratejisi: shadow → canary → full rollout adımları tanımlandı
- [ ] Monitoring: servis metrikleri + data drift + business metric ayrı ayrı tanımlandı
- [ ] Alert eşikleri ve sahipleri (on-call) belirlendi
- [ ] Retraining tetikleme koşulu belirlendi (zamana vs performansa dayalı)
- [ ] Maliyet tahmini yapıldı (compute + storage + ekip zamanı)
- [ ] Risk matrisi: teknik + iş + düzenleyici riskler listelendi
- [ ] İlk 3 aylık yol haritası zaman çizelgesine dönüştürüldü

**Süre hedefi:** 1–2 hafta

---

## Portföy Değerlendirme Kriterleri

### İşe Alım Yöneticisi Gözüyle

**Ne öne çıkar?**
- Gerçek veri + gerçek problem (Titanic değil)
- Uçtan uca: veri → model → servis (notebook yeterli değil)
- İş etkisi: "Bu model ne kattı?" sorusunu yanıtlıyor
- Temiz kod: GitHub'daki README 30 saniyede anlaşılıyor

**Ne hayal kırıklığı yaratır?**
- Sadece Jupyter notebook, src/ yok
- Leakage var (zamanla fark edilir)
- "AUC 0.95" ama baseline yok
- Hiç test yok, hiç monitoring yok

### Portföy Sunuş Şablonu (30 saniye)

```
"Bu proje [PROBLEM]'i çözüyor. [VERİ] kullandım.
[YÖNTEM] ile [METRİK] elde ettim.
Bu sayede [İŞ ETKİSİ] sağladım.
Prod'a aldım / A/B test tasarladım / [SONRAKI ADIM] var."
```

Örnek:
```
"Bu proje e-ticaret müşteri churn'ünü tahmin ediyor.
3 yıllık sipariş geçmişi ve 47 feature kullandım.
LightGBM + Optuna ile ROC-AUC 0.87 elde ettim.
Maliyet matrisine göre threshold seçerek aylık 2M TL churner değerini
yakalamayı hedefliyoruz. FastAPI + Docker ile servis edildi,
Evidently ile drift izleniyor."
```

### Sektör Notu — Portföy 2026

2026 araştırmalarına göre işe alım değerlendirmesi:
- İşe alım yöneticilerinin %62'si portföyü <30 saniye tarar → scannable olmalı
- Production-ready proje (Docker, tests, monitoring) %40+ daha fazla dikkat çekiyor
- LLM/RAG projesi portföye eklemek 2026'da artı puan — ama "ChatGPT wrapper" değil, gerçek problem çözümü
- A/B test analizi veya causal inference içeren proje "growth DS" ve "product DS" rolleri için çok değerli

---

## Portföy Sunumu: GitHub Repo Yapısı ve Standartlar

### Önerilen GitHub Organizasyonu

```
github.com/kullanici-adi/
├── ds-portfolio/                  # Ana portföy repo'su (pinned)
│   ├── README.md                  # Portföy özeti + proje linkleri
│   └── resume.pdf                 # Opsiyonel
├── churn-prediction/              # Proje-1 (bağımsız repo)
├── ab-test-analysis/              # Proje-2
├── timeseries-forecasting/        # Proje-3
├── nlp-complaint-classifier/      # Proje-4
├── recsys-two-stage/              # Proje-5
├── mlops-churn-platform/          # Proje-6
└── fraud-system-design/           # Proje-7
```

**Neden ayrı repo?** Her proje bağımsız olarak clone edilebilir, kendi CI/CD pipeline'ı ve dependency listesi olur. Ana portföy repo'su sadece bir "vitrin" görevi görür.

### README Standardı (Her Proje İçin)

Her README aşağıdaki bölümleri **bu sırayla** içermeli:

| Bölüm | İçerik | Uzunluk |
|-------|--------|---------|
| Başlık + tek satır açıklama | Ne yapıyor? | 1–2 satır |
| Badges | CI status, Python sürümü, lisans | 1 satır |
| Hızlı demo (GIF/screenshot) | Görsel sonuç | 1 görsel |
| Problem & Motivasyon | Neden bu proje? İş etkisi ne? | 3–5 satır |
| Veri | Kaynak, boyut, hedef değişken | 2–3 satır |
| Yöntem & Mimari | Akış diyagramı veya bullet list | 5–10 satır |
| Sonuçlar | Metrik tablosu + en önemli grafik | Tablo + 1 görsel |
| Kurulum & Çalıştırma | `pip install` + komutlar | 5 satır |
| Proje yapısı | Dizin ağacı | tree çıktısı |
| Sınırlılıklar & Gelecek çalışma | Dürüst değerlendirme | 3–5 satır |

**Kritik ipucu:** İşe alım yöneticisi 30 saniyede karar verir. README'nin ilk 3 satırı ve ilk görsel her şeyi belirler.

### Demo / Video Önerileri

| Format | Ne zaman? | Araç |
|--------|-----------|------|
| **Ekran GIF** | Dashboard, UI çıktısı, terminal komutu | [Peek](https://github.com/phw/peek), [Gifski](https://gif.ski/) |
| **Kısa video (2–3 dk)** | Projeyi anlatırken (Loom tarzı) | [Loom](https://www.loom.com/), OBS Studio |
| **Streamlit demo** | İnteraktif model çıktısı | Streamlit Cloud (ücretsiz) |
| **Hugging Face Space** | NLP / LLM demo | Gradio + HF Spaces (ücretsiz) |
| **Statik site** | Portföy vitrini | GitHub Pages + Jekyll / Hugo |

**Video yapı önerisi (2–3 dakika):**
1. **(0:00–0:20)** Problem: "Bu proje X sorununu çözüyor"
2. **(0:20–1:00)** Veri & yöntem: ekranda notebook / mimari diyagram göster
3. **(1:00–1:40)** Sonuçlar: metrik tablosu + en etkileyici grafik
4. **(1:40–2:20)** Canlı demo: API call veya dashboard etkileşimi
5. **(2:20–2:40)** Sınırlılıklar ve sonraki adım

**Portföy vitrini README.md şablonu (ana repo):**
```markdown
# Data Science Portfolio — [İsim]

[1 cümle: Kim olduğun + ne yaptığın]

## Projeler

| # | Proje | Alan | Teknolojiler | Demo |
|---|-------|------|-------------|------|
| 1 | [Churn Prediction](link) | ML / Tabular | LightGBM, Optuna, SHAP | [Streamlit](link) |
| 2 | [A/B Test Analysis](link) | İstatistik | Bayesian, CUPED, Bootstrap | [Notebook](link) |
| 3 | [Sales Forecasting](link) | Zaman Serisi | Prophet, LightGBM | [Rapor](link) |
| 4 | [Complaint Classifier](link) | NLP | BERT, LoRA, GPT-4 | [HF Space](link) |
| 5 | [RecSys](link) | Öneri Sistemi | Two-Tower, FAISS, LightGBM | [Demo](link) |
| 6 | [MLOps Platform](link) | MLOps | Docker, MLflow, FastAPI | [Video](link) |
| 7 | [Fraud System Design](link) | Sistem Tasarımı | Doküman | [PDF](link) |

## İletişim
- LinkedIn: [link]
- E-posta: [e-posta]
```

---

> **Araştırma kaynakları:**
> - [Data Science Portfolio Projects 2025 — ProjectPro](https://www.projectpro.io/article/data-science-portfolio-projects/954)
> - [How to Build a Data Science Portfolio (2026) — BrainStation](https://brainstation.io/career-guides/how-to-build-a-data-science-portfolio)
> - [Stand-out Portfolio with GitHub — KDnuggets](https://www.kdnuggets.com/develop-stand-out-data-science-portfolio-github)
> - [RecSys 2025 — Cold Start Session](https://recsys.acm.org/recsys24/session-7/)
> - [Cold Start Recommendation Resources — GitHub](https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation)
> - [RecSys 2025: LLMs and Personalization — Taboola Engineering](https://www.taboola.com/engineering/recsys-2025-ai-recommendation-trends/)

---

<div class="nav-footer">
  <span><a href="#file_katman_H_buyuk_veri">← Önceki: Katman H — Büyük Veri</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_mulakat">Sonraki: Mülakat Hazırlığı →</a></span>
</div>
