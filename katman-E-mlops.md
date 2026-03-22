# Katman E — MLOps ve Üretimleme

> Bu katmanda ne öğrenilir: Modeli notebook'tan prod'a taşıma. Kod paketleme, FastAPI servis, Docker, MLflow, izleme, drift tespiti ve CI/CD.
>
> Süre: 2–4 hafta. 2026 itibarıyla MLOps bilgisi DS rolleri için "nice to have" değil "must have".
>
> **Çapraz referanslar:** Veri doğrulama temelleri → [Katman A](katman-A-temeller.md) | Feature store & sistem tasarımı → [Katman F](katman-F-sistem-tasarimi.md)


<div class="prereq-box">
<strong>Önkoşul:</strong> <strong>Katman B veya D</strong> tamamlanmış ve en az bir çalışan model mevcut olmalı. Docker ve terminal kullanımı faydalıdır.
</div>

---

## E.1 Model Yaşam Döngüsü

### Sezgisel Açıklama

Model eğitmek işin yarısı. Asıl iş:
1. Modeli güvenilir şekilde serve et
2. Prod'da ne yapıyor izle
3. Bozulunca (drift) yeniden eğit
4. Yeni sürümü güvenli dağıt

Bu döngüyü otomatize eden altyapı = MLOps.

```
Veri toplama → Feature engineering → Model eğitimi → Değerlendirme
     ↑                                                       ↓
     ↑              Feedback loop ←──────────────── Deploy → Monitor
     ↑                                                       ↓
     └──────────────────────── Retrain trigger ─────────────┘
```

---

## E.2 Kod Paketleme (Notebook → src/)

### Sezgisel Açıklama

Notebook = keşif. `src/` = üretimde çalışan kod.

Kural: Aynı kod iki yerde olmamalı. Notebook'ta `from src.features import build_features` çağırılmalı.

```
proje/
├── notebooks/
│   └── 01_eda.ipynb          # Keşif, deneysel
├── src/
│   ├── __init__.py
│   ├── data.py               # Veri yükleme, doğrulama
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model eğitimi CLI
│   ├── predict.py            # Tahmin CLI
│   ├── evaluate.py           # Metrik hesabı
│   └── api.py                # FastAPI servisi
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── configs/
│   └── config.yaml           # Hyperparameter + path'ler
├── Dockerfile
├── pyproject.toml
└── README.md
```

### Kod Örneği — Train Script

```python
# src/train.py
import argparse
import logging
import yaml
import joblib
from pathlib import Path

import mlflow
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

from src.data import load_train_data
from src.features import build_features

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    logger.info("Veri yükleniyor...")
    df_train, df_val = load_train_data(cfg["data"])

    logger.info("Feature engineering...")
    X_train, y_train = build_features(df_train, cfg["features"], fit=True)
    X_val, y_val = build_features(df_val, cfg["features"], fit=False)

    logger.info("Model eğitiliyor...")
    with mlflow.start_run():
        # Parametreleri logla
        mlflow.log_params(cfg["model"]["params"])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))

        model = lgb.LGBMClassifier(**cfg["model"]["params"])
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # Metrikleri logla
        from sklearn.metrics import roc_auc_score, average_precision_score
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        ap = average_precision_score(y_val, val_proba)

        mlflow.log_metric("val_roc_auc", auc)
        mlflow.log_metric("val_pr_auc", ap)
        logger.info(f"Val ROC-AUC: {auc:.4f}, PR-AUC: {ap:.4f}")

        # ── Model Kaydetme Stratejisi ──
        # Birincil yöntem: MLflow log_model
        # → Model + metadata + bağımlılıklar birlikte versiyonlanır
        # → Model Registry ile stage geçişleri (Staging → Production)
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=cfg["model"].get("registry_name", "ChurnModel"),
            input_example=X_train[:3],   # Serving schema'sını otomatik çıkarır
        )
        logger.info("Model MLflow Registry'ye kaydedildi.")

        # Legacy/fallback: joblib ile lokal kopya
        # → Hızlı test, MLflow server yokken geliştirme ortamı
        model_path = Path(cfg["model"]["save_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Lokal kopya (joblib): {model_path}")

    return model, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
```

---

## E.3 FastAPI ile Model Servisi

```python
# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import numpy as np
import logging
import uuid
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="Müşteri churn olasılık tahmini"
)

# Model yükle (startup'ta)
MODEL_PATH = Path("model.pkl")       # Legacy/fallback
MLFLOW_MODEL_URI = os.getenv(         # Birincil: MLflow Registry
    "MLFLOW_MODEL_URI", "models:/ChurnModel/Production"
)
model = None

@app.on_event("startup")
async def load_model():
    global model
    # Birincil: MLflow Registry'den yükle
    try:
        import mlflow
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
        logger.info(f"Model MLflow'dan yüklendi: {MLFLOW_MODEL_URI}")
        return
    except Exception as e:
        logger.warning(f"MLflow yükleme başarısız ({e}), joblib fallback deneniyor...")
    # Fallback: Lokal joblib dosyası
    if MODEL_PATH.exists():
        import joblib
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model joblib'den yüklendi: {MODEL_PATH}")
    else:
        logger.error("Model yüklenemedi — ne MLflow ne de joblib!")

# Request / Response şemaları
class PredictRequest(BaseModel):
    user_id: str = Field(..., description="Kullanıcı ID")
    n_orders: int = Field(..., ge=0, description="Sipariş sayısı")
    total_spend: float = Field(..., ge=0, description="Toplam harcama (TL)")
    days_since_last_order: int = Field(..., ge=0, description="Son siparişten gün")
    avg_order_value: Optional[float] = Field(None, ge=0)
    country: str = Field(..., description="Ülke kodu")

    @validator("country")
    def validate_country(cls, v):
        allowed = {"TR", "DE", "GB", "US", "FR"}
        if v not in allowed:
            raise ValueError(f"Geçersiz ülke: {v}. İzin verilenler: {allowed}")
        return v

class PredictResponse(BaseModel):
    user_id: str
    churn_probability: float
    risk_level: str      # low / medium / high
    request_id: str
    latency_ms: float

@app.get("/health")
async def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil")

    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    try:
        # Feature vektörü oluştur
        features = np.array([[
            request.n_orders,
            request.total_spend,
            request.days_since_last_order,
            request.avg_order_value or (request.total_spend / max(request.n_orders, 1)),
            1 if request.country == "TR" else 0,
        ]])

        prob = float(model.predict_proba(features)[0, 1])

        # Risk seviyesi
        if prob < 0.3:
            risk = "low"
        elif prob < 0.7:
            risk = "medium"
        else:
            risk = "high"

        latency = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] user={request.user_id} prob={prob:.3f} "
                    f"risk={risk} latency={latency:.1f}ms")

        return PredictResponse(
            user_id=request.user_id,
            churn_probability=round(prob, 4),
            risk_level=risk,
            request_id=request_id,
            latency_ms=round(latency, 1)
        )

    except Exception as e:
        logger.error(f"[{request_id}] Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: list[PredictRequest]):
    """Toplu tahmin (batch serving)."""
    results = []
    for req in requests:
        result = await predict(req)
        results.append(result)
    return results
```

> **Senior Notu:** FastAPI'da Pydantic validation kritik. Üretimde gelecek kötü input'lar model'i patlattırır. Timeout, rate limiting ve authentication (API key veya JWT) ekle. Metrics için Prometheus middleware ekle.

---

## E.4 Docker

```dockerfile
# Dockerfile — multi-stage build
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# Final image — sadece gerekli
FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (libgomp = LightGBM için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini builder'dan kopyala
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

# Uygulama kodu
COPY src/ ./src/
COPY model.pkl .
COPY configs/ ./configs/

# Güvenlik: root olmayan kullanıcı
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model.pkl:/app/model.pkl:ro
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

---

## E.5 Experiment Tracking — MLflow

### Model Kaydetme Stratejisi

| Yöntem | Ne Zaman | Avantaj | Dezavantaj |
|--------|----------|---------|------------|
| `mlflow.sklearn.log_model()` | **Birincil (varsayılan)** | Versiyon, metadata, bağımlılıklar birlikte | Tracking server gerekli |
| `joblib.dump()` | Legacy / hızlı test | Basit, bağımsız | Versiyon yok, metadata yok |
| `model.save_model()` (native) | Framework-specific | En küçük dosya boyutu | Taşınabilirlik düşük |

> **Kural:** Üretimde her zaman MLflow `log_model` + Model Registry kullan. `joblib` sadece geliştirme ortamında hızlı iterasyon veya MLflow server'ı olmayan ortamlarda fallback olarak kullan.

### Model Registry Best Practices (2026)

1. **İsimlendirme:** Ürün odaklı isimler kullan (`ChurnDetector`, `FraudScorer`), versiyon numarasını isme ekleme — Registry halleder.
2. **Gated promotion:** Staging → Production geçişi otomatik test suite'ten geçmeli, manuel onay gereksin.
3. **Metadata:** Her run'a hyperparameters + veri sürümü (DVC commit hash) + feature listesi logla.
4. **Backend:** Üretimde PostgreSQL backend store + S3/GCS artifact store kullan (SQLite çok kullanıcılı erişimde bozulur).
5. **`input_example`:** `log_model`'a her zaman `input_example` geç — serving schema'sını otomatik çıkarır, train-serve uyumsuzluğunu önler.

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# MLflow Tracking Server başlat: mlflow ui --host 0.0.0.0 --port 5000

# Deney takibi
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction-v2")

with mlflow.start_run(run_name="lgbm-optuna-v1"):
    # Parametreler
    mlflow.log_params({
        "model_type": "LightGBM",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "feature_count": X_train.shape[1],
        "data_version": "abc1234",       # DVC commit hash
    })

    # Metrikler (adım bazlı)
    for epoch, (train_auc, val_auc) in enumerate(zip(train_aucs, val_aucs)):
        mlflow.log_metrics({"train_auc": train_auc, "val_auc": val_auc}, step=epoch)

    # Final metrikler
    mlflow.log_metrics({"final_val_auc": 0.87, "final_pr_auc": 0.64})

    # Artifact'lar
    mlflow.log_artifact("reports/feature_importance.png")
    mlflow.log_artifact("reports/calibration_plot.png")

    # Model kaydet — input_example ile schema otomatik çıkarılır
    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="ChurnModelProd",
        input_example=X_train[:3],
    )

# Model Registry işlemleri
client = MlflowClient()

# Stage geçişleri (Gated promotion workflow)
client.transition_model_version_stage(
    name="ChurnModelProd",
    version=3,
    stage="Staging"      # None → Staging → Production → Archived
)

# Production'daki en son modeli yükle
model = mlflow.sklearn.load_model("models:/ChurnModelProd/Production")
```

---

## E.6 Model İzleme (Monitoring)

### Drift Türleri

| Drift Türü | Tanım | Tespit |
|-----------|-------|--------|
| Covariate shift | X dağılımı değişti | KS testi, PSI |
| Prior shift | y dağılımı değişti | Tahmin ortalamasını izle |
| Concept drift | P(y\|X) değişti | Ground truth gelince karşılaştır |
| Feature drift | Bir feature aniden 0 veya sabit | Min/max/std izle |

### Kod Örneği — Drift İzleme

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
import pandas as pd
from scipy import stats
import numpy as np

# 1. Evidently raporu — HTML + JSON çıktı
def create_drift_report(df_reference: pd.DataFrame,
                          df_current: pd.DataFrame,
                          output_path: str = "drift_report.html"):
    """Kapsamlı drift raporu oluştur (HTML + JSON)."""
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
    ])
    report.run(reference_data=df_reference, current_data=df_current)

    # HTML rapor — tarayıcıda interaktif görselleştirme
    report.save_html(output_path)
    print(f"Drift raporu (HTML) kaydedildi: {output_path}")

    # JSON çıktı — programatik erişim ve alert entegrasyonu
    import json
    json_output = report.as_dict()
    json_path = output_path.replace(".html", ".json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"Drift raporu (JSON) kaydedildi: {json_path}")

    return json_output


# 1b. Alert entegrasyonu — Slack / Email webhook
def send_drift_alert(drift_results: dict, webhook_url: str,
                      channel: str = "ml-alerts"):
    """Drift tespit edilince Slack veya email webhook'a bildirim gönder."""
    import requests

    # Drift metriklerini özetle
    metrics = drift_results.get("metrics", [])
    drifted_features = []
    for m in metrics:
        result = m.get("result", {})
        if result.get("dataset_drift", False):
            drifted_features.append(
                f"Dataset drift tespit edildi! "
                f"Drifted features: {result.get('number_of_drifted_columns', '?')}"
                f"/{result.get('number_of_columns', '?')}"
            )

    if not drifted_features:
        return  # Drift yok, alert gönderme

    message = {
        "channel": channel,
        "text": (
            "🚨 *ML Model Drift Alert*\n"
            f"{''.join(drifted_features)}\n"
            f"Detay: drift_report.html"
        ),
    }
    try:
        resp = requests.post(webhook_url, json=message, timeout=10)
        resp.raise_for_status()
        print(f"Alert gönderildi: {channel}")
    except requests.RequestException as e:
        print(f"Alert gönderilemedi: {e}")

# 2. PSI (Population Stability Index) — manuel hesaplama
def calculate_psi(reference: np.ndarray, current: np.ndarray,
                   n_bins: int = 10) -> float:
    """
    PSI hesapla.
    < 0.1: Stabil
    0.1 – 0.2: Hafif değişim (izle)
    > 0.2: Ciddi değişim (retraining tetikle)
    """
    # Referans dağılımına göre bin sınırları
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-8  # Sınır dahil olsun

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current, bins=bins)[0]

    # Sıfır yerine küçük değer koy
    ref_pcts = np.maximum(ref_counts / len(reference), 1e-8)
    cur_pcts = np.maximum(cur_counts / len(current), 1e-8)

    psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
    return psi

# 3. KS testi — feature drift tespiti
def ks_drift_test(reference: np.ndarray, current: np.ndarray,
                   threshold: float = 0.05) -> dict:
    """KS testi ile covariate shift tespiti."""
    ks_stat, p_value = stats.ks_2samp(reference, current)
    return {
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "drift_detected": p_value < threshold
    }

# 4. Monitoring pipeline
def monitoring_report(df_reference: pd.DataFrame, df_current: pd.DataFrame,
                        features: list, prediction_col: str = "predicted_prob"):
    """Günlük/haftalık monitoring raporu."""
    report = {}

    for feature in features:
        if pd.api.types.is_numeric_dtype(df_reference[feature]):
            psi = calculate_psi(df_reference[feature].dropna(),
                                 df_current[feature].dropna())
            ks_result = ks_drift_test(df_reference[feature].dropna(),
                                       df_current[feature].dropna())
            report[feature] = {
                "psi": round(psi, 4),
                "ks_p_value": round(ks_result["p_value"], 4),
                "drift_psi": psi > 0.1,
                "drift_ks": ks_result["drift_detected"],
                "ref_mean": df_reference[feature].mean(),
                "cur_mean": df_current[feature].mean(),
                "mean_change_pct": (df_current[feature].mean() /
                                     df_reference[feature].mean() - 1)
            }

    # Prediction distribution
    report["prediction"] = {
        "ref_mean": df_reference[prediction_col].mean(),
        "cur_mean": df_current[prediction_col].mean(),
        "drift": abs(df_current[prediction_col].mean() -
                     df_reference[prediction_col].mean()) > 0.05
    }

    return pd.DataFrame(report).T
```

### PSI Eşikleri

```
PSI < 0.10 → Stabil, sorun yok
0.10 ≤ PSI < 0.20 → Hafif değişim, izlemeye devam
PSI ≥ 0.20 → Ciddi değişim → Retraining tetikle, alarmı tetikle
```

### Evidently Monitoring Dashboard

Evidently sadece tek seferlik rapor değil, sürekli izleme dashboard'u olarak da kullanılabilir:

```python
# Evidently + Streamlit ile canlı monitoring dashboard
# Kurulum: pip install evidently streamlit

# dashboard_app.py
import streamlit as st
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import pandas as pd
from datetime import datetime, timedelta

st.title("ML Model Monitoring Dashboard")

# Zaman aralığı seç
window = st.selectbox("Zaman penceresi", ["Son 1 gün", "Son 7 gün", "Son 30 gün"])

# Veriyi yükle (örnek — gerçekte DB'den çekilir)
df_reference = pd.read_parquet("data/reference.parquet")
df_current = pd.read_parquet("data/current.parquet")

# Rapor oluştur
report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=df_reference, current_data=df_current)

# Streamlit'te göster
st.header("Data Drift Özeti")
drift_dict = report.as_dict()
for metric in drift_dict["metrics"]:
    result = metric.get("result", {})
    if "drift_share" in result:
        drift_pct = result["drift_share"] * 100
        status = "Stabil" if drift_pct < 20 else "DİKKAT"
        st.metric("Drift Oranı", f"{drift_pct:.1f}%", delta=status)

# HTML raporu iframe ile göster
report.save_html("temp_report.html")
with open("temp_report.html", "r") as f:
    st.components.v1.html(f.read(), height=800, scrolling=True)
```

> **Çalıştırma:** `streamlit run dashboard_app.py` → Tarayıcıda interaktif drift dashboard'u.
> Cron job ile günlük rapor oluşturup `send_drift_alert()` ile Slack'e bildirim gönderebilirsin.

---

## E.7 Retraining Stratejileri

```python
# Retraining karar ağacı
class RetrainingController:
    def __init__(self, psi_threshold=0.2, perf_threshold=0.05):
        self.psi_threshold = psi_threshold
        self.perf_threshold = perf_threshold

    def should_retrain(self, psi_scores: dict, current_auc: float,
                        baseline_auc: float) -> dict:
        reasons = []

        # 1. Drift trigger
        max_psi = max(psi_scores.values())
        if max_psi > self.psi_threshold:
            reasons.append(f"PSI={max_psi:.3f} eşiği geçti ({self.psi_threshold})")

        # 2. Performance trigger
        perf_drop = baseline_auc - current_auc
        if perf_drop > self.perf_threshold:
            reasons.append(f"AUC {perf_drop:.3f} düştü ({self.perf_threshold} eşiği)")

        return {
            "should_retrain": len(reasons) > 0,
            "reasons": reasons,
            "max_psi": max_psi,
            "auc_drop": baseline_auc - current_auc,
        }
```

---

## E.8 CI/CD for ML (GitHub Actions)

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"  # Her Pazartesi 06:00 — haftalık retrain

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Python kur
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Bağımlılıkları yükle
        run: |
          pip install uv
          uv pip install --system -e ".[dev]"

      - name: Linting
        run: ruff check src/ tests/

      - name: Tip kontrolü
        run: mypy src/

      - name: Testler
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Veri doğrulama
        run: python scripts/validate_data.py

  train-and-evaluate:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Model eğit
        run: python src/train.py --config configs/prod.yaml

      - name: Modeli değerlendir
        run: |
          python src/evaluate.py --threshold 0.85
          # AUC < 0.85 ise pipeline başarısız

      - name: Docker image oluştur
        run: docker build -t churn-api:${{ github.sha }} .

      - name: Docker testi
        run: |
          docker run -d -p 8000:8000 --name test-api churn-api:${{ github.sha }}
          sleep 10
          curl -f http://localhost:8000/health
          docker stop test-api

  deploy-staging:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: echo "Staging'e deploy et (Kubernetes, ECS, vs.)"
```

### Sektör Notu — MLOps 2026

2026 itibarıyla:

- **Olgun MLOps praticesi:** %60–80 daha hızlı deployment, %70–90 daha az downtime, %10–20× daha hızlı sorun tespiti
- **Tool stack:** MLflow (experiment tracking) + FastAPI (serving) + Docker (paketleme) + GitHub Actions (CI/CD) + Evidently (monitoring) → açık kaynak, ücretsiz, production-ready
- **Cloud alternatifleri:** SageMaker Pipelines (AWS), Vertex AI Pipelines (GCP), Azure ML → managed ama vendor lock-in
- **Model serving:** vLLM (LLM için), Triton Inference Server (NVIDIA), BentoML → özel use-case'ler için
- **Feature store:** Feast (açık kaynak), Hopsworks, Databricks Feature Store → orta-büyük ölçek

**LLMOps — GenAI Gözlemlenebilirliği (2025 yeni):**
- LLM/RAG pipeline'ları için özel izleme araçları: **Langfuse** (açık kaynak, prompt versioning, trace), **Arize Phoenix** (LLM observability), **Weights & Biases Weave**
- Klasik MLOps metrikleri (drift, latency) artık yeterli değil; **hallucination oranı, faithfulness, context recall** de izleniyor

**Platform Konsolidasyonu:**
- **Databricks + MLflow** native entegrasyon güçlendi: Unity Catalog üzerinden model registry, Feature Store entegrasyonu
- **Vertex AI (Google)** ve **SageMaker (AWS)** managed MLflow endpoint'leri sunmaya başladı
- Küçük-orta ölçekli takımlar için **managed platforms** (Railway, Modal, Replicate) geleneksel K8s deployment'ın önüne geçiyor

**Drift Detection Olgunluğu:**
- PSI (Population Stability Index) hâlâ endüstri standardı: PSI < 0.1 stabil, 0.1–0.2 dikkat, > 0.2 alarm
- JS divergence ve Wasserstein distance PSI'ye alternatif dağılım karşılaştırma metrikleri
- **Population stability** = feature drift; **concept drift** = P(y|X) değişimi — ikisi ayrı izlenmeli

---

## E.9 Data Lineage — Verinin Yolculuğunu Takip Etme

### Sezgisel Açıklama

Data lineage = "Bu tahmin hangi veriden geldi?" sorusuna cevap verebilmek.

Üretimde bir model yanlış tahmin yaptığında, hatanın kaynağını bulmak için verinin **nereden geldiğini**, **hangi dönüşümlerden geçtiğini** ve **nereye gittiğini** bilmen gerekir. Audit, compliance ve debugging için kritik.

> **Çapraz referans:** Veri doğrulama ve kalite kontrolü temelleri icin bkz. [Katman A — Temeller](katman-A-temeller.md). Feature store mimarisi icin bkz. [Katman F — Sistem Tasarimi](katman-F-sistem-tasarimi.md).

### Lineage Grafiği (Kavramsal)

```
┌────────────┐     ┌──────────────┐     ┌──────────────────┐
│ PostgreSQL │────▶│ dbt staging  │────▶│ dbt mart         │
│ (raw data) │     │ stg_orders   │     │ fct_user_metrics │
└────────────┘     └──────────────┘     └──────┬───────────┘
                                               │
┌────────────┐     ┌──────────────┐            │
│ S3 bucket  │────▶│ dbt staging  │────────────┤
│ (clicklog) │     │ stg_clicks   │            │
└────────────┘     └──────────────┘            ▼
                                       ┌──────────────────┐
                                       │ Feature Store     │
                                       │ (Feast / manual)  │
                                       └──────┬───────────┘
                                               │
                                               ▼
                                       ┌──────────────────┐
                                       │ ML Model          │
                                       │ (train / serve)   │
                                       └──────────────────┘
```

### dbt ile Data Lineage

dbt (data build tool) SQL dönüşüm pipeline'larında lineage'i otomatik çıkarır:

```sql
-- models/staging/stg_orders.sql
-- Kaynak tanımla → lineage otomatik oluşur
SELECT
    order_id,
    user_id,
    order_date,
    total_amount
FROM {{ source('postgres', 'raw_orders') }}
WHERE order_date >= '2024-01-01'
```

```sql
-- models/mart/fct_user_metrics.sql
-- upstream bağımlılıklar → lineage grafiğinde otomatik görünür
SELECT
    user_id,
    COUNT(*) AS n_orders,
    SUM(total_amount) AS total_spend,
    DATEDIFF('day', MAX(order_date), CURRENT_DATE) AS days_since_last_order,
    AVG(total_amount) AS avg_order_value
FROM {{ ref('stg_orders') }}
GROUP BY user_id
```

```bash
# Lineage grafiğini görselleştir
dbt docs generate && dbt docs serve
# → Tarayıcıda interaktif DAG: hangi model hangi kaynağa bağlı
```

> **Neden önemli:** Bir feature'ın değeri aniden değişirse, lineage sayesinde sorunun kaynağını (raw veri mi, dönüşüm mü, kaynak mı) hızlıca bulabilirsin. Compliance (GDPR, KVKK) için de "bu kişinin verisi nerede kullanıldı?" sorusuna cevap verir.

---

## E.10 Schema Validation — Train-Serve Uyumluluğu

### Sezgisel Açıklama

En sinsi MLOps bug'ı: Model eğitimde 15 feature beklerken, serving'de 14 feature gelir — ya da bir sütunun tipi `float` yerine `str` olur. Model sessizce yanlış tahmin yapar, hata almadan.

Çözüm: **Pandera** ile input schema'sını hem train hem serve tarafında doğrula.

### Kod Örneği — Pandera ile Schema Validation

```python
# src/schema.py
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd

# ── Train ve Serve için ORTAK schema tanımı ──
# Bu dosya tek kaynak (single source of truth)
churn_input_schema = DataFrameSchema(
    columns={
        "n_orders": Column(int, Check.ge(0), nullable=False),
        "total_spend": Column(float, Check.ge(0), nullable=False),
        "days_since_last_order": Column(int, Check.ge(0), nullable=False),
        "avg_order_value": Column(float, Check.ge(0), nullable=True),
        "country": Column(str, Check.isin(["TR", "DE", "GB", "US", "FR"]),
                          nullable=False),
    },
    strict=True,    # Fazla sütun varsa hata ver
    coerce=True,    # Tipleri dönüştürmeyi dene (ör. "123" → 123)
)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Input'u doğrula. Hatalıysa SchemaError fırlatır."""
    return churn_input_schema.validate(df)
```

### Train-Serve Schema Uyumsuzluğu Senaryosu

```python
# ❌ SORUN: Train'de 5 feature, serve'de 4 feature
# Train zamanı:
#   X_train.columns = ["n_orders", "total_spend", "days_since_last_order",
#                       "avg_order_value", "country_TR"]
# Serve zamanı:
#   input sadece: {"n_orders": 5, "total_spend": 200, ...}  ← country_TR eksik!
# Model predict() çağrısında boyut uyumsuzluğu veya sessiz hata

# ✅ ÇÖZÜM: Schema'yı paylaş, her iki tarafta doğrula
# src/api.py içinde:
from src.schema import validate_input

@app.post("/predict")
async def predict(request: PredictRequest):
    # Pydantic validation'dan SONRA, model'e göndermeden ÖNCE
    input_df = pd.DataFrame([request.dict()])
    try:
        validated = validate_input(input_df)
    except pa.errors.SchemaError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Schema validation hatası: {e.failure_cases.to_dict()}"
        )
    # ... model.predict(validated) devam eder

# src/train.py içinde:
from src.schema import validate_input

def train(config_path):
    # ...
    X_train_df = build_features(df_train, cfg["features"])
    validate_input(X_train_df)  # Aynı schema — uyumsuzluk varsa BURADA patlar
    # ... model.fit(X_train_df, y_train)
```

> **Kural:** Schema tanımını **tek bir dosyada** (`src/schema.py`) tut. Hem train pipeline'ı hem API aynı dosyayı import etsin. Böylece bir feature eklendiğinde veya tipi değiştiğinde, diğer taraf otomatik olarak uyumsuzluğu yakalar.

---

## Katman E Kontrol Listesi

- [ ] Proje kodunu notebooks/ → src/ yapısına taşıdım
- [ ] FastAPI endpoint kurdum (/health + /predict)
- [ ] Pydantic ile input validation yaptım
- [ ] Pandera ile train-serve schema uyumluluğu sağladım
- [ ] Dockerfile yazdım ve container çalıştırdım
- [ ] MLflow ile deney takibi yaptım (log_model + input_example)
- [ ] Model Registry'de staging → production geçişi yaptım
- [ ] Model kaydetme stratejisi: MLflow birincil, joblib fallback
- [ ] Drift monitoring pipeline kurdum (PSI + KS testi + Evidently)
- [ ] Evidently HTML/JSON rapor + alert entegrasyonu kurdum
- [ ] Data lineage kavramını anladım (dbt docs generate)
- [ ] GitHub Actions CI/CD pipeline oluşturdum
- [ ] Proje-6 (MLOps Mini Platform) tamamlandı

---

## E.9 Production Monitoring — Prometheus & Grafana

### Sezgisel Açıklama

Bir uçağı enstrümansız uçurmayı düşün: hız, yükseklik, yakıt göstergelerin yok. Modelini production'a aldıktan sonra izlemezsen aynı kör uçuşu yapıyorsun. Prometheus metrikleri toplar, Grafana görselleştirir — model "kokpiti" bu ikili.

### Prometheus ile ML Metrikleri

```python
# pip install prometheus-client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrik tanımları
PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Toplam tahmin sayısı",
    ["model_name", "status"]  # label'lar
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Tahmin gecikme süresi (saniye)",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)
DRIFT_SCORE = Gauge(
    "model_drift_score",
    "PSI drift skoru (0=stabil, >0.2=kritik)",
    ["feature_name"]
)
MODEL_ACCURACY = Gauge(
    "model_accuracy_current",
    "Canlı model doğruluğu (rolling 24h)",
    ["model_name"]
)

def predict_with_metrics(model, features, model_name="churn_v2"):
    start = time.time()
    try:
        prediction = model.predict(features)
        PREDICTION_COUNTER.labels(model_name=model_name, status="success").inc()
        return prediction
    except Exception as e:
        PREDICTION_COUNTER.labels(model_name=model_name, status="error").inc()
        raise
    finally:
        PREDICTION_LATENCY.observe(time.time() - start)
```

### FastAPI + Prometheus Entegrasyonu

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app
import prometheus_client

app = FastAPI()

# /metrics endpoint ekle
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/predict")
async def predict(data: dict):
    with PREDICTION_LATENCY.time():
        result = predict_with_metrics(model, data["features"])
    return {"prediction": result.tolist()}
```

### Docker Compose ile İzleme Stack'i

```yaml
# docker-compose.monitoring.yml
version: "3.8"
services:
  ml-api:
    build: .
    ports: ["8000:8000"]

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_storage:/var/lib/grafana

volumes:
  grafana_storage:
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "ml-api"
    static_configs:
      - targets: ["ml-api:8000"]
    metrics_path: "/metrics"
```

### Alert Kuralları

```yaml
# alerts.yml — Prometheus alerting rules
groups:
  - name: ml_model_alerts
    rules:
      - alert: ModelAccuracyDrop
        expr: model_accuracy_current < 0.75
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model doğruluğu kritik seviyeye düştü"
          description: "{{ $labels.model_name }} modeli 5 dakikadır %75 altında"

      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, model_prediction_latency_seconds) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P95 gecikme 500ms üzerinde"

      - alert: HighDriftScore
        expr: model_drift_score > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.feature_name }} feature'ında ciddi drift"
```

> **Senior Notu:** İzlenecek metrikler iki katmanda düşünülmeli:
> - **Business metrikleri:** Revenue per prediction, conversion rate, customer impact (bunlar en önemli — ML metriği iyiyken business metriği kötüleşebilir)
> - **Teknik metrikleri:** Latency (p50/p95/p99), error rate, memory/CPU, drift score
>
> Pratik kural: Grafana'da 5 dashboard'dan fazlası varsa odak dağılır. Tek ekran: accuracy trend + latency trend + drift alert + error rate.

> **Sektör Notu (2026):** Prometheus + Grafana stack'i ML izleme için sektör standardı. Alternatif: Datadog ML Monitoring ($$$), Arize Phoenix (LLM için güçlü), W&B (training + serving). Açık kaynak yolunda Prometheus + Evidently (drift) + Grafana üçlüsü en yaygın seçim.

---

## E.10 Model Versioning ve Lifecycle Yönetimi

### Sezgisel Açıklama

Bir yazılım ekibi "hangi kod versiyonu production'da?" sorusunu her zaman yanıtlayabilir — git log yeter. ML'de ise "hangi model, hangi veriyle, hangi hiperparametrelerle eğitildi?" sorusu cevaplanabilir olmalı. Model registry bu sorunun cevabı.

### MLflow Model Registry — Tam Workflow

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 1. Model kaydet (training sonrası)
with mlflow.start_run(run_name="churn_lgbm_v3") as run:
    model = train_model(X_train, y_train)

    # Metrikler logla
    mlflow.log_metrics({
        "auc": evaluate_auc(model, X_test, y_test),
        "f1": evaluate_f1(model, X_test, y_test),
        "calibration_error": calibration_error(model, X_test, y_test)
    })

    # Parametreler logla
    mlflow.log_params(model.get_params())

    # Modeli kaydet
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="churn-predictor",  # registry'e kayıt
        input_example=X_train[:5]
    )

run_id = run.info.run_id
print(f"Run ID: {run_id}")
```

```python
# 2. Model Registry: Staging → Production geçişi
client = MlflowClient()

# En son versiyonu bul
latest_versions = client.get_latest_versions(
    "churn-predictor", stages=["None"]
)
version = latest_versions[0].version

# Staging'e al (QA testi için)
client.transition_model_version_stage(
    name="churn-predictor",
    version=version,
    stage="Staging",
    archive_existing_versions=False
)

# Testler geçtikten sonra Production'a al
client.transition_model_version_stage(
    name="churn-predictor",
    version=version,
    stage="Production",
    archive_existing_versions=True  # eski production'ı archive'a gönder
)

# Servis katmanında her zaman "Production" modelini yükle
model = mlflow.pyfunc.load_model("models:/churn-predictor/Production")
```

### Champion/Challenger Pattern

```python
import random
from typing import Literal

def champion_challenger_predict(
    features,
    champion_model,
    challenger_model,
    challenger_traffic: float = 0.1  # %10 trafik challenger'a
) -> dict:
    """
    Champion: mevcut production modeli
    Challenger: test edilen yeni model
    Her iki modelin tahminini logla, sadece champion'ı sun.
    """
    use_challenger = random.random() < challenger_traffic
    active_model = "challenger" if use_challenger else "champion"

    # Her iki modeli çalıştır (shadow mode)
    champion_pred = champion_model.predict(features)
    challenger_pred = challenger_model.predict(features)

    # İkisini de logla (analiz için)
    log_predictions({
        "champion": champion_pred,
        "challenger": challenger_pred,
        "served": active_model,
        "timestamp": time.time()
    })

    # Sadece seçilen modelin tahminini sun
    return {
        "prediction": challenger_pred if use_challenger else champion_pred,
        "model": active_model
    }
```

### Shadow Deployment

```
Shadow Deployment Akışı:

Kullanıcı isteği
       ↓
 Champion Model ─────────────────── Kullanıcıya yanıt (100%)
       ↓
 Challenger Model ──── Log (0% kullanıcıya, sadece kayıt)
       ↓
 Karşılaştırma paneli → Challenger daha iyi? → Promote
```

> **Senior Notu:** Model registry olmayan ekiplerde sık karşılaşılan senaryo: "Production'daki model kim tarafından, ne zaman, hangi veriyle eğitildi?" sorusuna kimse cevap veremez. Bu durum hem debug'ı imkânsız kılar hem de GDPR/denetim gereksinimlerini karşılamaz.
>
> Minimum viable model registry için bile şunları kaydet: model artifact + training data hash + feature list + evaluation metrics + deployment timestamp. MLflow bu beşini ücretsiz sağlar.

> **Sektör Notu (2026):** MLflow model registry sektör standardı (Databricks entegrasyonu ile güçlendi). Alternatifler: W&B Model Registry (deneysel + production tek yerde), Vertex AI Model Registry (GCP), SageMaker Model Registry (AWS). Açık kaynak tercihinde MLflow önce.

---

## E.11 LLMOps — LLM'leri Production'da Yönetme

### Sezgisel Açıklama

Klasik ML modelinin çıktısı sayısal: 0.87 churn skoru. Doğru mu yanlış mı? Gerçek değerle karşılaştır. LLM'nin çıktısı metin: "Merhaba! Size nasıl yardımcı olabilirim?" — bu iyi mi kötü mü? Subjektif. LLMOps, bu öznel çıktıları ölçülebilir kılma sanatı.

### Geleneksel MLOps vs LLMOps

| Kriter | Klasik MLOps | LLMOps |
|--------|-------------|---------|
| **Çıktı tipi** | Sayı / sınıf | Metin (doğası gereği belirsiz) |
| **Başarı ölçütü** | AUC, F1, RMSE | Faithfulness, relevancy, hallucination rate |
| **Retraining** | Veri drift'i tetikler | Prompt güncelleme çoğu zaman yeterli |
| **Monitoring** | Metrik izleme | Trace + eval + cost tracking |
| **Gecikme bütçesi** | ms'ler | 1–10 saniye (kabul edilebilir) |
| **En büyük risk** | Data drift, concept drift | Hallucination, jailbreak, PII sızıntısı |

### Langfuse ile LLM İzleme

```python
# pip install langfuse openai
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"  # veya self-hosted
)

@observe()  # her çağrıyı otomatik logla
def generate_response(user_question: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI()

    # Prompt metadata ekle
    langfuse_context.update_current_trace(
        name="rag-query",
        metadata={"user_id": "u123", "source": "chatbot"}
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": user_question}
        ]
    )

    answer = response.choices[0].message.content

    # Kullanıcı skoru ekle (feedback loop)
    langfuse_context.score_current_trace(
        name="user-feedback",
        value=None  # kullanıcı rating geldiğinde doldurulur
    )

    return answer

# Langfuse dashboard'da görürsün:
# - Her trace: input → output → latency → token cost
# - Günlük/haftalık cost trend
# - Hangi prompt versiyonu daha iyi?
```

### LLM Evaluation — RAGAS Metrikleri

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Context'e sadık mı? (hallucination ters ölçüsü)
    answer_relevancy,      # Soruyla ilgili mi?
    context_precision,     # Retrieve edilen context kaliteli mi?
    context_recall         # Gerekli bilgi context'te var mıydı?
)
from datasets import Dataset

# Test verisi
test_data = {
    "question": ["Python'da liste comprehension nedir?"],
    "answer": ["Liste comprehension, kısa söz dizimiyle liste oluşturma yöntemidir."],
    "contexts": [["Python'da [x for x in range(10)] sözdizimi..."]],
    "ground_truth": ["Python'da [expr for item in iterable] formatında liste oluşturma."]
}

dataset = Dataset.from_dict(test_data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

print(result)
# {'faithfulness': 0.95, 'answer_relevancy': 0.87,
#  'context_precision': 0.91, 'context_recall': 0.83}
```

### Prompt Versioning

```python
# Prompt'ları kod gibi versiyonla
import json
from pathlib import Path
from datetime import datetime

PROMPTS_DIR = Path("prompts/")

def save_prompt_version(name: str, template: str, metadata: dict) -> str:
    """Prompt'u versiyonlayarak kaydet."""
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_data = {
        "name": name,
        "version": version,
        "template": template,
        "metadata": metadata,  # author, purpose, test_score
        "created_at": datetime.now().isoformat()
    }

    path = PROMPTS_DIR / f"{name}_{version}.json"
    path.write_text(json.dumps(prompt_data, ensure_ascii=False, indent=2))
    return version

def load_prompt(name: str, version: str = "latest") -> str:
    """En güncel veya belirli versiyonu yükle."""
    if version == "latest":
        files = sorted(PROMPTS_DIR.glob(f"{name}_*.json"))
        path = files[-1]  # en yeni
    else:
        path = PROMPTS_DIR / f"{name}_{version}.json"

    return json.loads(path.read_text())["template"]

# Kullanım:
save_prompt_version(
    name="rag_system_prompt",
    template="Sen yardımcı bir asistansın. Verilen context dışına çıkma.\n\nContext: {context}\n\nSoru: {question}",
    metadata={"author": "team", "ragas_faithfulness": 0.95}
)
```

> **Senior Notu:** LLM'leri izlerken en kritik üç şey:
> 1. **Cost tracking:** Token başına maliyet hızla artabilir. Günlük budget alert kur.
> 2. **Latency SLA:** Kullanıcı toleransı ortalama 3 saniye. P95 > 5s ise streaming'e geç.
> 3. **Hallucination rate:** RAGAS faithfulness < 0.8 → RAG pipeline'ını incele (chunking? retrieval quality?).
>
> Tip: Production'da LLM output'larının %5–10'unu manuel değerlendir. Otomatik metrikler yetmez — insan gözetimi şart.

> **Sektör Notu (2026):** LLMOps araçları hızla olgunlaşıyor. **Langfuse** (açık kaynak, self-hostable) en popüler. Alternatifler: **Arize Phoenix** (hallucination detection güçlü), **W&B Weave** (W&B ekosistemindeyseniz), **Helicone** (basit, proxy tabanlı). Büyük şirketlerde kendi LLM gateway'leri yaygınlaşıyor (cost control + privacy).

---

<div class="nav-footer">
  <span><a href="#file_katman_D_derin_ogrenme">← Önceki: Katman D — Derin Öğrenme</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_F_sistem_tasarimi">Sonraki: Katman F — Sistem Tasarımı →</a></span>
</div>
