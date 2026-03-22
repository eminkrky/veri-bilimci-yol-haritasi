# Katman F — Sistem Tasarımı (DS/ML için)

> Bu katmanda ne öğrenilir: ML sistemlerinin mimarisi. Online/offline ayrımı, feature store, latency bütçesi, maliyet optimizasyonu, model compression, canary deployment, data mesh ve gerçek dünya sistem tasarım örnekleri.
>
> Süre: 1–2 hafta. Senior DS mülakatlarında sistem tasarımı soruları artıyor.
>
> **Çapraz referanslar:** Model serving optimizasyonu → [Katman D — Derin Öğrenme](katman-D-derin-ogrenme.md) | MLOps pipeline'ları → [Katman E — MLOps](katman-E-mlops.md)


<div class="prereq-box">
<strong>Önkoşul:</strong> <strong>Katman E (MLOps)</strong> tamamlanmış olmalı. Temel bulut ve dağıtık sistem kavramları faydalıdır.
</div>

---

## F.1 Online vs Offline Serving

### Sezgisel Açıklama

İki temel servis modu:

**Batch scoring (offline):**
- Sabah çalış, gün boyunca kullan
- Örnek: Günlük churn listesi, aylık segment atama
- Latency önemli değil, throughput önemli

**Online scoring (realtime):**
- Kullanıcı isteğiyle tetiklenir
- Örnek: Fraud tespiti (<100ms), öneri sistemi (<50ms)
- Latency kritik

```
Hangi modu seç?

Latency gereksinimi var mı?
  ├── Hayır → Batch scoring
  └── Evet → Online scoring
       ├── <10ms → Feature store + hafif model
       ├── <100ms → Online feature fetch + model inference
       └── >1 saniye → Kabul edilemez (mimaryi yeniden düşün)
```

### Latency Bütçesi Örneği

```
Fraud scoring: <100ms total

Kaynak          | Bütçe | Optimizasyon
----------------|-------|------------------
Network + API   | 10ms  | CDN, coğrafi dağıtım
Feature fetch   | 20ms  | Redis cache, precompute
Model inference | 5ms   | Model sıkıştırma, ONNX
Postprocessing  | 5ms   | Vektörize işlemler
Güvenlik payı   | 60ms  | Buffer
----------------|-------|------------------
Toplam          | 100ms |
```

---

## F.2 Feature Store

### Sezgisel Açıklama

Training-serving skew: eğitimde Python pandas ile hesaplanan feature, serving'de Java ile farklı hesaplandı. Model "bozuldu" gibi görünür ama asıl sorun feature tutarsızlığı.

Feature store bunu çözer: tek bir feature tanımı, hem training hem serving aynı kodu çalıştırır.

### Mimari

```
Ham Veri (Data Lake/Warehouse)
          ↓
Feature Pipeline (Spark, dbt, Beam)
          ↓
┌─────────────────────┬──────────────────────┐
│   Offline Store     │    Online Store       │
│   (S3/GCS/HDFS)     │    (Redis/DynamoDB)   │
│   Parquet/Iceberg   │    Low-latency KV     │
│   Tarihsel veriler  │    Güncel değerler    │
│   Training için     │    Serving için       │
└─────────────────────┴──────────────────────┘
          ↓                      ↓
   Model Training          Model Serving
   (batch read)            (real-time read)
```

### Kod Örneği — Basit Feature Store (Feast)

```python
# feature_repo/features.py
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from datetime import timedelta

# Entity tanımı
user = Entity(
    name="user_id",
    description="Kullanıcı ID",
)

# Feature source (offline)
user_stats_source = FileSource(
    path="data/user_stats.parquet",
    timestamp_field="event_timestamp",
)

# Feature view tanımı
user_stats_fv = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=1),   # Online store'da ne kadar tutulsun?
    schema=[
        Field(name="n_orders_30d", dtype=Int64),
        Field(name="total_spend_30d", dtype=Float32),
        Field(name="days_since_last_order", dtype=Int64),
        Field(name="avg_order_value", dtype=Float32),
    ],
    source=user_stats_source,
)

# Training: offline retrieval
# feature_store.get_historical_features(entity_df, features)

# Serving: online retrieval
# feature_store.get_online_features(entities, features)
```

### Feature Store Karar Tablosu

| Soru | Online | Offline |
|------|--------|---------|
| Latency gereksinimi? | <10ms | Dakika-saat kabul |
| Feature güncelleme sıklığı? | Realtime/dakika | Günlük/saatlik |
| Depolama maliyeti? | Yüksek (RAM) | Düşük (disk) |
| Tarihsel veri gerekiyor mu? | Hayır | Evet |
| Kullanım? | Model serving | Model training |

**Pratik tavsiye:** Online store = sadece aktif kullanıcıların son N günlük feature'ları. Offline = tüm tarih. Bu kombinezon maliyeti %80+ azaltır.

---

## F.3 Sistem Tasarım Örnekleri

### Örnek 1: Gerçek Zamanlı Fraud Tespiti

**Gereksinimler:**
- Her ödeme işleminde <100ms fraud skoru
- Günlük 10M işlem (120 TPS ortalama, peak 500 TPS)
- %99.99 uptime
- Yanlış pozitif oranı: <%1 (müşteri deneyimi)
- Recall: >%90 (fraud kaçırma maliyeti yüksek)

**Mimari:**

```
[Ödeme İsteği]
      ↓
[API Gateway] (rate limiting, auth)
      ↓
[Feature Fetcher] ── [Redis: User profile, transaction history]
      ↓              ── [Kafka: Real-time aggregations (Flink)]
[Model Server] ── [Primary model: LightGBM <5ms]
      ↓         ── [Backup model: Logistic Regression <1ms]
[Karar Motoru] ── Rule engine + model score → Block/Allow/Challenge
      ↓
[Audit Log] ── Elasticsearch (sorgulanabilir kayıt)
              ── S3 (uzun dönem arşiv)

[Monitoring] ── Prometheus/Grafana (latency, throughput)
             ── Feature drift dashboard (PSI)
             ── Ground truth pipeline (label delay yönetimi)
```

**Tasarım Kararları:**

```python
# Feature kategorileri
REALTIME_FEATURES = [
    "transaction_amount",           # Anlık
    "merchant_category",            # Anlık
    "card_present",                 # Anlık
    "ip_country_mismatch",          # Anlık (GeoIP)
]

# Redis'ten gelecek (precomputed)
CACHED_FEATURES = [
    "user_avg_transaction_30d",     # Günlük güncelleme
    "user_n_transactions_1h",       # Streaming aggregate
    "user_n_failed_auths_24h",      # Streaming aggregate
    "merchant_fraud_rate_30d",      # Günlük güncelleme
]

# Latency bütçesi
LATENCY_BUDGET = {
    "api_gateway": 5,           # ms
    "feature_fetch_redis": 3,   # ms (cache hit)
    "feature_fetch_db": 20,     # ms (cache miss)
    "model_inference": 5,       # ms
    "decision_logic": 2,        # ms
    "response": 5,              # ms
    "safety_margin": 60,        # ms
}  # Toplam: ~100ms
```

**Shadow Mode (Güvenli Model Rollout):**

```python
class ModelServer:
    def __init__(self, production_model, shadow_model=None):
        self.production = production_model
        self.shadow = shadow_model

    def predict(self, features: dict) -> float:
        prod_score = self.production.predict(features)

        if self.shadow:
            # Shadow: sonucu sakla ama kararı etkileme
            shadow_score = self.shadow.predict(features)
            self._log_shadow(prod_score, shadow_score, features)

        return prod_score

    def _log_shadow(self, prod, shadow, features):
        # Karşılaştırma için asenkron log
        import asyncio
        asyncio.ensure_future(self._async_log(prod, shadow))
```

---

### Örnek 2: Öneri Sistemi (YouTube/Netflix Benzeri)

**Gereksinimler:**
- Her sayfa yüklemesinde <50ms öneri
- 50M kullanıcı, 5M içerik
- Çeşitlilik + alaka + novelty dengesi

**Mimari:**

```
[Kullanıcı İsteği]
        ↓
┌──────────────────────────────────────────┐
│          1. RETRIEVAL (Adayları Daralt)  │
│                                          │
│  Two-Tower Model → User embedding        │
│  FAISS ANN arama → Top 1000 aday        │
│  Rule-based filtreler (ülke, yaş, vb.)  │
└──────────────────────────────────────────┘
        ↓ 1000 aday
┌──────────────────────────────────────────┐
│          2. PRE-RANKING (Opsiyonel)      │
│                                          │
│  Hafif model → Hızlı eleme              │
│  1000 → 200 aday                        │
└──────────────────────────────────────────┘
        ↓ 200 aday
┌──────────────────────────────────────────┐
│          3. RANKING                      │
│                                          │
│  Zengin feature + LightGBM              │
│  CTR, CVR, engagement tahmin            │
│  200 → Top 20                           │
└──────────────────────────────────────────┘
        ↓ 20 sıralı öneri
┌──────────────────────────────────────────┐
│          4. POST-PROCESSING              │
│                                          │
│  Çeşitlilik (diversity) uygula          │
│  Business rule (sponsored, stok)         │
│  Fatigue filtering (az önce gördün)     │
└──────────────────────────────────────────┘
        ↓ Final liste
[Kullanıcıya Göster]
```

**Offline Değerlendirme vs Online Metrikler:**

```python
# Offline metrikler (training sonrası)
OFFLINE_METRICS = {
    "NDCG@10": "Sıralama kalitesi",
    "Recall@100": "Retrieval coverage",
    "Coverage": "Kaç farklı item önerildi?",
    "Novelty": "Yeni item oranı",
    "Serendipity": "Beklenmedik ama beğenilecek",
}

# Online metrikler (A/B test sonrası)
ONLINE_METRICS = {
    "CTR": "Click-through rate",
    "Completion_rate": "İzleme/okuma bitiş oranı",
    "Session_length": "Oturum süresi",
    "Diversity_score": "Önerilen item çeşitliliği",
    "Return_visit_rate": "Geri dönüş oranı (long-term)",
}

# Guardrail metrikler
GUARDRAIL_METRICS = {
    "Latency_p99": "<50ms",
    "Error_rate": "<%0.1",
    "Filter_bubble_score": "Çeşitlilik korunuyor mu?",
}
```

---

### Örnek 3: A/B Test Platformu

**Bileşenler:**

```python
# 1. Assignment Service — deterministik atama
import hashlib

def assign_variant(user_id: str, experiment_id: str,
                    traffic_pct: float = 0.5) -> str:
    """
    User → Variant atama.
    Deterministik: aynı user her zaman aynı variant.
    Salting: farklı deneyler için farklı atama.
    """
    hash_input = f"{user_id}:{experiment_id}"
    hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    bucket = (hash_val % 10000) / 10000.0  # 0-1 arası

    if bucket < traffic_pct:
        return "treatment"
    else:
        return "control"

# 2. Exposure logging
def log_exposure(user_id: str, experiment_id: str, variant: str,
                  timestamp: float, metadata: dict = None):
    """
    Assignment anında log. NOT: kullanım anında değil.
    Bu fark analiz kalitesini etkiler.
    """
    event = {
        "event_type": "exposure",
        "user_id": user_id,
        "experiment_id": experiment_id,
        "variant": variant,
        "timestamp": timestamp,
        "session_id": metadata.get("session_id") if metadata else None,
    }
    # Kafka'ya publish et (async)
    kafka_producer.send("experiment_events", event)

# 3. Analysis pipeline
def analyze_experiment(experiment_id: str,
                         primary_metric: str = "revenue_per_user",
                         alpha: float = 0.05) -> dict:
    """Standart A/B test analizi."""
    # Exposure log'ından kullanıcıları al
    exposed = get_exposed_users(experiment_id)

    control = exposed[exposed["variant"] == "control"][primary_metric]
    treatment = exposed[exposed["variant"] == "treatment"][primary_metric]

    # SRM kontrolü
    srm = check_srm(len(control), len(treatment))
    if srm["detected"]:
        return {"error": "SRM detected", "srm": srm}

    # Analiz
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(treatment, control)
    effect = treatment.mean() - control.mean()
    relative_lift = effect / control.mean()

    # Bootstrap CI
    boots = [np.mean(np.random.choice(treatment, len(treatment), replace=True)) -
             np.mean(np.random.choice(control, len(control), replace=True))
             for _ in range(10_000)]
    ci = np.percentile(boots, [2.5, 97.5])

    return {
        "experiment_id": experiment_id,
        "n_control": len(control),
        "n_treatment": len(treatment),
        "control_mean": control.mean(),
        "treatment_mean": treatment.mean(),
        "effect": effect,
        "relative_lift": relative_lift,
        "p_value": p_value,
        "significant": p_value < alpha,
        "ci_95": ci.tolist(),
    }
```

---

### Örnek 4: Search Ranking Sistemi

**Gereksinimler:**
- Kullanıcı sorgusu → en alakalı sonuçları <200ms'de döndür
- Milyonlarca doküman/ürün havuzu
- Pozisyon bias'ını azalt, tıklama + dönüşüm optimize et
- Freshness: yeni içerik dezavantajlı olmasın

**Mimari:**

```
[Kullanıcı Sorgusu: "kırmızı koşu ayakkabısı"]
        ↓
┌──────────────────────────────────────────────┐
│  1. QUERY UNDERSTANDING                      │
│                                              │
│  Tokenization + spell correction             │
│  Query expansion (eş anlamlılar, stemming)   │
│  Intent classification (navigational /       │
│     informational / transactional)           │
│  Query → embedding (BERT-based encoder)      │
└──────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  2. RETRIEVAL (Candidate Generation)         │
│                                              │
│  Inverted index (Elasticsearch/Solr) → BM25  │
│  ANN search (FAISS/ScaNN) → embedding match  │
│  Birleşim: BM25 ∪ ANN → Top 1000 aday      │
└──────────────────────────────────────────────┘
        ↓ 1000 aday
┌──────────────────────────────────────────────┐
│  3. RANKING (L1 → L2)                       │
│                                              │
│  L1: Hafif model (LightGBM)                 │
│    Features: BM25 score, embedding sim,      │
│    CTR history, freshness, popularity        │
│    1000 → 100 aday                           │
│                                              │
│  L2: Ağır model (Cross-encoder / LLM-based) │
│    Query-document cross attention            │
│    100 → 20 sıralı sonuç                    │
└──────────────────────────────────────────────┘
        ↓ 20 sonuç
┌──────────────────────────────────────────────┐
│  4. RE-RANKING & BUSINESS LOGIC              │
│                                              │
│  Diversity injection (aynı seller sınırı)    │
│  Sponsored sonuçları entegre et             │
│  Pozisyon bias correction (IPW)              │
│  Freshness boost (yeni ürünlere ek skor)    │
└──────────────────────────────────────────────┘
        ↓ Final SERP
[Kullanıcıya Göster]
```

**Tasarım Kararları:**

```python
# Search ranking feature'ları
QUERY_FEATURES = [
    "query_length",                 # Kısa query → navigational olabilir
    "query_embedding",              # BERT-based, 768d → PCA ile 128d
    "query_intent",                 # {navigational, transactional, informational}
    "query_frequency",              # Popüler query mi?
]

DOCUMENT_FEATURES = [
    "doc_embedding",                # İçerik embedding
    "historical_ctr",              # Bu dokümanın genel CTR'ı
    "freshness_score",             # Yayın tarihine göre decay
    "quality_score",               # Editoryal / otomatik kalite skoru
    "seller_rating",               # E-commerce: satıcı puanı
]

CROSS_FEATURES = [
    "bm25_score",                  # Lexical relevance
    "embedding_cosine_sim",        # Semantic relevance
    "query_doc_click_history",     # Bu query-doc çifti önceden tıklandı mı?
]

# Pozisyon bias correction — Inverse Propensity Weighting
def correct_position_bias(clicks, positions, n_positions=10):
    """
    Üst pozisyondaki tıklamalar bias'lı → IPW ile düzelt.
    Propensity: P(click | position) — examination probability.
    """
    from collections import Counter
    import numpy as np

    # Pozisyon bazlı propensity tahmini
    pos_clicks = Counter()
    pos_impressions = Counter()
    for click, pos in zip(clicks, positions):
        pos_impressions[pos] += 1
        if click:
            pos_clicks[pos] += 1

    propensity = {pos: pos_clicks[pos] / pos_impressions[pos]
                  for pos in pos_impressions if pos_impressions[pos] > 100}

    # Normalize: pozisyon 1'in propensity'si = 1.0
    max_prop = propensity.get(1, max(propensity.values()))
    propensity = {pos: p / max_prop for pos, p in propensity.items()}

    return propensity

# Offline evaluation — Interleaving
def team_draft_interleaving(ranking_a: list, ranking_b: list, k: int = 10) -> list:
    """
    İki ranking'i interleave et → hangi model daha iyi tıklama alıyor?
    A/B testten daha hızlı sonuç verir (aynı sayfada karşılaştırma).
    """
    result = []
    team_a, team_b = [], []
    i, j = 0, 0

    while len(result) < k and (i < len(ranking_a) or j < len(ranking_b)):
        if len(team_a) <= len(team_b) and i < len(ranking_a):
            if ranking_a[i] not in result:
                result.append(ranking_a[i])
                team_a.append(ranking_a[i])
            i += 1
        elif j < len(ranking_b):
            if ranking_b[j] not in result:
                result.append(ranking_b[j])
                team_b.append(ranking_b[j])
            j += 1

    return result, team_a, team_b
```

> **Çapraz referans:** Embedding modelleri ve FAISS kullanımı → [Katman D — Derin Öğrenme](katman-D-derin-ogrenme.md)

---

### Örnek 5: Dynamic Pricing Sistemi

**Gereksinimler:**
- Gerçek zamanlı fiyat optimizasyonu (otel odaları, uçak bileti, e-commerce)
- Talep tahmini + fiyat elastisitesi modeli
- Business constraint: minimum marj, fiyat tutarlılığı, yasal sınırlar
- A/B test ile fiyat etkisini ölç

**Mimari:**

```
┌──────────────────────────────────────────────┐
│  1. DEMAND ESTIMATION                        │
│                                              │
│  Tarihsel satış + dışsal faktörler           │
│  (mevsimsellik, tatiller, rakip fiyat,       │
│   hava durumu, etkinlikler)                  │
│  Model: LightGBM / Prophet / temporal fusion │
│  Çıktı: demand_forecast(item, date, price)   │
└──────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  2. PRICE ELASTICITY MODEL                   │
│                                              │
│  Fiyat-talep ilişkisi: ε = ΔQ/Q ÷ ΔP/P    │
│  Segmente göre elastisite (premium vs        │
│    budget müşteri farklı tepki verir)        │
│  Causal inference: IV veya RDD ile           │
│    gerçek elastisiteyi tahmin et             │
└──────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  3. PRICE OPTIMIZATION                       │
│                                              │
│  Objective: Revenue = Price × Demand(Price)  │
│  Constraint: min_margin, max_price_change,   │
│    rate parity, yasal limitler               │
│  Solver: scipy.optimize veya OR-Tools        │
└──────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────┐
│  4. A/B TEST & GUARDRAILS                    │
│                                              │
│  Fiyat değişikliğini test et                │
│  Switchback design (zaman bazlı, geo bazlı) │
│  Metrikler: revenue/unit, conversion rate,   │
│    customer satisfaction, return rate         │
│  Guardrail: %10'dan fazla fiyat artışı yok  │
└──────────────────────────────────────────────┘
        ↓
[Fiyat API → Frontend / Booking Engine]
```

**Tasarım Kararları:**

```python
import numpy as np
from scipy.optimize import minimize_scalar

# --- 1. Talep tahmini (basitleştirilmiş) ---
def estimate_demand(base_demand: float, price: float,
                     reference_price: float, elasticity: float) -> float:
    """
    Log-linear talep modeli.
    base_demand: referans fiyattaki beklenen talep
    elasticity: genelde negatif (fiyat artınca talep düşer)
    """
    return base_demand * (price / reference_price) ** elasticity

# --- 2. Gelir optimizasyonu ---
def optimize_price(base_demand: float, reference_price: float,
                    elasticity: float, min_price: float,
                    max_price: float, unit_cost: float) -> dict:
    """
    Kar = (Price - Cost) × Demand(Price) → maximize et.
    """
    def neg_profit(price):
        demand = estimate_demand(base_demand, price, reference_price, elasticity)
        profit = (price - unit_cost) * demand
        return -profit  # minimize negative = maximize positive

    result = minimize_scalar(neg_profit, bounds=(min_price, max_price),
                               method="bounded")
    optimal_price = result.x
    optimal_demand = estimate_demand(base_demand, optimal_price,
                                      reference_price, elasticity)
    return {
        "optimal_price": round(optimal_price, 2),
        "expected_demand": round(optimal_demand, 1),
        "expected_revenue": round(optimal_price * optimal_demand, 2),
        "expected_profit": round((optimal_price - unit_cost) * optimal_demand, 2),
    }

# Örnek kullanım
result = optimize_price(
    base_demand=100,        # Referans fiyatta 100 birim/gün talep
    reference_price=50.0,   # Mevcut fiyat $50
    elasticity=-1.5,        # Elastik ürün
    min_price=30.0,
    max_price=80.0,
    unit_cost=20.0,
)
# result = {"optimal_price": 50.0, "expected_demand": 100.0, ...}

# --- 3. Switchback A/B test tasarımı ---
def create_switchback_schedule(regions: list, n_days: int,
                                 treatment_pct: float = 0.5,
                                 seed: int = 42) -> dict:
    """
    Switchback design: her gün her bölge rastgele treatment/control.
    Fiyat deneylerinde user-level randomization bias yaratır
    (aynı ürünü farklı fiyata gören kullanıcılar).
    Switchback bunu çözer: zaman × bölge biriminde randomize et.
    """
    np.random.seed(seed)
    schedule = {}
    for day in range(n_days):
        for region in regions:
            schedule[(day, region)] = (
                "treatment" if np.random.random() < treatment_pct
                else "control"
            )
    return schedule
```

> **Çapraz referans:** A/B test istatistikleri ve switchback design → [Katman C — İstatistik](katman-C-istatistik.md)

---

## F.4 Güvenlik ve Gizlilik

```python
# PII maskeleme
import re

def mask_pii(text: str) -> str:
    """Log'larda PII maskele."""
    # E-posta maskele: abc@example.com → a**@example.com
    text = re.sub(r'\b([a-zA-Z0-9._%+-])[a-zA-Z0-9._%+-]*(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                   r'\1**\2', text)
    # Telefon maskele
    text = re.sub(r'\b(\d{3})\d{4}(\d{4})\b', r'\1****\2', text)
    # Kredi kartı maskele
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?(\d{4})\b', r'****-****-****-\1', text)
    return text

# Differential privacy (özet istatistiklere gürültü)
import numpy as np

def dp_mean(data: np.ndarray, epsilon: float = 1.0,
             sensitivity: float = 1.0) -> float:
    """Laplace mekanizması ile DP ortalama."""
    true_mean = np.mean(data)
    noise = np.random.laplace(0, sensitivity / epsilon)
    return true_mean + noise

# RBAC — erişim kontrolü
ROLE_PERMISSIONS = {
    "ds_junior": ["read_data", "run_notebooks"],
    "ds_senior": ["read_data", "run_notebooks", "access_pii", "deploy_model"],
    "ds_lead":   ["read_data", "run_notebooks", "access_pii", "deploy_model", "admin"],
}
```

---

## F.5 Maliyet Optimizasyonu

### GPU/CPU Maliyet Karşılaştırması (2026 itibarıyla referans fiyatlar)

```
AWS us-east-1 — 2026 itibarıyla yaklaşık saatlik fiyatlar (on-demand / spot)

Instance Türü       | GPU/CPU          | vCPU | RAM   | On-Demand  | Spot (~)   | Kullanım Senaryosu
--------------------|------------------|------|-------|------------|------------|-----------------------------
ml.m5.xlarge        | CPU only         | 4    | 16GB  | $0.23/saat | $0.07      | Hafif model serving, batch
ml.c6g.2xlarge      | CPU (Graviton)   | 8    | 16GB  | $0.27/saat | $0.08      | CPU inference (%15 ucuz x86'dan)
ml.g5.xlarge        | 1× A10G (24GB)   | 4    | 16GB  | $1.41/saat | $0.42      | Orta boy model training/serving
ml.g5.2xlarge       | 1× A10G (24GB)   | 8    | 32GB  | $1.89/saat | $0.57      | LLM fine-tuning, medium
ml.p4d.24xlarge     | 8× A100 (40GB)   | 96   | 1.1TB | $32.77/saat| $9.83      | Büyük model training
ml.inf2.xlarge      | AWS Inferentia2  | 4    | 16GB  | $0.76/saat | —          | Inference-only (en ucuz GPU alt.)
--------------------|------------------|------|-------|------------|------------|-----------------------------

Pratik Kurallar:
• Training: Spot instance → %60–70 tasarruf (checkpoint ile interrupt tolere)
• Serving (CPU yeterli mi?): LightGBM, XGBoost, küçük NN → CPU yeterli, GPU gereksiz
• Serving (GPU gerekli mi?): Transformer, >100M param → GPU veya Inferentia
• ARM (Graviton): x86'dan %10–20 ucuz, aynı CPU performansı
```

### Autoscaling Maliyet Simülasyonu

```python
import numpy as np

def simulate_autoscaling_cost(
    hourly_traffic: list[int],     # 24 saatlik trafik profili (req/saat)
    requests_per_instance: int,     # Bir instance'ın kapasitesi (req/saat)
    cost_per_instance_hour: float,  # $/saat
    min_instances: int = 1,
    max_instances: int = 20,
    scale_up_threshold: float = 0.7,   # %70 dolulukta scale up
    scale_down_threshold: float = 0.3, # %30 dolulukta scale down
) -> dict:
    """
    Autoscaling vs sabit kapasite maliyet karşılaştırması.
    """
    # --- Sabit kapasite: peak'e göre boyutla ---
    peak_traffic = max(hourly_traffic)
    fixed_instances = min(max_instances,
                          max(min_instances,
                              int(np.ceil(peak_traffic / requests_per_instance))))
    fixed_cost_daily = fixed_instances * 24 * cost_per_instance_hour

    # --- Autoscaling: saatlik ayarla ---
    autoscale_cost_daily = 0
    current_instances = min_instances
    hourly_log = []

    for hour, traffic in enumerate(hourly_traffic):
        needed = max(min_instances,
                     int(np.ceil(traffic / requests_per_instance)))
        needed = min(needed, max_instances)

        utilization = traffic / (current_instances * requests_per_instance) \
                      if current_instances > 0 else 1.0

        if utilization > scale_up_threshold:
            current_instances = min(needed + 1, max_instances)
        elif utilization < scale_down_threshold:
            current_instances = max(needed, min_instances)

        autoscale_cost_daily += current_instances * cost_per_instance_hour
        hourly_log.append({
            "hour": hour, "traffic": traffic,
            "instances": current_instances,
            "utilization": round(traffic / (current_instances * requests_per_instance), 2),
        })

    savings_pct = (1 - autoscale_cost_daily / fixed_cost_daily) * 100

    return {
        "fixed_instances": fixed_instances,
        "fixed_cost_daily_usd": round(fixed_cost_daily, 2),
        "autoscale_cost_daily_usd": round(autoscale_cost_daily, 2),
        "savings_pct": round(savings_pct, 1),
        "monthly_savings_usd": round((fixed_cost_daily - autoscale_cost_daily) * 30, 2),
        "hourly_log": hourly_log,
    }

# Örnek: tipik web trafiği profili (gece düşük, gündüz yüksek)
hourly_traffic = [
    200, 150, 100, 80, 80, 100,     # 00:00–05:00 (gece)
    300, 600, 900, 1200, 1500, 1800, # 06:00–11:00 (sabah ramp-up)
    2000, 1900, 1700, 1500, 1600,    # 12:00–16:00 (öğleden sonra)
    1800, 2200, 2500, 2000, 1500,    # 17:00–21:00 (akşam peak)
    800, 400,                         # 22:00–23:00 (gece)
]

result = simulate_autoscaling_cost(
    hourly_traffic=hourly_traffic,
    requests_per_instance=500,
    cost_per_instance_hour=1.41,  # g5.xlarge
)
# Tipik sonuç: %35–50 tasarruf
```

### FinOps Yaklaşımı: Tag'leme, Budget Alert, Birim Maliyet

```python
# --- FinOps prensipleri: ML harcamalarını görünür ve hesap verebilir kıl ---

# 1. Maliyet tag'leme — her resource'a kim/ne/neden tag'ı
REQUIRED_COST_TAGS = {
    "team":        "ml-fraud | ml-recommendation | ml-platform",
    "environment": "dev | staging | production",
    "project":     "fraud-v3 | recsys-reranker | churn-model",
    "cost_center": "engineering | data-science | research",
    "model_name":  "fraud_lgbm_v3 | recsys_twotower_v2",
}
# Kural: tag'sız resource → otomatik uyarı + 7 gün sonra terminate

# 2. Budget alert yapılandırması
BUDGET_ALERTS = {
    "ml-fraud-team": {
        "monthly_budget_usd": 5000,
        "alerts": [
            {"threshold_pct": 50, "action": "email_team_lead"},
            {"threshold_pct": 80, "action": "email_team_lead + slack_channel"},
            {"threshold_pct": 100, "action": "email_vp_eng + auto_stop_dev_instances"},
            {"threshold_pct": 120, "action": "pagerduty_oncall + freeze_non_prod"},
        ],
    },
}

# 3. Birim maliyet metrikleri (FinOps'un ML'e özel katkısı)
UNIT_COST_METRICS = {
    "cost_per_1k_predictions": "Serving maliyeti / 1000 prediction",
    "cost_per_training_run":   "Bir model eğitiminin toplam maliyeti",
    "cost_per_experiment":     "Bir A/B testin infra maliyeti",
    "gpu_utilization_pct":     "GPU'nun gerçekten kullanıldığı süre oranı",
    "cost_per_revenue_dollar": "ML harcaması / ML'in ürettiği gelir",
}
# Hedef: cost_per_1k_predictions'ı her çeyrekte %10 azalt
```

### Geleneksel Maliyet Tahmini

```python
# Maliyet tahmini şablonu
def estimate_monthly_cost(n_predictions_daily: int,
                            model_latency_ms: float,
                            n_gpus_training: int) -> dict:
    """Kaba maliyet tahmini (AWS us-east-1 fiyatları baz)."""
    # Serving: Lambda/ECS Fargate
    n_requests_monthly = n_predictions_daily * 30
    serving_cost = n_requests_monthly * 0.0000002  # $0.20 per 1M

    # Training: g5.xlarge spot ~$0.42/saat (2026 itibarıyla)
    training_hours_monthly = 4 * n_gpus_training  # Haftalık 1 saat
    training_cost = training_hours_monthly * 0.42

    # Storage: S3 + Redis
    storage_cost = 50  # Sabit $50/ay baz

    total = serving_cost + training_cost + storage_cost
    return {
        "serving_cost": serving_cost,
        "training_cost": training_cost,
        "storage_cost": storage_cost,
        "total_monthly_usd": total,
    }
```

> **Çapraz referans:** Spot instance yönetimi ve training pipeline → [Katman E — MLOps](katman-E-mlops.md)

---

## F.6 Model Compression (Sıkıştırma)

Üretimde model boyutunu ve inference süresini azaltmak kritik. Üç temel teknik: **pruning**, **knowledge distillation**, **quantization**.

### Pruning (Budama)

```
Pruning Türleri:

Unstructured Pruning              Structured Pruning
─────────────────────             ─────────────────────
• Bireysel ağırlıkları sıfırla   • Tüm nöron/filtre/katman kaldır
• Sparse matris → özel HW gerek  • Dense matris kalır → standart HW
• Daha yüksek sıkıştırma oranı   • Daha kolay deploy, gerçek hızlanma
• Örnek: %90 ağırlık sıfır       • Örnek: %50 filtre kaldır

Ne zaman hangisi?
├── Özel donanım var (sparse tensor core) → Unstructured
├── Standart GPU/CPU deploy       → Structured
└── Mobile / edge device           → Structured + quantization
```

```python
import torch
import torch.nn.utils.prune as prune

# --- Unstructured pruning örneği ---
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

# Conv2d katmanlarında %30 ağırlığı sıfırla (L1 norm'a göre en düşükler)
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Pruning mask'ını kalıcı hale getir
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# --- Structured pruning örneği (filtre bazlı) ---
# Bir Conv2d katmanındaki en düşük L2 norm'lu %40 filtreyi kaldır
conv_layer = model.layer1[0].conv1
prune.ln_structured(conv_layer, name='weight', amount=0.4, n=2, dim=0)

# Sonuç: kalan filtrelerle daha küçük ve hızlı model
print(f"Sıfır ağırlık oranı: "
      f"{torch.sum(conv_layer.weight == 0).item() / conv_layer.weight.nelement():.1%}")
```

### Knowledge Distillation (Bilgi Damıtma)

```
Teacher-Student Paradigması:

┌──────────────────────┐
│   TEACHER MODEL      │
│   (Büyük, yavaş,     │
│    yüksek doğruluk)  │
│   ResNet-152 / BERT  │
│   Params: 60M+       │
└─────────┬────────────┘
          │ Soft labels (probability distribution)
          │ Temperature T ile softmax → bilgi transferi
          ↓
┌──────────────────────┐
│   STUDENT MODEL      │
│   (Küçük, hızlı,     │
│    kabul edilir acc.) │
│   MobileNet / TinyBERT│
│   Params: 5M          │
└──────────────────────┘

Loss = α × CrossEntropy(student, hard_labels)
     + (1-α) × KL_Divergence(student_soft, teacher_soft) × T²

• T (temperature): Yüksek T → soft distribution daha bilgi verici
  T=1 → normal softmax, T=5–20 → daha "yumuşak" olasılıklar
• α: Hard label vs soft label dengesi (genelde 0.1–0.5)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Knowledge distillation loss: teacher'ın soft label'ları ile öğren."""
    def __init__(self, temperature: float = 5.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, true_labels):
        # Hard label loss (normal cross-entropy)
        hard_loss = F.cross_entropy(student_logits, true_labels)

        # Soft label loss (teacher'ın bilgisi)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # T² ile scale (gradient magnitude düzeltmesi)
        total_loss = (self.alpha * hard_loss +
                      (1 - self.alpha) * soft_loss * self.temperature ** 2)
        return total_loss

# Kullanım
distill_loss = DistillationLoss(temperature=5.0, alpha=0.3)
# Training loop'ta:
# loss = distill_loss(student_out, teacher_out, labels)
# loss.backward()
```

### ONNX Export + Quantization

```python
import torch
import numpy as np

# --- 1. PyTorch model → ONNX export ---
model = ...  # Eğitilmiş PyTorch modeli
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # Örnek input shape

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},     # Dinamik batch boyutu
        "output": {0: "batch_size"},
    },
    opset_version=17,
)

# --- 2. ONNX quantization (FP32 → INT8) ---
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QInt8,    # INT8 ağırlıklar
)

# --- 3. Quantized model ile inference ---
import onnxruntime as ort

# Orijinal vs quantized karşılaştırma
import os
original_size = os.path.getsize("model.onnx") / (1024 * 1024)
quantized_size = os.path.getsize("model_quantized.onnx") / (1024 * 1024)
print(f"Orijinal:  {original_size:.1f} MB")
print(f"Quantized: {quantized_size:.1f} MB")
print(f"Sıkıştırma: {(1 - quantized_size/original_size)*100:.0f}%")
# Tipik sonuç: %60–75 boyut azalması

# Inference benchmark
session = ort.InferenceSession("model_quantized.onnx")
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

import time
times = []
for _ in range(100):
    start = time.perf_counter()
    result = session.run(None, {"input": input_data})
    times.append((time.perf_counter() - start) * 1000)

print(f"Ortalama inference: {np.mean(times):.1f}ms (p99: {np.percentile(times, 99):.1f}ms)")
```

**Model Compression Karar Tablosu:**

| Teknik | Boyut Azalması | Hız Artışı | Doğruluk Kaybı | Ne Zaman? |
|--------|---------------|------------|----------------|-----------|
| Pruning (structured) | %30–60 | %20–40 | <%2 | GPU/CPU serving, edge |
| Pruning (unstructured) | %50–90 | HW bağımlı | <%1 | Sparse tensor HW varsa |
| Distillation | %80–95 | %3–10× | %1–3 | Büyük model → küçük model |
| Quantization (INT8) | %60–75 | %2–4× | <%1 | Her yerde, ilk dene |
| ONNX export | — | %10–30 | 0 | Framework bağımsız deploy |

> **Çapraz referans:** Model mimarileri (ResNet, BERT) ve training → [Katman D — Derin Öğrenme](katman-D-derin-ogrenme.md)

---

## F.7 Canary Deployment Protokolü

Shadow mode'da model doğrulandıktan sonra, canary deployment ile kademeli olarak production trafiğine açılır. Bu bölüm, shadow'dan tam rollout'a kadar olan süreci kapsar.

### Shadow → Canary Geçiş Adımları

```
Model Rollout Pipeline:

1. SHADOW MODE (0% gerçek trafik etkisi)
   │ Model production'da çalışır ama kararı etkilemez
   │ Kontrol: shadow vs production prediction uyumu
   │ Süre: 3–7 gün
   │ Geçiş kriteri: hata oranı <%1, latency bütçe içinde
   ↓
2. CANARY %1 (minimum trafik)
   │ Gerçek kullanıcılara servis başlar
   │ Acil sorunları yakala (crash, timeout, NaN)
   │ Süre: 1–2 saat
   │ Geçiş kriteri: hata yok, metrikler stabil
   ↓
3. CANARY %5 (küçük ölçek doğrulama)
   │ İstatistiksel olarak anlamlı veri topla
   │ Business metriklerini izle (CTR, revenue)
   │ Süre: 6–24 saat
   │ Geçiş kriteri: guardrail metrikler yeşil
   ↓
4. CANARY %25 (orta ölçek)
   │ Farklı segmentlerde performansı kontrol et
   │ Edge case'leri yakala (nadir kullanıcı tipleri)
   │ Süre: 1–3 gün
   │ Geçiş kriteri: tüm segmentlerde iyileşme veya nötr
   ↓
5. CANARY %100 (full rollout)
   │ Tüm trafik yeni modele yönlendirilir
   │ Eski model 7 gün bekleme (hızlı rollback için)
   │ Sonra eski model archive edilir
   ↓
6. BAKE PERIOD (7 gün izleme)
   Geç ortaya çıkan sorunları yakala
   Başarılı → eski model sil | Sorunlu → rollback
```

### Traffic Splitting ve Otomatik Canary Controller

```python
import time
import hashlib
from dataclasses import dataclass
from typing import Optional

@dataclass
class CanaryConfig:
    """Canary deployment konfigürasyonu."""
    model_name: str
    canary_version: str
    stable_version: str
    traffic_pct: float = 0.01            # Başlangıç: %1
    stages: list = None                   # Kademeli artış
    rollback_on_error_rate: float = 0.02  # %2 hata → rollback
    rollback_on_latency_p99_ms: float = 100.0
    rollback_on_metric_drop_pct: float = 5.0  # %5 düşüş → rollback
    min_samples_per_stage: int = 1000

    def __post_init__(self):
        if self.stages is None:
            self.stages = [0.01, 0.05, 0.25, 0.50, 1.0]


class CanaryDeploymentController:
    """
    Otomatik canary deployment yöneticisi.
    Metrikler iyiyse trafik artır, kötüyse rollback.
    """
    def __init__(self, config: CanaryConfig):
        self.config = config
        self.current_stage_idx = 0
        self.metrics_log = []

    def route_request(self, user_id: str) -> str:
        """Kullanıcıyı canary veya stable modele yönlendir."""
        hash_val = int(hashlib.sha256(
            f"{user_id}:{self.config.model_name}".encode()
        ).hexdigest(), 16)
        bucket = (hash_val % 10000) / 10000.0

        if bucket < self.config.traffic_pct:
            return self.config.canary_version
        return self.config.stable_version

    def collect_metrics(self, canary_metrics: dict,
                         stable_metrics: dict) -> dict:
        """Stage metriklerini karşılaştır."""
        comparison = {
            "canary_error_rate": canary_metrics.get("error_rate", 0),
            "stable_error_rate": stable_metrics.get("error_rate", 0),
            "canary_latency_p99": canary_metrics.get("latency_p99_ms", 0),
            "stable_latency_p99": stable_metrics.get("latency_p99_ms", 0),
            "canary_primary_metric": canary_metrics.get("primary_metric", 0),
            "stable_primary_metric": stable_metrics.get("primary_metric", 0),
            "canary_sample_count": canary_metrics.get("sample_count", 0),
        }
        self.metrics_log.append(comparison)
        return comparison

    def evaluate_and_advance(self, canary_metrics: dict,
                               stable_metrics: dict) -> dict:
        """
        Metrikleri değerlendir → ilerle, bekle veya rollback.
        """
        comparison = self.collect_metrics(canary_metrics, stable_metrics)

        # --- Rollback kontrolleri ---
        # 1. Hata oranı çok yüksek mi?
        if comparison["canary_error_rate"] > self.config.rollback_on_error_rate:
            return self._rollback(
                reason=f"Error rate {comparison['canary_error_rate']:.3f} > "
                       f"threshold {self.config.rollback_on_error_rate}")

        # 2. Latency bütçeyi aşıyor mu?
        if comparison["canary_latency_p99"] > self.config.rollback_on_latency_p99_ms:
            return self._rollback(
                reason=f"Latency p99 {comparison['canary_latency_p99']:.0f}ms > "
                       f"threshold {self.config.rollback_on_latency_p99_ms}ms")

        # 3. Primary metric düşüş kontrolü
        if comparison["stable_primary_metric"] > 0:
            drop_pct = ((comparison["stable_primary_metric"] -
                         comparison["canary_primary_metric"]) /
                        comparison["stable_primary_metric"]) * 100
            if drop_pct > self.config.rollback_on_metric_drop_pct:
                return self._rollback(
                    reason=f"Primary metric {drop_pct:.1f}% düştü > "
                           f"threshold {self.config.rollback_on_metric_drop_pct}%")

        # --- Yeterli sample var mı? ---
        if comparison["canary_sample_count"] < self.config.min_samples_per_stage:
            return {"action": "wait",
                    "reason": "Yetersiz sample, bekleniyor",
                    "current_traffic_pct": self.config.traffic_pct}

        # --- Sonraki stage'e geç ---
        return self._advance()

    def _advance(self) -> dict:
        """Trafik yüzdesini bir sonraki stage'e artır."""
        self.current_stage_idx += 1
        if self.current_stage_idx >= len(self.config.stages):
            self.config.traffic_pct = 1.0
            return {"action": "complete",
                    "reason": "Full rollout tamamlandı",
                    "current_traffic_pct": 1.0}

        self.config.traffic_pct = self.config.stages[self.current_stage_idx]
        return {"action": "advance",
                "reason": f"Metrikler pozitif, trafik artırıldı",
                "current_traffic_pct": self.config.traffic_pct}

    def _rollback(self, reason: str) -> dict:
        """Tüm trafiği stable modele geri yönlendir."""
        self.config.traffic_pct = 0.0
        return {"action": "rollback",
                "reason": reason,
                "current_traffic_pct": 0.0,
                "alert": "PagerDuty + Slack bildirimi gönderildi"}

# --- Kullanım örneği ---
config = CanaryConfig(
    model_name="fraud_detector",
    canary_version="fraud_lgbm_v4",
    stable_version="fraud_lgbm_v3",
    stages=[0.01, 0.05, 0.25, 0.50, 1.0],
)
controller = CanaryDeploymentController(config)

# Her stage'de metrik topla ve değerlendir
decision = controller.evaluate_and_advance(
    canary_metrics={"error_rate": 0.005, "latency_p99_ms": 45,
                     "primary_metric": 0.92, "sample_count": 5000},
    stable_metrics={"error_rate": 0.004, "latency_p99_ms": 42,
                     "primary_metric": 0.91, "sample_count": 50000},
)
# decision = {"action": "advance", "current_traffic_pct": 0.05, ...}
```

**Rollback Kriterleri Özet Tablosu:**

| Metrik | Eşik | Aksiyon |
|--------|------|---------|
| Error rate | >%2 | Otomatik rollback |
| Latency p99 | >100ms (bütçenin %100'ü) | Otomatik rollback |
| Primary metric düşüşü | >%5 | Otomatik rollback |
| NaN/Inf prediction | >0 | Anında rollback |
| Memory/CPU spike | >%90 sustained | Otomatik rollback + alert |
| Yeterli sample yok | <min_samples | Bekle, advance etme |

> **Çapraz referans:** CI/CD pipeline'ında canary entegrasyonu → [Katman E — MLOps](katman-E-mlops.md)

---

## F.8 Data Mesh ve Data Contract

### Data Mesh Nedir?

Geleneksel merkezi data team modeli ölçeklenmez: her veri isteği merkezi ekibe bağımlı → darboğaz. Data mesh, veri yönetimini **domain ekiplerine** dağıtır.

```
Geleneksel (Merkezi)              Data Mesh (Dağıtık)
──────────────────────            ──────────────────────
Tek data team her şeyi yapar     Her domain kendi verisini yönetir
Merkezi data lake / warehouse    Domain-oriented veri ürünleri
"Bana veri çek" kuyruğu          Self-serve veri altyapısı
Darboğaz, yavaş                  Ölçeklenebilir, hızlı

Data Mesh'in 4 Prensibi:

1. Domain Ownership (Alan Sahipliği)
   Fraud ekibi → fraud verilerinin sahibi
   Payment ekibi → ödeme verilerinin sahibi
   Her domain kendi pipeline + kalitesinden sorumlu

2. Data as a Product (Ürün Olarak Veri)
   Veri bir üründür: SLA, dokümantasyon, keşfedilebilirlik
   Kalite garantisi veren, versiyonlanan, güvenilir

3. Self-Serve Platform (Öz-Servis Altyapı)
   Platform team altyapıyı sağlar (infra, tooling)
   Domain team'ler kendi pipeline'larını kurar
   "Paved road" — standart yollar, ama esneklik de var

4. Federated Governance (Federatif Yönetişim)
   Global standartlar (naming, format, PII kuralları)
   Lokal uygulama (her domain kendi içinde uygular)
   Interoperability: domain'ler arası veri kullanılabilir
```

### Data Contract: Schema, SLA, Ownership

```python
# Data contract — veri üreticisi ile tüketicisi arasındaki anlaşma
# YAML/JSON formatında tanımlanır, CI/CD'de doğrulanır

DATA_CONTRACT_EXAMPLE = {
    "contract_id": "payments.transactions.v3",
    "version": "3.2.0",
    "owner": {
        "team": "payments-platform",
        "contact": "payments-oncall@company.com",
        "slack": "#payments-data",
    },

    # --- Schema tanımı ---
    "schema": {
        "type": "record",
        "fields": [
            {"name": "transaction_id",   "type": "string",  "required": True,
             "description": "Unique transaction identifier (UUID v4)"},
            {"name": "user_id",          "type": "string",  "required": True,
             "pii": True},
            {"name": "amount_cents",     "type": "int64",   "required": True,
             "constraints": {"min": 0, "max": 100_000_000}},
            {"name": "currency",         "type": "string",  "required": True,
             "constraints": {"enum": ["USD", "EUR", "TRY", "GBP"]}},
            {"name": "merchant_id",      "type": "string",  "required": True},
            {"name": "status",           "type": "string",  "required": True,
             "constraints": {"enum": ["pending", "completed", "failed", "refunded"]}},
            {"name": "event_timestamp",  "type": "timestamp", "required": True},
        ],
    },

    # --- SLA tanımları ---
    "sla": {
        "freshness": "max 5 minutes lag",       # Streaming: 5dk gecikme max
        "availability": "99.9%",                 # Aylık uptime
        "completeness": ">99.5% rows present",   # Eksik satır toleransı
        "update_frequency": "real-time (Kafka)",
        "retention": "7 years (regulatory)",
    },

    # --- Kalite kuralları ---
    "quality_rules": [
        {"rule": "transaction_id is unique",     "severity": "critical"},
        {"rule": "amount_cents >= 0",            "severity": "critical"},
        {"rule": "event_timestamp < now() + 1h", "severity": "warning"},
        {"rule": "null rate of user_id < 0.1%",  "severity": "critical"},
    ],

    # --- Breaking change politikası ---
    "evolution_policy": {
        "backward_compatible": ["add optional field", "widen enum"],
        "breaking_change": ["remove field", "change type", "narrow enum"],
        "breaking_change_process": "30 gün önceden bildirim + migration plan",
    },

    # --- Tüketiciler ---
    "consumers": [
        {"team": "ml-fraud", "usage": "fraud model training + serving features"},
        {"team": "analytics", "usage": "revenue dashboards"},
        {"team": "finance", "usage": "reconciliation"},
    ],
}
```

### ML Pipeline'da Data Contract Doğrulama

```python
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class ContractViolation:
    rule: str
    severity: str  # "critical" | "warning"
    details: str

def validate_data_contract(df: pd.DataFrame,
                            contract: dict) -> list[ContractViolation]:
    """
    ML pipeline'ın başında data contract'ı doğrula.
    Contract ihlali varsa → pipeline durur (critical) veya uyarı verir.
    """
    violations = []

    # 1. Schema kontrolü — beklenen kolonlar var mı?
    expected_fields = {f["name"] for f in contract["schema"]["fields"]}
    actual_fields = set(df.columns)
    missing = expected_fields - actual_fields
    if missing:
        violations.append(ContractViolation(
            rule="schema_completeness",
            severity="critical",
            details=f"Eksik kolonlar: {missing}",
        ))

    # 2. Required field null kontrolü
    for field in contract["schema"]["fields"]:
        if field.get("required") and field["name"] in df.columns:
            null_rate = df[field["name"]].isnull().mean()
            if null_rate > 0.001:  # >%0.1 null → ihlal
                violations.append(ContractViolation(
                    rule=f"null_check_{field['name']}",
                    severity="critical",
                    details=f"{field['name']} null oranı: {null_rate:.4f}",
                ))

    # 3. Constraint kontrolü (min/max, enum)
    for field in contract["schema"]["fields"]:
        constraints = field.get("constraints", {})
        col = field["name"]
        if col not in df.columns:
            continue

        if "min" in constraints:
            below_min = (df[col] < constraints["min"]).sum()
            if below_min > 0:
                violations.append(ContractViolation(
                    rule=f"range_check_{col}",
                    severity="critical",
                    details=f"{col}: {below_min} satır min ({constraints['min']}) altında",
                ))

        if "enum" in constraints:
            invalid = set(df[col].dropna().unique()) - set(constraints["enum"])
            if invalid:
                violations.append(ContractViolation(
                    rule=f"enum_check_{col}",
                    severity="warning",
                    details=f"{col}: geçersiz değerler {invalid}",
                ))

    # 4. Freshness kontrolü
    if "event_timestamp" in df.columns:
        max_ts = pd.to_datetime(df["event_timestamp"]).max()
        lag = pd.Timestamp.now() - max_ts
        if lag > pd.Timedelta(minutes=30):
            violations.append(ContractViolation(
                rule="freshness_check",
                severity="warning",
                details=f"Veri gecikmesi: {lag}",
            ))

    return violations

# Pipeline'da kullanım
def ml_training_pipeline(data_path: str, contract: dict):
    """Data contract doğrulaması ile başlayan ML pipeline."""
    df = pd.read_parquet(data_path)

    # Contract doğrula
    violations = validate_data_contract(df, contract)

    critical_violations = [v for v in violations if v.severity == "critical"]
    if critical_violations:
        for v in critical_violations:
            print(f"[CRITICAL] {v.rule}: {v.details}")
        raise ValueError(
            f"Data contract ihlali: {len(critical_violations)} critical violation. "
            f"Pipeline durduruldu. Data owner'a bildirim gönderildi."
        )

    warnings = [v for v in violations if v.severity == "warning"]
    for v in warnings:
        print(f"[WARNING] {v.rule}: {v.details}")

    # Contract geçti → training devam
    print(f"Data contract doğrulandı. {len(df)} satır, {len(warnings)} uyarı.")
    # ... model training ...
```

> **Çapraz referans:** Data quality monitoring ve pipeline orchestration → [Katman E — MLOps](katman-E-mlops.md)

---

## Sektör Notu — ML System Design 2026

2026 itibarıyla üretim sistemlerinden öğrenimler:

- **Uber Michelangelo:** 10 trilyon feature hesabı/gün. Online store: Redis cluster, Offline: Hive+S3. Training-serving skew'in ana nedeni: "Python'da bir şey, Java'da başka şey hesaplanıyor."

- **Airbnb Zipline:** Sub-10ms feature latency, milyonlarca model. En önemli ders: feature pipeline ile model pipeline ayrı versiyonlanmalı.

- **DoorDash Fabricator:** Feature engineering süresini %90 azalttı. Sadece SQL + metadata tanımla, sistem online+offline store'a otomatik deploy.

- **Google/YouTube:** p99 <5ms feature serving. Embedding boyutu (128d) ile retrieval kalitesi tradeoff — küçük embedding → hızlı, büyük → kaliteli.

- **FinOps for AI (2026 trendi):** Kuruluşların %98'i artık AI harcamalarını aktif olarak yönetiyor (2024'te %31'di). GPU tüketimi, token bazlı faturalandırma ve model retrain döngüleri yeni finansal volatilite kaynakları. Pre-deployment architecture costing — altyapı provision edilmeden önce maliyet analizi — standart pratik haline geldi.

- **Data Mesh olgunlaşması:** Contract-driven data consumption, SLO/SLA tanımları ve data quality dashboard'ları artık radikal kavramlar değil, standart uygulama. ML modelleri artık domain data product'larının bir output port'u olarak tasarlanıyor.

---

## Katman F Kontrol Listesi

- [ ] Online vs offline serving farkını ve ne zaman hangisini seçeceğimi biliyorum
- [ ] Feature store kavramını (offline + online) açıklayabilirim
- [ ] Training-serving skew nedir ve nasıl önlenir biliyorum
- [ ] Latency bütçesi tablosu hazırlayabilirim
- [ ] Fraud tespiti veya öneri sistemi mimarisini whiteboard'da çizebilirim
- [ ] Shadow mode deployment nedir, nasıl uygulanır biliyorum
- [ ] Maliyet optimizasyon tekniklerini ve FinOps prensiplerini biliyorum
- [ ] GPU vs CPU maliyet trade-off'unu açıklayabilirim
- [ ] Model compression tekniklerini (pruning, distillation, quantization) biliyorum
- [ ] ONNX export ve quantization uygulayabilirim
- [ ] Canary deployment protokolünü ve rollback kriterlerini açıklayabilirim
- [ ] Data mesh prensiplerini ve data contract kavramını biliyorum
- [ ] Search ranking sistemi mimarisini çizebilirim
- [ ] Dynamic pricing sistemi tasarlayabilirim
- [ ] Proje-7 (Sistem Tasarım Dokümanı) tamamlandı

---

<div class="nav-footer">
  <span><a href="#file_katman_E_mlops">← Önceki: Katman E — MLOps</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_G_senior_davranislar">Sonraki: Katman G — Senior Davranışlar →</a></span>
</div>
