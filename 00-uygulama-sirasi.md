# 00 — Uygulama Sırası (Sırayla Yapılacaklar)

> Bu dosya, rehberdeki tüm içerikleri **en verimli öğrenme sırasında** kullanman için tasarlandı.
>
> **Kural:** Her aşamada mutlaka bir **teslimat (deliverable)** üret. Teslimat yoksa öğrenme kalıcı olmaz.

---

## Genel Strateji

**%20 teori / %80 pratik.**

Senior veri bilimci olma yolculuğu doğrusal değil, sarmal. Her katmanda bir öncekini derinleştirirsin. Ama başlangıçta sıra önemli: temelsiz ağaç olmaz.

```
Setup → Matematik → Analitik Temel → İstatistik Kasları → Klasik ML
    → Deney/Nedensellik → Uzmanlık → MLOps → Senior Davranışlar
```

**Tahmini toplam süre:** Haftada ~15 saat (günde 2–3 saat) çalışarak **5–7 ay**. Tam zamanlı (haftada ~40 saat) çalışarak **2–3 ay**.

---

## Aşama 0 — Setup (1 gün)

> **Süre:** Haftada 15 saat çalışarak 1 gün | Tam zamanlı: 1 gün

> **Detay için bkz.** `katman-A-temeller.md`

### Yapılacaklar

1. **Python ortamı** kur
   - Python 3.11+ indir
   - `uv` paket yöneticisini kur: `pip install uv`
   - `uv venv .venv && source .venv/bin/activate`
   - Temel paketler: `uv pip install pandas numpy scikit-learn matplotlib seaborn jupyter`

2. **Geliştirme ortamı**
   - VSCode indir + Python, Pylance, Jupyter, GitLens, Ruff eklentilerini ekle
   - Alternatif: JupyterLab

3. **Git/GitHub**
   - `git init` ile ilk repo oluştur
   - GitHub'da public repo aç (portföy için)
   - SSH key ayarla

4. **Proje klasör şablonu** oluştur:

```
ds-portfoy/
├── README.md
├── notebooks/
│   └── 00_setup_test.ipynb
├── src/
│   └── __init__.py
├── data/
│   └── raw/
├── reports/
│   └── figures/
└── requirements.txt
```

### Teslimat

- [ ] Boş repo GitHub'da
- [ ] Çalışır notebook (import pandas, sklearn başarılı)
- [ ] README.md (sadece "Bu repo X için" düzeyinde bile olsa)

---

## Aşama 0.5 — Matematik Hızlandırma (3–5 gün, isteğe bağlı)

> **Süre:** Haftada 15 saat çalışarak 1 hafta | Tam zamanlı: 3–5 gün

> **Detay için bkz.** `katman-0-matematik.md`

> "Matematiği atlayabilir miyim?" sorusu sık sorulur. Cevap: **ML'i kullanan için hayır, icat eden için evet değil.** Sezgisel anlamak zorunlu, ezber değil.

### Yapılacaklar

1. **Lineer cebir** — vektörler, matris çarpımı, özdeğer sezgisi
   - 3Blue1Brown "Essence of Linear Algebra" (YouTube, ~3 saat)
   - NumPy ile pratik: `np.dot`, `np.linalg.eig`, broadcasting

2. **Kalkülüs** — türev, zincir kuralı, gradient descent sezgisi
   - 3Blue1Brown "Essence of Calculus" (ilk 4 video yeterli)
   - Gradient descent'i NumPy ile lineer regresyon üzerine sıfırdan uygula

3. **Olasılık** — dağılımlar, koşullu olasılık, Bayes
   - StatQuest (YouTube) — Normal, Binomial, Bayes videoları
   - Beta-Binomial konjugat: prior → posterior görselleştir

### Pratik

```python
import numpy as np

# Gradient descent ile lineer regresyon
X = np.random.randn(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

w = np.zeros(2)
lr = 0.01

for i in range(200):
    grad = -2 * X.T @ (y - X @ w) / len(y)
    w -= lr * grad

print(f"Öğrenilen ağırlıklar: {w}")  # [~3, ~2] beklenir
```

### Teslimat

- [ ] El yazısıyla veya Python ile 5 temel ispat/gösteri:
  1. Matris çarpımı (elle + `@` operatörü kıyası)
  2. Gradient descent ile lineer regresyon
  3. Sigmoid türevi numpy ile
  4. Bayes teoremi spam filtresi hesabı
  5. Beta dağılımı prior → posterior plot

---

## Aşama 1 — Analitik Temel (1–2 hafta)

> **Süre:** Haftada 15 saat çalışarak 2 hafta | Tam zamanlı: 1 hafta

> **Detay için bkz.** `katman-A-temeller.md`

### Yapılacaklar

1. **Python/Pandas:**
   - Veri okuma, temizleme, tip dönüşümleri
   - GroupBy, merge/join, zaman serisi
   - `SettingWithCopyWarning` tuzağını tanı ve düzelt
   - Polars'a giriş (Pandas'tan hızlı)

2. **SQL:**
   - SELECT, JOIN, CTE (WITH), subquery
   - Window functions: ROW_NUMBER, RANK, LAG, LEAD, SUM OVER
   - Funnel + cohort retention sorguları
   - En az 10 pencere fonksiyonu sorusu çöz (DataLemur)

3. **EDA & Raporlama:**
   - `df.describe()`, `df.isna().mean()`, histogram, boxplot
   - Korelasyon heatmap
   - 1 sayfa executive summary yaz

### Günlük Plan (1–2 haftalık)

| Gün | Konu |
|-----|------|
| 1–2 | Python temelleri: veri tipleri, dosya işlemleri, hata yönetimi |
| 3–4 | Pandas: groupby, merge, zaman serisi |
| 5–6 | SQL: JOIN + CTE + window functions |
| 7–8 | SQL: funnel, cohort, sessionization |
| 9–10 | EDA: eksik değer, uç değer, dağılım analizi |
| 11–14 | Proje-0 tamamla |

### Teslimat: Proje-0 Analitik Paket

```
proje-0-analitik/
├── analysis.ipynb    # EDA + grafikler
├── queries.sql       # funnel + retention sorguları
└── README.md         # 1 sayfa özet
```

---

## Aşama 2 — İstatistik Kasları (1–2 hafta)

> **Süre:** Haftada 15 saat çalışarak 2 hafta | Tam zamanlı: 1 hafta

> **Detay için bkz.** `katman-A-temeller.md` (istatistik bölümü)

### Yapılacaklar

1. **Güven aralığı (CI) + bootstrap**
   - Parametrik CI formülü
   - Bootstrap ile non-parametrik CI (10,000 resample)

2. **Hipotez testleri**
   - p-value nedir, ne değildir
   - t-test, chi-square, Mann-Whitney seçimi
   - Effect size: Cohen's d, uplift

3. **Pratik anlamlılık**
   - İstatistiksel anlamlılık ≠ iş etkisi
   - Çoklu karşılaştırma: Bonferroni, Benjamini-Hochberg

4. **Bayesian giriş**
   - Prior × likelihood → posterior
   - Beta-Binomial A/B test

### Teslimat

```python
# Bootstrap CI notebook
def bootstrap_ci(data, stat_fn=np.mean, n=10_000, alpha=0.05):
    boots = [stat_fn(np.random.choice(data, size=len(data), replace=True))
             for _ in range(n)]
    return np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])

# Parametrik vs bootstrap CI kıyası — bu notebooku GitHub'a yükle
```

---

## Aşama 3 — Klasik ML (2–4 hafta)

> **Süre:** Haftada 15 saat çalışarak 4 hafta | Tam zamanlı: 2 hafta

> **Detay için bkz.** `katman-B-klasik-ml.md`

### Yapılacaklar — sıra önemli

1. **Problem framing:** maliyet matrisi, metrik seçimi, iş etkisi
2. **Split stratejileri:** zaman bazlı, group bazlı (aynı kullanıcı iki tarafta olmasın)
3. **Baseline:** logistic regression (veya naïve tahmin)
4. **Güçlü model:** LightGBM / XGBoost
5. **Hyperparameter tuning:** Optuna ile (Grid Search değil)
6. **Calibration + threshold seçimi:** maliyet fonksiyonuyla optimal eşik
7. **Error analysis:** segment bazlı hata analizi (ülke? zaman? kullanıcı tipi?)
8. **SHAP:** global + lokal açıklanabilirlik

### Günlük Plan (2–4 haftalık)

| Hafta | Konu |
|-------|------|
| 1 | Klasik ML kavramları: doğrusal modeller, ağaçlar |
| 2 | Boosting: LightGBM, feature engineering, leakage önleme |
| 3 | Değerlendirme: metrik seçimi, calibration, threshold |
| 4 | Proje-1 tamamla (uçtan uca churn modeli) |

### Teslimat: Proje-1 Churn Tahmini

```
proje-1-churn/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_features.py
├── README.md
└── model_card.md
```

**Checklist:**
- [ ] Zaman bazlı split (gelecek bilgisi train'de yok)
- [ ] Feature pencereleri doğru (leakage yok)
- [ ] Baseline → LightGBM → ablation
- [ ] SHAP summary plot
- [ ] Error analysis raporu
- [ ] Threshold seçimi (maliyet matrisine göre)

---

## Aşama 4 — Deney/Nedensellik (1–3 hafta)

> **Süre:** Haftada 15 saat çalışarak 3 hafta | Tam zamanlı: 1–2 hafta

> **Detay için bkz.** `katman-C-deney-nedensellik.md`

### Yapılacaklar

1. **A/B test tasarım şablonu** (primary + guardrail metric)
2. **Power analizi ve MDE hesabı**
3. **CUPED** — varyans azaltma, CI'yi daralt
4. **Sequential testing** tuzakları — peeking neden yanlış?
5. **Network effects ve karıştırıcı etkiler**
6. **Causal inference giriş:** DiD, propensity score, DoWhy

### Teslimat: Proje-2 A/B Test Paketi

```
proje-2-ab-test/
├── ab_test_analysis.ipynb    # frekantist + bootstrap + CUPED
├── power_analysis.ipynb      # MDE hesabı
├── peeking_simulation.ipynb  # peeking etkisi gösterimi
└── README.md
```

**Checklist:**
- [ ] A/B test tasarım dokümanı yazıldı (primary + guardrail metric tanımlı)
- [ ] Power analizi ile sample size hesaplandı
- [ ] CUPED ile varyans azaltma uygulandı ve CI daraltıldı
- [ ] Peeking simülasyonu çalıştırıldı, false positive artışı gösterildi
- [ ] En az bir causal inference yöntemi (DiD veya propensity score) uygulandı
- [ ] Sonuçlar iş diline çevrildi (uplift % ve tahmini gelir etkisi)

---

## Aşama 5 — Uzmanlık Seçimi (2–6 hafta)

> **Süre:** Haftada 15 saat çalışarak 4–6 hafta | Tam zamanlı: 2–3 hafta

> **Detay için bkz.** `katman-D-derin-ogrenme.md`

Birini seç (sonra diğerlerine geç):

| Seçenek | Proje | Dosya |
|---------|-------|-------|
| NLP | Proje-4: Şikayet sınıflandırma | `katman-D-derin-ogrenme.md` |
| RecSys | Proje-5: Two-stage öneri sistemi | `katman-D-derin-ogrenme.md` |
| CV | Görüntü sınıflandırma | `katman-D-derin-ogrenme.md` |
| LLM/RAG | RAG uygulaması | `katman-D-derin-ogrenme.md` |
| Reinforcement Learning / Contextual Bandits | Dinamik fiyatlama veya öneri optimizasyonu | `katman-D-derin-ogrenme.md` |
| AutoML (AutoGluon) | Hızlı prototipleme + baseline karşılaştırma | `katman-B-klasik-ml.md` |

### Seçenek Detayları

#### NLP — Doğal Dil İşleme

Metin sınıflandırma ile başla (müşteri şikayeti kategorilendirme). Önce TF-IDF + Logistic Regression baseline kur, sonra Hugging Face Transformers ile fine-tuned BERT/DistilBERT modeline geç. Tokenizer mantığını, attention mekanizmasını sezgisel düzeyde öğren. Türkçe NLP için `dbmdz/bert-base-turkish-cased` modelini dene. **Kaynak:** Hugging Face NLP Course (ücretsiz), "Natural Language Processing with Transformers" (O'Reilly).

#### RecSys — Öneri Sistemleri

İki aşamalı mimariyle başla: candidate generation (ANN/FAISS ile) + ranking (LightGBM veya neural ranker). Collaborative filtering (ALS) ile baseline kur, sonra content-based ve hybrid yaklaşımlara geç. Cold-start problemini çözmek için popularity fallback ve feature-based yaklaşımları uygula. Offline metrikleri (NDCG, MAP) ile online metrikleri (CTR, session süresi) ayırt et. **Kaynak:** "Recommendation Systems" (Aggarwal), Google RecSys kursları.

#### CV — Bilgisayarlı Görme

Transfer learning ile başla: pretrained ResNet/EfficientNet üzerine kendi veri setini fine-tune et. Data augmentation teknikleri (Albumentations kütüphanesi) kritik. Confusion matrix ile sınıflar arası hata analizini yap. Object detection (YOLO) ve segmentation (SAM) konularına göz at ama önce sınıflandırmayı sağlamca otur. **Kaynak:** fast.ai "Practical Deep Learning for Coders" (ücretsiz), PyTorch tutorials.

#### LLM/RAG — Büyük Dil Modelleri ve Retrieval-Augmented Generation

RAG pipeline kur: doküman chunking → embedding (OpenAI veya sentence-transformers) → vector DB (ChromaDB/Qdrant) → LLM ile cevap üretimi. Chunk boyutu, overlap, retrieval stratejisi (hybrid search) deneylerini yap. Evaluation için RAGAS framework kullan (faithfulness, relevance, context recall). Fine-tuning kararı: ne zaman RAG yeterli, ne zaman LoRA/QLoRA gerekli? **Kaynak:** LangChain/LlamaIndex dokümanları, "Building LLM Apps" (Hugging Face).

#### Reinforcement Learning / Contextual Bandits

Klasik RL yerine iş problemlerine daha yakın olan contextual bandits ile başla. Epsilon-greedy, Thompson Sampling ve LinUCB algoritmalarını öğren. Kullanım alanları: dinamik fiyatlama, reklam optimizasyonu, kişiselleştirilmiş öneri sıralama. Vowpal Wabbit veya `coba` kütüphanesi ile pratik yap. Offline policy evaluation (OPE) ile gerçek A/B test öncesi politika karşılaştırması yap. **Kaynak:** "Bandit Algorithms for Website Optimization" (White), Sutton & Barto Bölüm 2 (bandits).

#### AutoML (AutoGluon) — Hızlı Prototipleme

AutoGluon ile saatler içinde güçlü baseline modeli kur. `TabularPredictor` ile train et, feature importance ve leaderboard raporlarını analiz et. AutoML çıktısını benchmark olarak kullan: elle kurduğun modelin AutoML'i geçip geçemediğini ölç. Dikkat: AutoML "sihirli değnek" değil — problem framing, feature engineering ve data leakage kontrolü hâlâ senin sorumluluğunda. **Kaynak:** AutoGluon documentation, "AutoML: Methods, Systems, Challenges" (Springer).

**Tavsiye:** 2026 iş piyasasında NLP/LLM ve MLOps birlikte isteniyor. Contextual bandits ise kişiselleştirilmiş ürün deneyimleri sunan şirketlerde (e-ticaret, fintech, medya) giderek daha çok aranıyor.

**Checklist:**
- [ ] Seçilen uzmanlık alanında en az 1 uçtan uca proje tamamlandı
- [ ] Baseline model kuruldu ve güçlü model ile karşılaştırıldı
- [ ] Model performansı iş metriğine çevrildi (örneğin: "precision %85 → yanlış sınıflandırma maliyeti %40 azaldı")
- [ ] Proje README'sinde problem tanımı, yaklaşım, sonuçlar ve learned lessons var
- [ ] Kod notebooks'tan src/'ye taşındı (reusable modüller)
- [ ] En az 1 blog yazısı veya teknik sunum hazırlandı

---

## Aşama 6 — MLOps + Sistem Tasarımı (2–6 hafta)

> **Süre:** Haftada 15 saat çalışarak 6 hafta | Tam zamanlı: 2–3 hafta

> **Detay için bkz.** `katman-E-mlops.md` ve `katman-F-sistem-tasarimi.md`

### Yapılacaklar

1. **Kod paketleme:** `notebooks/` → `src/` (üretilebilir kod)
2. **Data validation:** Pandera veya Great Expectations ile veri kalitesi kontrolleri
3. **FastAPI servis:** `/health` + `/predict` endpoint'leri
4. **Docker:** Dockerfile yazma, image build
5. **MLflow:** deney takibi, model registry
6. **İzleme:** latency + data drift (Evidently)
7. **Evidently drift raporu** oluşturma
8. **Feature store:** Feast ile offline/online feature tutarlılığı
9. **CI/CD:** GitHub Actions ile otomatik test

### Data Validation Örneği (Pandera)

```python
import pandera as pa
from pandera import Column, Check

# Train/serving verisinin şemasını tanımla
schema = pa.DataFrameSchema({
    "age":       Column(int, Check.in_range(0, 120), nullable=False),
    "income":    Column(float, Check.greater_than(0), nullable=True),
    "city":      Column(str, Check.isin(["Istanbul", "Ankara", "Izmir", "Diger"])),
    "churn":     Column(int, Check.isin([0, 1])),
})

# Veri her pipeline çalıştığında validate et
validated_df = schema.validate(df, lazy=True)  # lazy=True: tüm hataları topla
```

### Evidently Drift Raporu Örneği

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Referans (train) vs güncel (production) veriyi karşılaştır
drift_report = Report(metrics=[
    DataDriftPreset(),
    TargetDriftPreset(),
])

drift_report.run(
    reference_data=df_train,   # train seti
    current_data=df_prod,      # production'dan gelen son 7 gün
)

# HTML rapor kaydet — CI/CD'de artifact olarak sakla
drift_report.save_html("reports/drift_report.html")

# Programatik erişim — alert sistemi için
result = drift_report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    print("UYARI: Data drift tespit edildi! Model retraining gerekebilir.")
```

### Feature Store Mini Setup (Feast)

```python
# feature_store.yaml (proje kökünde)
"""
project: churn_features
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
"""

# features.py — feature tanımları
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

customer = Entity(name="customer_id", join_keys=["customer_id"])

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="total_purchases_30d", dtype=Int64),
        Field(name="avg_session_duration", dtype=Float32),
        Field(name="days_since_last_login", dtype=Int64),
    ],
    source=FileSource(
        path="data/customer_features.parquet",
        timestamp_field="event_timestamp",
    ),
)

# Kullanım: feast apply → feast materialize → online serving
# Bu sayede train ve serving'de aynı feature logic kullanılır
```

### Teslimat: Proje-6 MLOps Mini Platform

```
proje-6-mlops/
├── src/
│   ├── api.py          # FastAPI
│   ├── train.py
│   ├── monitor.py
│   └── validate.py     # Pandera schema checks
├── feature_store/
│   ├── feature_store.yaml
│   └── features.py
├── tests/
├── reports/
│   └── drift_report.html
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ml-pipeline.yml
└── README.md
```

**Checklist:**
- [ ] Model MLflow registry'de
- [ ] FastAPI container ayağa kalkıyor
- [ ] Data validation (Pandera/Great Expectations) pipeline'a entegre
- [ ] Evidently drift raporu otomatik üretiliyor
- [ ] Feature store (Feast) kuruldu, offline/online tutarlılık sağlandı
- [ ] Drift tespit edildiğinde alert mekanizması var
- [ ] GitHub Actions testi geçiyor
- [ ] Model card yazılı
- [ ] README'de mimari diyagram (Mermaid) var

---

## Aşama 7 — Senior Seviyeye Geçiş (Sürekli)

> **Süre:** Bu aşama süreklidir. İlk geçiş için haftada 15 saat çalışarak 4–6 hafta yoğun pratik. Sonrası kariyer boyu devam.

> **Detay için bkz.** `katman-G-senior-davranislar.md` ve `katman-F-sistem-tasarimi.md`

Bu aşama bir defalık değil, sürekli bir pratik:

1. **Dokümantasyon:** Her projede model card + karar dokümanı (ADR)
2. **Stakeholder dili:** AUC değil, "kaç müşteri daha kaldı, etkisi ne?"
3. **Etki ölçümü:** İş metrikleriyle bağlantı kur (OKR çerçevesi)
4. **Mentorluk alışkanlığı:** Code review checklist ile kendi kodunu gözden geçir
5. **Technical debt:** Her sprint'in %20'si eski borcu öde
6. **Fairness ve model ethics:** Her üretim modeli için adalet kontrolü yap

### Gerçekçi Vaka Çalışması — STAR Formatı

Mülakatlarda ve iş içerisinde etki anlatırken STAR formatını kullan:

> **Situation:** E-ticaret şirketinde müşteri kayıp oranı (churn) %18'e çıktı, yönetim müşterileri elde tutmak için hedefli kampanya yapmak istiyordu ama kimlere odaklanacağını bilmiyordu.
>
> **Task:** Churn olasılığı yüksek müşterileri 30 gün önceden tespit eden bir model kur ve kampanya bütçesini optimize et.
>
> **Action:** 12 aylık işlem verisinden 45 feature ürettim (RFM, session süresi, destek talebi sayısı). LightGBM ile model kurdum, SHAP ile en etkili faktörleri belirledim. Threshold'u maliyet matrisine göre optimize ettim: false negative (kaçırılan churn) maliyeti false positive'den 5x fazlaydı. Modeli FastAPI ile deploy edip, haftalık batch prediction pipeline'ı kurdum.
>
> **Result:** Hedefli kampanya ile churn oranı %18 → %12'ye düştü (6 puan iyileşme). Kampanya bütçesi %35 azaldı (sadece yüksek riskli müşterilere odaklanıldı). Yıllık tahmini tasarruf: ~2.4M TL.

### OKR Bağlantısı — Somut Örnek

Modelini her zaman şirket OKR'larına bağla:

| OKR | Model Katkısı | Metrik |
|-----|---------------|--------|
| **O:** Müşteri elde tutma oranını artır | Churn prediction modeli | Churn oranı %18 → %12 |
| **KR1:** Kayıp riski yüksek müşterilerin %80'ini tespit et | Recall@top20% | %82 (hedef: %80) |
| **KR2:** Kampanya maliyetini %30 azalt | Precision-based targeting | %35 azalma (hedef: %30) |
| **KR3:** Q3 sonuna kadar production'a al | Deployment tarihi | 2 hafta erken teslim |

**Böylece:** "AUC 0.87 olan bir model kurdum" değil, "Müşteri kayıp oranını 6 puan düşürerek yıllık 2.4M TL tasarruf sağlayan bir sistem kurdum" dersin.

### Fairness ve Model Ethics Kontrol Listesi

Her production modeli için aşağıdaki kontrolleri yap:

| Kontrol | Açıklama | Araç |
|---------|----------|------|
| **Demografik parite** | Model tahminleri korumalı gruplara (cinsiyet, yaş, etnik köken) göre anlamlı farklılık gösteriyor mu? | `fairlearn`, `aequitas` |
| **Eşit fırsat (equalized odds)** | True positive rate gruplar arası eşit mi? | `fairlearn.metrics` |
| **Calibration farkı** | Model her grup için eşit derecede kalibre mi? | Calibration curve per group |
| **Proxy değişken kontrolü** | Posta kodu, isim gibi değişkenler korumalı özellikler için proxy görevi görüyor mu? | SHAP interaction analizi |
| **Veri temsili** | Eğitim verisinde azınlık grupları yeterince temsil ediliyor mu? | `df.groupby("group").size()` |
| **Açıklanabilirlik** | Red kararı alan bir kullanıcı "neden?" diye sorsa, anlaşılır bir açıklama verebiliyor musun? | SHAP local explanation |
| **Dokümantasyon** | Model card'da fairness metrikleri ve bilinen kısıtlamalar yazılı mı? | Model card şablonu |

```python
# Fairlearn ile hizli fairness kontrolu
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score

metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "recall": recall_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=df_test["gender"],
)

print(metric_frame.by_group)        # Grup bazlı metrikler
print(metric_frame.difference())    # Gruplar arası max fark
# Fark > 0.05 ise alarma geç ve müdahale stratejisi belirle
```

### Teslimat: Proje-7 Sistem Tasarım Dokümanı

- Real-time fraud scoring mimarisi
- Mimari diyagram (Mermaid veya draw.io)
- Latency bütçesi tablosu
- Retraining stratejisi
- Maliyet tahmini
- Fairness raporu (en az 2 metrik, grup bazlı sonuçlar)

**Checklist:**
- [ ] En az 1 proje için STAR formatında etki hikayesi yazıldı
- [ ] OKR bağlantısı kuruldu: model metriği → iş metriği → parasal etki
- [ ] Sistem tasarım dokümanı tamamlandı (mimari diyagram + latency bütçesi)
- [ ] Fairness kontrol listesi en az 1 model için uygulandı
- [ ] Model card'da fairness metrikleri ve bilinen kısıtlamalar yer alıyor
- [ ] ADR (Architecture Decision Record) yazıldı: hangi model, neden, alternatifler
- [ ] En az 1 junior'a code review veya mentorluk yapıldı (veya simülasyonu)
- [ ] Stakeholder sunumu hazırlandı (teknik olmayan kitleye yönelik)

---

## En Kısa "İşe Girilebilir" Rota

Zaman çok kısıtlıysa (1–2 aylık sprint):

```
Aşama 0 → Aşama 1 → Aşama 3
(Proje-0 + Proje-1 + SQL mülakat seti + 2–3 ML case)
```

**Somut zaman planı:** Haftada 15 saat çalışarak ~6 hafta. Tam zamanlı (haftada 40 saat) çalışarak ~3 hafta.

Bu rota junior-mid pozisyonlar için yeterli. Senior için Aşama 4–7 gerekli (ek 3–5 ay).

---

## Haftalık Çalışma Ritmi

### Günlük plan (2–3 saat)
- **30 dk** — Teori (okuma, video)
- **90 dk** — Pratik (kod, proje)
- **30 dk** — Not + özet + "yarın ne yapacağım" listesi

### Haftalık plan
| Gün | Aktivite |
|-----|---------|
| Pazartesi–Çarşamba | Yeni konu öğren |
| Perşembe | Mini proje veya ödev teslimi |
| Cuma | Haftalık review + rapor (bulgular + grafik) |
| Hafta sonu | Kaggle / okuma / izleme (zorunlu değil) |

### "Takildim" protokolu
1. 15 dk kendi basina dene
2. Hata mesajini + kodu dokumana/Stack Overflow'a bak
3. 15 dk daha
4. ChatGPT/Claude'a sor (ama cevabi anlamadan gecme!)
5. Not tut: "Bu konuda takildim, X ogrendim"

---

## Aylik Check-in

Her ay sonunda kendine sor:

- [ ] Bu ay kac teslimat urettim?
- [ ] En zayif konu neydi? (Buna odaklan)
- [ ] Portfoyume ne ekledim?
- [ ] Mulakat pratigi yaptim mi? (DataLemur'dan en az 10 SQL)
- [ ] Bir senior birinin calismasini inceledi mi veya sen birinin kodunu reviewladin mi?

---

## Sektor Notu — 2026 Is Piyasasi

2026 itibariyla is ilanlarinin %80'inden fazlasinda ML becerileri isteniyor. Deep learning ve LLM ozellikleri ise %30'dan fazla ilanda belirmis durumdadir (uc yil once bu oran %10'du). Sirketlerin %40'indan fazlasi AI Agent'lar ile cok adimli is sureclerini otomasyona gecirmeye baslamis durumda ve bu alanda is gunu %15–20 artmis bulunuyor. Ayrica 2026'da rekabet avantaji, modelin karmasikligi yerine veri pipeline'inin hizi ve temizligine kaydi — proje basarisinin %80'i otomatik veri etiketleme ve veri hijyenine baglidir.

Senior DS rolleri icin:

- **MLOps bilgisi** artik "guzel olur" degil, "gerekli" kategorisine gecti
- **Deployment + monitoring** becerileri tek basina modelleme bilgisinden daha fazla ucret getiriyor
- **Causal inference** FAANG ve buyuk teknoloji sirketlerinde ayrim yaratan beceri olmaya devam ediyor
- **LLM entegrasyonu** — API kullanmak degil, RAG pipeline kurabilmek ve fine-tuning kararini verebilmek
- **Small Language Models (SLM)** — Dar alanlarda GPT-4 seviyesinde dogruluk, %50–70 daha dusuk maliyet; is odakli uygulamalarda yukselen trend
- **Data-centric AI** — Model merkezli yaklasimdan veri merkezli yaklasima gecis; veri kalitesi, etiketleme tutarliligi ve data validation artik temel beklenti

Bu yol haritasi bu gercekleri yansitacak sekilde tasarlandi.

---

*Kaynaklar: [AI and Data Scientist Roadmap](https://roadmap.sh/ai-data-scientist), [Data Science Trends 2025–2030](https://emerline.com/blog/top-data-science-trends), [Data Scientist Career Path](https://www.scaler.com/blog/data-scientist-career-path/), [Data Science Career Roadmap](https://www.coursera.org/resources/job-leveling-matrix-for-data-science-career-pathways)*

---

<div class="nav-footer">
  <span><a href="#file_README">← Önceki: Giriş</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_01_yetkinlik_matrisi">Sonraki: Yetkinlik Matrisi →</a></span>
</div>
