# Çalışma Günlüğü

> Her çalışma seansında buraya kısa not düş: ne öğrendin, neyi anlamakta zorlandın, bir sonraki adım nedir. Birikim zamanla motivasyon kaynağına dönüşür.

**Format:**
```
## YYYY-MM-DD — [Konu veya Başlık]
- Bugün ne yaptım
- Neyi anladım / neyi anlamadım
- Bir sonraki adım
```

---

## 2026-03-22 — Yol Haritası Yeniden Yapılandırma

**Ne yapıldı:**

Orijinal monolitik dosya (`veri_bilimci_yol_haritasi.md`, ~3000 satır) okundu ve 14 ayrı kategorik dosyaya bölündü. Her dosya zenginleştirildi ve güncel araştırmayla desteklendi.

Oluşturulan dosyalar:

| Dosya | İçerik |
|-------|--------|
| `README.md` | Ana dizin, öğrenme yolu diyagramı |
| `00-uygulama-sirasi.md` | Aşama 0–7 öğrenme planı, günlük şablonlar |
| `01-yetkinlik-matrisi.md` | Junior/Mid/Senior karşılaştırma tabloları |
| `katman-0-matematik.md` | Lineer cebir, kalkülüs, olasılık teorisi |
| `katman-A-temeller.md` | Python, Pandas, SQL, istatistik, EDA |
| `katman-B-klasik-ml.md` | LightGBM, SHAP, Optuna, kalibrasyon |
| `katman-C-deney-nedensellik.md` | A/B testi, CUPED, DiD, DoWhy |
| `katman-D-derin-ogrenme.md` | PyTorch, Transformers, LoRA, RAG, RecSys |
| `katman-E-mlops.md` | FastAPI, Docker, MLflow, Evidently, CI/CD |
| `katman-F-sistem-tasarimi.md` | Feature store, latency, assignment service |
| `katman-G-senior-davranislar.md` | ICE, OKR, model card, ADR, code review |
| `katman-H-buyuk-veri.md` | Polars, DuckDB, Spark, Delta Lake |
| `projeler.md` | 8 proje şablonu, portföy kriterleri |
| `mulakat.md` | 30 SQL sorusu, ML case studies, behavioral |
| `kaynaklar.md` | Kurslar, kitaplar, topluluklar, newsletter |
| `calisma-gunlugu.md` | Bu dosya |

**Yapı standardı:** Her katman dosyası şu formatı takip eder:
- Sezgisel açıklama
- Matematik/teori detayı (gereken yerlerde)
- Çalıştırılabilir Python/SQL kodu
- Senior notu (production deneyimi)
- Sektör notu (2025–2026 trendleri)

**Bir sonraki adım:**
- [x] Review Pass 1: Her dosyayı oku, yapı uyumu ve kod doğruluğunu kontrol et
- [x] Review Pass 2: Çapraz referansları hizala, tekrarları gider, akışı düzelt
- [ ] Katman-0 ve Katman-A üzerinde pratik alıştırma başlat

---

### 2026-03-22 — Kapsamlı İyileştirme (Faz 1-5)

Tüm 16 dosya üzerinde sistematik iyileştirme yapıldı:

**Faz 1 — Hızlı Kazanım:**
- calisma-gunlugu.md: Örnek seanslar, şablonlar, ilerleme tablosu eklendi
- 00-uygulama-sirasi.md: Aşama 5-7 derinleştirildi, kontrol listeleri ve süreler eklendi
- projeler.md: İskelet kodlar çalıştırılabilir hale getirildi, cold start stratejisi eklendi

**Faz 2 — Yüksek Etki:**
- katman-D: GNN, LLM guardrailing, alignment, model compression, distributed training eklendi
- katman-C: A/A test, SPRT, Synthetic Control, RDD, IV derinleştirildi
- katman-B: Imbalanced learning, LIME, AutoML, fairness metrikleri, counterfactual explanations eklendi
- katman-H: Streaming ML, credentials, Apache Iceberg, Ray, Dask-ML eklendi

**Faz 3 — İnceleme:**
- kaynaklar.md: Podcast, newsletter, Türkçe kaynaklar eklendi
- 01-yetkinlik-matrisi.md: Nicel ölçütler, puan sistemi, Staff/Principal detayı eklendi
- katman-A: Data validation (Pandera, Great Expectations), DuckDB genişletildi
- katman-F: FinOps, model compression, canary deployment, data mesh, 2 yeni senaryo eklendi
- katman-G: STAR vaka çalışmaları, OKR somutlaştırma, etik/KVKK bölümü eklendi

**Faz 4 — Cilalama:**
- katman-0: SVD, convexity, notasyon tutarlılığı, 5 alıştırma eklendi
- katman-E: MLflow tutarsızlığı düzeltildi, Evidently tamamlandı, data lineage, schema validation eklendi
- mulakat.md: 5 ileri SQL sorusu, counter-examples, 2 sistem tasarım senaryosu, behavioral sorular eklendi

**Faz 5 — Ortak İşlemler:**
- Tüm dosyalarda "2025 itibarıyla" → "2026 itibarıyla" güncellendi
- Çapraz referanslar eklendi
- README güncellendi

---

## 2026-03-22 — İkinci Tur Kapsamlı Review + PDF İyileştirme

**Ne yapıldı:**

**Kritik hata düzeltmeleri:**
- `01-yetkinlik-matrisi.md`: 6 kırık link düzeltildi (03-problem-cerceveleme.md vb. → gerçek katman dosyaları)
- `katman-A-temeller.md`: 4 kırık link düzeltildi + `katman-E-data-pipeline.md` → `katman-E-mlops.md`
- `00-uygulama-sirasi.md`: Aşama 5-7 boyunca ASCII karakter bozulması giderildi (ş→ş, ğ→ğ, ü→ü, ö→ö vb.)

**İçerik genişletme:**
- `katman-E-mlops.md`: 963 → 1392 satır; E.9 Prometheus/Grafana monitoring, E.10 MLflow Model Versioning, E.11 LLMOps (Langfuse + RAGAS) eklendi
- `katman-G-senior-davranislar.md`: 902 → ~1350 satır; ICE kodu tamamlandı, automated stakeholder raporu, 3 OKR örneği, STAR hikayeleri, Teknik Borç Quadrantı eklendi
- `katman-C-deney-nedensellik.md`: Çoklu test düzeltmeleri (Bonferroni/BH/Holm, `multipletests()`) + IV zayıf araç senior notu
- `katman-D-derin-ogrenme.md`: Fine-tuning vs RAG karar rehberi + RAGAS evaluation kodu
- `katman-A-temeller.md`: Pandas vs Polars — 2026 Seçim Rehberi bölümü eklendi
- `kaynaklar.md`: Tüm URL'ler eklendi, blog ve newsletter bölümleri güncellendi

**PDF / build iyileştirmeleri:**
- `requirements.txt`: Oluşturuldu
- `build_pdf.py`: CSS `pre { white-space: pre-wrap; word-break: break-word; }` düzeltmesi
- `build_pdf.py`: Kapak başlık font-size 32pt → 27pt (3-satır sorunu giderildi)
- `build_pdf.py`: `.nav-footer` margin-top 3em → 1.5em + `page-break-before: avoid` (boş sayfa giderildi)
- `build_pdf.py`: `td code { font-size: 7pt }` (tablo hücresinde kod taşması giderildi)
- PDF metadata eklendi (author, description, keywords)
- PDF yeniden üretildi: 2.1 MB

---

## Çalışma Seansı Şablonu

```markdown
## YYYY-MM-DD — [Konu]

**Çalıştığım katman:** [örn. Katman B — Klasik ML]
**Süre:** [örn. 2 saat]

**Bugün ne yaptım:**
- ...

**Anladığım kavramlar:**
- ...

**Hâlâ anlamadığım / kafama takılan:**
- ...

**Yazdığım/çalıştırdığım kod:**
- [dosya adı veya kısa açıklama]

**Çapraz referanslar:**
- [Bu seansta hangi dosyaları takip ettim, örn. `katman-B-klasik-ml.md`]

**Bir sonraki seans hedefi:**
- ...
```

---

## Örnek Çalışma Seansları

Aşağıdaki üç örnek, farklı katmanlardan gerçekçi seans girdileri gösterir. Kendi girdilerini bu formatta tutabilirsin.

### Katman-0 Örneği: Gradient Descent ve Learning Rate

```markdown
## 2026-03-25 — Gradient Descent Kodlaması: lr Seçimi

**Çalıştığım katman:** Katman-0 — Matematik Temelleri
**Süre:** 2.5 saat

**Bugün ne yaptım:**
- Gradient descent'i sıfırdan NumPy ile kodladım (linear regression üzerinde).
- lr=0.1 ile loss sürekli patladı (diverge). lr=0.0001 ile çok yavaş converge etti.
- Learning rate schedule (step decay) denedim — 50 epoch sonra lr'yi yarıya düşürdüm.
- Son olarak lr=0.01 + momentum=0.9 kombinasyonuyla 200 epoch'ta convergence sağladım.

**Anladığım kavramlar:**
- Learning rate çok büyükse loss landscape'te zıplama olur, çok küçükse local minima'da takılır.
- Momentum, gradyan yönündeki tutarlılığı kullanarak saddle point'lerden kurtulmaya yardımcı olur.
- Loss curve'ü plot etmek debugging için vazgeçilmez — sayısal değerlere bakmak yetmiyor.

**Hâlâ anlamadığım / kafama takılan:**
- Adam optimizer neden pratikte SGD+momentum'dan daha stabil? Adaptive lr tam olarak neyi adapt ediyor?
- Learning rate warmup ne zaman gerekli? Transformer'larda neden warmup kullanılıyor?

**Yazdığım/çalıştırdığım kod:**
- `notebooks/gradient_descent_lr_experiment.py` — 4 farklı lr ile loss curve karşılaştırması
- Matplotlib ile yan yana plot: diverge vs converge görselleştirmesi

**Çapraz referanslar:**
- `katman-0-matematik.md` — Kalkülüs bölümü, kısmi türev ve gradyan vektörü tanımı
- `katman-D-derin-ogrenme.md` — Optimizer karşılaştırma tablosu (SGD, Adam, AdamW)

**Bir sonraki seans hedefi:**
- Adam optimizer'ı sıfırdan kodla, SGD+momentum ile aynı problem üzerinde karşılaştır.
```

---

### Katman-B Örneği: LightGBM + Optuna Tuning + Overfitting

```markdown
## 2026-04-02 — İlk LightGBM Modeli ve Optuna Tuning

**Çalıştığım katman:** Katman-B — Klasik ML
**Süre:** 3 saat

**Bugün ne yaptım:**
- Kaggle Tabular Playground veri setinde ilk LightGBM modelimi kurdum.
- Baseline: default parametrelerle train AUC=0.95, validation AUC=0.78 — ciddi overfitting.
- Optuna ile 50 trial hyperparameter search yaptım (num_leaves, min_child_samples, reg_alpha, reg_lambda).
- En iyi trial: num_leaves=31, min_child_samples=50, reg_alpha=1.0 — val AUC=0.84'e çıktı.
- Overfitting'i azaltmak için ek olarak feature_fraction=0.7 ve bagging_fraction=0.8 ekledim.

**Anladığım kavramlar:**
- num_leaves düşürmek en etkili regularization — tree complexity'yi doğrudan kontrol ediyor.
- min_child_samples artırmak leaf'lerin daha genelleyici olmasını sağlıyor.
- Optuna'nın pruning (MedianPruner) özelliği, kötü trial'ları erken keserek zaman kazandırıyor.
- Train-val gap > 0.10 ise overfitting alarmı — bu eşiği aklımda tutacağım.

**Hâlâ anlamadığım / kafama takılan:**
- Early stopping round sayısı nasıl seçilmeli? 50 mi 100 mü? Veri boyutuna göre mi değişir?
- SHAP değerlerini Optuna objective içinde kullanmak mantıklı mı? (feature importance-aware tuning)

**Yazdığım/çalıştırdığım kod:**
- `notebooks/lgbm_optuna_tuning.py` — Optuna study + best params logging
- SHAP summary plot ile top-10 feature görselleştirmesi

**Çapraz referanslar:**
- `katman-B-klasik-ml.md` — LightGBM bölümü, hyperparameter açıklamaları ve Optuna entegrasyonu
- `katman-A-temeller.md` — Pandas ile feature engineering pipeline
- `projeler.md` — Proje 2: Churn Prediction şablonu (bu modeli oraya entegre edeceğim)

**Bir sonraki seans hedefi:**
- Stratified K-Fold cross-validation ekle, tek split'e güvenmek riskli.
- SHAP dependence plot ile top-3 feature'ın non-linear etkisini incele.
```

---

### Katman-E Örneği: FastAPI + Docker + Healthcheck

```markdown
## 2026-04-10 — FastAPI ile Model Servisi ve Docker Containerization

**Çalıştığım katman:** Katman-E — MLOps
**Süre:** 4 saat

**Bugün ne yaptım:**
- Eğittiğim LightGBM modelini FastAPI ile serve ettim (/predict endpoint).
- Pydantic ile input validation: feature sayısı veya tipi yanlışsa 422 döndürüyor.
- Dockerfile yazdım: python:3.11-slim base, multi-stage build ile image boyutu 180MB'a düştü.
- Docker healthcheck eklendi: /health endpoint'i model loaded + memory usage döndürüyor.
- docker-compose.yml ile port mapping (8000:8000) ve environment variable yönetimi.
- Locust ile basit load test: 100 concurrent user, p95 latency = 45ms — kabul edilebilir.

**Anladığım kavramlar:**
- FastAPI'nin async yapısı I/O-bound işler için ideal ama model inference CPU-bound — uvicorn worker sayısını artırmak gerekiyor.
- Multi-stage Docker build, final image'den build dependency'lerini çıkararak boyutu %60 azaltıyor.
- Healthcheck endpoint'i sadece 200 OK döndürmemeli, model state'i de kontrol etmeli (model is not None).
- Pydantic V2'nin model_validator dekoratörü ile cross-field validation yapılabiliyor.

**Hâlâ anlamadığım / kafama takılan:**
- Model versiyonlama nasıl yapılmalı? MLflow model registry mi, custom S3 path convention mı?
- Kubernetes'e geçişte HPA (Horizontal Pod Autoscaler) nasıl konfigüre edilir?
- A/B test için traffic splitting: Istio vs application-level routing?

**Yazdığım/çalıştırdığım kod:**
- `api/main.py` — FastAPI app, /predict ve /health endpoint'leri
- `Dockerfile` + `docker-compose.yml` — containerization setup
- `tests/test_api.py` — pytest ile endpoint testleri (happy path + edge cases)

**Çapraz referanslar:**
- `katman-E-mlops.md` — FastAPI bölümü, Docker best practices, CI/CD pipeline örnekleri
- `katman-B-klasik-ml.md` — Modelin eğitim pipeline'ı (bu modeli serve ediyoruz)
- `katman-F-sistem-tasarimi.md` — Latency budget ve serving architecture pattern'ları
- `projeler.md` — Proje 5: End-to-End ML Pipeline şablonu

**Bir sonraki seans hedefi:**
- GitHub Actions ile CI pipeline: her push'ta pytest + Docker build + image push.
- MLflow ile model artifact logging ve versiyonlama entegrasyonu.
```

---

## Haftalık Özet Şablonu

Her hafta sonu (Pazar) aşağıdaki şablonu doldur. Haftalık ritmi görmek, çalışma disiplinini korumak için kritik.

```markdown
## Haftalık Özet: YYYY-MM-DD → YYYY-MM-DD (Hafta #N)

**Çalışılan katmanlar:**
| Katman | Konu | Süre |
|--------|------|------|
| Katman-0 | Gradient descent, loss functions | 2.5 sa |
| Katman-B | LightGBM tuning, SHAP | 3 sa |
| Katman-E | FastAPI, Docker | 4 sa |

**Toplam çalışma süresi:** 9.5 saat
**Günlük ortalama:** 1.4 saat (hedef: 2 saat)

**Katman ilerleme durumu:**
- Katman-0: %30 → %40 (+10%)
- Katman-B: %10 → %25 (+15%)
- Katman-E: %0 → %15 (+15%)

**Bu hafta en iyi öğrendiğim şey:**
- [1-2 cümle]

**Bu hafta en çok zorlandığım şey:**
- [1-2 cümle]

**Sonraki hafta önceliği:**
- [ ] [Hedef 1]
- [ ] [Hedef 2]
- [ ] [Hedef 3]
```

---

## Aylık Retrospektif Şablonu

Her ayın son günü aşağıdaki retrospektifi doldur. Büyük resmi görmek ve rotayı düzeltmek için kullan.

```markdown
## Aylık Retrospektif: [Ay Yılı] (örn. Nisan 2026)

### Hedef vs Gerçekleşen

| Hedef | Durum | Not |
|-------|-------|-----|
| Katman-0'ı %60'a taşı | %45 — Geride | Olasılık teorisi bölümü beklenenden uzun sürdü |
| İlk LightGBM projesini bitir | %100 — Tamamlandı | Optuna + SHAP + CV pipeline tamam |
| FastAPI serving'i öğren | %80 — Devam ediyor | Docker tamam, CI/CD kaldı |
| 2 mülakat sorusu çöz | %50 — Kısmen | SQL sorularını çözdüm, ML case study kaldı |

### Ay boyunca toplam çalışma
- **Toplam saat:** 38 saat
- **Çalışılan gün sayısı:** 22 / 30
- **En verimli gün:** Cumartesi (ort. 3 saat)
- **En az verimli gün:** Çarşamba (ort. 0.5 saat)

### Neler iyi gitti?
- ...

### Neler kötü gitti? Neden?
- ...

### Sonraki ay öncelikleri (max 3)
1. [ ] [Öncelik 1 — hangi katman, hangi konu]
2. [ ] [Öncelik 2]
3. [ ] [Öncelik 3]

### Çapraz referans kontrolü
- [ ] İlgili katman dosyalarındaki checkbox'ları güncelledim
- [ ] `01-yetkinlik-matrisi.md` üzerindeki seviyemi gözden geçirdim
- [ ] `projeler.md` dosyasındaki proje durumlarını güncelledim
```

---

## İlerleme Takip Tablosu

Bu tabloyu her hafta güncelle. Genel ilerlemeyi tek bakışta görmek için kullan.

| Katman / Proje | Kapsam | Tamamlanma (%) | Son Güncelleme | Referans Dosya |
|----------------|--------|:--------------:|:--------------:|----------------|
| Katman-0 — Matematik | Lineer cebir, kalkülüs, olasılık | %0 | — | `katman-0-matematik.md` |
| Katman-A — Temeller | Python, Pandas, SQL, istatistik, EDA | %0 | — | `katman-A-temeller.md` |
| Katman-B — Klasik ML | LightGBM, SHAP, Optuna, kalibrasyon | %0 | — | `katman-B-klasik-ml.md` |
| Katman-C — Deney/Nedensellik | A/B testi, CUPED, DiD, DoWhy | %0 | — | `katman-C-deney-nedensellik.md` |
| Katman-D — Derin Öğrenme | PyTorch, Transformers, LoRA, RAG | %0 | — | `katman-D-derin-ogrenme.md` |
| Katman-E — MLOps | FastAPI, Docker, MLflow, CI/CD | %0 | — | `katman-E-mlops.md` |
| Katman-F — Sistem Tasarımı | Feature store, latency, serving | %0 | — | `katman-F-sistem-tasarimi.md` |
| Katman-G — Senior Davranışlar | ICE, OKR, model card, ADR | %0 | — | `katman-G-senior-davranislar.md` |
| Katman-H — Büyük Veri | Polars, DuckDB, Spark, Delta Lake | %0 | — | `katman-H-buyuk-veri.md` |
| Proje 1 — EDA Portfolio | End-to-end EDA + storytelling | %0 | — | `projeler.md` |
| Proje 2 — Churn Prediction | LightGBM + pipeline + deployment | %0 | — | `projeler.md` |
| Proje 3 — A/B Test Analizi | İstatistiksel test + raporlama | %0 | — | `projeler.md` |
| Proje 5 — E2E ML Pipeline | Train → serve → monitor | %0 | — | `projeler.md` |
| Mülakat Hazırlığı | SQL, ML case, behavioral | %0 | — | `mulakat.md` |

> **Not:** Yüzdeleri subjektif olarak belirle. Bir katmanın tamamlanması demek, o katmandaki tüm konuları anlayıp en az bir pratiğini yapmış olmak demektir. Mükemmellik değil, yeterlilik hedefle.

---

## Motivasyon Notu

> "Öğrenme, her gün küçük birikimlerle olur. Büyük sıçramalar bekleme; küçük adımları sürdür."

Bu günlük senin yolculuğunun belgesi. Her girdi, gelecekteki sana bir mesajdır. Ama sadece yazmak yetmez — aksiyon al:

- **Bir katmanı bitirdiğinde** yukarıdaki ilerleme tablosunda yüzdeyi güncelle ve `01-yetkinlik-matrisi.md` dosyasındaki seviyeni gözden geçir.
- **Bir projeyi tamamladığında** `projeler.md` dosyasına sonuçları yaz, GitHub repo linkini ekle.
- **Bir mülakata girdiğinde** ne sorulduğunu, nerede takıldığını, neyi iyi yaptığını buraya yaz — bir sonraki mülakat için en iyi hazırlık bu notlar olacak.
- **Motivasyonun düştüğünde** bu günlüğün ilk girdisine dön ve ne kadar yol aldığını gör.
- **Her hafta sonu** haftalık özeti doldur. **Her ay sonu** retrospektifi yap. Tutarlılık, yoğunluktan daha değerli.

---

*Bu günlük 2026-03-22 tarihinde oluşturulmuştur.*
