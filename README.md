# Veri Bilimci Yol Haritası — Emin

> **Amaç:** Veri bilimi alanında işe girilebilir seviyeden başlayıp senior seviyeye kadar giden, Türkçe anlatımlı, tam kapsamlı bir öğrenme ve proje yolculuğu.

**📊 Proje İstatistikleri:** 16 dosya | ~14.700 satır | 8 proje | 35 SQL sorusu | 9 katman

**⏱️ Tahmini Süre:** Bu yol haritasını başından sonuna tamamlamak: **~6-9 ay** (haftada 15-20 saat) veya **~2-3 ay** (tam zamanlı)

Bu rehber tek bir büyük dosya yerine **katmanlı, bölünmüş** bir yapıya ayrılmıştır. Her dosya bağımsız okunabilir, birlikte de bir bütün oluşturur.

---

## Quick Start — Profiline Göre Hızlı Başlangıç

| Profil | Önerilen Yol | Tahmini Süre |
|--------|-------------|:------------:|
| **Sıfırdan başlayan** | Aşama 0 → 1 → 2 → 3 (`katman-0` → `katman-A` → `katman-B` → `katman-C`) | 4-6 ay |
| **Python/SQL bilen** | Aşama 2 → 3 → 4 → 5 (`katman-B` → `katman-C` → `katman-D` → `katman-E`) | 3-4 ay |
| **ML bilen, senior hedefleyen** | Aşama 4 → 5 → 6 → 7 (`katman-D` → `katman-E` → `katman-F` → `katman-G`) | 2-3 ay |

> Her profil için ilgili katmanları sırasıyla çalış, arada [`projeler.md`](./projeler.md) dosyasından eşleşen projeyi yap.

---

## Nasıl Kullanılır?

1. Önce bu README'yi oku — genel resmi gör.
2. [`00-uygulama-sirasi.md`](./00-uygulama-sirasi.md) dosyasıyla başla — hangi sırayla ne öğreneceğini belirle.
3. [`01-yetkinlik-matrisi.md`](./01-yetkinlik-matrisi.md) ile hedefini netleştir — "Senior ne demek?"
4. Katman dosyalarını sırayla çalış: `katman-0` → `katman-A` → ... → `katman-H`
5. Her katmanı bitirince ilgili projeyi [`projeler.md`](./projeler.md) dosyasından yap.
6. Mülakat zamanı yaklaştığında [`mulakat.md`](./mulakat.md) dosyasına gir.
7. [`kaynaklar.md`](./kaynaklar.md) her zaman açık tutulabilir, derinleşmek için.

---

## Dosya Listesi

| Dosya | İçerik | Tahmini Süre |
|-------|--------|-------------|
| [`00-uygulama-sirasi.md`](./00-uygulama-sirasi.md) | Aşama 0–7 öğrenme planı, kontrol listeleri, süreler, teslimatlar | 15 dk okuma |
| [`01-yetkinlik-matrisi.md`](./01-yetkinlik-matrisi.md) | Junior→Staff/Principal yetkinlik matrisi, nicel ölçütler, puan sistemi | 20 dk okuma |
| [`katman-0-matematik.md`](./katman-0-matematik.md) | Lineer cebir, kalkülüs, olasılık, SVD, convexity, 5 alıştırma | 3–5 gün çalışma |
| [`katman-A-temeller.md`](./katman-A-temeller.md) | Python, Pandas, SQL, istatistik, EDA, Pandera, Great Expectations, DuckDB | 3–4 hafta |
| [`katman-B-klasik-ml.md`](./katman-B-klasik-ml.md) | LightGBM, SHAP, Optuna, imbalanced learning, LIME, AutoML, fairness metrikleri | 2–4 hafta |
| [`katman-C-deney-nedensellik.md`](./katman-C-deney-nedensellik.md) | A/B test, A/A test, SPRT, CUPED, Synthetic Control, RDD, IV, DiD, DoWhy | 1–3 hafta |
| [`katman-D-derin-ogrenme.md`](./katman-D-derin-ogrenme.md) | DL, NLP, CV, RecSys, LLM/RAG, GNN, alignment, model compression, distributed training | 4–8 hafta |
| [`katman-E-mlops.md`](./katman-E-mlops.md) | FastAPI, Docker, MLflow, Evidently, CI/CD, data lineage, schema validation | 2–4 hafta |
| [`katman-F-sistem-tasarimi.md`](./katman-F-sistem-tasarimi.md) | Sistem tasarımı, feature store, FinOps, model compression, canary deployment, data mesh | 1–2 hafta |
| [`katman-G-senior-davranislar.md`](./katman-G-senior-davranislar.md) | ICE, OKR, STAR vaka çalışmaları, etik/KVKK, liderlik, dokümantasyon | Sürekli |
| [`katman-H-buyuk-veri.md`](./katman-H-buyuk-veri.md) | Spark, Polars, DuckDB, Delta Lake, streaming ML, Apache Iceberg, Ray, Dask-ML | 1–2 hafta |
| [`projeler.md`](./projeler.md) | Proje-0'dan Proje-7'ye çalıştırılabilir portföy seti, cold start stratejisi | Proje başına 1–4 hafta |
| [`mulakat.md`](./mulakat.md) | 35 SQL sorusu, ileri SQL, ML case study, sistem tasarım senaryoları, behavioral sorular | 2–4 hafta |
| [`kaynaklar.md`](./kaynaklar.md) | Ücretsiz kurslar, kitaplar, podcast, newsletter, Türkçe kaynaklar, topluluklar | Referans |
| [`calisma-gunlugu.md`](./calisma-gunlugu.md) | İlerleme günlüğü, örnek seanslar, şablonlar, ilerleme tablosu | Sürekli güncellenir |

---

## Öğrenme Katmanları (Görsel Özet)

```
Katman 0 — Matematik (Lineer cebir, kalkülüs, olasılık)
     ↓
Katman A — Temeller (Python, İstatistik, SQL, Görselleştirme)
     ↓
Katman B — Klasik ML (Modeller, değerlendirme, feature engineering)
     ↓
Katman C — Deney/Nedensellik (A/B, CUPED, causal)
     ↓
Katman D — Derin Öğrenme + Uzmanlık (DL, NLP, CV, RecSys, LLM/RAG)
     ↓
Katman E — MLOps (Servis, izleme, drift, CI/CD)
     ↓
Katman F — Sistem Tasarımı (Feature store, latency, mimari)
     ↓
Katman G — Senior Davranışlar (Etki, iletişim, liderlik)
     ↓
Katman H — Büyük Veri (Spark, Dask, cloud — gerektiğinde)
```

---

## Kaynak Tabanı

Bu rehber aşağıdaki kaynaklara dayanmaktadır:

- roadmap.sh/ai-data-scientist
- Andrew Ng ML & MLOps Specialization (Coursera/DeepLearning.AI)
- fast.ai (pratik DL)
- Kaggle Grandmaster tavsiyeleri
- Stanford İstatistik müfredatı
- FAANG mülakat rehberleri (DataLemur, InterviewQuery)
- Chip Huyen — Designing Machine Learning Systems
- Causal Inference: The Mixtape (Scott Cunningham)

---

## Stil Kılavuzu

Her katman dosyasında şu yapı tekrarlanır:

```
## [Konu]

### Sezgisel Açıklama
Analoji veya basit örnek ile kavramı anla.

### Matematik Detayı
Formül, türev veya loss fonksiyonu (gerektiğinde).

### Kod Örneği
Çalıştırılabilir Python kodu.

> **Senior Notu:** Üretimde dikkat edilecek nokta.

### Sektör Notu
Araştırma sonuçları, hangi araçlar öne çıkıyor, ne tercih ediliyor.
```

---

## Başlangıç Noktaları

### "Nereden başlayacağım?" sorusuna göre:

| Durum | Başlangıç Noktası |
|-------|------------------|
| Sıfırdan başlıyorum | `katman-0-matematik.md` → `katman-A-temeller.md` |
| Python biliyorum, ML yeni | `katman-B-klasik-ml.md` |
| ML biliyorum, prod tecrübem yok | `katman-E-mlops.md` + `katman-F-sistem-tasarimi.md` |
| Mülakat yaklaşıyor | `mulakat.md` + `00-uygulama-sirasi.md` |
| Hızlı portföy lazım | `projeler.md` → Proje-0 + Proje-1 |

---

*Son güncelleme: 2026-03-22*

---

<div class="nav-footer">
  <span></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_00_uygulama_sirasi">Sonraki: Uygulama Sırası →</a></span>
</div>
