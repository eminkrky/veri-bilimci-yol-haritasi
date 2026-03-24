# Veri Bilimci Yol Haritası — 0 → Senior

<div class="learning-objectives">
<strong>Bu Bölümde Ne Öğreneceksiniz</strong>
<ul>
<li>Rehberin yapısını ve öğrenme yollarını tanıyacaksınız</li>
<li>Hangi profilden olursanız olun, nereden başlamanız gerektiğini öğreneceksiniz</li>
<li>Kitabın dosya yapısını ve navigasyonunu anlayacaksınız</li>
</ul>
</div>

<div class="industry-quote">
"Veriyi anlama, işleme, değer çıkarma, görselleştirme ve iletişim kurma becerisi — önümüzdeki on yılların en kritik yetkinliği olacak."
<span class="quote-author">— DJ Patil, ABD İlk Baş Veri Bilimcisi</span>
</div>

> **Türkçe yazılmış, kapsamlı, ücretsiz veri bilimi öğrenme rehberi.**
> Sıfırdan başlayan → senior seviyeye ulaşmak isteyen herkes için.

---

## Genel Bakış

Bu rehber, veri biliminde **işe girilebilir seviyeden senior'a** kadar giden, Türkçe anlatımlı, tam kapsamlı bir öğrenme yolculuğudur. Tek bir dosya yerine **katmanlı, modüler** yapıda tasarlanmıştır: her dosya bağımsız okunabilir, birlikte tam bir müfredat oluşturur.

| | |
|---|---|
| **Toplam içerik** | ~15.000 satır, 16 dosya |
| **Öğrenme katmanı** | 9 katman (Katman 0 → H) |
| **Portföy projesi** | 8 proje (Proje-0 → Proje-7) |
| **SQL sorusu** | 35+ (temel → ileri) |
| **Tahmini süre** | 6–9 ay (haftada 15–20 saat) · 2–3 ay (tam zamanlı) |
| **Son güncelleme** | Mart 2026 |

---

## Hızlı Başlangıç

Profiline göre nereye atlayacağını bul:

| Profil | Başlangıç Noktası | Tahmini Süre |
|--------|-------------------|:------------:|
| Sıfırdan başlıyorum | `katman-0` → `katman-A` → `katman-B` → `katman-C` | 4–6 ay |
| Python/SQL biliyorum, ML yeni | `katman-B` → `katman-C` → `katman-D` → `katman-E` | 3–4 ay |
| ML biliyorum, senior hedefliyorum | `katman-D` → `katman-E` → `katman-F` → `katman-G` | 2–3 ay |
| Mülakat yaklaşıyor | `mulakat.md` + `00-uygulama-sirasi.md` | 4–6 hafta |
| Hızlı portföy lazım | `projeler.md` → Proje-0 + Proje-1 | 3–6 hafta |

---

## Öğrenme Katmanları

```
Katman 0 — Matematik
    Lineer cebir · Kalkülüs · Olasılık teorisi
    ↓
Katman A — Temeller
    Python/Pandas · NumPy · SQL · İstatistik · EDA · Görselleştirme
    ↓
Katman B — Klasik Makine Öğrenmesi
    Doğrusal modeller · LightGBM · SHAP · Optuna · Fairness
    ↓
Katman C — Deney Tasarımı ve Nedensellik
    A/B test · Power analizi · CUPED · DiD · DoWhy · RDD
    ↓
Katman D — Derin Öğrenme
    PyTorch · NLP/Transformer · CV · RecSys · LLM/RAG · LoRA
    ↓
Katman E — MLOps
    FastAPI · Docker · MLflow · Evidently · CI/CD · Drift tespiti
    ↓
Katman F — ML Sistem Tasarımı
    Feature store · Latency bütçesi · Canary deploy · FinOps
    ↓
Katman G — Senior Davranışlar
    Etki odaklı çalışma · OKR · Mentorluk · Etik · Dokümantasyon
    ↓
Katman H — Büyük Veri  (gerektiğinde)
    Spark · Polars · DuckDB · Delta Lake · Streaming · Ray
```

---

## Dosya Rehberi

### Temel Belgeler
| Dosya | Açıklama | Süre |
|-------|----------|------|
| [`00-uygulama-sirasi.md`](./00-uygulama-sirasi.md) | 8 aşamalı sıralı öğrenme planı, kontrol listeleri, teslimatlar | 15 dk okuma |
| [`01-yetkinlik-matrisi.md`](./01-yetkinlik-matrisi.md) | Junior → Senior yetkinlik matrisi, nicel ölçütler | 20 dk okuma |

### Öğrenme Katmanları
| Dosya | İçerik | Süre |
|-------|--------|------|
| [`katman-0-matematik.md`](./katman-0-matematik.md) | Lineer cebir, kalkülüs, olasılık, SVD | 3–5 gün |
| [`katman-A-temeller.md`](./katman-A-temeller.md) | Python, Pandas, SQL (CTE, window), istatistik, EDA, Pandera | 3–4 hafta |
| [`katman-B-klasik-ml.md`](./katman-B-klasik-ml.md) | LightGBM, SHAP, Optuna, calibration, fairness, AutoML | 3–4 hafta |
| [`katman-C-deney-nedensellik.md`](./katman-C-deney-nedensellik.md) | A/B test, CUPED, Synthetic Control, RDD, IV, DiD | 2–3 hafta |
| [`katman-D-derin-ogrenme.md`](./katman-D-derin-ogrenme.md) | PyTorch, NLP, CV, RecSys, LLM/RAG, GNN, model compression | 4–6 hafta |
| [`katman-E-mlops.md`](./katman-E-mlops.md) | FastAPI, Docker, MLflow, Evidently, GitHub Actions, feature store | 2–3 hafta |
| [`katman-F-sistem-tasarimi.md`](./katman-F-sistem-tasarimi.md) | Sistem mimarisi, feature store, latency, canary deploy, data mesh | 2–3 hafta |
| [`katman-G-senior-davranislar.md`](./katman-G-senior-davranislar.md) | ICE, OKR, STAR vaka çalışmaları, etik, liderlik, dokümantasyon | Sürekli |
| [`katman-H-buyuk-veri.md`](./katman-H-buyuk-veri.md) | Spark, Polars, DuckDB, Delta Lake, streaming ML, Ray | 2–3 hafta |

### Uygulama ve Referans
| Dosya | Açıklama | Süre |
|-------|----------|------|
| [`projeler.md`](./projeler.md) | 8 portfolyo projesi (Proje-0 → Proje-7), standartlar, soğuk başlangıç stratejisi | Proje başına 1–4 hafta |
| [`mulakat.md`](./mulakat.md) | 35+ SQL sorusu, ML case study, sistem tasarım senaryoları, behavioral | 2–4 hafta |
| [`kaynaklar.md`](./kaynaklar.md) | Ücretsiz kurslar, kitaplar, podcast, newsletter, Türkçe kaynaklar | Referans |
| [`calisma-gunlugu.md`](./calisma-gunlugu.md) | İlerleme günlüğü, şablonlar | Sürekli |

---

## Nasıl Kullanılır?

1. **Bu README'yi oku** — genel resmi gör, profilini belirle.
2. **[`00-uygulama-sirasi.md`](./00-uygulama-sirasi.md)** ile başla — öğrenme sırasını netleştir.
3. **[`01-yetkinlik-matrisi.md`](./01-yetkinlik-matrisi.md)** ile hedefini ölç — "Senior ne demek?"
4. **Katmanları sırayla** çalış: `katman-0` → `A` → `B` → … → `H`
5. **Her katmanı bitirince** ilgili projeyi [`projeler.md`](./projeler.md)'den yap.
6. **Mülakat hazırlığında** [`mulakat.md`](./mulakat.md)'e gir, STAR hikayelerini hazırla.
7. **[`kaynaklar.md`](./kaynaklar.md)** her zaman referans olarak açık tutulabilir.

> **Altın kural:** Her katman için "Teslim edilebilir bir çıktı" üret — proje, notebook veya özet. Çıktı olmayan öğrenme yapışmaz.

---

## İçerik Yazım Anlayışı

Her konu bu sırayla anlatılır:

```
1. Sezgisel açıklama   → Analoji veya günlük hayat örneği
2. Matematik detayı    → Formül, türev, loss fonksiyonu (isteğe bağlı)
3. Kod örneği          → Çalıştırılabilir Python / SQL snippet
4. Senior Notu         → Üretimde dikkat, yaygın hatalar, best practice
5. Sektör Notu         → 2025–2026 trendleri, araç önerileri
```

**Araç tercihleri:** Cloud-agnostic, açık kaynak öncelikli (MLflow, Docker, FastAPI, Evidently). AWS/GCP yan bilgi olarak yer alır.

---

## PDF

Tüm içerik tek bir kitap formatında PDF olarak da mevcuttur: [`veri-bilimci-yol-haritasi.pdf`](./veri-bilimci-yol-haritasi.pdf)

PDF'i yeniden oluşturmak için:

```bash
pip install markdown weasyprint
python3 build_pdf.py
```

---

## Katkı ve Kaynaklar

Bu rehber şu kaynaklara dayanmaktadır:

- [roadmap.sh/ai-data-scientist](https://roadmap.sh/ai-data-scientist)
- Andrew Ng — ML & MLOps Specialization (DeepLearning.AI)
- fast.ai — Practical Deep Learning
- Chip Huyen — *Designing Machine Learning Systems*
- Scott Cunningham — *Causal Inference: The Mixtape*
- DataLemur, InterviewQuery — mülakat soruları

---

*Son güncelleme: Mart 2026*

---

<div class="whats-next">
<strong>Sırada Ne Var?</strong>
<p>Bir sonraki bölümde sizi adım adım 0'dan Senior'a taşıyacak 8 aşamalı öğrenme planını bulacaksınız.</p>
</div>

<div class="nav-footer">
  <span></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_00_uygulama_sirasi">Sonraki: Uygulama Sırası →</a></span>
</div>
