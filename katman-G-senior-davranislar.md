# Katman G — Senior Davranışlar: Etki, Liderlik, İletişim

> Bu katmanda ne öğrenilir: Senior DS'i ayırt eden davranışlar. Etki odaklı proje seçimi, stakeholder iletişimi, dokümantasyon kültürü, mentorluk ve teknik borç yönetimi.
>
> Süre: Sürekli pratik. Bu katman bir seferlik öğrenilmez, alışkanlık haline getirilir.


<div class="prereq-box">
<strong>Önkoşul:</strong> Tüm teknik katmanlar (<strong>A–F</strong>) tamamlanmış ya da aktif olarak tamamlanıyor olmalı. Bu katman teknik bilgiyi iş etkisine dönüştürme rehberidir.
</div>

---

## G.1 Etki (Impact) Odaklı Çalışma

### Sezgisel Açıklama

Junior: "Bu modelimin AUC'si 0.87."
Senior: "Bu model sayesinde aylık 2M TL churn kaybını önleyeceğiz."

Fark: Junior teknik çıktı anlatır. Senior iş etkisi anlatır.

### ICE Önceliklendirme Çerçevesi

```
ICE = Impact × Confidence × Ease

Impact: Tahmini iş etkisi (1–10)
  1: Küçük operasyonel iyileştirme
  5: Müşteri deneyimini etkileyen
  10: Gelir veya maliyet üzerinde büyük etki

Confidence: Etki gerçekleşme güveni (1–10)
  1: Çok spekülatif
  5: Bazı kanıtlar var
  10: Net causal evidence

Ease: Yapılabilirlik kolaylığı (1–10)
  1: 6+ ay çalışma, yüksek belirsizlik
  5: 1–2 ay, çoğunlukla tanımlı
  10: 1 hafta, template hazır
```

```python
import pandas as pd

def prioritize_projects(projects: list[dict]) -> pd.DataFrame:
    """
    ICE skoruna göre proje önceliklendir.

    projects = [
        {"name": "Churn model", "impact": 8, "confidence": 7, "ease": 5},
        {"name": "Data pipeline hızlandırma", "impact": 4, "confidence": 9, "ease": 8},
        {"name": "GenAI chatbot", "impact": 9, "confidence": 3, "ease": 2},
    ]
    """
    df = pd.DataFrame(projects)
    df["ice_score"] = df["impact"] * df["confidence"] * df["ease"]
    return df.sort_values("ice_score", ascending=False)

# Kullanım
projects = [
    {"name": "Churn Modeli (LightGBM)", "impact": 8, "confidence": 7, "ease": 5,
     "est_revenue": "2M TL/yıl", "est_weeks": 4},
    {"name": "SQL Pipeline Optimizasyonu", "impact": 4, "confidence": 9, "ease": 8,
     "est_revenue": "100K TL/yıl (maliyet)", "est_weeks": 1},
    {"name": "GenAI Müşteri Chatbot", "impact": 9, "confidence": 3, "ease": 2,
     "est_revenue": "Belirsiz", "est_weeks": 16},
    {"name": "A/B Test Çerçevesi", "impact": 7, "confidence": 8, "ease": 6,
     "est_revenue": "3M TL/yıl (iyileştirilmiş kararlar)", "est_weeks": 6},
]
print(prioritize_projects(projects))
```

### OKR ile DS Bağlantısı

```
Şirket Objective:   "Müşteri kayıplarını azalt"
DS Key Result:      "Q3'te churn oranını %10 azalt"
Model Metric:       "Recall@threshold > 0.75" (leading indicator)

Bağlantı: Model recall artarsa → Daha çok churner yakalanır →
          Kampanya gönderilir → Churn azalır → KR'a katkı
```

#### Somut Çeyreklik OKR Planı (Q3 Örneği)

```markdown
## Objective: Müşteri kaybını veri bilimi ile ölçülebilir şekilde azalt

### Key Result 1: Churn tahmin modelinin production recall@0.35 değerini 0.75'ten 0.82'ye çıkar
- Model metriği: Recall@threshold (haftalık izlenir)
- Geri bağlantı: Recall ↑ → Daha fazla churner yakalanır → Kampanya hedefi genişler

### Key Result 2: Model-destekli kampanya ile aylık kurtarılan müşteri sayısını 500'den 750'ye çıkar
- Model metriği: Precision@threshold ≥ 0.70 (yanlış alarm maliyetini sınırla)
- Geri bağlantı: Precision sabit kalırken recall artışı → Net kurtarılan müşteri ↑

### Key Result 3: Model retraining süresini 5 günden 1 güne düşür (pipeline otomasyonu)
- Model metriği: Pipeline execution time + data validation pass rate ≥ %99
- Geri bağlantı: Hızlı retraining → Drift'e hızlı tepki → Stabil production AUC ≥ 0.83
```

**KR → Model Metriği Geri Bağlantı Zinciri:**

```
KR1 (Recall 0.82)
  └─ Model: feature engineering iyileştirme + hyperparameter tuning
       └─ Haftalık metrik: Recall@threshold, PR-AUC
            └─ Aylık iş etkisi: Yakalanan churner sayısı

KR2 (750 kurtarılan müşteri)
  └─ Model: threshold optimizasyonu + kampanya segmentasyonu
       └─ Haftalık metrik: Precision@threshold, kampanya conversion rate
            └─ Aylık iş etkisi: Kurtarılan müşteri × LTV = gelir etkisi

KR3 (1 gün retraining)
  └─ MLOps: Airflow DAG otomasyonu + data validation
       └─ Haftalık metrik: Pipeline süresi, test pass rate
            └─ Aylık iş etkisi: DS zamanı tasarrufu (4 gün/ay × 2 DS = 8 adam-gün)
```

**Haftalık OKR Check-in Şablonu:**

```markdown
## Haftalık DS OKR Check-in — Hafta [X/13]

**Tarih:** [YYYY-MM-DD]
**Katılımcılar:** DS Lead, Product Owner, Growth Lead

### KR1: Recall 0.75 → 0.82 (şu an: [____])
- Bu hafta ne yapıldı: [örn. 3 yeni feature eklendi, A/B validation tamamlandı]
- Engeller: [örn. feature store'da latency sorunu]
- Güven seviyesi: [Yeşil / Sarı / Kırmızı]

### KR2: 500 → 750 kurtarılan müşteri/ay (şu an: [____])
- Bu hafta ne yapıldı: [örn. kampanya segmenti güncellendi]
- Engeller: [örn. Growth ekibi kampanya bütçesi onayı bekliyor]
- Güven seviyesi: [Yeşil / Sarı / Kırmızı]

### KR3: 5 gün → 1 gün retraining (şu an: [____] gün)
- Bu hafta ne yapıldı: [örn. Airflow DAG'ı yazıldı, test ortamında çalıştı]
- Engeller: [örn. staging environment erişimi yok]
- Güven seviyesi: [Yeşil / Sarı / Kırmızı]

### Aksiyon Maddeleri
- [ ] [Kişi]: [Aksiyon] — Deadline: [tarih]
- [ ] [Kişi]: [Aksiyon] — Deadline: [tarih]
```

> **Senior Notu:** "AUC 0.02 arttı" = hiçbir şey ifade etmez. "AUC artışı sayesinde aylık 500 müşteri daha tutuldu, bu 1.2M TL değeri" = anlamlı.

### Vaka Çalışması 1: Model Drift Tespiti ve Yönetimi (STAR Formatı)

```markdown
## Senaryo: E-Ticaret Churn Modelinde Sessiz Performans Düşüşü

### Situation (Durum)
Bir e-ticaret şirketinde production'daki churn tahmin modeli 6 aydır çalışıyor.
Son 2 aydır kampanya conversion rate'i %18'den %9'a düştü. Growth ekibi
"kampanya mesajları etkisini kaybetti" diye şikayet ediyor. Kimse modele
bakmayı düşünmüyor.

### Task (Görev)
Senior DS olarak kök neden analizi yapıp conversion düşüşünün gerçek kaynağını
bulmak ve çözüm önerisi sunmak.

### Action (Aksiyon)

**Adım 1: Monitoring metrikleri kontrolü**
- Production AUC'yi kontrol ettim: 0.84 → 0.71'e düşmüş (2 ayda kademeli)
- PSI (Population Stability Index) kontrolü: 3 feature'da PSI > 0.25
  - `days_since_last_order`: PSI = 0.31 (dağılım kaymış)
  - `avg_order_value_trend`: PSI = 0.28
  - `n_orders_90d`: PSI = 0.19

**Adım 2: Kök neden araştırması**
- Veri kaynağını inceledim: 2 ay önce ürün kataloğu yenilendi, yeni kategoriler
  eklendi. Sipariş davranışı değişti ama model eski dağılıma göre eğitilmişti.
- Causal analiz: Katalog değişikliği → Sipariş paterni değişimi → Feature
  dağılımı kayması → Model tahmin kalitesi düşüşü → Yanlış müşterilere
  kampanya → Conversion düşüşü
  (→ Bkz. katman-C: causal inference temelleri)

**Adım 3: Stakeholder iletişimi**
- Growth ekibine: "Kampanya mesajları değil, altındaki model bozulmuş.
  Yanlış müşterilere gönderiyoruz."
- VP'ye: "2 ayda tahmini 800K TL kayıp. Düzeltme 1 hafta sürer,
  kurtarma potansiyeli aylık 400K TL."

**Adım 4: Çözüm implementasyonu**
- Modeli yeni veri ile retrain ettim (son 6 ay veri, yeni katalog dahil)
- 3 drift eden feature'ı yeniden mühendislik yaptım
- Otomatik drift alerting sistemi kurdum (PSI > 0.2 → Slack alert)
- Haftalık otomatik AUC raporu oluşturdum

### Result (Sonuç)
- Yeni model AUC: 0.86 (production, retrain sonrası)
- Kampanya conversion rate: %9 → %16'ya geri döndü (4 hafta içinde)
- Yıllık kurtarılan gelir tahmini: ~1.6M TL
- Kalıcı monitoring sistemi: Bir daha sessiz drift yaşanmadı

### Öğrenilen Dersler
- Monitoring olmadan model deploy etmek "kör uçuş" demek
- Performans düşüşünü ilk fark eden genellikle iş ekibi olur — ama yanlış
  yere bakar
- Senior DS'in işi: "Sorun nerede?" sorusunu doğru yere yönlendirmek
```

### Vaka Çalışması 2: Stakeholder'ı ML Yerine Basit Kural ile İkna Etme (STAR Formatı)

```markdown
## Senaryo: Karmaşık Model vs. If/Else Kuralı — ROI Temelli Karar

### Situation (Durum)
Fintech şirketinde fraud detection ekibi, "derin öğrenme ile fraud tespiti"
projesi talep ediyor. CTO heyecanlı, 3 aylık proje planı hazırlanmış.
Mevcut sistem: Manuel kurallar (if/else), precision düşük (%45), çok
false positive var. Aylık fraud kaybı: ~2M TL.

### Task (Görev)
Senior DS olarak projeyi değerlendirmek. "En iyi çözüm nedir?" sorusunu
cevaplamak — "en havalı çözüm" değil.

### Action (Aksiyon)

**Adım 1: Mevcut durumu analiz et**
- Mevcut kuralları inceledim: 12 kural var, 3'ü tüm fraud tespitlerinin
  %78'ini kapsıyor
- False positive analizi: Kuralların %60'ı gereksiz — güncellenmemiş
  threshold'lar yüzünden
- Fraud pattern analizi: %85'i 4 temel pattern'a düşüyor
  (yüksek tutar + yeni hesap + gece saati + farklı şehir)

**Adım 2: ROI karşılaştırması hazırla**

| Kriter | DL Modeli | Optimize Edilmiş Kurallar |
|--------|-----------|--------------------------|
| Geliştirme süresi | 12 hafta | 2 hafta |
| Tahmini precision | %82 | %74 |
| Tahmini recall | %88 | %80 |
| Bakım karmaşıklığı | Yüksek (GPU, MLOps) | Düşük (SQL + if/else) |
| İlk 6 ay net kazanç | -200K TL (yatırım) | +800K TL |
| Explainability | Düşük (black box) | Yüksek (kural bazlı) |
| Regülatör uyumluluk | Ek çalışma gerekir | Hazır |

**Adım 3: İki aşamalı strateji öner**
- Faz 1 (2 hafta): Mevcut kuralları optimize et, threshold'ları güncelle,
  3 yeni kural ekle → Hızlı kazanım
- Faz 2 (Q+1): Veri topla, DL model geliştir, A/B test ile karşılaştır →
  Kanıta dayalı geçiş

**Adım 4: CTO'ya sunum**
- "DL modeli 3 ay sonra belki %82 precision verir. Ama 2 haftada kuralları
  optimize edersek precision %45'ten %74'e çıkar. İlk 6 ayda 800K TL
  kazanırız. Sonra DL'e geçiş kararını veriyle veririz."

### Result (Sonuç)
- Faz 1 deploy edildi: 2 haftada precision %45 → %71
- False positive %60 azaldı → Operasyon ekibinin manuel review yükü yarıya indi
- 6 ayda 1.2M TL fraud kaybı önlendi
- Faz 2 başlatıldı ama A/B test sonucu: DL modeli kurallara göre sadece
  +%5 precision artışı sağladı → Kurallarla devam kararı

### Öğrenilen Dersler
- "En iyi model" her zaman "en karmaşık model" değildir
- ROI analizi teknik kararları iş kararına çevirir
- Fazlı yaklaşım (crawl → walk → run) riski azaltır ve güven inşa eder
- Senior DS: "Hayır, buna ML gerekmiyor" diyebilme cesareti gösterir
```

---

## G.2 Stakeholder İletişimi

### Sezgisel Açıklama

Farklı kitleler farklı dil konuşur:
- **Engineering:** Teknik detay, latency, architecture
- **Product:** Kullanıcı etkisi, özellik davranışı
- **Business:** Gelir, maliyet, risk, ROI
- **C-suite:** Strateji, rekabet avantajı, risk

Senior her biriyle doğru dili konuşur.

### Karar Açıklama Şablonu (Piramit Prensibi)

```markdown
## [Proje/Karar Başlığı]

### Sonuç (Öne Çek)
"Yeni churn modelimizi deploy edersek Q3'te 500 müşteri daha kalalacak (~1.5M TL)."

### Kanıtlar
- Validation AUC: 0.87 (baseline 0.74'ten +17%)
- Backtest: Son 3 çeyrekte ortalama %11 daha fazla churner yakalanıyor
- A/B test simülasyonu: İstatistiksel güç %85

### Riskler ve Kısıtlar
- Model, kampanya ile birlikte çalışıyor — kampanya etkinliği bağımsız risk
- Prod'da feature store hazır değil → geçici batch serving ile başlayacağız
- Personalization team'le alignment gerekiyor (aynı kullanıcıya çift mesaj riski)

### Öneri + Sonraki Adım
Deploy → Sahip: Emin | Deadline: 15 Mart
Kampanya tasarımı → Sahip: Growth | Deadline: 20 Mart
```

### Teknik Borcu İş Diline Çevirmek

```python
# Teknik borç kategorileri
TECH_DEBT_MATRIX = {
    "Entangled features": {
        "teknik": "Feature A çıkarınca pipeline kırılıyor",
        "iş": "Her model güncelleme 3 gün yerine 2 hafta alıyor",
        "maliyet_tahmin": "Yılda 4 × 10 gün = 40 gün DS zamanı = 200K TL"
    },
    "Monitoring yok": {
        "teknik": "Data drift tespit edilemiyor",
        "iş": "Model bozulunca tespit 30–60 gün gecikiyor",
        "maliyet_tahmin": "1 bozuk model = ortalama 500K TL kayıp"
    },
    "Dokümantasyon eksik": {
        "teknik": "Feature hesapları yorumsuz",
        "iş": "Yeni DS onboarding süresi: 3 hafta (hedef: 3 gün)",
        "maliyet_tahmin": "Yılda 2 yeni kişi × 15 gün = 30 gün kayıp"
    },
}
```

---

## G.3 Dokümantasyon Kültürü

### Model Card Şablonu

```markdown
# Model Card: Churn Tahmin Modeli v2.1

**Son Güncelleme:** 2024-03-22
**Sahip:** Emin Atabey
**Durum:** Production

---

## Amaç ve Kapsam

**Amaç:** 30 gün içinde ayrılacak kullanıcıları tahmin et, kampanya hedeflemesi yap.
**Kullanım kapsamı:** Yurt içi kullanıcılar, premium segment.
**Uygun OLMAYAN kullanım:** Fiyatlandırma kararları, kredi skorlama.

---

## Eğitim Verisi

| Özellik | Değer |
|---------|-------|
| Dönem | 2022-01 – 2023-12 |
| N (eğitim) | 485,000 |
| N (validation) | 58,000 |
| Churn oranı | %8.3 |
| Özellik sayısı | 47 |

**Bilinen önyargılar:**
- Yeni kullanıcılar (<30 gün) eksik temsil ediliyor (geçmiş verisi az)
- Ülke dağılımı TR ağırlıklı (%65), model yurt dışında daha düşük performans

---

## Metrikler

| Metrik | Validation | Production (son 30 gün) |
|--------|-----------|------------------------|
| ROC-AUC | 0.87 | 0.84 |
| PR-AUC | 0.64 | 0.61 |
| Threshold | 0.35 | 0.35 |
| Precision@threshold | 0.72 | 0.69 |
| Recall@threshold | 0.78 | 0.75 |

---

## Feature'lar (İlk 10, SHAP değerine göre)

1. `days_since_last_order` — Negatif etki (büyük = churn riski)
2. `n_orders_90d` — Negatif etki (az = risk)
3. `avg_order_value_trend` — Düşüş = risk
4. `support_tickets_30d` — Pozitif etki (fazla şikayet = risk)
...

---

## İzleme

- **PSI kontrol sıklığı:** Günlük
- **Performance kontrol:** Haftalık (ground truth 14 gün gecikmeli)
- **Retraining trigger:** PSI > 0.2 veya AUC < 0.80
- **Retraining sıklığı:** Aylık (trigger yoksa)

---

## Fairness ve Risk

- Yaş ve cinsiyet feature olarak kullanılmıyor
- Gelir tahmini proxy'si: ortalama sipariş değeri (dolaylı bias riski)
- Kampanya bütçesi limitli → Model yüksek recall tercih ediyor, precision ikincil

---

## Sorumlular

| Rol | Kişi | İletişim |
|-----|------|---------|
| Model sahibi | Emin Atabey | e.atabey@... |
| MLOps | — | — |
| İş sahibi | Growth Team | — |
```

### ADR (Architecture Decision Record)

```markdown
# ADR-0003: Model Serving Framework Seçimi

**Tarih:** 2024-03-01
**Durum:** Kabul edildi

## Bağlam

Churn modelini prod'a almak için serving infrastructure gerekiyor.
3 seçenek değerlendirildi.

## Değerlendirilen Seçenekler

| Seçenek | Avantaj | Dezavantaj |
|---------|---------|------------|
| FastAPI + Docker | Kontrol, şeffaflık, düşük maliyet | Ops yükü |
| SageMaker Endpoints | Managed, otomatik ölçekleme | Pahalı, vendor lock-in |
| Batch scoring (Airflow) | Basit, mevcut altyapı | Realtime değil |

## Karar

FastAPI + Docker seçildi.

## Gerekçe

- Latency gereksinimi: <500ms → batch uygun değil
- Maliyet: SageMaker ~3× daha pahalı; FastAPI sabit maliyet
- Kontrol: Monitoring ve logging tam kontrolde
- Bağımlılık: Mevcut container ekibi var

## Sonuçlar

- Docker image bakımı gerekiyor (pozitif: ops kası geliştiriliyor)
- Otomatik ölçekleme manuel kurulumu var (Kubernetes HPA)
- Monitoring Prometheus/Grafana ile — mevcut altyapıyla uyumlu

## Alternatife Geçiş Kriteri

Eğer günlük >10M istek veya ML mühendis/DS oranı >1:3 olursa
SageMaker geçişini yeniden değerlendir.
```

### Deney Dokümanı Şablonu

```markdown
## Deney: Yeni Öneri Algoritması (Deney #47)

**Tarih:** 2024-03-15 → 2024-03-29
**Sahip:** Emin
**Durum:** Tamamlandı — Deploy kararı

### Problem
Mevcut öneri CTR'si Aralık'tan beri %12 düşüş gösteriyor.
Hipotez: Katalog genişledi ama retrieval modeli güncellenmedi.

### Hipotez
Yeni two-tower retrieval modeli, CTR'yi en az %5 artırır.

### Tasarım
- Randomization unit: user_id
- Trafik: %50/%50
- Süre: 14 gün (hafta içi + sonu dengesi)
- Primary metric: CTR (click-through rate)
- Guardrail: Latency p99 <50ms, Refund rate

### Analiz (ön belirlenmiş)
t-test (two-sided, α=0.05) + Bootstrap CI + CUPED

### Sonuçlar
| Metrik | Control | Treatment | Lift | p-value |
|--------|---------|-----------|------|---------|
| CTR | 0.082 | 0.091 | +10.9% | 0.001 |
| Revenue/user | 42.1 TL | 44.8 TL | +6.4% | 0.023 |
| Latency p99 | 38ms | 41ms | +7.9% | — |
| Refund rate | 2.1% | 2.2% | +4.8% | 0.41 |

### Karar
**Deploy.** Primary metric anlamlı artış (%10.9, p<0.001).
Guardrail metrikleri normal sınırlarda.

**Kısıt:** Latency artışı dikkat gerektiriyor (+3ms).
Model optimizasyonu backlog'a eklendi.
```

---

## G.4 Mentorluk Yaklaşımı

### Code Review Checklist

```python
# Code review ederken kontrol et:

CODE_REVIEW_CHECKLIST = {
    "Fonksiyonellik": [
        "Beklenen davranış doğru çalışıyor mu?",
        "Edge case'ler düşünülmüş mü? (None, boş liste, sıfır, negatif)",
        "Error handling anlamlı mı?",
    ],
    "Veri/ML özel": [
        "Leakage riski var mı?",
        "Preprocessing sadece train'e mi fit edildi?",
        "Zaman bazlı split doğru mu?",
        "Assert/kontrol yeterli mi?",
    ],
    "Kod kalitesi": [
        "Değişken isimleri açıklayıcı mı?",
        "Tekrar eden kod var mı? (DRY prensibi)",
        "Fonksiyon tek bir şey yapıyor mu?",
        "Type hints var mı?",
    ],
    "Test": [
        "Birim test var mı?",
        "Edge case'ler test edilmiş mi?",
        "Mock/fixture doğru kullanılmış mı?",
    ],
    "Dokümantasyon": [
        "Docstring var mı? (karmaşık fonksiyonlarda)",
        "Anlaşılması zor mantık yorumlanmış mı?",
        "README güncellendi mi?",
    ],
}
```

### Soru Sorma Sanatı

Senior'ın görevi cevap vermek değil, doğru soru sormak:

```
"Bu model neden iyi performans gösteriyor?" yerine:
→ "Validation'dan train AUC'si ne kadar farklı?"
→ "Bu feature'ı production'da hangi kaynaktan alıyoruz?"
→ "Bu skor dağılımı gerçekten olasılık mı, kalibre mi?"

"Bu A/B test iyi görünüyor" yerine:
→ "SRM kontrolü yaptın mı?"
→ "Guardrail metrikleri nasıl?"
→ "Novelty effect olabilir mi, deney süresi yeterli mi?"
```

---

## G.5 Teknik Borç Yönetimi

```
Teknik borç türleri (ML özelinde):

1. Entangled features: Her şey her şeyle karışık
   → Feature dependency graph çiz, izole et

2. Undocumented decisions: "Bu neden böyle?"
   → ADR yaz, docstring ekle

3. Data debt: Pipeline güvenilmez, monitoring yok
   → Data validation (Pandera/Great Expectations) ekle
   → Drift monitoring kur

4. Dependency debt: Eski kütüphane
   → uv/pip-audit ile dependency scanner
   → Major version kırılımlarını takip et

5. Test borcu: Test yok
   → Önce kritik path test, sonra genişlet
   → %70 coverage hedefi yeterli başlangıç
```

```python
# Boy Scout Rule: "Bulduğundan daha temiz bırak"
# Her pull request'te küçük bir iyileştirme yap

def refactor_example():
    """
    ÖNCE (borç):
    x = df['col1'].values
    y = []
    for i in range(len(x)):
        if x[i] > 0:
            y.append(x[i] * 2)
    result = sum(y) / len(y)

    SONRA (temiz):
    """
    import numpy as np
    x = df['col1'].values
    result = np.mean(x[x > 0] * 2)
    return result
```

---

## G.6 Kariyer Büyümesi için Davranışlar

### Her Ay Yapılacaklar

```markdown
### Teknik Büyüme
- [ ] 1 makale/blog oku + not al (ML Engineering, MLOps)
- [ ] 1 yeni araç/teknik dene (mini proje)
- [ ] DataLemur'dan 10 SQL sorusu çöz

### Etki ve Görünürlük
- [ ] 1 analiz/proje için executive summary yaz
- [ ] İç toplantıda bulgu paylaş (demo, show & tell)
- [ ] Bir iş kararına DS perspektifi koy

### Dokümantasyon
- [ ] 1 model card veya ADR yaz/güncelle
- [ ] En az 1 PR'da ayrıntılı code review yap

### Mentorluk
- [ ] Bir junior/meslektaşın sorusuna "soru sorarak" cevap ver
- [ ] Kendi "takıldım" anlarını not et
```

### Sektör Notu — Senior DS Liderliği 2026

Araştırma bulguları:
- Haftalık liderlik iletişimi alan çalışanların mutluluk oranı %77 — iletişim almayanlarda %41
- "Hedef ve strateji çok net" diyen çalışanların %89'u mutlu (genel ortalama %67)
- Yöneticilerle teknikalite değil iş etkisi konuşmak kariyer gelişimini hızlandırıyor

Pratik çıkarım: Her proje sonunda "bu çalışma ne kattı?" sorusunu 1 paragraflı yazı ile yanıtla. Zaman içinde bu alışkanlık hem seni hem ekibini etkiler.

---

## G.7 Etik ve Sorumluluk (Responsible AI)

2026 itibarıyla model geliştirme sürecinde etik ve sorumluluk, "iyi niyetli ekstra" değil zorunlu bir mühendislik pratiği haline gelmiştir. Aşağıdaki çerçeve, her modelin production'a alınmadan önce geçmesi gereken minimum etik kontrol noktalarını tanımlar.

### Model Bias Tespiti Kontrol Listesi

```markdown
## Pre-deployment Bias Kontrolü

- [ ] **1. Veri temsil analizi:** Eğitim verisinde korunan gruplar
      (cinsiyet, yaş, etnisite, gelir düzeyi, coğrafya) yeterli ve dengeli
      temsil ediliyor mu? Alt grup başına minimum N kontrolü yapıldı mı?

- [ ] **2. Proxy değişken taraması:** Korunan özellikler doğrudan
      kullanılmasa bile, proxy görevi gören feature'lar var mı?
      (örn. posta kodu → etnisite, isim → cinsiyet, cihaz türü → gelir)

- [ ] **3. Alt grup performans karşılaştırması:** Model metrikleri (AUC,
      precision, recall) her alt grup için ayrı ayrı hesaplandı mı?
      Gruplar arası performans farkı kabul edilebilir sınırda mı? (≤%5 fark)

- [ ] **4. Fairness metrikleri hesaplaması:** En az 2 fairness metriği
      raporlandı mı?
      - Demographic Parity: P(ŷ=1|A=0) ≈ P(ŷ=1|A=1)
      - Equalized Odds: TPR ve FPR gruplar arasında eşit mi?
      - Calibration: Her grup için tahmin olasılıkları kalibre mi?
      (→ Bkz. katman-B: SHAP, fairness metrikleri detayı)

- [ ] **5. Intersectional analiz:** Tek özellik değil, kesişimsel gruplar
      da kontrol edildi mi? (örn. genç + düşük gelir + kırsal)

- [ ] **6. Feedback loop riski:** Model kararları gelecek eğitim verisini
      etkiliyor mu? (örn. kredi reddi → veri yok → sürekli ret döngüsü)

- [ ] **7. Adversarial test:** Model, kasıtlı manipülasyona karşı dayanıklı
      mı? Edge case'lerde mantıksız çıktılar üretiyor mu?

- [ ] **8. Human-in-the-loop değerlendirme:** Yüksek etkili kararlar
      (kredi, işe alım, sağlık) için insan denetimi mekanizması var mı?
```

### Fairness Metrikleri Referansı

```python
# Temel fairness metrikleri — katman-B'de SHAP ve model açıklanabilirlik
# ile birlikte kullanılır

FAIRNESS_METRICS = {
    "Demographic Parity (DP)": {
        "tanım": "Pozitif tahmin oranı tüm gruplarda eşit olmalı",
        "formül": "P(ŷ=1|A=a) = P(ŷ=1|A=b) ∀ a,b",
        "kabul_eşiği": "0.8 ≤ DP ratio ≤ 1.25 (4/5 kuralı)",
        "uygun_senaryo": "İşe alım, reklam hedefleme",
    },
    "Equalized Odds (EO)": {
        "tanım": "TPR ve FPR tüm gruplarda eşit olmalı",
        "formül": "P(ŷ=1|Y=y,A=a) = P(ŷ=1|Y=y,A=b) ∀ y,a,b",
        "kabul_eşiği": "TPR farkı ≤ 0.05, FPR farkı ≤ 0.05",
        "uygun_senaryo": "Fraud tespiti, risk skorlama",
    },
    "Predictive Parity": {
        "tanım": "Precision tüm gruplarda eşit olmalı",
        "formül": "P(Y=1|ŷ=1,A=a) = P(Y=1|ŷ=1,A=b)",
        "kabul_eşiği": "Precision farkı ≤ 0.05",
        "uygun_senaryo": "Tıbbi tanı, kampanya hedefleme",
    },
    "Calibration": {
        "tanım": "Tahmin olasılıkları her grupta gerçek oranı yansıtmalı",
        "formül": "P(Y=1|ŷ=p,A=a) ≈ p ∀ a",
        "kabul_eşiği": "Brier score farkı ≤ 0.02",
        "uygun_senaryo": "Kredi skorlama, sigorta fiyatlama",
    },
}

# Not: Bu metriklerin hepsi aynı anda sağlanamaz (impossibility theorem).
# İş bağlamına göre öncelikli metrik seçilmelidir.
# Detay → katman-B: fairness-accuracy trade-off
```

### KVKK / GDPR Temel Farkındalık

```markdown
## Veri Bilimciler İçin Kişisel Veri İşleme Özeti

### Temel İlkeler (KVKK & GDPR ortak)
1. **Amaç sınırlılığı:** Veri yalnızca toplandığı amaç için işlenir.
   Model eğitimi için toplanan veri, farklı bir modelde kullanılacaksa
   yeniden rıza/değerlendirme gerekir.

2. **Veri minimizasyonu:** Modele yalnızca gerekli feature'lar dahil edilir.
   "Belki lazım olur" yaklaşımı hukuki risk taşır.

3. **Saklama süresi:** Eğitim verisi belirsiz süre saklanamaz.
   Retention policy tanımlı olmalı ve model card'da belirtilmeli.
   (→ Bkz. katman-E: model card şablonunda saklama süresi alanı)

### Unutulma Hakkı (Right to Erasure)
- Kullanıcı veri silme talep ederse:
  - Eğitim verisinden çıkarılmalı (veya anonimleştirme yeterli mi? → DPO ile konuş)
  - Modelin o kişinin verisini "hatırlayıp hatırlamadığı" değerlendirilmeli
    (membership inference riski)
  - Retraining gerekiyorsa süreç ve maliyet planlanmalı

### Model Açıklanabilirlik Gereksinimleri
- KVKK Madde 11 / GDPR Madde 22: Otomatik karar alma süreçlerinde
  kişi "kararın mantığını öğrenme" hakkına sahip
- Pratikte: SHAP/LIME gibi açıklama araçları ile bireysel tahmin
  gerekçesi üretilebilmeli
  (→ Bkz. katman-B: SHAP değerleri ve feature importance)
- Black box modeller (derin öğrenme) regülatör risk taşır;
  explainability wrapper zorunlu

### DS için Pratik Kurallar
- [ ] Model card'da veri kaynağı ve işleme amacı belirtildi mi?
- [ ] Korunan özellikler (cinsiyet, yaş, etnisite) feature olarak
      kullanılıyorsa hukuki gerekçe var mı?
- [ ] Veri silme talebi gelirse pipeline nasıl tepki verir, test edildi mi?
- [ ] Model kararı bireysel düzeyde açıklanabilir mi?
- [ ] DPO (Data Protection Officer) ile model review yapıldı mı?
```

### Responsible AI Framework Özeti

```
Responsible AI — 6 Temel Sütun (2026 itibarıyla)

┌─────────────────────────────────────────────────────────┐
│  1. FAIRNESS (Adalet)                                   │
│     Tüm gruplar için eşit performans ve fırsat          │
│     → Fairness metrikleri + alt grup analizi             │
│                                                         │
│  2. TRANSPARENCY (Şeffaflık)                            │
│     Model kararları açıklanabilir ve denetlenebilir     │
│     → Model card + SHAP + karar logları                 │
│     (→ Bkz. katman-B: SHAP; katman-E: model card)      │
│                                                         │
│  3. ACCOUNTABILITY (Hesap verebilirlik)                 │
│     Her modelin bir sahibi ve denetim süreci var        │
│     → Ownership matrix + audit trail                    │
│                                                         │
│  4. PRIVACY (Mahremiyet)                                │
│     Kişisel veri korunur, minimum veri ilkesi uygulanır │
│     → KVKK/GDPR uyumluluk + differential privacy       │
│                                                         │
│  5. SAFETY (Güvenlik)                                   │
│     Model beklenmedik durumlarda güvenli davranır        │
│     → Guardrails + fallback mekanizmaları               │
│                                                         │
│  6. HUMAN OVERSIGHT (İnsan denetimi)                    │
│     Yüksek etkili kararlarda insan onayı mekanizması    │
│     → Human-in-the-loop + escalation policy             │
└─────────────────────────────────────────────────────────┘

Her production modeli için minimum:
✓ Model card (şeffaflık)
✓ Fairness raporu (adalet)
✓ Owner + reviewer atanmış (hesap verebilirlik)
✓ Veri işleme amacı belgelenmiş (mahremiyet)
✓ Fallback / degradation planı (güvenlik)
✓ Otomatik karar eşiği tanımlı (insan denetimi)
```

---

## G.8 Senior Davranış Simülasyonu — Alıştırma Soruları

Aşağıdaki senaryolarda kendinizi Senior DS olarak konumlandırın. Her soru için STAR formatında (Situation → Task → Action → Result) bir yanıt taslağı hazırlayın.

### Senaryo 1: Acil Model Kararı

```
Cuma günü saat 17:00. Production'daki öneri modeli son 2 saattir
%40 daha fazla "stokta yok" ürün öneriyor. Müşteri şikayetleri artıyor.
Engineering ekibi "model tarafı, bize ait değil" diyor.
Product Manager "hemen kapat" diyor.

Sorular:
a) İlk 30 dakikada hangi 3 kontrolü yaparsınız?
b) "Modeli kapat" kararı doğru mu? Alternatif ne olabilir?
c) Pazartesi günü hangi kalıcı çözümü önerirsiniz?
d) Bu olayı nasıl dokümante edersiniz? (post-mortem)
```

### Senaryo 2: Etik İkilem

```
Kredi skorlama modeliniz production'da. Performans analizi sırasında
fark ediyorsunuz: model, 18-25 yaş grubunda diğer gruplara göre %15
daha yüksek red oranı veriyor. Bu yaş grubu için precision aslında
daha yüksek (doğru red), ama equalized odds sağlanmıyor.

Sorular:
a) Bu durum "bias" mı yoksa "gerçek risk farkı" mı? Nasıl ayırt edersiniz?
b) Hangi fairness metriğini önceliklendirirsiniz ve neden?
   (→ Bkz. katman-B: fairness metrikleri)
c) İş ekibine ve regülatöre nasıl farklı anlatırsınız?
d) Teknik çözüm olarak ne önerirsiniz? (threshold ayarı, reweighting,
   adversarial debiasing, vb.)
```

### Senaryo 3: Proje Önceliklendirme Çatışması

```
Q2 planlama toplantısındasınız. 3 proje talebi var:

1. CEO: "Rakipler GenAI chatbot çıkardı, biz de yapmalıyız" (belirsiz ROI)
2. Ops VP: "Envanter tahmin modeli aylık 500K TL tasarruf sağlar" (kanıtlı)
3. Growth Lead: "Churn modeli güncellenmeli, drift var" (acil ama glamorous değil)

Kaynağınız: 2 DS, 1 çeyrek.

Sorular:
a) ICE framework ile 3 projeyi skorlayın.
b) CEO'ya GenAI projesini erteleme kararını nasıl anlatırsınız?
c) "Hepsini yapalım" baskısına nasıl karşı koyarsınız?
d) Seçtiğiniz 1-2 projenin OKR'ını yazın.
   (→ Bkz. G.1: OKR şablonu)
```

---

## Katman G Kontrol Listesi

- [ ] ICE framework ile en az 3 projeyi önceliklendirdim
- [ ] Bir analiz için executive summary yazdım
- [ ] Model card şablonunu doldurdum (→ katman-E: model card standartları)
- [ ] En az 1 ADR yazdım
- [ ] Deney dokümanı şablonunu kullandım
- [ ] Bir code review'da checklist uyguladım
- [ ] Teknik borç kategorilerini mevcut projemde analiz ettim
- [ ] OKR bağlantısını bir iş hedefiyle kurdum (haftalık check-in şablonu ile)
- [ ] Vaka çalışmalarını okudum ve kendi deneyimimle karşılaştırdım
- [ ] Model bias kontrol listesini en az 1 modelde uyguladım (→ katman-B: SHAP, fairness)
- [ ] KVKK/GDPR temel kurallarını projemde kontrol ettim
- [ ] Responsible AI framework'ünün 6 sütununu modelde değerlendirdim
- [ ] En az 2 alıştırma senaryosunu STAR formatında yanıtladım
- [ ] Causal analiz perspektifini bir kök neden araştırmasında kullandım (→ katman-C)

### Çapraz Referans Haritası

| Bu Katman (G) | İlgili Katman | Konu |
|----------------|---------------|------|
| G.1 Vaka 1: Drift analizi | **Katman-C** | Causal inference, kök neden zinciri |
| G.7 Fairness metrikleri | **Katman-B** | SHAP değerleri, model açıklanabilirlik, fairness-accuracy trade-off |
| G.7 KVKK/Model açıklanabilirlik | **Katman-B** | SHAP/LIME bireysel açıklama |
| G.3 Model card | **Katman-E** | Model card standartları, versiyon yönetimi |
| G.7 Responsible AI | **Katman-E** | Model dokümantasyonu, audit trail |
| G.8 Senaryo 2: Etik ikilem | **Katman-B** | Fairness metrikleri, bias tespiti |
| G.8 Senaryo 3: Proje önceliklendirme | **Katman-C** | Causal evidence ile confidence skorlama |

---

<div class="nav-footer">
  <span><a href="#file_katman_F_sistem_tasarimi">← Önceki: Katman F — Sistem Tasarımı</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_H_buyuk_veri">Sonraki: Katman H — Büyük Veri →</a></span>
</div>
