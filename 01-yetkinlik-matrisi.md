# 01 — Senior Veri Bilimci Nedir? Yetkinlik Matrisi

> **Senior** sadece "daha iyi model kuran" değildir. Belirsizliği yönetir, iş etkisini ölçer, kararlar aldırır ve ekibi büyütür.

---

## Senior Veri Bilimci Tanımı

Senior bir veri bilimci şu soruları bağımsız yanıtlayabilir:

1. **Hangi problemi çözmeliyim?** (Problem framing)
2. **Bu problemi ML ile mi, başka bir yolla mı çözeyim?**
3. **Veri güvenilir mi? Neyi ölçüyor?**
4. **Model üretimde ne kadar dayanıklı?**
5. **Sonuçları iş kararına nasıl çeviririm?**

Junior bu soruları sormaz ya da cevaplamaya çalışmaz. Mid-level sorar ama yardım ister. Senior bağımsız yönetir.

---

## Yetkinlik Matrisi

### 1) Problem Çerçeveleme (Framing)

> Ilgili katman: [Katman B — Klasik ML (Problem Framing)](katman-B-klasik-ml.md)

| Junior | Mid | Senior |
|--------|-----|--------|
| Verilen problemi çözer | Problemi netleştirmeye çalışır | Problemi yeniden tanımlar, ML gerekip gerekmediğini sorgular |
| Modeli kurar | Metrik seçimini bilir | Metriği iş etkisiyle ilişkilendirir |
| — | — | "Bu aslında operasyon sorunu, ML değil" diyebilir |

**Nicel ölçütler:**
- **Junior:** Verilen 1 problemi çözer, scope değiştirmez.
- **Mid:** Çeyrek başına 2+ problemi bağımsız netleştirir, metrik önerir.
- **Senior:** Çeyrek başına 1+ problemi ML-dışı çözüme yönlendirir; her projede ROI tahmini sunar.

**Senior soru seti:**
- İş hedefi nedir? ("Churn azalt" → Hangi kullanıcı segmentinde? Ne kadar?)
- Karar alma mekanizması ne? (Tahmin → hangi aksiyon tetikleniyor?)
- Yanlış pozitif/negatif maliyeti ne?
- Basit bir kural (if/else) bu problemi daha iyi çözmez mi?

### 2) Veri Hakimiyeti

> Ilgili katman: [Katman A — Temeller (EDA & Veri Kalitesi)](katman-A-temeller.md)

| Junior | Mid | Senior |
|--------|-----|--------|
| Verilen veriyi kullanır | Veri kalitesini kontrol eder | Ölçüm tanımlarını sorgular, upstream problemi tespit eder |
| — | Eksik değerle başa çıkar | MCAR/MAR/MNAR ayrımı yapar, bilgi kaybını minimize eder |
| — | — | Point-in-time correctness: eğitim-servis paritesini garantiler |

**Nicel ölçütler — SQL & veri becerisi:**
- **Junior:** Temel SELECT/JOIN/GROUP BY sorguları yazar; verilen tabloyu kullanır.
- **Mid:** Window function, CTE, subquery kullanır; veri kalitesi kontrolü scriptleri yazar.
- **Senior:** Query performans optimizasyonu yapar (EXPLAIN ANALYZE); data modeling (star/snowflake schema) tasarlar; upstream data contract tanımlar.

**Senior kontrol listesi:**
- [ ] Feature'lar prediction time'da gerçekten mevcut mu?
- [ ] Label nasıl tanımlandı? Kim tanımladı? Ne zaman netleşti?
- [ ] Upstream değişiklik feature dağılımını bozar mı?
- [ ] Bias kaynakları: selection bias, survivorship bias, measurement bias

### 3) Modelleme Ustalığı

> Ilgili katman: [Katman B — Klasik ML (Modelleme Ustalığı)](katman-B-klasik-ml.md)

| Junior | Mid | Senior |
|--------|-----|--------|
| Çeşitli modeller dener | Overfitting/underfitting teşhis eder | Ablation: her komponentin değerini ölçer |
| AUC'ye bakar | Calibration bilir | Maliyet matrisine göre threshold seçer |
| — | SHAP kullanır | SHAP'ın ne söylemediğini de bilir (nedensellik değil) |

**Nicel ölçütler — Model deploy:**
- **Junior:** 1 model production'a deploy eder (gözetim altında).
- **Mid:** 3+ model deploy + monitoring dashboard kurulumu; model performance raporu üretir.
- **Senior:** 5+ model deploy + otomatik retraining pipeline; model registry yönetimi; cross-team model standardı belirler.

**Modelleme döngüsü (senior tempo):**
```
1. Baseline kur (2 saatte)
2. Güçlü model kur (1 günde)
3. Ablation: her feature grubunun katkısını ölç
4. Hata analizi: hangi segmentte kötü? Neden?
5. Calibration + threshold optimizasyonu
6. Sonucu iş etkisine çevir
```

### 4) Üretim ve Yaşam Döngüsü

> Ilgili katman: [Katman E — MLOps](katman-E-mlops.md)

| Junior | Mid | Senior |
|--------|-----|--------|
| Notebook'ta model | FastAPI servisi kurar | Tüm yaşam döngüsünü yönetir |
| — | Docker ile paketler | Drift izler, retraining kararı verir |
| — | — | Training-serving skew'i önceden önler |

**Nicel ölçütler — MLOps olgunluğu:**
- **Junior:** Notebook'tan script'e geçiş; basit Docker container oluşturma.
- **Mid:** CI/CD pipeline kurulumu; MLflow/W&B ile experiment tracking; 2+ servisi Docker Compose ile yönetir.
- **Senior:** End-to-end ML pipeline (Airflow/Kubeflow); otomatik drift detection + alert sistemi; infra maliyetini %20+ optimize eder.

**Prod checklist:**
- [ ] Model MLflow/W&B'de versiyonlandı mı?
- [ ] Drift monitoring kuruldu mu? (data + prediction + performance)
- [ ] Retraining koşulları tanımlı mı? (zaman, drift trigger, performans trigger)
- [ ] Rollback planı var mı?
- [ ] Model card yazıldı mı?

### 5) İletişim ve Liderlik

> Ilgili katman: [Katman G — Senior Davranışlar (İletişim & Liderlik)](katman-G-senior-davranislar.md)

| Junior | Mid | Senior |
|--------|-----|--------|
| Teknik detay verir | Bulgularını anlatır | "Bu kararın riski şu, alternatifleri şunlar" der |
| Sonuçları paylaşır | Görsel sunar | Karar aldırır |
| — | — | Teknik borcu iş diline çevirir, önceliklendirir |

**Nicel ölçütler — İletişim çıktıları:**
- **Junior:** Teknik rapor / notebook dokümantasyonu yazar; takım içi sunum yapar.
- **Mid:** Stakeholder sunumu hazırlar (ayda 1+); dashboard ile self-service insight sunar.
- **Senior:** Roadmap ve strateji dokümanı yazar; C-suite'e karar aldıran brief hazırlar; teknik blog / konferans konuşması yapar.

**Stakeholder iletişim şablonu:**
```
1. Sonuç (öne çek — "Churn oranı bu çeyrekte %15 düşebilir")
2. Kanıtlar (2–3 bullet, teknik olmayan dille)
3. Riskler ve kısıtlar ("Eğer X olursa...")
4. Öneri + Beklenen etki
5. Sonraki adım + sahip
```

### 6) Deney ve Nedensellik (FAANG Seviyesi Ayrım)

> Ilgili katman: [Katman C — Deney Tasarımı ve Nedensellik](katman-C-deney-nedensellik.md)

| Junior | Mid | Senior |
|--------|-----|--------|
| A/B test sonucu okur | Power analizi yapar | CUPED ile varyansı azaltır |
| p-value bilir | Guardrail metric izler | Network effect, novelty effect, non-compliance yönetir |
| — | — | Gözlemsel veriden causal inference: DiD, PSM, DoWhy |

**Nicel ölçütler:**
- **Junior:** A/B test sonuçlarını doğru okur ve raporlar.
- **Mid:** Bağımsız power analizi yapar; yılda 2+ deney tasarlar.
- **Senior:** Causal inference yöntemi (DiD, IV, RDD) uygular; deney platformu/framework'üne katkı sağlar.

---

## Sektör Beklentileri (2026 itibarıyla)

Araştırmalara göre şirketlerin senior DS'ten beklentileri:

**Teknik:**
- ML bilgisi (modelleme + deployment) — ilanların %77'sinde
- Cloud bilgisi (AWS/GCP/Azure) — ilanların %15'inde aktif aranıyor
- **LLM/GenAI farkındalığı — ilanların %30+'ında zorunlu hale geldi.** AI-related ilanların %31'i LLM, RAG, prompt engineering, vector database gibi becerileri açıkça istiyor. Geleneksel "Data Scientist" ilanlarının %27'si artık başlıkta bile AI/GenAI/LLM ifadesi taşıyor.
- **MLOps/model monitoring — "nice to have"den kesin "must have"e geçti.** MLOps mühendisliği son 5 yılda 9.8x büyüme gösterdi (LinkedIn Emerging Jobs). Kubernetes (%17.6), Docker (%15.4), CI/CD pipeline bilgisi standart beklenti haline geldi.
- **Causal inference — FAANG'da ayrıştırıcı faktör.** A/B test ötesinde DiD, PSM, DoWhy gibi yöntemler; özellikle Meta, Netflix, Uber gibi şirketlerde senior/staff ayrımını belirliyor.

**Non-teknik (ayırt edici):**
- Belirsiz problemleri yapılandırma
- C-suite ile konuşabilme (teknik bilgi vermek değil, karar aldırmak)
- Proaktif risk tespiti ("bu modeli deploy ederken şu riskleri görmemiz lazım")
- Mentorluk ve ekip büyütme

**Kıdemle beklenti dağılımı (2026):** AI skill gerektiren ilanların %73'ü mid (%43) ve senior (%30) seviyeyi hedefliyor; entry-level yalnızca %6. Bu, "AI öğrenmek yetmez, deneyimle birleştirmek gerekir" mesajını net veriyor.

> **Senior kural:** "Ben modeli yaptım" değil, "Bu kararın riski/alternatifi şu, önerim bu."

---

## Kariyer Basamakları

```
Junior DS    → Verilen problemi çözer, gözetim altında çalışır
Mid DS       → Problemi netleştirir, bağımsız deliver eder
Senior DS    → Problemi bulur, çözer, etkiyi ölçer, ekibi büyütür
Staff DS     → Çoklu ekip etki, platform/altyapı kararları
Principal DS → Şirket düzeyinde teknik yön, dışarıya sözcü
```

### Staff DS — Detaylı Beklentiler

Staff seviyesi, Senior'dan sonra gelen bireysel katkıcı (IC) rolüdür — yöneticilik değildir. 5-10 yıl deneyim beklenir. Temel fark: **etki alanının tek takımdan çoklu takıma genişlemesi.**

| Alan | Somut Katkı |
|------|-------------|
| **Teknik derinlik** | Takımdaki hiç kimsenin çözemeyeceği belirsiz, karmaşık problemleri formüle eder ve çözer |
| **Cross-team leadership** | 2+ takımın ortak kullandığı ML platform/pipeline kararlarını yönlendirir |
| **Mimari etki** | Feature store, model registry, experiment platform gibi altyapı tasarımına sahip olur |
| **Mentorluk** | Design doc ve code review üzerinden senior DS'leri geliştirir |
| **Stakeholder yönetimi** | Proje timeline, gereksinim ve risk yönetimini bağımsız yürütür |
| **Netlik sağlama** | Belirsiz problem alanlarını yapılandırarak takımın etkili çalışmasını sağlar |

### Principal DS — Detaylı Beklentiler

Principal, teknik derinlik ile iş stratejisinin kesiştiği noktadadır. Etki alanı **organizasyon genelindedir** (org-wide impact).

| Alan | Somut Katkı |
|------|-------------|
| **Teknik vizyon** | Şirketin AI/ML yol haritasını üst yönetime sunar ve şekillendirir |
| **Mimari yönetim** | Platform tasarımı, ML deployment stratejisi ve araştırma yatırımlarında trade-off analizi yapar |
| **Risk ve compliance** | AI sistemlerinin riskini değerlendirir; regülasyon kısıtlarını teknik kararlara entegre eder |
| **Cross-product scope** | Tek ürün değil, birden fazla ürün/iş hattı genelinde çalışır |
| **Pipeline excellence** | Daha doğru, daha hafif, production-ready pipeline'lar tasarlar |
| **Dış temsil** | Konferanslarda konuşma, açık kaynak katkı, sektörel danışmanlık |
| **Deney liderliği** | Organizasyon genelinde deneyselleştirme (experimentation) kültürünü kurar |

Bu rehber **Junior → Senior** arasını kapatmak için tasarlandı; Staff/Principal hedefleyenler için ise yukarıdaki çerçeveyi pusula olarak kullanabilir.

---

## Öz-değerlendirme Puan Tablosu

Her yetkinlik alanı için kendini **1-5 ölçeğinde** değerlendir. Her 3 ayda bir güncelle ve gelişim trendini izle.

**Puan ölçeği:**
| Puan | Seviye | Tanım |
|------|--------|-------|
| 1 | Başlangıç | Kavramı duydum, henüz uygulamadım |
| 2 | Temel | Gözetim altında uygulayabiliyorum |
| 3 | Bağımsız | Bağımsız çalışabiliyor, standart senaryoları çözüyorum |
| 4 | İleri | Karmaşık senaryolarda çözüm üretiyorum, başkalarına öğretebiliyorum |
| 5 | Uzman | Yeni yaklaşımlar geliştiriyorum, organizasyon genelinde etki yaratıyorum |

### Değerlendirme Tablosu

| # | Yetkinlik Alanı | Puan (1-5) | Hedef Puan | Notlar / Aksiyon Planı |
|---|----------------|------------|------------|----------------------|
| 1 | Problem Çerçeveleme (Framing) | ___ | ___ | |
| 2 | Veri Hakimiyeti & SQL | ___ | ___ | |
| 3 | Modelleme Ustalığı | ___ | ___ | |
| 4 | Üretim & MLOps | ___ | ___ | |
| 5 | İletişim & Liderlik | ___ | ___ | |
| 6 | Deney & Nedensellik | ___ | ___ | |
| 7 | LLM / GenAI | ___ | ___ | |
| 8 | Cloud & Altyapı | ___ | ___ | |
| | **Toplam** | __/40 | __/40 | |

**Seviye referansları (toplam puan):**
- **8-15:** Junior seviye — temel becerileri geliştirmeye odaklan
- **16-24:** Mid seviye — bağımsızlık ve derinlik kazanma aşaması
- **25-32:** Senior seviye — etki ve liderlik boyutunu güçlendir
- **33-37:** Staff seviye — cross-team etki ve platform düşüncesi
- **38-40:** Principal seviye — organizasyonel vizyon ve dış temsil

---

## Öz-değerlendirme Soruları

Her 3 ayda bir kendine sor:

### Teknik
- [ ] Son 3 ayda production'a çıkardığım bir model var mı?
- [ ] Model drift ile karşılaştım ve yönettim mi?
- [ ] Bir A/B test tasarladım ve sonucunu iş kararına çevirdim mi?
- [ ] Bir causal inference sorusuna yaklaştım mı?
- [ ] LLM/GenAI tabanlı bir çözümü değerlendirdim veya prototipledim mi?

### Liderlik
- [ ] Bir problemi "ML değil, başka çözüm lazım" diyerek yeniden çerçeveledim mi?
- [ ] Bir stakeholder'a teknik riski iş diline çevirerek anlatabildim mi?
- [ ] Bir junior veya meslektaşımın çalışmasını review ettim mi?

### Sürekli Öğrenme
- [ ] Bu çeyrekte hangi yeni aracı/tekniği öğrendim ve uyguladım?
- [ ] Portföyüme gerçek etki gösteren bir proje ekledim mi?

---

## Yetkinlik Özeti

```
Teknik Derinlik:
  ✓ Matematik temeli (sezgisel, ezber değil)
  ✓ ML modelleme (baseline → boosting → ablation)
  ✓ Feature engineering (leakage'siz, prod-ready)
  ✓ İstatistik + A/B test + causal inference
  ✓ DL temeli + NLP/RecSys/LLM'den en az biri
  ✓ LLM/GenAI (RAG, prompt engineering, vector DB) — 2026 beklentisi
  ✓ MLOps (servis, izleme, CI/CD, Kubernetes/Docker)
  ✓ Sistem tasarımı (feature store, latency, mimari)

Ürün/İş:
  ✓ Problem framing (ML gerekip gerekmediğini sorgula)
  ✓ Metrik → iş etkisi bağlantısı
  ✓ Stakeholder iletişimi (karar aldırmak)
  ✓ Technical debt yönetimi

Dokümantasyon:
  ✓ Model card
  ✓ Data card
  ✓ ADR (Architecture Decision Record)
  ✓ Deney dokümanı (hipotez + tasarım + sonuç)
```

---

<div class="nav-footer">
  <span><a href="#file_00_uygulama_sirasi">← Önceki: Uygulama Sırası</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_0_matematik">Sonraki: Katman 0 — Matematik →</a></span>
</div>
