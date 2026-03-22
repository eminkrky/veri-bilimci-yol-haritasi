# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje Özeti

Türkçe yazılmış, kapsamlı bir veri bilimi öğrenme rehberi: "Veri Bilimci Yol Haritası (0 → Senior)". Yazılım projesi değil, içerik/müfredat projesidir. Build/test/lint komutu yoktur.

**Hedef kitle:** Sıfırdan başlayan → senior seviyeye ulaşmak isteyen herkes.

## Dosya Yapısı

Proje **katmanlara göre bölünmüş** çoklu dosya yapısındadır:

```
veribilimikitabı/
├── CLAUDE.md
├── README.md                          # Ana index, içindekiler, genel bakış
├── 00-uygulama-sirasi.md              # Aşama 0–7 sıralı öğrenme planı
├── 01-yetkinlik-matrisi.md            # Senior veri bilimci tanımı
├── katman-0-matematik.md              # Lineer cebir, kalkülüs, olasılık
├── katman-A-temeller.md               # Python, istatistik, SQL, görselleştirme
├── katman-B-klasik-ml.md              # Doğrusal modeller, ağaçlar, boosting, değerlendirme
├── katman-C-deney-nedensellik.md      # A/B test, power, CUPED, causal inference
├── katman-D-derin-ogrenme.md          # DL, NLP, CV, RecSys, LLM/RAG
├── katman-E-mlops.md                  # Paketleme, servis, izleme, drift, CI/CD
├── katman-F-sistem-tasarimi.md        # Online/offline, feature store, maliyet
├── katman-G-senior-davranislar.md     # Etki, liderlik, dokümantasyon
├── katman-H-buyuk-veri.md             # Spark, Dask, dağıtık hesaplama
├── projeler.md                        # Portföy projeleri (Proje-0 → Proje-7)
├── mulakat.md                         # Mülakat hazırlığı (SQL, ML, case, behavioral)
├── kaynaklar.md                       # Ücretsiz/temel kaynak listesi
└── calisma-gunlugu.md                 # Değişiklik günlüğü
```

## İçerik Yazım Kuralları

### Dil
- **Türkçe** ana dil. Tüm açıklamalar Türkçe yazılır.
- **Teknik terimler karma:** İngilizce teknik terimler (overfitting, leakage, drift, boosting, gradient descent vb.) olduğu gibi kullanılır + Türkçe açıklama eşlik eder. Sektör standardına uygun.
- Örnek: "Overfitting (aşırı öğrenme) — model eğitim verisini ezberler…"

### Format & Stil
- `##` ana bölümler, `###` alt bölümler, `####` detay başlıkları.
- Fenced code blocks (```python, ```sql, ```bash) kod örnekleri için.
- Her konuda **dengeli yaklaşım:** önce sezgisel açıklama → sonra isteğe bağlı matematik detayı → sonra çalıştırılabilir kod örneği.
- Blockquote (`>`) ipucu, uyarı ve "senior notu" için kullanılır.

### Konu Anlatım Şablonu

Her konu/kavram şu sırayla anlatılmalı:

1. **Sezgisel açıklama** — Kavramı günlük hayattan analoji veya basit örnekle anlat
2. **Matematik detayı** (isteğe bağlı bölüm) — Formül, türev, loss fonksiyonu
3. **Kod örneği** — Çalıştırılabilir Python/SQL snippet (scikit-learn, statsmodels, pandas vb.)
4. **Senior notu** — Üretimde dikkat edilecek noktalar, yaygın hatalar, best practice

### Öncelikli Derinleştirme Alanları

Aşağıdaki bölümler diğerlerine göre **daha detaylı** yazılmalı:

1. **ML & İstatistik** (katman-B, katman-C): Algoritmaların arkasındaki matematik + sezgisel açıklama + scikit-learn/statsmodels kod örnekleri. Deney tasarımı, power analizi, causal inference derinlemesine.
2. **MLOps & Üretim** (katman-E): Cloud-agnostic, açık kaynak araç odaklı (MLflow, Docker, FastAPI, Evidently). AWS/GCP'ye bağımlı olmadan.
3. **Mülakat hazırlığı** (mulakat.md): Çözümlü case study'ler, SQL soruları, ML case'leri genişletilecek.

### Araç Tercihleri (MLOps)
- **Cloud-agnostic / açık kaynak** öncelikli: MLflow, Docker, FastAPI, Evidently, Great Expectations
- Bulut servisleri (AWS/GCP) varsa yan bilgi olarak, ana anlatım açık kaynak üzerinden

## İçerik Geliştirme Süreci

### Araştırma Zorunluluğu
Her bölüm veya alan yazılmadan/güncellemeden önce **internet üzerinden detaylı araştırma** yapılmalıdır:
- **Sektör trendleri:** O alandaki güncel araçlar, frameworkler, yaklaşımlar neler? Hangisi yükseliyor, hangisi düşüşte?
- **Gelecek yönelimleri:** 2025–2027 arasında bu alanda ne değişecek? Hangi teknolojiler öne çıkacak?
- **Best practice'ler:** Büyük şirketler (FAANG, unicorn'lar) bu konuyu nasıl uyguluyor?
- **Topluluk görüşleri:** Reddit, HackerNews, Medium, arXiv'deki güncel tartışmalar neler?
- **Araç karşılaştırmaları:** Alternatif araçların artıları/eksileri, hangisi hangi senaryoda tercih edilmeli?

Bu araştırma sonuçları doğrudan içeriğe yansıtılmalı: güncel tavsiyeleri, sektörel kararları ve gelecek öngörülerini bölüme "Sektör Notu" veya "Güncel Trend" başlığıyla ekle.

### Kalite Kontrol — İki Geçişli Review
İçerik üretimi tamamlandıktan sonra **tüm doküman 2 kez baştan sona gözden geçirilmelidir:**

**1. Geçiş — Eksiklik Taraması:**
- Her bölümde anlatım şablonuna (sezgisel açıklama → matematik → kod → senior notu) uyulmuş mu?
- Eksik kalan konular, atlanmış kavramlar, yetersiz derinlik var mı?
- Bölümler arası çapraz referanslar tutarlı mı?
- Kod örnekleri çalıştırılabilir ve güncel mi?

**2. Geçiş — Tutarlılık ve Kalite:**
- Dil ve terim kullanımı tutarlı mı? (karma terim politikası doğru uygulanmış mı?)
- Tekrar eden içerik var mı? (aynı şey birden fazla dosyada anlatılmış mı?)
- Akış mantıklı mı? Bir önceki bölümü bilmeden bu bölüm anlaşılır mı?
- Kaynak linkleri geçerli ve güncel mi?

Her review geçişinde bulunan eksikler doğrudan düzeltilmeli, sonraya bırakılmamalıdır.

### Durma Kuralı
İçerik geliştirme süreci **tüm dosyalar tamamlanana kadar durmamalıdır.** Agent bir bölümü bitirdiğinde hemen sonraki bölüme geçmeli, tüm katmanlar + projeler + mülakat bölümü + kaynaklar yazıldıktan sonra 2 geçişli review yapmalıdır.

## Commit Kuralları

- Conventional Commits formatı: `docs: <açıklama>`, `feat: <açıklama>`, `fix: <açıklama>`
- Branch stratejisi: `main`, `feature/<konu>`, `experiment/<isim>`
- Türkçe commit mesajları tercih edilir

## Çalışma Günlüğü

Her önemli değişiklikte `calisma-gunlugu.md` dosyasına tarih + yapılan iş özeti eklenmelidir.
