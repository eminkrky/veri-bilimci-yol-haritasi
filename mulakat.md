# Mülakat Hazırlığı: SQL + ML + İstatistik + Sistem Tasarımı + Behavioral

> Bu dosya hem teknik hem davranışsal mülakat için kapsamlı bir hazırlık seti. Her bölüm çözümlü örnekler içeriyor.
>
> **Strateji:** Mülakat hazırlığı 4–6 hafta, yoğun 6–8 saat/hafta.

---

## Mülakat Süreci Genel Yapısı

### Teknik Şirketlerde Tipik Aşamalar

```
1. Recruiter Screen (30 dk)
   → Geçmiş, motivasyon, genel uyum

2. Technical Phone Screen (45–60 dk)
   → SQL veya Python + temel ML sorusu

3. Onsite / Virtual Loop (4–6 round)
   → SQL/Data analysis
   → Statistics + A/B Testing
   → ML Modeling
   → System Design (senior rollerde)
   → Behavioral (tüm rollerde)

4. Team Interview + Bar Raiser
   → Şirkete özgü kültür uyumu
```

### Hazırlık Dağılımı

```
%40 — Teknik pratik (SQL, Python, ML)
%30 — Behavioral hikaye hazırlığı
%20 — Sistem tasarımı
%10 — Şirkete özgü araştırma
```

---

## A) SQL Mülakat Seti (35 Soru + Çözümler)

### A1. Temel Seviye (1–10)
*(→ Bkz. Katman F: SQL İpuçları — NULL yönetimi, QUALIFY, STRING_AGG)*

**S1: Günlük Aktif Kullanıcı (DAU)**
```sql
SELECT
    DATE(event_ts) AS date,
    COUNT(DISTINCT user_id) AS dau
FROM events
WHERE event_ts >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY 1
ORDER BY 1;
```

**S2: En son siparişi getir (her kullanıcı için)**
```sql
-- Yöntem 1: Window function
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_ts DESC) AS rn
    FROM orders
)
SELECT user_id, order_id, amount, order_ts
FROM ranked WHERE rn = 1;

-- Yöntem 2: Subquery (bazı DB'lerde daha hızlı)
SELECT o.*
FROM orders o
INNER JOIN (
    SELECT user_id, MAX(order_ts) AS last_ts
    FROM orders
    GROUP BY user_id
) latest ON o.user_id = latest.user_id AND o.order_ts = latest.last_ts;
```

**S3: Her kullanıcının 2. siparişi**
```sql
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_ts) AS rn
    FROM orders
)
SELECT * FROM ranked WHERE rn = 2;
```

**S4: Aylık gelir + MoM büyüme oranı**
```sql
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', order_ts) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY 1
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
          / NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;
```

**S5: Median sipariş tutarı (PERCENTILE_CONT)**
```sql
-- Standart SQL (BigQuery, Snowflake, PostgreSQL)
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount
FROM orders;

-- MySQL / Redshift alternatifi
WITH ranked AS (
    SELECT amount,
        ROW_NUMBER() OVER (ORDER BY amount) AS rn,
        COUNT(*) OVER () AS total
    FROM orders
)
SELECT AVG(amount) AS median_amount
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2.0), CEIL((total + 1) / 2.0));
```

---

### A2. Orta Seviye (11–20)
*(→ Bkz. Katman C: İstatistik — funnel metriklerinde Simpson Paradoxu riski)*

**S6: Funnel dönüşüm analizi**
```sql
WITH steps AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_name = 'page_view' THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN event_name = 'add_to_cart' THEN 1 ELSE 0 END) AS carted,
        MAX(CASE WHEN event_name = 'checkout' THEN 1 ELSE 0 END) AS checked_out,
        MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM events
    WHERE event_ts >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY user_id
)
SELECT
    COUNT(*) AS n_viewed,
    SUM(carted) AS n_carted,
    SUM(checked_out) AS n_checked_out,
    SUM(purchased) AS n_purchased,
    ROUND(100.0 * SUM(carted) / COUNT(*), 1) AS view_to_cart_pct,
    ROUND(100.0 * SUM(purchased) / NULLIF(SUM(carted), 0), 1) AS cart_to_purchase_pct
FROM steps
WHERE viewed = 1;
```

**S7: Cohort retention (tam çözüm)**
```sql
WITH first_order AS (
    SELECT user_id, MIN(DATE_TRUNC('month', order_ts)) AS cohort_month
    FROM orders GROUP BY user_id
),
orders_monthly AS (
    SELECT
        o.user_id,
        f.cohort_month,
        DATE_TRUNC('month', o.order_ts) AS activity_month,
        DATEDIFF('month', f.cohort_month, DATE_TRUNC('month', o.order_ts)) AS period
    FROM orders o JOIN first_order f USING (user_id)
),
retention AS (
    SELECT cohort_month, period, COUNT(DISTINCT user_id) AS n_users
    FROM orders_monthly GROUP BY 1, 2
),
cohort_size AS (
    SELECT cohort_month, n_users AS cohort_n FROM retention WHERE period = 0
)
SELECT
    r.cohort_month,
    r.period,
    r.n_users,
    c.cohort_n,
    ROUND(100.0 * r.n_users / c.cohort_n, 1) AS retention_pct
FROM retention r JOIN cohort_size c USING (cohort_month)
ORDER BY 1, 2;
```

**S8: 80/20 Pareto — Gelirin %80'ini getiren kullanıcı oranı**
```sql
WITH user_revenue AS (
    SELECT user_id, SUM(amount) AS total_revenue
    FROM orders GROUP BY user_id
),
ranked AS (
    SELECT *,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC) AS cum_revenue,
        SUM(total_revenue) OVER () AS grand_total,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS rn,
        COUNT(*) OVER () AS n_users
    FROM user_revenue
)
SELECT
    COUNT(*) AS users_for_80pct_revenue,
    (COUNT(*) * 100.0 / MAX(n_users)) AS pct_of_users
FROM ranked
WHERE cum_revenue / grand_total <= 0.80;
```

**S9: Duplicate event temizleme**
```sql
-- Her user-event için sadece ilkini tut
WITH deduped AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, event_name, DATE(event_ts)
            ORDER BY event_ts
        ) AS rn
    FROM events
)
SELECT * FROM deduped WHERE rn = 1;
```

**S10: Sessionization (30 dk kesim)**
```sql
WITH events_with_prev AS (
    SELECT
        user_id, event_ts,
        LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts
    FROM events
),
session_marks AS (
    SELECT *,
        CASE
            WHEN prev_ts IS NULL
              OR DATEDIFF('minute', prev_ts, event_ts) > 30
            THEN 1 ELSE 0
        END AS new_session
    FROM events_with_prev
),
session_ids AS (
    SELECT *,
        SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_num
    FROM session_marks
)
SELECT
    user_id,
    session_num,
    MIN(event_ts) AS session_start,
    MAX(event_ts) AS session_end,
    DATEDIFF('minute', MIN(event_ts), MAX(event_ts)) AS duration_min,
    COUNT(*) AS n_events
FROM session_ids
GROUP BY 1, 2;
```

---

### A3. İleri Seviye (21–35)
*(→ Bkz. Katman D: Sistem tasarımı — SCD, late-arriving data pipeline senaryoları)*

**S11: SCD Type 2 ile Point-in-Time Correct Join**
```sql
-- Kullanıcı ülkesi değişmiş — sipariş tarihindeki ülkeyi al
SELECT
    o.order_id,
    o.user_id,
    o.amount,
    o.order_ts,
    u.country  -- Sipariş tarihindeki ülke (güncel değil!)
FROM orders o
JOIN user_country_history u
    ON o.user_id = u.user_id
    AND o.order_ts >= u.valid_from
    AND o.order_ts < COALESCE(u.valid_to, CURRENT_TIMESTAMP);
```

**S12: Recursive CTE ile hiyerarşi**
```sql
-- Organizasyon ağacı — bir çalışanın tüm altındakileri
WITH RECURSIVE org AS (
    -- Base case: başlangıç çalışan
    SELECT employee_id, manager_id, name, 0 AS level
    FROM employees
    WHERE employee_id = 100  -- Başlangıç kişi

    UNION ALL

    -- Recursive step: altındakiler
    SELECT e.employee_id, e.manager_id, e.name, org.level + 1
    FROM employees e
    JOIN org ON e.manager_id = org.employee_id
)
SELECT * FROM org ORDER BY level, name;
```

**S13: Late arriving data — geç gelen etiket yönetimi**
```sql
-- Etiket geldiğinde mevcut prediction ile birleştir
WITH predictions AS (
    SELECT user_id, predicted_prob, prediction_ts
    FROM model_predictions
    WHERE prediction_ts = '2024-03-01'
),
labels AS (
    -- 30 gün sonra gelen ground truth
    SELECT user_id, churned, labeled_ts
    FROM user_labels
    WHERE labeled_ts = '2024-04-01'
)
SELECT
    p.user_id,
    p.predicted_prob,
    p.prediction_ts,
    l.churned AS actual_label,
    l.labeled_ts
FROM predictions p
LEFT JOIN labels l USING (user_id);
```

**S14: Gap Analysis — Ardışık gün boşluğu bulma**
*(→ Bkz. Katman A2: Sessionization mantığıyla aynı LAG/LEAD yaklaşımı)*
```sql
-- Kullanıcının aktif olduğu günler arasındaki boşlukları bul
WITH daily_activity AS (
    SELECT DISTINCT user_id, DATE(event_ts) AS activity_date
    FROM events
),
with_prev AS (
    SELECT
        user_id,
        activity_date,
        LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date) AS prev_date
    FROM daily_activity
),
gaps AS (
    SELECT
        user_id,
        prev_date AS gap_start,
        activity_date AS gap_end,
        activity_date - prev_date - 1 AS gap_days   -- boşluk uzunluğu
    FROM with_prev
    WHERE activity_date - prev_date > 1              -- sadece boşluklar
)
SELECT *
FROM gaps
ORDER BY gap_days DESC;

-- Yorum: gap_days >= 7 filtrelemesiyle "1 haftadan uzun sessiz kalan"
-- kullanıcıları tespit edebilirsiniz.
```

**S15: Moving Percentile — Kayan pencerede yüzdelik hesaplama**
*(→ Bkz. Katman A1-S5: PERCENTILE_CONT temel kullanımı)*
```sql
-- Son 7 gün içinde sipariş tutarının p50 ve p90 değeri (günlük)
-- PostgreSQL / BigQuery stili
SELECT
    DATE(order_ts) AS order_date,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount)
        AS p50_amount,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount)
        AS p90_amount
FROM orders
WHERE order_ts >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY 1
ORDER BY 1;

-- Gerçek "kayan pencere" yüzdelik (her gün için son 7 gün):
WITH daily_orders AS (
    SELECT
        DATE(order_ts) AS order_date,
        amount
    FROM orders
),
rolling_window AS (
    SELECT
        d1.order_date,
        d2.amount
    FROM (SELECT DISTINCT order_date FROM daily_orders) d1
    JOIN daily_orders d2
        ON d2.order_date BETWEEN d1.order_date - INTERVAL '6 days'
                              AND d1.order_date
)
SELECT
    order_date,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) AS rolling_p50,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount) AS rolling_p90,
    COUNT(*) AS n_orders
FROM rolling_window
GROUP BY 1
ORDER BY 1;
```

**S16: Self-Join ile N-day Retention**
*(→ Bkz. Katman A2-S7: Cohort retention'ın günlük versiyonu)*
```sql
-- Day-1, Day-7, Day-30 retention: kayıt sonrası geri dönüş oranı
WITH signups AS (
    SELECT user_id, DATE(created_at) AS signup_date
    FROM users
),
activity AS (
    SELECT DISTINCT user_id, DATE(event_ts) AS active_date
    FROM events
)
SELECT
    s.signup_date,
    COUNT(DISTINCT s.user_id) AS cohort_size,
    -- Day-1 retention
    COUNT(DISTINCT CASE
        WHEN a1.active_date = s.signup_date + INTERVAL '1 day'
        THEN s.user_id END) AS d1_retained,
    -- Day-7 retention
    COUNT(DISTINCT CASE
        WHEN a7.active_date = s.signup_date + INTERVAL '7 days'
        THEN s.user_id END) AS d7_retained,
    -- Day-30 retention
    COUNT(DISTINCT CASE
        WHEN a30.active_date = s.signup_date + INTERVAL '30 days'
        THEN s.user_id END) AS d30_retained,
    -- Oranlar
    ROUND(100.0 * COUNT(DISTINCT CASE
        WHEN a1.active_date = s.signup_date + INTERVAL '1 day'
        THEN s.user_id END) / COUNT(DISTINCT s.user_id), 1) AS d1_pct,
    ROUND(100.0 * COUNT(DISTINCT CASE
        WHEN a7.active_date = s.signup_date + INTERVAL '7 days'
        THEN s.user_id END) / COUNT(DISTINCT s.user_id), 1) AS d7_pct,
    ROUND(100.0 * COUNT(DISTINCT CASE
        WHEN a30.active_date = s.signup_date + INTERVAL '30 days'
        THEN s.user_id END) / COUNT(DISTINCT s.user_id), 1) AS d30_pct
FROM signups s
LEFT JOIN activity a1  ON s.user_id = a1.user_id
LEFT JOIN activity a7  ON s.user_id = a7.user_id
LEFT JOIN activity a30 ON s.user_id = a30.user_id
GROUP BY 1
ORDER BY 1;
```

**S17: PIVOT / UNPIVOT — Satırdan sütuna, sütundan satıra**
*(→ Bkz. Katman A2-S6: Funnel analizi CASE WHEN ile benzer pivot mantığı)*
```sql
-- PIVOT: Aylık geliri kategorilere göre sütunlara aç
-- Yöntem 1: Manuel pivot (tüm SQL dialektlerinde çalışır)
SELECT
    DATE_TRUNC('month', order_ts) AS month,
    SUM(CASE WHEN category = 'electronics' THEN amount ELSE 0 END) AS electronics,
    SUM(CASE WHEN category = 'clothing'    THEN amount ELSE 0 END) AS clothing,
    SUM(CASE WHEN category = 'food'        THEN amount ELSE 0 END) AS food
FROM orders
GROUP BY 1
ORDER BY 1;

-- Yöntem 2: SQL Server / Snowflake PIVOT syntax
SELECT *
FROM (
    SELECT DATE_TRUNC('month', order_ts) AS month, category, amount
    FROM orders
) src
PIVOT (
    SUM(amount) FOR category IN ('electronics', 'clothing', 'food')
) AS pvt;

-- UNPIVOT: Sütunları satırlara dönüştür (yukarıdaki sonucu geri çevir)
-- BigQuery stili UNPIVOT
SELECT month, category, revenue
FROM monthly_category_revenue
UNPIVOT (revenue FOR category IN (electronics, clothing, food));

-- Alternatif: UNION ALL ile elle unpivot
SELECT month, 'electronics' AS category, electronics AS revenue
FROM monthly_category_revenue
UNION ALL
SELECT month, 'clothing', clothing FROM monthly_category_revenue
UNION ALL
SELECT month, 'food', food FROM monthly_category_revenue;
```

**S18: JSON / ARRAY Field Sorgusu (BigQuery + PostgreSQL)**
*(→ Bkz. Katman F: SQL İpuçları — modern SQL özellikleri)*
```sql
-- PostgreSQL: JSONB sütununda arama
-- Tablo: user_profiles (user_id INT, metadata JSONB)
-- metadata örneği: {"plan": "premium", "tags": ["ml", "python"], "score": 85}

-- Plan türüne göre filtrele
SELECT user_id, metadata->>'plan' AS plan_type
FROM user_profiles
WHERE metadata->>'plan' = 'premium';

-- JSON içindeki sayısal değere göre sıralama
SELECT user_id, (metadata->>'score')::INT AS score
FROM user_profiles
ORDER BY score DESC;

-- ARRAY içinde arama: "ml" etiketine sahip kullanıcılar
SELECT user_id
FROM user_profiles
WHERE metadata->'tags' ? 'ml';

-- BigQuery: UNNEST ile array açma
-- Tablo: events (user_id INT64, event_params ARRAY<STRUCT<key STRING, value STRING>>)
SELECT
    user_id,
    ep.key,
    ep.value
FROM events,
UNNEST(event_params) AS ep
WHERE ep.key = 'page_name';

-- BigQuery: JSON_EXTRACT ile nested field
SELECT
    user_id,
    JSON_EXTRACT_SCALAR(metadata, '$.plan') AS plan_type,
    JSON_EXTRACT_SCALAR(metadata, '$.score') AS score
FROM user_profiles
WHERE JSON_EXTRACT_SCALAR(metadata, '$.plan') = 'premium';

-- PostgreSQL: jsonb_array_elements ile array'i satıra dönüştür
SELECT
    user_id,
    tag.value AS tag
FROM user_profiles,
LATERAL jsonb_array_elements_text(metadata->'tags') AS tag(value);
```

---

## B) ML Mülakat Senaryoları (15 Çözümlü Case)

### B1. Data Leakage Tespiti
*(→ Bkz. Katman A3-S13: Late arriving data; Katman D2: Pipeline tasarımı)*

**Soru:** "Model training'de AUC 0.99, production'da 0.62. Ne oldu?"

**Cevap yapısı:**
```
1. İlk hipotez: Leakage var (gap çok büyük)

2. Kontrol listesi:
   □ Feature window doğru mu? (gelecek bilgisi mevcut mu?)
   □ Target ile direkt ilişkili feature var mı?
   □ Preprocessing train'e mi fit edildi, yoksa tüm veriye mi?
   □ Test seti özellikle kolay örnekler mi içeriyor?
   □ Label definition doğru mu? (eğitimde future bilgi var mı?)

3. En yaygın sebepler:
   a) Zaman bazlı split yapılmamış → test'te önceki dönem var
   b) Feature: "last_login_date" → çok yakın geçmişten (target'ı açıklıyor)
   c) "campaign_sent" feature → kampanya churn sonrası belirlendi (tersine nedensellik)
   d) Scaler tüm veriye fit edilmiş → test bilgisi training'e sızmış

4. Düzeltme:
   → Katı zaman bazlı split
   → Feature penceresini target'tan önce kes
   → Preprocessing sadece train'e fit
   → Leakage controlled validation yapı
```

### B2. Imbalanced Classification (%1 pozitif)
*(→ Bkz. Katman D4: Fraud detection sistemi — %0.1 pozitif oranıyla gerçek dünya uygulaması)*

**Soru:** "Fraud tespiti, %1 pozitif. Nasıl yaklaşırsın?"

```python
# Yaklaşım önerisi:

# 1. Doğru metrik seç
# ❌ Accuracy (her zaman %99 → trivial)
# ✓ PR-AUC, F1-beta (β>1 recall ağırlıklı)

# 2. Model seviyesinde
model = lgb.LGBMClassifier(
    class_weight="balanced",   # Loss hesabında ağırlık
    # veya
    scale_pos_weight=99,       # XGBoost için: negative/positive oranı
)

# 3. Sampling (dikkatli!)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
# NOT: SMOTE sadece training set'e uygula, validation/test'e asla

# 4. Threshold optimizasyonu
# Default 0.5 değil, precision-recall tradeoff'una göre
thresholds = np.linspace(0.01, 0.99, 100)
f1_scores = [f1_score(y_val, (y_prob >= t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

### B3. Model Drift — Production'da Performans Düşüşü
*(→ Bkz. Katman D2: Churn pipeline monitoring; Katman B6: Data drift & train-serve skew)*

**Soru:** "Deploy ettiğin model 2 ayda AUC 0.87'den 0.73'e düştü. Diagnosis yapı."

```
Sistematik teşhis:

1. Data drift var mı?
   → PSI her feature için hesapla
   → Hangi feature'lar değişti?

2. Label drift var mı?
   → Prediction ortalaması değişti mi?
   → Gerçek churn oranı değişti mi?

3. Concept drift var mı?
   → P(y|X) değişti mi? (model doğru feature'ları alıyor ama ilişki değişti)
   → Dış faktörler: sezonluk, pazar değişimi, ürün değişikliği

4. Data pipeline sorunu var mı?
   → Bir feature kaynağı bozuldu mu?
   → NULL oranı arttı mı?

5. Düzeltme seçenekleri:
   a) Retrain: mevcut yapı, yeni veri
   b) Online learning: yeni örneklerle artımlı güncelleme
   c) Model mimarisini güncelle (feature set değişti)
   d) Segment modeli: drift belirli bir segmentte yoğunsa
```

### B4. Feature Importance Yanlış Yorumlama
*(→ Bkz. Katman C: Nedensellik ve A/B test; Katman B5: SHAP counter-example)*

**Soru:** "SHAP değeri yüksek olan feature'ı müşteriye 'değiştir' diyebilir miyiz?"

```
Cevap: HAYIR. SHAP korelasyonu gösterir, nedenselliği değil.

Örnek:
  Feature: "son_30_gün_login_sayısı" → SHAP yüksek (negatif ilişki)
  "Daha az giriş yapan = churn riski" — doğru gözlem

  AMA: "Login sayısını artırırsak churn düşer mi?"
  → Belki. Belki sadece ilgisiz kullanıcılar zaten az giriş yapıyordur.
  → Nedensellik için A/B test gerekir.

SHAP'ın söyleyebileceği:
  ✓ Model hangi feature'a ne kadar dikkat ediyor?
  ✓ Bu örnekte tahmin neden bu değer?
  ✓ Global trend: genel olarak nasıl bir ilişki?

SHAP'ın söyleyemeyeceği:
  ✗ Bu feature churn'e gerçekten yol açıyor mu?
  ✗ Bu feature'ı değiştirirsek ne olur?
```

### B5. SHAP'ın Yanıltıcı Olduğu Somut Durum (Counter-Example)
*(→ Bkz. Katman B4: SHAP yorumlama; Katman C: Nedensellik)*

**Soru:** "SHAP grafiğinde 'müşteri destek arama sayısı' en önemli feature. Destek hattını kapatırsak churn düşer mi?"

```
Senaryo:
  → Churn modeli eğitildi, SHAP global importance:
    1. support_calls_30d (en yüksek SHAP)
    2. days_since_last_purchase
    3. plan_type

  SHAP'a göre: Çok destek arayan = yüksek churn riski ✓ (korelasyon doğru)

  AMA yanıltıcı yorum:
  "Destek aramalarını azaltırsak churn düşer"
  → HAYIR. Destek araması churn'ün SONUCU, sebebi değil.
  → Gerçek neden: ürün kalitesi / bug'lar → kullanıcı destek arar → çözemezse churn

  Doğru okuma:
  → support_calls_30d bir "proxy variable" (vekil değişken)
  → Müdahale destek hattına değil, ürün kalitesine olmalı
  → Nedensellik testi: Destek kalitesini artırma A/B testi
    → Eğer destek kalitesi arttığında churn düşerse → nedensel ilişki var
    → Eğer düşmezse → sorun daha derinde (ürün-market uyumu)

  Genel kural:
  SHAP "model neye bakıyor?" sorusunu yanıtlar,
  "Neyi değiştirmeliyiz?" sorusunu yanıtlamaz.
  İkincisi için DAG (yönlü asiklik çizge) + müdahale deneyi gerekir.
```

### B6. "Model İyi Görünüyor Ama Production'da Başarısız" — Ek Senaryolar
*(→ Bkz. Katman B1: Data leakage; Katman B3: Model drift; Katman D2: Pipeline tasarımı)*

**Senaryo 1: Data Drift (Dağılım Kayması)**

```
Durum:
  → E-ticaret churn modeli, AUC=0.88 (test set)
  → 3 ay sonra production AUC=0.71

Teşhis:
  → Feature dağılımları karşılaştırıldı (PSI hesabı):
    • avg_order_value: PSI=0.35 (kırmızı alarm, >0.25 = ciddi kayma)
    • Neden? Pandemi sonrası ortalama sipariş tutarı %40 arttı
    • Model düşük tutarlara optimize edilmişti

  → Diğer drift kaynakları:
    • Yeni ödeme yöntemi (BNPL) eklendi → feature'da yok
    • Mobil uygulamadaki UI değişikliği kullanıcı davranışını değiştirdi

Çözüm:
  1. Kısa vade: Retrain (son 90 gün veriyle)
  2. Orta vade: PSI monitör + otomatik retrain trigger (PSI > 0.20 → alert)
  3. Uzun vade: Feature store'a yeni feature'lar ekle (ödeme yöntemi, platform)

Ders: Model "bozulmaz", dünya değişir. Monitoring olmadan model rotu çürür.
```

**Senaryo 2: Train-Serve Skew (Eğitim-Servis Çarpıklığı)**

```
Durum:
  → Offline AUC=0.91, online AUC=0.65
  → Aynı model, aynı ağırlıklar — fark nereden?

Teşhis:
  → Feature pipeline farklı:
    Training:                    Serving:
    ─────────                    ────────
    Python pandas ile            Java Spark ile
    hesaplanan feature'lar       hesaplanan feature'lar

  → Farklar:
    1. NULL handling: pandas NaN → 0, Spark NULL → model'e NaN geçiyor
    2. Timestamp zone: Training UTC, serving local time
    3. Aggregation window: Training'de tam 30 gün,
       serving'de "son 30 gün" ama event gecikmesi nedeniyle 28 gün

Çözüm:
  1. Feature pipeline'ı TEK kaynak yap (dbt/Feast/Tecton)
  2. Training ve serving aynı kodu çalıştırmalı
  3. Integration test: Training feature ↔ serving feature karşılaştır
     → Her deploy öncesi 1000 örnek için feature değerlerini karşılaştır
     → Fark > %1 ise deploy'u blokla

Ders: Model problemi değil, mühendislik problemi.
  "Aynı model" farklı veriyle farklı sonuç verir.
  Feature parity testi CI/CD pipeline'ına eklenmelidir.
```

### B7. Cold Start Problemi — Öneri Sistemi
*(→ Bkz. Katman D4: RecSys mimarisi; Katman F2: Online/offline öneri sistemi tasarımı)*

**Soru:** "Yeni kullanıcı ve yeni ürün için öneri nasıl yaparsınız? (cold start)"

```
Cold start iki boyutlu bir sorundur:
  1. Yeni kullanıcı: Hiç geçmişi yok, collaborative filtering çalışmaz.
  2. Yeni ürün:     Hiç etkileşim almamış, embedding öğrenemez.

── Yeni Kullanıcı Stratejileri ──────────────────────────────────────

1. Popularity-based fallback (en basit, güçlü baseline):
   → Tüm kullanıcılara trending / en çok satan ürünleri göster.
   → Segmentlere böl: yeni kullanıcı ülkesi, cihaz türü, kayıt kanalına
     göre özelleştirilmiş "popüler" listesi.
   → Avantaj: Her zaman çalışır, sıfır latency.
   → Dezavantaj: Kişisel değil, filter bubble riski.

2. Onboarding soruları (explicit feedback):
   → Kayıt sırasında "Hangi kategoriler ilginizi çekiyor?" sorusu.
   → 3-5 kategori seçimi → anında içerik tabanlı öneri başlat.
   → Netflix tarzı: ilk giriş ekranında tür seçimi.

3. Content-based filtering (metadata embedding):
   → Kullanıcı profil bilgileri (yaş grubu, konum, referral source)
     → benzer profildeki kullanıcıların beğendikleri.
   → Ürün metadata'sı (kategori, fiyat aralığı, marka) → TF-IDF veya
     BERT embedding ile kullanıcı tercihine benzer ürünler.

4. Transfer learning / cross-domain:
   → Kullanıcı başka bir platformdan (login via Google) geliyorsa
     demografik bilgi çıkar.
   → Aynı şirketin farklı ürününden davranış transferi.

── Yeni Ürün Stratejileri ───────────────────────────────────────────

1. Content-based bootstrap:
   → Ürün açıklaması, kategorisi, fiyatı, görseli → embedding üret.
   → Benzer embedding'li ürünlerin aldığı kullanıcılara göster.

2. Explore-exploit (epsilon-greedy veya Thompson Sampling):
   → Yeni ürünü %10 oranında rastgele kullanıcılara göster (explore).
   → Yeterli tıklama/satın alma birikince normal ranker devreye girer.

3. Hybrid öneri mimarisi:

   def recommend(user_id, n=10):
       history = get_user_history(user_id)
       if len(history) >= MIN_INTERACTIONS:          # warm user
           return collaborative_filter(user_id, n)
       elif has_profile(user_id):                    # cold user + profil
           return content_based(user_id, n)
       else:                                         # tam cold
           return popularity_based(segment=get_segment(user_id), n=n)

── Değerlendirme ────────────────────────────────────────────────────

  Cold start kullanıcıları ayrı segment olarak izle:
  → İlk 7 gün retention oranı (cold vs warm user)
  → Click-through rate farkı
  → Kaçıncı etkileşimden sonra kişiselleşme başlıyor? (ramp-up curve)

Senior Notu:
  Cold start bir "başlangıç sorunu" değil, sürekli bir sorundur.
  Kullanıcıların %20-30'u herhangi bir anda "cold" sayılabilir
  (seyrek kullanıcılar, yeni segment, mevsimsel kullanıcılar).
  Sistemi bu oran düşünülerek tasarla.
```

### B8. Concept Drift — Ne Zaman Yeniden Eğit?
*(→ Bkz. Katman E3: Model monitoring; Katman B6: Production'da başarısız model)*

**Soru:** "Model ne zaman yeniden eğitilmeli? Nasıl karar verirsin?"

```
Drift türleri önce netleştirilmeli:

  1. Data drift (covariate shift): P(X) değişti, P(Y|X) aynı.
     → Feature dağılımları kaydı.
     → Örnek: Müşteri tabanı genişledi, yeni demografi geldi.

  2. Concept drift: P(Y|X) değişti.
     → Gerçek ilişki değişti.
     → Örnek: COVID → "seyahat" feature'ı anlamını yitirdi.

  3. Label shift: P(Y) değişti.
     → Pozitif/negatif oranı kaydı.

── Tespit Yöntemleri ────────────────────────────────────────────────

1. Population Stability Index (PSI) — feature drift için:
   PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
   → PSI < 0.10: stabil (yeşil)
   → 0.10 ≤ PSI < 0.25: hafif kayma (sarı, izle)
   → PSI ≥ 0.25: ciddi kayma (kırmızı, alarm)

2. KS testi (Kolmogorov-Smirnov):
   from scipy import stats
   stat, p = stats.ks_2samp(train_feature, prod_feature)
   if p < 0.05:
       print("Drift tespit edildi!")

3. Performans monitörü (label varsa):
   → Günlük/haftalık AUC, F1, precision@k izle.
   → Baseline'dan %5 düşüş → sarı alarm.
   → %10 düşüş → kırmızı alarm, retrain tetikle.

4. Evidently AI (açık kaynak):
   from evidently.report import Report
   from evidently.metric_preset import DataDriftPreset
   report = Report(metrics=[DataDriftPreset()])
   report.run(reference_data=train_df, current_data=prod_df)
   report.save_html("drift_report.html")

── Retrain Stratejileri ─────────────────────────────────────────────

Scheduled retraining (takvim bazlı):
  → Her hafta / her ay sabit aralıkla retrain.
  → Basit, öngörülebilir. Drift olmasa bile retrain eder (maliyet).
  → Yavaş değişen sistemler için uygundur.

Triggered retraining (tetiklemeli):
  → PSI veya performans eşiği aşılınca retrain.
  → Daha verimli, ama tespit gecikmesi riski var.
  → Gerçek zamanlı kritik sistemler için tercih edilir.

Hybrid (önerilen):
  → Scheduled + trigger: "En az ayda bir, ama drift tespitinde hemen."

── Champion-Challenger Pattern ──────────────────────────────────────

  Mevcut model (champion) → %90 trafik
  Yeni retrained model (challenger) → %10 trafik
  → 1-2 hafta A/B test
  → Challenger kazanırsa otomatik promote

  Avantaj: Production'a güvenle geçiş. Kötü model tüm trafiği etkilemez.

── Gradual Rollout ───────────────────────────────────────────────────

  Yeni model: %1 → %5 → %20 → %50 → %100
  Her aşamada metrik kontrol, sorun yoksa bir sonraki aşama.
  Canary deployment ile birleştirilebilir.

Senior Notu:
  Retrain ≠ fix. Eğer concept drift yapısal ise (iş modeli değişti,
  pandemi vb.) eski veriyi atmak gerekebilir. "Ne kadar geriye git?"
  sorusunun cevabı drift hızına bağlıdır. Expanding window değil,
  sliding window dene.
```

### B9. Production'da Fairness Bug
*(→ Bkz. Katman G2: Etik ve sorumluluk; Katman B2: Imbalanced classification)*

**Soru:** "Deploy ettiğin modelin belirli bir demografik grupta %30 daha yüksek false positive oranı oluşturduğunu keşfettin. Ne yaparsın?"

```
Acil triage adımları (ilk 2 saat):

  1. Ciddiyeti doğrula:
     → Tek bir gün mi, sürekli mi? (noise vs gerçek sorun)
     → Hangi grup, kaç kullanıcı etkileniyor?
     → İş etkisi: Para kaybı, yasal risk, reputasyon hasarı?

  2. Modeli durdur veya fallback'e geç:
     → Eğer yüksek risk (kredi, işe alım, sağlık): modeli durdur.
     → Eğer düşük risk: Izlemeye devam, düzeltme planla.

  3. Stakeholder bildirimi:
     → Legal / compliance ekibi (GDPR, EEOC gibi düzenlemeler)
     → Ürün yöneticisi ve etkilenen ekipler
     → Üst yönetim (gerekirse)

── Fairness Metrikleri ──────────────────────────────────────────────

  Demografik Eşitlik (Demographic Parity):
    P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
    → Her grupta pozitif tahmin oranı eşit olmalı.

  Eşitlenmiş Şans (Equalized Odds):
    FPR ve TPR her grupta eşit olmalı.
    → Daha güçlü kısıt, çoğu durumda tercih edilir.

  Fırsat Eşitliği (Equal Opportunity):
    TPR (recall) her grupta eşit olmalı.
    → Hassas uygulamalar için (tıbbi tanı, kredi verme).

  Python ile hesap:
  from sklearn.metrics import confusion_matrix

  def group_fpr(y_true, y_pred, group_mask):
      tn, fp, fn, tp = confusion_matrix(
          y_true[group_mask], y_pred[group_mask]
      ).ravel()
      return fp / (fp + tn)

  fpr_a = group_fpr(y_true, y_pred, group == 'A')
  fpr_b = group_fpr(y_true, y_pred, group == 'B')
  print(f"FPR farkı: {abs(fpr_a - fpr_b):.3f}")  # hedef < 0.05

── Düzeltme Stratejileri ────────────────────────────────────────────

1. Grup başına eşik (threshold per group) — en hızlı çözüm:
   → Etkilenen grupta eşiği düşür (örn. 0.5 → 0.35)
   → FPR düşer ama TPR de değişir, dikkatli calibrate et.
   → Dezavantaj: Yasal açıdan "differential treatment" sayılabilir.

2. Veri yeniden ağırlıklandırma (reweighting):
   → Eksik temsil edilen grubu up-weight ederek retrain.
   from sklearn.utils.class_weight import compute_sample_weight
   weights = compute_sample_weight('balanced', y=group_labels)

3. Fairness-aware algoritmalar (fairlearn):
   from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
   mitigator = ExponentiatedGradient(
       estimator=base_model,
       constraints=EqualizedOdds()
   )
   mitigator.fit(X_train, y_train, sensitive_features=A_train)

4. Upstream çözüm — veri kalitesi:
   → Etiket gürültüsü var mı? (belirli grupta yanlış etiket fazla mı?)
   → Feature'larda proxy discrimination var mı?
     (posta kodu → etnik grup proxy'si olabilir)

── Post-mortem ──────────────────────────────────────────────────────

  → Neden deployment öncesi tespit edilmedi?
  → Fairness testi CI/CD'ye ekle (bias check as a gate)
  → Model kartı (model card) güncelle — bilinen kısıtlamaları belgele
  → Fairness metriğini production dashboard'a ekle

Senior Notu:
  Fairness tek bir metrik değildir. Demographic parity ve equalized
  odds matematiksel olarak aynı anda sağlanamaz (Impossibility Theorem,
  Chouldechova 2017). Hangi metriğin öncelikli olduğuna iş + etik +
  hukuk ekibiyle birlikte karar ver. Teknik çözüm tek başına yeterli değil.
```

### B10. Feature Store Tasarımı
*(→ Bkz. Katman F2: Feature store mimarisi; Katman E2: MLOps pipeline)*

**Soru:** "Bir e-ticaret platformunda feature store nasıl tasarlarsın?"

```
Feature store neden gerekli?
  → Aynı feature farklı ekiplerce farklı hesaplanıyor → tutarsızlık
  → Training-serving skew (B6 senaryosu) → performans kaybı
  → Feature hesaplama maliyeti tekrar tekrar ödeniyor → israf
  → Yeni model geliştirme yavaş → feature keşfedilemez

── Mimari: Offline + Online Store ───────────────────────────────────

  ┌─────────────────────────────────────────────────────────┐
  │                   Feature Pipeline                       │
  │  Raw Events → dbt/Spark → Feature hesaplama             │
  └────────────────┬───────────────────────────────────────-┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
   Offline Store          Online Store
   (S3 / Data Warehouse)  (Redis / DynamoDB)
   Batch training için    Real-time serving için
   Point-in-time query    < 5ms latency
   Terabayt ölçek         Gigabayt ölçek

Offline Store:
  → Historical feature değerleri, timestamp ile birlikte
  → Point-in-time correctness kritik:
    "Model eğitiminde, o anki bilgiyi kullan — gelecek bilgiyi değil"
  → Örnek: 2024-01-15'teki kullanıcı segmenti, o günkü değer olmalı
    (bugünkü değer değil — data leakage kaynağı!)

Online Store:
  → En güncel feature değerleri (son N dakika)
  → Düşük latency: Redis Sorted Sets, Cassandra, DynamoDB
  → TTL (time-to-live) ile eski değerler otomatik temizlenir

── Feast ile Somut Uygulama ─────────────────────────────────────────

  # Feature tanımlama
  from feast import Entity, Feature, FeatureView, ValueType
  from feast.infra.offline_stores.file_source import FileSource

  user_entity = Entity(name="user_id", value_type=ValueType.INT64)

  user_source = FileSource(
      path="s3://data-lake/user_features/",
      timestamp_field="event_timestamp",
  )

  user_feature_view = FeatureView(
      name="user_features",
      entities=["user_id"],
      ttl=timedelta(days=7),
      features=[
          Feature(name="total_orders_30d", dtype=ValueType.INT64),
          Feature(name="avg_order_value_30d", dtype=ValueType.FLOAT),
          Feature(name="churn_score", dtype=ValueType.FLOAT),
      ],
      source=user_source,
  )

  # Training için point-in-time join
  training_df = store.get_historical_features(
      entity_df=entity_df,  # user_id + event_timestamp içerir
      features=["user_features:total_orders_30d",
                "user_features:avg_order_value_30d"],
  ).to_df()

  # Serving için online lookup
  feature_vector = store.get_online_features(
      features=["user_features:churn_score"],
      entity_rows=[{"user_id": 12345}],
  ).to_dict()

── Feature Freshness SLA ────────────────────────────────────────────

  Feature tipi          | Freshness hedefi | Güncelleme sıklığı
  ──────────────────────|──────────────────|───────────────────
  Anlık sepet içeriği   | < 1 saniye       | Event-driven (Kafka)
  Son 1 saat aktivite   | < 5 dakika       | Micro-batch (Spark)
  Son 30 gün toplamları | < 1 gün          | Günlük batch (dbt)
  Kullanıcı segmenti    | < 1 hafta        | Haftalık batch

── Backfill Stratejisi ───────────────────────────────────────────────

  Yeni feature eklendiğinde geçmiş için hesaplama:
  1. Full backfill: Tüm tarihi veri için hesapla (pahalı, tek seferlik)
  2. Incremental: Partition by date, sadece eksik günleri hesapla
  3. On-demand: Training sırasında lazy hesapla

  Backfill tamamlanmadan modeli eğitme!
  → Eksik değerler imputation'a giderse feature anlamsızlaşır.

Senior Notu:
  Feature store'un değeri, içindeki feature'ların kalitesiyle ölçülür.
  Teknik altyapı hazır olsa da "feature keşif kültürü" olmadan
  feature store boş bir raf olur. Her feature için:
  owner, description, computation logic, freshness SLA belgele.
  Feature catalog (Datahub, Amundsen) ile entegre et.
```

### B11. Multi-Label Classification
*(→ Bkz. Katman B2: Imbalanced classification; Katman D1: Metin sınıflandırma)*

**Soru:** "Her ürüne birden fazla etiket tahmin eden bir model nasıl kurarsın? (örn. haber kategorisi)"

```
Problem tanımı:
  Girdi: Haber metni
  Çıktı: {Teknoloji, Siyaset, Spor, Ekonomi, ...} içinden 0 veya daha fazla etiket

  Örnek:
  "Apple yeni çip teknolojisiyle rekor satış açıkladı"
  → [Teknoloji ✓, Ekonomi ✓, Siyaset ✗, Spor ✗]

── Problem Çerçeveleme Yöntemleri ───────────────────────────────────

1. Binary Relevance (en yaygın):
   → Her etiket için bağımsız binary sınıflandırıcı.
   → L etiket → L ayrı model.
   → Avantaj: Basit, paralel eğitilebilir, yorumlanabilir.
   → Dezavantaj: Etiketler arası korelasyonu görmez.

   from sklearn.multiclass import OneVsRestClassifier
   from sklearn.linear_model import LogisticRegression
   clf = OneVsRestClassifier(LogisticRegression())
   clf.fit(X_train, y_train)  # y_train: multilabel binary matrix

2. Classifier Chain (etiket korelasyonunu yakalar):
   → Etiket 1 tahmin → Etiket 2 = f(X, Etiket_1) → ...
   → Sıralama önemli, ensemble ile çözülür.
   from sklearn.multioutput import ClassifierChain
   chain = ClassifierChain(LogisticRegression(), order='random', cv=5)

3. Label Powerset (nadiren, küçük etiket seti için):
   → Etiket kombinasyonlarını tek sınıf olarak gör.
   → 3 etikette 2^3=8 sınıf → 10 etikette 1024 sınıf (patlar).

4. Neural (önerilen büyük ölçekte):
   → Tek model, çoklu sigmoid çıktı.
   import torch.nn as nn
   class MultiLabelClassifier(nn.Module):
       def __init__(self, n_labels):
           super().__init__()
           self.bert = BertModel.from_pretrained('bert-base-uncased')
           self.classifier = nn.Linear(768, n_labels)
           self.sigmoid = nn.Sigmoid()

       def forward(self, input_ids, attention_mask):
           outputs = self.bert(input_ids, attention_mask)
           return self.sigmoid(self.classifier(outputs.pooler_output))

   Loss: BCEWithLogitsLoss (etiket başına binary cross-entropy)

── Değerlendirme Metrikleri ─────────────────────────────────────────

  from sklearn.metrics import f1_score, hamming_loss

  # Hamming Loss: yanlış etiket oranı (düşük = iyi)
  hl = hamming_loss(y_true, y_pred)
  print(f"Hamming Loss: {hl:.4f}")  # 0.0 mükemmel, 1.0 en kötü

  # Micro F1: tüm (örnek, etiket) çiftlerini dengeler
  # Nadir etiketler dezavantajlı → nadir ama önemli etiket varsa dikkat
  micro_f1 = f1_score(y_true, y_pred, average='micro')

  # Macro F1: her etiket eşit ağırlık
  # Nadir etiketleri de önemser → imbalanced label seti için tercih
  macro_f1 = f1_score(y_true, y_pred, average='macro')

  # Samples F1: örnek bazında hesap (çok etiketli için en anlamlı)
  samples_f1 = f1_score(y_true, y_pred, average='samples')

── Etiket Başına Eşik Optimizasyonu ─────────────────────────────────

  Global 0.5 eşiği çoğu zaman suboptimal:
  → Nadir etiket için 0.3, baskın etiket için 0.7 daha iyi olabilir.

  from sklearn.metrics import f1_score
  import numpy as np

  def optimize_threshold(y_true, y_prob):
      thresholds = np.arange(0.1, 0.9, 0.05)
      best_thresholds = []
      for i in range(y_true.shape[1]):
          best_t, best_f1 = 0.5, 0
          for t in thresholds:
              f1 = f1_score(y_true[:, i], y_prob[:, i] > t, zero_division=0)
              if f1 > best_f1:
                  best_f1, best_t = f1, t
          best_thresholds.append(best_t)
      return best_thresholds

Senior Notu:
  Etiket dağılımını mutlaka incele. Nadir etiketler (örneklerin %1'i)
  için binary relevance modeli tahmin bile üretmeyebilir. Class weight
  veya focal loss kullan. Etiket sayısı 50+ ise hierarchical label
  yapısı (üst kategori → alt kategori) modeli hem daha iyi hem daha
  yorumlanabilir yapar.
```

### B12. Anomaly Detection Sistemi
*(→ Bkz. Katman B2: Imbalanced classification; Katman F1: Gerçek zamanlı sistem tasarımı)*

**Soru:** "Kredi kartı işlemlerinde anomali tespiti için sistem tasarla."

```
Zorluklar:
  → Etiket yok veya çok az (hileli işlemlerin %0.1'i etiketli)
  → Gerçek zamanlı karar: < 100ms latency
  → Sürekli değişen saldırı pattern'ları (adversarial drift)
  → False positive maliyeti yüksek (meşru müşteri engellenir)

── Denetimli vs Denetimsiz Yaklaşım ─────────────────────────────────

  Denetimli (labeled fraud varsa):
  → XGBoost, LightGBM ile sınıflandırma
  → Avantaj: Yüksek precision/recall (etiket kaliteli ise)
  → Dezavantaj: Yeni fraud pattern'larını kaçırır (known unknowns)

  Denetimsiz (label yok):
  → Isolation Forest, Autoencoder, LOF, One-Class SVM
  → Avantaj: Zero-day fraud (hiç görülmemiş pattern) yakalar
  → Dezavantaj: Threshold ayarı zor, false positive yüksek

  Önerilen: Ensemble (ikisi birlikte)

── Isolation Forest ─────────────────────────────────────────────────

  Sezgi: Anomaliler rastgele bölünmede hızla izole edilir.

  from sklearn.ensemble import IsolationForest
  import numpy as np

  iso_forest = IsolationForest(
      n_estimators=100,
      contamination=0.01,  # Beklenen anomali oranı (%1)
      random_state=42
  )
  iso_forest.fit(X_train_normal)  # Sadece normal işlemlerle eğit

  scores = iso_forest.decision_function(X_test)
  # Düşük skor = anormal
  # -0.5 altı genellikle anomali

── Autoencoder ile Anomali Tespiti ──────────────────────────────────

  Sezgi: Normal işlemleri yeniden oluşturmayı öğren.
  Anomaliler = yüksek reconstruction error.

  import torch
  import torch.nn as nn

  class TransactionAutoencoder(nn.Module):
      def __init__(self, input_dim=50, latent_dim=8):
          super().__init__()
          self.encoder = nn.Sequential(
              nn.Linear(input_dim, 32),
              nn.ReLU(),
              nn.Linear(32, latent_dim),
          )
          self.decoder = nn.Sequential(
              nn.Linear(latent_dim, 32),
              nn.ReLU(),
              nn.Linear(32, input_dim),
          )

      def forward(self, x):
          z = self.encoder(x)
          return self.decoder(z)

  # Eğitim: sadece normal işlemler
  # Anomali skoru: MSE(input, reconstruction)
  def anomaly_score(model, x):
      with torch.no_grad():
          recon = model(x)
      return ((x - recon) ** 2).mean(dim=1)

── Eşik Kalibrasyonu ────────────────────────────────────────────────

  Label olmadan eşik nasıl ayarlanır?

  1. Domain knowledge: "Günde 100 işlemi manuel inceleme kapasitemiz var"
     → Top-100 highest score → %X percentile bul → eşik yap.

  2. Precision@k: Analistler k işlemi inceliyor → bunların kaçı gerçek fraud?
     → Precision@100 = 0.30 → Kabul edilebilir mi? İş ile karar ver.

  3. Cost-based optimization:
     FP maliyeti: Müşteri şikayeti, destek maliyeti = 5 TL
     FN maliyeti: Kaçan fraud = ortalama 250 TL
     → Threshold'u minimize et: E[maliyet] = FP×5 + FN×250

── Feedback Loop ────────────────────────────────────────────────────

  Analist kararları → model güncelleme döngüsü:
  → Analist "fraud" işaretlerse → labeled veri birikiyor
  → 3 ayda bir supervised model ile karşılaştır
  → Yeterli label birikince supervised model devreye al

  Dikkat: Analistler sadece yüksek skorlu işlemleri görüyor
  → Selection bias! Düşük skorlu ama gerçek fraud'ları görmüyor.
  → Çözüm: %1 rastgele örneklemi analist incelemesine gönder.

Senior Notu:
  Anomali tespiti "model kurma" değil, "karar sistemi kurma" sorunudur.
  Modelin skoru ham girdi, son karar kural motoru + iş logik ile verilir.
  Eşik = sabitleme, açıklama, gözden geçirme döngüsü gerektirir.
  Model kartına: false positive oranı, incelenen işlem başı maliyet,
  son değerlendirme tarihi yazılmalı.
```

### B13. Öneri Sistemi — Diversity vs Accuracy Trade-off
*(→ Bkz. Katman D4: RecSys; Katman B7: Cold start problemi)*

**Soru:** "Kullanıcılar hep aynı tür içerik öneriliyor diye şikayet ediyor. Ne yaparsın?"

```
Sorun: Filter Bubble (filtre balonu)
  → Collaborative filtering sadece geçmiş davranışa bakıyor.
  → Beğendiğin içerikle benzer içerik öneriyor.
  → Sonuç: Kullanıcı giderek daha dar bir içerik alanında hapsolur.
  → Uzun vadede: Engagement düşüşü, churn artışı.

── Diversity Metrikleri ─────────────────────────────────────────────

  Intra-List Diversity (ILD):
  → Bir kullanıcıya gösterilen N öneri arasındaki ortalama mesafe.
  → Yüksek ILD = çeşitli öneri listesi.

  def intra_list_diversity(item_embeddings, recommended_ids):
      embeddings = item_embeddings[recommended_ids]
      n = len(recommended_ids)
      total_dist = 0
      for i in range(n):
          for j in range(i+1, n):
              dist = 1 - cosine_similarity(
                  embeddings[i].reshape(1,-1),
                  embeddings[j].reshape(1,-1)
              )[0][0]
              total_dist += dist
      return total_dist / (n * (n - 1) / 2)

  Coverage:
  → Toplam önerilen unique item sayısı / katalog büyüklüğü
  → Sistemin "keşfettiği" alan genişliği

  Serendipity:
  → Kullanıcının beklemeyeceği ama beğeneceği öneriler.
  → Ölçmek zor: Explicit "sürpriz mıydı?" anketi gerektirebilir.

── Maximal Marginal Relevance (MMR) ─────────────────────────────────

  Greedy algoritma: Her adımda hem relevance hem diversity optimize et.

  MMR skoru:
  score(i) = λ × relevance(i) - (1-λ) × max_similarity(i, selected)

  → λ = 1.0: Sadece relevance (standart öneri)
  → λ = 0.5: Dengeli (önerilen başlangıç değeri)
  → λ = 0.0: Sadece diversity

  def mmr_rerank(candidates, selected, item_embeddings, lambda_=0.5):
      scores = {}
      for item in candidates:
          relevance = item.score
          if selected:
              max_sim = max(
                  cosine_similarity(
                      item_embeddings[item.id].reshape(1,-1),
                      item_embeddings[s.id].reshape(1,-1)
                  )[0][0]
                  for s in selected
              )
          else:
              max_sim = 0
          scores[item.id] = lambda_ * relevance - (1-lambda_) * max_sim
      return max(scores, key=scores.get)

── Exploration-Exploitation Stratejileri ────────────────────────────

  Epsilon-greedy öneride:
  → %90 en iyi öneriyi göster (exploit)
  → %10 rastgele farklı kategoriden öneri ekle (explore)
  → Basit ama etkili başlangıç noktası

  Thompson Sampling (Bayesian bandit):
  → Her kategori için Beta dağılımı tut (tıklanma/tıklanmama)
  → Posterior'dan örnekle, en yüksek örneklenen kategoriyi seç
  → Doğal exploration-exploitation dengesi

  Diversity-aware bandits:
  → Ödül = accuracy + diversity_bonus
  → diversity_bonus = ILD(son 10 öneri) × ağırlık

── A/B Test Tasarımı ────────────────────────────────────────────────

  Hipotez: MMR ile (λ=0.6) çeşitlilik artar, engagement düşmez.

  Birincil metrik: 7-günlük retention (churn metriğini ölçer)
  İkincil metrikler:
    → Günlük aktif kullanıcı başına tıklama sayısı
    → Farklı kategori sayısı (per-session category diversity)
    → Oturum uzunluğu

  Dikkat: CTR tek başına yanıltıcı.
  Çeşitli öneri listesinde CTR düşebilir ama retention artabilir.
  → Doğru metrik: uzun vadeli LTV (customer lifetime value).

Senior Notu:
  Diversity ile accuracy arasındaki trade-off kullanıcı segmentine göre
  değişir. Yeni kullanıcılar keşif ister, power user'lar kaliteli
  içerik ister. Kişiselleştirilmiş λ değeri (her kullanıcıya farklı
  diversity ağırlığı) son adım olarak uygulanabilir.
```

### B14. NLP Model Deployment — Latency vs Accuracy
*(→ Bkz. Katman E2: Model servis etme; Katman D2: NLP modelleri)*

**Soru:** "BERT modelini production'a alacaksın, 200ms latency SLA var ama model 500ms. Ne yaparsın?"

```
Önce profil çıkar:
  → 500ms nerede harcanıyor?
  → Model inference: 400ms
  → Tokenizer: 30ms
  → Network/preprocessing: 70ms

  Bottleneck = inference → buraya odaklan.

── Strateji 1: Quantization (INT8) ──────────────────────────────────

  Sezgi: 32-bit float yerine 8-bit integer → 4x küçük model, ~3x hızlı.

  from transformers import AutoModelForSequenceClassification
  import torch

  model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

  # Dynamic quantization (en kolay, CPU için)
  quantized_model = torch.quantization.quantize_dynamic(
      model,
      {torch.nn.Linear},
      dtype=torch.qint8
  )
  # Sonuç: ~3x hızlanma, <1% accuracy kaybı (genellikle)

  # ONNX + INT8 (GPU için de çalışır)
  # ort.quantization.quantize_dynamic() ile ONNX modelini quantize et

── Strateji 2: Knowledge Distillation ──────────────────────────────

  Sezgi: Büyük model (teacher) → küçük modeli (student) eğitir.
  Student, teacher'ın davranışını taklit eder.

  DistilBERT:
  → BERT'in %40 daha küçük, %60 daha hızlı versiyonu
  → GLUE benchmark'ta BERT'in %97 performansı
  → Hugging Face'den hazır: "distilbert-base-uncased"

  from transformers import DistilBertForSequenceClassification
  student = DistilBertForSequenceClassification.from_pretrained(
      "distilbert-base-uncased"
  )
  # Domain-specific: Önce distilbert-base ile fine-tune, teacher soft
  # labellarıyla distillation loss ekle

  Hiyerarşi (hız ↑, accuracy ↓):
  BERT-large → BERT-base → DistilBERT → TinyBERT → MobileBERT

── Strateji 3: ONNX Export ──────────────────────────────────────────

  PyTorch → ONNX → ONNXRuntime (2-3x hızlanma)

  import torch
  from transformers import AutoTokenizer, AutoModel

  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased")

  dummy_input = tokenizer("test", return_tensors="pt")
  torch.onnx.export(
      model,
      tuple(dummy_input.values()),
      "bert.onnx",
      opset_version=13,
      input_names=['input_ids', 'attention_mask'],
      output_names=['last_hidden_state'],
      dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'}}
  )

  import onnxruntime as ort
  sess = ort.InferenceSession("bert.onnx",
         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
  # ~2x hızlanma CPU'da, GPU'da daha fazla

── Strateji 4: Batching ve Async ────────────────────────────────────

  Dynamic batching (Triton Inference Server veya TorchServe):
  → Gelen istekleri ~5ms beklet, biriktir, batch olarak çalıştır.
  → 10 isteği ayrı ayrı: 10 × 500ms = 5000ms toplam GPU süresi
  → 10 isteği batch: ~600ms toplam GPU süresi = 10x throughput artışı
  → Latency biraz artar ama P99 iyileşir (queuing azalır)

  Async inference:
  → FastAPI + asyncio: I/O bound kısımlar (tokenize, pre/post process)
    non-blocking yap
  → Model inference CPU-bound: thread pool executor'e at

── Strateji 5: Fallback Katmanı ─────────────────────────────────────

  Basit sorgular için BERT'e gerek yok:

  def predict(text):
      if is_simple_query(text):           # kısa, net sorgular
          return tfidf_classifier(text)   # ~5ms
      elif medium_complexity(text):
          return distilbert(text)          # ~80ms
      else:
          return full_bert(text)           # ~500ms

  Routing logic: Güven skoru + query uzunluğu + kelime dağarcığı

── Sonuç: Latency Bütçesi ──────────────────────────────────────────

  Strateji kombinasyonu ile hedef:
  ONNX export:          500ms → 250ms  (-50%)
  INT8 quantization:    250ms → 100ms  (-60%)
  Dynamic batching:     P99 → 150ms   (kuyruk azalır)
  Fallback (%30 istek): Ortalama ~80ms

  → P50 latency: ~80ms ✓
  → P99 latency: ~180ms (SLA: 200ms) ✓

Senior Notu:
  Latency optimizasyonu accuracy-latency Pareto frontier problemidir.
  Her optimizasyon adımını A/B test veya shadow deployment ile doğrula.
  %2 accuracy kaybı kabul edilebilir mi? Bu iş kararı. Offline metrik
  (F1) yerine online metrik (CTR, dönüşüm) üzerinde etkiye bak.
```

### B15. Zaman Serisi Tahmin — Sezonellik ve Trend
*(→ Bkz. Katman B4: Feature engineering; Katman C3: Backtesting)*

**Soru:** "E-ticaret satış tahmini yapman gerekiyor. Sezonellik ve trend nasıl ele alırsın?"

```
Zaman serisi bileşenleri:
  Satış(t) = Trend(t) + Sezonellik(t) + Tatil_Etkisi(t) + Gürültü(t)

  → Trend: Uzun vadeli artış/azalış (yıllık %15 büyüme)
  → Sezonellik: Tekrar eden pattern (haftalık: Cuma > Pazartesi,
                 yıllık: Kasım-Aralık zirve)
  → Tatil etkisi: Kara Cuma, Ramazan, Yılbaşı
  → Gürültü: Açıklanamayan varyans

── STL Decomposition ────────────────────────────────────────────────

  import pandas as pd
  from statsmodels.tsa.seasonal import STL
  import matplotlib.pyplot as plt

  # Günlük satış verisi
  stl = STL(sales_series, period=7, robust=True)
  result = stl.fit()

  # Bileşenleri incele
  result.plot()
  plt.show()

  # Trend güçlü mü? Sezonellik ne kadar büyük?
  trend_strength = 1 - result.resid.var() / (result.trend + result.resid).var()
  seasonal_strength = 1 - result.resid.var() / (result.seasonal + result.resid).var()
  print(f"Trend gücü: {trend_strength:.2f}")    # >0.6 = güçlü trend
  print(f"Sezonellik: {seasonal_strength:.2f}") # >0.6 = güçlü sezonellik

── Prophet ile Tahmin ───────────────────────────────────────────────

  from prophet import Prophet
  import pandas as pd

  # Prophet formatı: 'ds' (datetime) + 'y' (değer)
  df = sales_df.rename(columns={'date': 'ds', 'sales': 'y'})

  model = Prophet(
      changepoint_prior_scale=0.05,   # Trend esnekliği (düşük = stabil)
      seasonality_prior_scale=10,     # Sezonellik kuvveti
      yearly_seasonality=True,
      weekly_seasonality=True,
      daily_seasonality=False,        # Günlük veri için False
  )

  # Tatil günleri ekle (kritik!)
  from prophet.make_holidays import make_holidays_df
  tr_holidays = make_holidays_df(year_list=[2023, 2024, 2025], country='TR')
  model.add_country_holidays(country_name='TR')

  # Ek regressor: Kampanya
  model.add_regressor('is_campaign_day')
  df['is_campaign_day'] = (df['ds'].isin(campaign_dates)).astype(int)

  model.fit(df)
  future = model.make_future_dataframe(periods=90)  # 90 gün tahmin
  forecast = model.predict(future)

  # Tahmin aralıkları otomatik gelir (yhat_lower, yhat_upper)

── Feature Engineering Yaklaşımı (ML modeli için) ───────────────────

  Zaman serisi özelliklerini manuel oluştur → XGBoost / LightGBM ile:

  def create_time_features(df):
      df = df.copy()
      df['hour'] = df['ds'].dt.hour
      df['dayofweek'] = df['ds'].dt.dayofweek
      df['month'] = df['ds'].dt.month
      df['quarter'] = df['ds'].dt.quarter
      df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
      df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)

      # Lag features (geçmiş değerler)
      for lag in [1, 7, 14, 28]:
          df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

      # Rolling features
      for window in [7, 14, 28]:
          df[f'sales_roll_mean_{window}'] = (
              df['sales'].shift(1).rolling(window).mean()
          )
          df[f'sales_roll_std_{window}'] = (
              df['sales'].shift(1).rolling(window).std()
          )

      # Fourier features (sezonellik için)
      for k in range(1, 4):
          df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['dayofyear'] / 365.25)
          df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['dayofyear'] / 365.25)

      return df

── Değerlendirme Metrikleri ─────────────────────────────────────────

  # MAPE (Mean Absolute Percentage Error) — iş iletişimi için
  def mape(y_true, y_pred):
      return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  # Dikkat: y_true=0 olursa sonsuz hata → WAPE tercih et

  # WAPE (Weighted Absolute Percentage Error) — daha robust
  def wape(y_true, y_pred):
      return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

  # RMSE — büyük hataları daha çok cezalandırır
  from sklearn.metrics import mean_squared_error
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))

── Backtesting (doğru değerlendirme) ────────────────────────────────

  Zaman serisi cross-validation — forward chaining:

  from sklearn.model_selection import TimeSeriesSplit
  tscv = TimeSeriesSplit(n_splits=5, gap=7)  # gap=7: 1 hafta boşluk

  # Fold 1: Train [0:100]   Test [107:121]
  # Fold 2: Train [0:200]   Test [207:221]
  # ...

  for train_idx, test_idx in tscv.split(X):
      model.fit(X[train_idx], y[train_idx])
      preds = model.predict(X[test_idx])
      print(f"WAPE: {wape(y[test_idx], preds):.1f}%")

  ÖNEMLI: Gelecek bilgisi sızdırma (leakage) riski:
  → Rolling mean hesaplarken shift(1) kullan, shift(0) değil!
  → Tatil flag'leri gelecek bilgisi değil (takvimden) — güvenli.
  → Kampanya tarihleri gelecek bilgisi — ama plan biliniyorsa kullanılabilir.

── Production Değerlendirmeleri ─────────────────────────────────────

  SKU başına model vs global model:
  → 10.000 ürün için 10.000 ayrı model → yönetilemez
  → Global model + ürün embedding → ölçeklenebilir
  → Hiyerarşik tahmin: Kategori → Alt kategori → SKU (top-down reconcile)

  Yeniden eğitim sıklığı:
  → Hızlı değişen ürünler (moda, elektronik): haftalık
  → Stabil ürünler (FMCG): aylık

  Uncertainty quantification:
  → Sadece nokta tahmini değil, güven aralıkları üret
  → Stok kararı için: "300-400 adet" → lojistik planlama için daha yararlı

Senior Notu:
  Mükemmel model değil, karar vermeye yarayan model yap.
  WAPE=%15 ile WAPE=%12 arasındaki fark teknik başarı olabilir
  ama iş kararını değiştirmiyorsa (aynı stok miktarı seçiliyorsa)
  iyileştirme "boşa" gider. Tahmin modelini karar modeline bağla:
  Yanlış tahmin → hangi iş kararı değişiyor → hangi maliyet değişiyor?
```

---

## C) İstatistik ve Deney Senaryoları (10 Soru)

### C1. Peeking Sorunu
*(→ Bkz. Katman D3: A/B test katmanı — dinamik fiyatlandırmada peeking riski)*

**Soru:** "Deney 7 günde 'anlamlı' çıktı. Bitirelim mi?"

```
Cevap: HAYIR (tek başına bu yeterli değil).

Neden?
  → Peeking: Her gün bakınca Tip I hata α=0.05'ten çok yüksek çıkar.
  → 14 gün deneyde her gün bakan: ~%30 yanlış pozitif
  → Sadece sonda bakan: ~%5 yanlış pozitif

Ne yapmalı?
  1. Planlanan süreyi (14 gün) tamamla
  2. Veya Sequential test kullan (O'Brien-Fleming, mSPRT)
  3. Veya Bayesian monitoring (her an bakılabilir)

Ek kontroller:
  → SRM var mı?
  → Guardrail metrikler nasıl?
  → Novelty effect olabilir mi?
```

### C2. Simpson Paradoxu
*(→ Bkz. Katman A2-S6: Funnel analizi — segment bazlı vs toplam metrik farkı)*

**Soru:** "Genel dönüşüm arttı ama ülke bazında ikisi de düştü. Nasıl?"

```python
# Simülasyon
import pandas as pd

# Before
data_before = {
    "TR": {"users": 800, "conversions": 80},  # %10
    "DE": {"users": 200, "conversions": 30},  # %15
}
# After (TR trafiği azaldı, DE arttı)
data_after = {
    "TR": {"users": 300, "conversions": 27},  # %9 (azaldı)
    "DE": {"users": 700, "conversions": 98},  # %14 (azaldı)
}

# Toplam before: (80+30)/(800+200) = 11%
# Toplam after: (27+98)/(300+700) = 12.5% (arttı!)

# Neden? Ülke karışımı değişti:
# TR (düşük conversion) → DE (yüksek conversion) karışımı arttı
# Gerçekte her ülkede düştü, ama karışım etkisi toplam trendi maskeledi

# Çözüm: Segment bazlı analizle başla, sonra "karışıklık" etkisini ayrıştır
```

### C3. Çoklu Karşılaştırma
*(→ Bkz. Katman D3: Dynamic pricing — çok metrikli deney tasarımı)*

**Soru:** "20 metriğe baktık, 3'ü anlamlı. Ne yapmalıyız?"

```
Problem:
  α=0.05, 20 test → Beklenen yanlış pozitif: 20 × 0.05 = 1
  Gördüğümüz 3 anlamlı sonuçtan 1'i şans eseri olabilir!

Çözümler:
  1. Bonferroni düzeltme:
     α_adjusted = 0.05 / 20 = 0.0025
     → Çok konservatif, Tip II hata artar

  2. Benjamini-Hochberg (FDR) — pratik tercih:
     → Yanlış pozitif oranını kontrol eder
     → Bonferroni'den daha az konservatif
     from statsmodels.stats.multitest import multipletests
     rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

  3. Pre-registration:
     → Deneyi başlatmadan primary metric tek olmalı
     → Secondary metrikler "bilgi amaçlı" kaydedilmeli
```

---

## D) Sistem Tasarımı Senaryoları (7 Soru)

### D1. YouTube Öneri Sistemi
*(→ Bkz. Katman B: Model seçimi trade-off'ları; Katman A: Feature sorgulama SQL'leri)*

**Soru:** "YouTube'un öneri sistemini tasarla."

```
1. Ölçek gereksinimleri:
   - 2 milyar kullanıcı, 800M video
   - <50ms öneri latency
   - Kişiselleştirilmiş

2. İki aşamalı:
   Retrieval: 800M → 10,000
     - User embedding + video embedding
     - FAISS ANN search
     - Rule-based filtreler (dil, yaş kısıtı)

   Ranking: 10,000 → 10-20
     - CTR, watch time, like rate tahmin
     - LightGBM veya neural ranker
     - Diversity + freshness bias

3. Feature store:
   - User: uzun vadeli ilgi (embedding) — günlük güncelleme
   - User: kısa vadeli (son 24 saat) — real-time
   - Video: kalite sinyalleri, trend — saatlik

4. Offline metrikler: NDCG@20, watch time prediction MAE
5. Online metrikler (A/B): CTR, session watch time, return visit rate
6. Guardrails: Latency p99, filter bubble score
```

### D2. Customer Churn Pipeline (Batch)
*(→ Bkz. Katman B3: Model drift teşhisi; Katman A3-S13: Late arriving data)*

**Soru:** "Günlük churn riski listesi üretecek sistem tasarla."

```
Gereksinimler:
  - Her sabah 06:00'da güncel liste
  - 5M aktif kullanıcı
  - CRM sistemiyle entegrasyon

Mimari:
  [Airflow DAG]
       ↓
  [Data Pull] → Bigquery'den son 90 günlük veri
       ↓
  [Feature Engineering] → Spark veya dbt
       ↓
  [Model Inference] → LightGBM batch predict
       ↓
  [Output] → PostgreSQL veya S3
       ↓
  [CRM Entegrasyon] → Salesforce / Klaviyo API

Feature versiyonlama:
  → Training'de kullanılan feature pipeline = serving'de aynı kod
  → Makefile: make features → make predict → make upload

Monitoring:
  → Feature drift (PSI) — günlük alert
  → Prediction distribution — haftalık rapor
  → Ground truth (30 gün sonra) — aylık performance review
```

### D3. Dynamic Pricing (Dinamik Fiyatlandırma) Sistemi
*(→ Bkz. Katman B: Model drift; Katman C: A/B test; Katman D1: Feature store mimarisi)*

**Soru:** "Otel / uçak / ride-sharing için dinamik fiyatlandırma sistemi tasarla."

```
1. Problem Tanımı & İş Hedefi:
   - Amaç: Geliri maksimize ederken doluluk oranını %85+ tutmak
   - Kısıtlar: Fiyat aralığı [min, max], regülatör kurallar, müşteri memnuniyeti
   - Latency: Fiyat sorgusu < 200ms

2. Demand Forecasting (Talep Tahmini):
   Input feature'lar:
     • Tarih/saat, gün tipi (hafta içi/sonu, tatil)
     • Geçmiş talep (son 7/30/365 gün — mevsimsellik)
     • Dış sinyaller: hava durumu, etkinlikler, rakip fiyatları
     • Mevcut doluluk / envanter durumu
   Model:
     • LightGBM (baseline) veya LSTM (zaman serisi) → talep miktarı tahmini
     • Çıktı: her fiyat noktası için beklenen talep

3. Price Elasticity (Fiyat Esnekliği):
   → Talebin fiyata duyarlılığı: %1 fiyat artışı → %X talep düşüşü
   → Segment bazlı esneklik (iş seyahati vs tatil → farklı esneklik)
   → Causal estimation: Geçmiş fiyat deneyleri veya IV (instrumental variable)

4. Optimization (Fiyat Optimizasyonu):
   → Objective: max(revenue) = price × demand(price)
   → Constraint: doluluk >= %85, fiyat ∈ [min, max]
   → Yöntem: Grid search (basit) veya convex optimization (üretim)
   → Segment × zaman × envanter bazında optimal fiyat hesapla

5. A/B Test Katmanı:
   → Yeni fiyatlandırma modelini doğrulamak için:
     Control: Mevcut fiyat mantığı
     Treatment: ML-driven dinamik fiyat
   → Primary metric: Revenue per available unit (RevPAU)
   → Guardrails: Müşteri memnuniyeti (NPS), iptal oranı, şikayet
   → Dikkat: Fiyat A/B testi → network effect riski
     (kullanıcılar fiyat karşılaştırması yapabilir)
     → Çözüm: Coğrafi veya zaman bazlı randomizasyon

6. Mimari:
   [Event Stream] → Kafka
        ↓
   [Feature Store] → Real-time: mevcut doluluk, son arama sayısı
                   → Batch: tarihsel talep, esneklik katsayıları
        ↓
   [Demand Model] → Talep tahmini (her 15 dk güncelleme)
        ↓
   [Optimizer] → Optimal fiyat hesaplama
        ↓
   [Price API] → Frontend/API'ye serve (cache: 5 dk TTL)
        ↓
   [Monitoring] → Fiyat dağılımı, gelir trendi, müşteri tepkisi

7. Monitoring & Feedback Loop:
   → Günlük: Gerçek talep vs tahmin karşılaştırması (MAPE)
   → Haftalık: Revenue trend + elasticity katsayısı drift
   → Aylık: Model retrain (yeni sezon verisiyle)
```

### D4. Real-time Fraud Detection (Gerçek Zamanlı Dolandırıcılık Tespiti)
*(→ Bkz. Katman B2: Imbalanced classification; Katman B6: Train-serve skew; Katman A3-S14: Gap analysis)*

**Soru:** "Saniyede 50.000 işlem alan bir ödeme sistemi için fraud detection tasarla."

```
1. Problem Tanımı & Ölçek:
   - 50K TPS (transactions per second)
   - Latency bütçesi: < 50ms (kullanıcı deneyimi bozulmamalı)
   - Fraud oranı: ~%0.1 (1000'de 1) → ciddi imbalanced problem
   - Hedef: Precision >= %70 (yanlış blokları minimize et)
            Recall >= %90 (fraud'ların %90'ını yakala)

2. Feature Engineering (4 katman):
   Real-time (< 1 sn):
     • Transaction amount, currency, merchant category
     • Cihaz fingerprint, IP geolocation
     • Son 5 dk'daki işlem sayısı (velocity)
     • Kart-merchant ilk kez mi? (boolean)

   Near-real-time (1–60 sn, streaming):
     • Son 1 saatteki toplam harcama
     • Farklı merchant sayısı (son 1 saat)
     • Coğrafi hız: son işlemden bu işleme mesafe/zaman

   Batch (günlük):
     • Kullanıcı profili: ortalama işlem tutarı, sık merchant'lar
     • Kart yaşı, hesap yaşı
     • Geçmiş fraud geçmişi

   Graph-based (haftalık):
     • Merchant-kart ilişki ağı (fraud ring tespiti)
     • Paylaşılan cihaz/IP cluster'ları

3. Model Mimarisi (Katmanlı Karar):
   Katman 1 — Rule Engine (< 5ms):
     → Bilinen fraud pattern'ları: kara liste, imkansız coğrafi hız
     → Kesin fraud → anında blok (model'e bile gitmez)
     → Kesin temiz → bypass (model yükünü azalt)

   Katman 2 — ML Model (< 30ms):
     → LightGBM (hızlı inference, düşük latency)
     → Input: Real-time + near-real-time + batch feature'lar
     → Output: fraud_probability [0, 1]
     → class_weight="balanced" veya focal loss

   Katman 3 — Decision Engine (< 5ms):
     → Threshold'a göre karar:
       score > 0.9  → Blokla (otomatik)
       0.5 < score <= 0.9 → 3D Secure / OTP doğrulama (challenge)
       score <= 0.5 → Onayla
     → Threshold'lar segment bazlı ayarlanabilir

4. Streaming Altyapı:
   [Ödeme İsteği] → [API Gateway]
        ↓
   [Kafka Topic: raw_transactions]
        ↓                    ↓
   [Flink: Feature Agg]    [Rule Engine]
        ↓                    ↓
   [Feature Store (Redis)]  [Blok / Bypass kararı]
        ↓
   [Model Server (TF Serving / Triton)]
        ↓
   [Decision Engine] → Approve / Challenge / Block
        ↓
   [Kafka Topic: decisions] → [Feedback DB]

5. Feedback Loop:
   → Blok kararları:
     • Kullanıcı itiraz → insan inceleme → label düzeltme
     • İtiraz oranı arttıysa → threshold ayarla
   → Onaylanan işlemler:
     • 30 gün sonra chargeback gelirse → pozitif label
     • Gelmezse → negatif label (gecikmeli ground truth)
   → Label gecikmesi: Modelin en zor kısmı
     • Çözüm: Semi-supervised learning (label'sız dönemde)
     • Veya: Proxy label (24 saat içinde hesap şikayeti)

6. Monitoring & Alerting:
   → Real-time dashboard:
     • Fraud oranı (saatlik), blok oranı, challenge oranı
     • Model latency p50/p95/p99
     • Feature freshness (Redis'teki feature'lar ne kadar güncel?)
   → Alert kuralları:
     • Blok oranı > %5 → olası model hatası veya saldırı
     • Latency p99 > 100ms → kapasite artır
     • Feature NULL oranı > %2 → pipeline sorunu

7. Etik & Compliance:
   → Bias tespiti: demografik gruplara göre false positive oranı
   → Açıklanabilirlik: SHAP ile "neden blokladık?" müşteriye gösterilebilir
   → GDPR/KVKK: Kişisel veri retention süresi, anonimleştirme
```

---

## E) Behavioral (Davranışsal) Sorular

### STAR Yöntemi

**S**ituation — Bağlam ve sorun
**T**ask — Senin görevi / sorumluluğun
**A**ction — Ne yaptın?
**R**esult — Sonuç ne oldu? (sayısal!)

### Sık Sorulan Sorular + Örnek Yanıtlar

**S: "Başarısız bir ML projesini anlat."**

```
Situation: Müşteri segmentasyonu projesi (k-means), 6 hafta çalışma.

Task: 8 anlamlı segment üretmek, pazarlama ekibine teslim etmek.

Action:
  → k=8 seçtim, silhouette score makul görünüyordu.
  → Segmentleri pazarlamaya teslim ettim.
  → 3 hafta sonra "segmentler çok benzer, ayırt edilemiyor" geri bildirimi.

Result:
  → Revizyonda anladım: feature'lar normalize edilmemişti, büyük ölçekli
    feature (GMV) her şeyi baskılıyordu.
  → Yeniden: StandardScaler + feature selection + k=5 → çok daha tutarlı.

Ne öğrendim:
  → Kümeleme validasyonu sadece sayısal metrik değil,
    iş kullanıcısıyla early checkpoint şart.
  → Üretimdeki "müşteri" validation'ı model validation'dan önce gelir.
```

**S: "Stakeholder'ı ikna etmek zorunda kaldığın bir durum?"**

```
Situation: Ürün direktörü churn modeli threshold'u 0.7'de sabit tutmak istiyordu.

Task: Threshold'un 0.35 olmasının neden daha iyi olduğunu göstermek.

Action:
  → Maliyet matrisi hesabı yaptım:
    • Threshold 0.7: ayda 200 müşteri yakalanıyor, 800 kaçıyor
    • Threshold 0.35: 650 yakalanıyor, 350 kaçıyor
    → Kampanya maliyeti hesabıyla: net kazanç 3× daha fazla

  → "Yanlış pozitif maliyeti ne?" diye sordum → kampanya başına 15 TL
  → Maliyet matrisiyle her threshold için beklenen kazancı gösterdim.

Result:
  → Threshold 0.35 onaylandı.
  → 3 aylık A/B test sonrası churn rate %9 düştü.

Öğrenme: "Hangi threshold?" sorusu teknik değil iş sorusudur.
Teknik cevap değil, parasal cevap isteği ikna eder.
```

**S: "Zaman baskısı altında nasıl karar verdin?"**

```
Situation: Black Friday 3 gün önce, tavsiye modeli production'da çöktü.

Task: 3 günde çalışır hale getirmek VEYA fallback çalıştırmak.

Action:
  → 2 saat root cause: feature pipeline değişikliği model'in input format'ını bozmuş.
  → Karar: Full fix yerine fallback (popularity-based öneri).
  → Stakeholder'a bildirdim: "Model X iyileşme yerine fallback öneriyorum.
    Fallback %Y daha düşük CTR ama %0 downtime riski."
  → Fallback deploy, Black Friday boyunca çalıştı.
  → Bir hafta sonra full fix.

Result:
  → Black Friday hasarsız geçti.
  → Model aynı haftada hotfix deploy edildi.

Öğrenme: Zaman baskısında "en iyi çözüm" değil "en güvenli yeterli çözüm" seç.
```

**S: "En gurur duyduğun teknik karar neydi?"**
*(→ Bkz. Katman D: Sistem tasarımı kararları; Katman B: Model seçimi trade-off'ları)*

```
Situation: E-ticaret platformunda ürün sıralama modeli yenilenmesi projesi.
  Mevcut sistem: El ile yazılmış kurallar (satış adedi × marj × stok durumu).
  Sorun: Yeni kategorilerde sıralama kalitesi çok düşük, müşteri şikayetleri %15 artmış.

Task: Kural tabanlı sistemi ML tabanlı sıralama ile değiştirmek.
  2 seçenek: (A) Büyük neural ranker, (B) LightGBM + feature engineering.

Action:
  → İlk impulsım neural ranker'dı (akademik makalelerde SOTA).
  → AMA: Latency bütçesi 20ms, veri boyutu 50K ürün (neural için yetersiz).
  → Karar: LightGBM + iyi feature engineering.
  → Neden? 3 sebep belgeledim:
    1. Latency: LightGBM < 5ms vs neural ranker ~80ms
    2. Veri: 50K ürün → neural overfit riski yüksek
    3. Açıklanabilirlik: Kategori yöneticileri SHAP ile neden bir ürünün
       üst sırada olduğunu görebilmeli
  → Bunu 1-sayfalık trade-off dokümanı olarak ekiple paylaştım.

Result:
  → NDCG@10: 0.72 → 0.89 (%24 iyileşme)
  → Müşteri şikayetleri %40 azaldı (3 aylık A/B test)
  → Inference latency: 3ms (bütçenin çok altında)
  → Ek bonus: Kategori yöneticileri SHAP dashboard'u aktif kullanıyor

Öğrenme: "En iyi model" değil "probleme en uygun model" seçmek
  teknik olgunluğun göstergesi. Trade-off'ları belgelemek gelecekteki
  seni de korur.
```

**S: "Ekip içinde teknik bir anlaşmazlık yaşadığın durumu anlat."**
*(→ Bkz. Katman C: İstatistik tartışmaları; Katman D: Mimari kararlar)*

```
Situation: A/B test platformu modernizasyonu projesinde ML mühendisiyle
  anlaşmazlık. Ben Bayesian A/B test öneriyordum, o Frequentist yaklaşımda
  ısrar ediyordu.

Task: Platformun test metodolojisini belirlemek (tüm şirket kullanacak).

Action:
  → İlk yaklaşım: Mail zincirinde tartışma → verimsiz, kimse ikna olmadı.
  → Adım 1: Tartışmayı kişiselleştirmekten kaçındım. "Sen yanlışsın" yerine
    "Her iki yaklaşımın güçlü yanlarını karşılaştıralım" dedim.
  → Adım 2: Somut karşılaştırma yaptım — aynı 5 geçmiş deney üzerinde
    her iki yöntemi uyguladım:
    • Frequentist: 3/5 deneyda "anlamlı değil" → karar verilemedi
    • Bayesian: 5/5 deneyda net posterior dağılımı → karar verildi
    • AMA: Bayesian'da prior seçimi hassas → 1 deneyda farklı prior
      farklı sonuç verdi
  → Adım 3: Hibrit çözüm önerdim:
    • Primary metric → Frequentist (standarizasyon, regülatör uyum)
    • Early monitoring → Bayesian (peeking sorunu yok)
    • Raporlama → Her ikisini de göster

Result:
  → Hibrit yaklaşım kabul edildi.
  → ML mühendisi: "Prior seçimi sorununu göstermen ikna etti" dedi.
  → Platform 6 ayda 40+ deney yürüttü, peeking hatası sıfıra düştü.

Öğrenme:
  → Teknik anlaşmazlıkta veriyle konuş, ego ile değil.
  → "Benim yöntemim" yerine "hangi yöntem bu probleme uygun?" çerçevesi kur.
  → Hibrit çözümler çoğunlukla saf yaklaşımlardan daha pratiktir.
```

---

## F) Kısa İpuçları ve Sık Hata Listesi

### Mülakat Sırasında

```
✓ "Soruyu netleştirebilir miyim?" diye sor
✓ Sessiz kalmak yerine "şöyle düşünüyorum..." de
✓ Trade-off'ları aktif ifade et ("A seçenek şu avantajda, ama B seçenek...")
✓ Sayısal olmaya çalış ("çok iyi" değil "AUC 0.87, PR-AUC 0.64")
✓ "Bilmiyorum, ama böyle araştırırdım" kabul edilebilir

✗ İlk gelen çözümü sunma (30 saniye düşün)
✗ Varsayımları açıklamadan geçme
✗ Sadece teknik → iş etkisi söyle
✗ "Kaggle'da AUC 0.99 elde ettim" → prod'da ne oldu?
```

### SQL İpuçları

```sql
-- NULL yönetimi: COALESCE, NULLIF, IS NULL
SELECT COALESCE(amount, 0) AS amount_filled     -- NULL → 0
SELECT NULLIF(denominator, 0)                   -- 0 → NULL (bölme hatası önle)

-- Sıfıra bölme önle
SELECT numerator / NULLIF(denominator, 0) AS ratio

-- QUALIFY (Snowflake, BigQuery) — daha temiz window filter
SELECT * FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_ts DESC) = 1;

-- STRING_AGG / GROUP_CONCAT (kategorileri birleştir)
SELECT user_id, STRING_AGG(category, ', ' ORDER BY category) AS categories
FROM orders GROUP BY user_id;
```

### Sektör Notu — Mülakat Trendleri 2026

2026 itibarıyla mülakat bulgularına göre:
- SQL hâlâ en önemli teknik beceri (DS mülakatlarının %90'ında soruluyor)
- System design soruları senior rollerde artıyor (%70+ ilanda)
- Causal inference ve A/B test bilgisi growth DS/product DS için ayırt edici
- LLM/RAG bilgisi artık standart beklenti haline geldi (2026 itibarıyla çoğu DS ilanında yer alıyor)
- Behavioral sorular daha sistematik (STAR + "sonuç ne oldu?" baskısı)
- Take-home assignment azalıyor → live coding / panel tercih ediliyor

**Tavsiye edilen kaynaklar:**
- DataLemur (datalemur.com) — gerçek mülakat SQL soruları
- StrataScratch — ileri seviye SQL + Python
- InterviewQuery — case study soruları
- "Ace the Data Science Interview" (Singh & Huo) — 200+ soru kitabı

---

## Mülakat Hazırlık Kontrol Listesi

### SQL
- [ ] Window function (ROW_NUMBER, RANK, LAG, LEAD, SUM OVER) rahat kullanıyorum
- [ ] CTE ile çok adımlı sorgu yazabiliyorum
- [ ] Funnel + cohort retention sorgusu yazdım
- [ ] Sessionization (LAG ile gap tespiti) mantığını biliyorum
- [ ] N-day retention (self-join) sorgusu yazabiliyorum
- [ ] PIVOT/UNPIVOT ve JSON/ARRAY sorgusu yazabiliyorum
- [ ] Gap analysis ve moving percentile hesaplama yapabiliyorum
- [ ] DataLemur'dan 20+ SQL sorusu çözdüm

### İstatistik ve A/B Testi
- [ ] p-value'yu sezgisel ve doğru açıklayabilirim
- [ ] Peeking'in neden yanlış olduğunu simülasyonla gösterdim
- [ ] CUPED'i anlatabilir ve hesaplayabilirim
- [ ] Simpson's Paradox için bir örnek verebilirim
- [ ] Power analizi yapabilirim (MDE, n hesabı)

### Makine Öğrenmesi
- [ ] Leakage tespiti için sistematik yaklaşımım var
- [ ] Model drift sorununu adım adım teşhis edebilirim
- [ ] SHAP yorumlamasında "correlation ≠ causation" tuzağını biliyorum
- [ ] SHAP'ın yanıltıcı olduğu somut bir counter-example verebilirim
- [ ] Data drift ve train-serve skew farklarını açıklayabilirim
- [ ] İmbalanced sınıf stratejilerini karşılaştırabilirim
- [ ] LightGBM + Optuna ile uçtan uca model kurabiliyorum

### Sistem Tasarımı
- [ ] YouTube (veya e-ticaret) öneri sistemi iki aşamasını açıkladım
- [ ] Feature store online/offline farkını biliyorum
- [ ] Latency bütçesi nedir, nasıl planlanır?
- [ ] Drift monitoring ve retraining trigger'larını açıklayabilirim
- [ ] Dynamic pricing sistemi tasarlayabiliyorum (demand → elasticity → optimization)
- [ ] Real-time fraud detection mimarisini açıklayabilirim (streaming + katmanlı karar)

### Behavioral
- [ ] En az 5 STAR hikayesi hazırladım (conflict, failure, success, pride, disagreement)
- [ ] "Neden bu şirket?" sorusuna özgün bir cevabım var
- [ ] Zayıf yönümü ve geliştirme planımı somut anlatabilirim
- [ ] "En gurur duyduğum teknik karar" hikayem hazır (trade-off odaklı)
- [ ] "Ekip içinde anlaşmazlık" hikayem hazır (veriyle ikna odaklı)

---

<div class="nav-footer">
  <span><a href="#file_projeler">← Önceki: Portföy Projeleri</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_kaynaklar">Sonraki: Kaynaklar →</a></span>
</div>
