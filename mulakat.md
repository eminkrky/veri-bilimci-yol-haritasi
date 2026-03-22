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
