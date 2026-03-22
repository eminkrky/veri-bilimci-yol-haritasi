# Katman A — Temeller (Python, İstatistik, SQL, Görselleştirme)

> Bu katmanda ne öğrenilir: Veri biliminin günlük araç takımı. Python/Pandas ile veri manipülasyonu, analitik SQL, istatistik temeli ve görsel hikâyecilik.
>
> Süre: 3–4 hafta yoğun çalışma. Bu katmanı sağlam kurmadan üzerine ML inşa edilmez.


<div class="prereq-box">
<strong>Önkoşul:</strong> <strong>Katman 0 (Matematik Temelleri)</strong> tamamlanmış olmalı — özellikle lineer cebir ve olasılık bölümleri.
</div>

---

## A.1 Python — DS Odaklı Temeller

### Sezgisel Açıklama

Python'u öğrenmek = notebook'ta kod yazmak değil. Senior DS için Python "üretimde çalışan, test edilmiş, okunabilir" kod yazmak demek. Notebook keşif içindir; `src/` klasörü gerçek iş içindir.

### Temel Veri Yapıları

```python
# Hangi yapı ne zaman?
liste = [1, 2, 3]          # Sıralı, değiştirilebilir, index ile erişim
demet = (1, 2, 3)          # Değişmez, hashable — dict key, namedtuple
kume = {1, 2, 3}           # Tekil kontrol O(1), küme işlemleri
sozluk = {"k": "v"}        # Key-value lookup O(1)

# List comprehension vs generator
kareler = [x**2 for x in range(1000)]      # Hepsini bellekte tut
kareler_gen = (x**2 for x in range(1000))  # Lazy — büyük veri için
```

### OOP ve Type Hints

```python
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class ModelConfig:
    """Model konfigürasyonu — tip güvenli, boilerplate az."""
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: Optional[int] = 6
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = []

# Tip ipuçları ile fonksiyon
def load_data(path: str, sep: str = ",") -> "pd.DataFrame":
    """CSV yükle, temel kalite kontrolü yap."""
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)

    df = pd.read_csv(path, sep=sep)
    logger.info(f"Yüklendi: {path}, şekil: {df.shape}")

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Toplam eksik değer: {n_missing}")

    return df
```

> **Senior Notu:** `mypy` ile statik tip kontrolü ekle. CI pipeline'ına `mypy src/` adımı koy. Tip hatalarını runtime'da değil, commit öncesi yakala.

### Test Kültürü (pytest)

```python
# src/features.py
import pandas as pd
from datetime import datetime
from typing import Optional

def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Tarih string'ini datetime'a çevirir. Hatalı/None → None döndürür."""
    if value is None or value == "":
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
```

```python
# tests/test_features.py
import pytest
from datetime import datetime
from src.features import parse_date

@pytest.mark.parametrize("input_val, expected", [
    ("2024-01-15", datetime(2024, 1, 15)),
    ("",  None),
    (None, None),
    ("bozuk-tarih", None),
    ("2024-13-01", None),   # Geçersiz ay
])
def test_parse_date(input_val, expected):
    assert parse_date(input_val) == expected
```

> **Senior Notu:** Test yazılmayan fonksiyon prod'da kırılmak için yazılmıştır. Özellikle feature engineering fonksiyonları — edge case'ler ürünü bozar.

### Sektör Notu — Python 2026

2026 itibarıyla Python ekosisteminde:
- **uv** paket yöneticisi pip+venv'in yerini alıyor (10× daha hızlı)
- **Ruff** linter/formatter olarak black+flake8'i geride bırakıyor
- **Polars** pandas'tan 10–30× hızlı, lazy evaluation ile büyük veri için tercihen kullanılıyor
- **DuckDB** yerel OLAP analizi için SQL-over-Parquet standart hale geldi

---

## A.2 NumPy — Sayısal Hesaplama

### Sezgisel Açıklama

NumPy vektörize işlemler yapar — Python döngüleri yerine C kodu çalıştırır. 1M elemanlı toplama için Python döngüsü ~100ms, NumPy ~1ms.

### Kod Örneği

```python
import numpy as np

# Broadcasting — farklı şekillerdeki dizileri otomatik genişlet
X = np.random.randn(1000, 50)   # 1000 örnek, 50 feature
mu = X.mean(axis=0)              # (50,) — her feature'ın ortalaması
std = X.std(axis=0)              # (50,)
X_normalized = (X - mu) / std   # Broadcasting: (1000,50) - (50,) → (1000,50)

# Sayısal kararlılık: log-sum-exp
def log_softmax_naive(x):
    return np.log(np.exp(x) / np.sum(np.exp(x)))  # Overflow riski!

def log_softmax_stable(x):
    c = np.max(x)
    return x - c - np.log(np.sum(np.exp(x - c)))  # Stabil

# Hız karşılaştırma
import time
x = np.random.randn(1000, 50)
w = np.random.randn(50)

# Yavaş: Python döngüsü
start = time.time()
result_loop = [sum(x[i, j] * w[j] for j in range(50)) for i in range(1000)]
t_loop = time.time() - start

# Hızlı: NumPy
start = time.time()
result_numpy = x @ w
t_numpy = time.time() - start

print(f"Python döngüsü: {t_loop*1000:.1f}ms")
print(f"NumPy: {t_numpy*1000:.1f}ms")
print(f"Hız farkı: {t_loop/t_numpy:.0f}×")
```

---

## A.3 Pandas — Derin Pratik

### Sezgisel Açıklama

Pandas = sütun bazlı tablo. Her sütun ayrı bir `Series`. DataFrame gerçek verinin %80'ini karşılar — ama tuzaklar çok.

### DataFrame Mental Modeli

```python
import pandas as pd
import numpy as np

# Veri tipleri ve bellek
df = pd.read_csv("orders.csv")
print(df.dtypes)
print(df.memory_usage(deep=True) / 1024**2, "MB")

# Kategorik sütunlar — 10× daha az bellek
df["country"] = df["country"].astype("category")
df["device"] = df["device"].astype("category")
print(df.memory_usage(deep=True) / 1024**2, "MB")  # Düşmüş olmalı
```

### SettingWithCopyWarning Tuzağı

```python
# KÖTÜ — sessiz hata!
df[df["amount"] > 100]["status"] = "premium"   # Çalışmaz

# İYİ — her zaman .loc kullan
df.loc[df["amount"] > 100, "status"] = "premium"

# Kopya ne zaman gerekli?
df_sub = df[df["country"] == "TR"].copy()  # Bağımsız çalışacaksan .copy()
df_sub["new_col"] = 1  # Şimdi orijinal df etkilenmez
```

### GroupBy Derin Pratik

```python
# Named aggregation — okunabilir
user_stats = df.groupby("user_id").agg(
    siparis_sayisi=("order_id", "count"),
    toplam_harcama=("amount", "sum"),
    ortalama_sepet=("amount", "mean"),
    son_siparis=("order_date", "max"),
    ilk_siparis=("order_date", "min"),
    std_harcama=("amount", "std"),
).reset_index()

# transform — grup içi değeri satır bazında yay
df["kullanici_ort_harcama"] = df.groupby("user_id")["amount"].transform("mean")
df["harcama_norm"] = df["amount"] / df["kullanici_ort_harcama"]

# apply — karmaşık mantık (dikkat: yavaş!)
def recency_score(group):
    son_gun = (pd.Timestamp.now() - group["order_date"].max()).days
    return son_gun

recency = df.groupby("user_id").apply(recency_score)
```

### Zaman Serisi

```python
df["order_date"] = pd.to_datetime(df["order_date"])

# Temporal özellikler
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["hour"] = df["order_date"].dt.hour

# Resample — günlük → haftalık
daily_revenue = df.set_index("order_date")["amount"].resample("D").sum()
weekly = daily_revenue.resample("W").sum()
monthly = daily_revenue.resample("MS").sum()   # Ay başı

# Rolling window
df_daily = df.groupby("order_date")["amount"].sum().reset_index()
df_daily["rolling_7d"] = df_daily["amount"].rolling(7, min_periods=1).mean()
df_daily["rolling_30d"] = df_daily["amount"].rolling(30, min_periods=1).mean()
```

### Sektör Notu — Polars ile Hız

```python
# Polars — Pandas'tan 5–30× hızlı (Rust tabanlı, Arrow columnar format)
import polars as pl

df_pl = pl.read_csv("large_data.csv")

# Lazy evaluation (query optimizer çalışır)
# ⚠️ DİKKAT: Polars v0.20+ ile .groupby() → .group_by() olarak değişti.
# Eski API kullanırsan DeprecationWarning alırsın.
result = (
    df_pl.lazy()
    .filter(pl.col("amount") > 0)
    .group_by("user_id")          # v0.20+: group_by (alt çizgili)
    .agg([
        pl.col("amount").sum().alias("total"),
        pl.col("amount").count().alias("n_orders"),
        pl.col("order_date").max().alias("last_order"),
    ])
    .sort("total", descending=True)
    .collect()  # Eager: hesaplamayı başlat
)
```

> **API Değişiklik Notu (v0.20+):** `.groupby()` → `.group_by()`, `.apply()` → `.map_elements()`, `.map()` → `.map_batches()`. Eski örnekleri kopyalarken dikkat — StackOverflow cevaplarının çoğu hâlâ eski API kullanıyor.

### Polars ile Zaman Serisi Feature Engineering

```python
import polars as pl

# Zaman serisi özellikleri — Polars'ın gerçek gücü
df_pl = pl.read_csv("orders.csv", try_parse_dates=True)

features = (
    df_pl.lazy()
    .sort("user_id", "order_date")
    .with_columns([
        # Bir önceki siparişten bu yana geçen gün
        (pl.col("order_date") - pl.col("order_date").shift(1).over("user_id"))
            .dt.total_days()
            .alias("days_since_last_order"),

        # Kullanıcı bazında kümülatif toplam
        pl.col("amount")
            .cum_sum()
            .over("user_id")
            .alias("cumulative_spend"),

        # Son 3 siparişin hareketli ortalaması
        pl.col("amount")
            .rolling_mean(window_size=3, min_periods=1)
            .over("user_id")
            .alias("rolling_3_avg"),

        # Kullanıcının kaçıncı siparişi
        pl.col("order_id")
            .cum_count()
            .over("user_id")
            .alias("order_sequence"),
    ])
    .collect()
)
```

### Polars vs Pandas — 2026 Benchmark Özeti

| İşlem | Pandas | Polars | Fark |
|-------|--------|--------|------|
| 1 GB CSV yükleme | ~8s | ~1.6s | **5×** |
| Filtreleme (1 GB) | ~2s | ~0.4s | **4.6×** |
| GroupBy + Agg | ~3s | ~1.1s | **2.6×** |
| Sıralama | ~5s | ~0.4s | **11.7×** |
| Bellek kullanımı (1 GB CSV) | ~1.4 GB | ~179 MB | **87% daha az** |

> Polars, Arrow columnar format ve multi-thread paralelizm sayesinde özellikle 1 GB+ veri setlerinde parlıyor. <1 GB veri için fark ihmal edilebilir — Pandas'ın ekosistem desteği daha geniş.

### Pandas vs Polars — 2026 Seçim Rehberi

#### Sezgisel Açıklama

Pandas Python'ın veri bilimi standardı — 2008'den beri. Polars ise 2021'de Rust ile yeniden yazıldı: Pandas API'sine benziyor ama çok çekirdekli (multi-threaded) çalışıyor, memory'yi daha verimli kullanıyor. 2026 itibarıyla yeni projelerde hangisini seç?

#### Yan Yana Kod Karşılaştırması

```python
import pandas as pd
import polars as pl

# --- VERİ OKUMA ---
# Pandas
df_pd = pd.read_csv("data.csv")

# Polars (eager - hemen çalışır)
df_pl = pl.read_csv("data.csv")

# Polars (lazy - sorgu planı oluşturur, collect() ile çalıştırır)
df_lazy = pl.scan_csv("data.csv")  # büyük dosyalar için tercih et


# --- FİLTRELEME ---
# Pandas
sonuc_pd = df_pd[df_pd["yas"] > 30]

# Polars
sonuc_pl = df_pl.filter(pl.col("yas") > 30)


# --- GRUPLAMA ---
# Pandas
ozet_pd = df_pd.groupby("sehir")["gelir"].mean().reset_index()

# Polars
ozet_pl = df_pl.group_by("sehir").agg(pl.col("gelir").mean())


# --- YENİ KOLON OLUŞTURMA ---
# Pandas
df_pd["gelir_log"] = df_pd["gelir"].apply(lambda x: x ** 0.5)  # yavaş

# Polars (vectorized, çok daha hızlı)
df_pl = df_pl.with_columns(
    pl.col("gelir").sqrt().alias("gelir_log")
)


# --- JOIN ---
# Pandas
sonuc_pd = df_pd.merge(diger_df, on="musteri_id", how="left")

# Polars
sonuc_pl = df_pl.join(diger_pl, on="musteri_id", how="left")


# --- ZINCIRLEME (method chaining) ---
# Pandas (verbose)
df_pd_clean = df_pd[df_pd["yas"] > 18]
df_pd_clean = df_pd_clean.groupby("sehir")["gelir"].mean().reset_index()
df_pd_clean = df_pd_clean.rename(columns={"gelir": "ort_gelir"})

# Polars (clean chaining)
df_pl_clean = (
    df_pl
    .filter(pl.col("yas") > 18)
    .group_by("sehir")
    .agg(pl.col("gelir").mean().alias("ort_gelir"))
    .sort("ort_gelir", descending=True)
)
```

#### Hız ve Bellek Karşılaştırması (Pratik Rehber)

| Veri Boyutu | Pandas | Polars | Öneri |
|-------------|--------|--------|-------|
| < 100 MB | Hızlı | Hızlı | Pandas (daha geniş ekosistem) |
| 100 MB – 2 GB | Yavaşlayabilir | Hızlı | Polars tercih et |
| 2 GB – 50 GB | Bellek sorunu | Lazy API ile verimli | Polars (lazy scan) |
| > 50 GB | Çöker | Zorlanır | DuckDB veya Spark |

#### Karar Rehberi

```
Yeni proje mi?
├── Evet → Polars tercih et (2026 standardı yükseliyor)
│
└── Hayır (mevcut Pandas kodu var)
    ├── Kritik performans sorunu var mı?
    │   ├── Evet → Darboğaz noktalarını Polars'a taşı
    │   └── Hayır → Pandas'ta kal, boşuna yeniden yazma
    │
    └── Yeni pipeline yazılıyor mu?
        ├── Büyük veri (>500MB) → Polars lazy API
        └── Küçük veri → İkisi de OK, Polars öğren
```

**Polars'ta sık karşılaşılan tuzaklar:**
```python
# ❌ Polars'ta .apply() kullanma (Pandas alışkanlığı)
df_pl.with_columns(
    pl.col("gelir").apply(lambda x: x * 1.18)  # yavaş, vectorize değil
)

# ✅ Polars expression API kullan
df_pl.with_columns(
    (pl.col("gelir") * 1.18).alias("gelir_kdv")  # hızlı, paralel
)

# ❌ Pandas gibi iterasyon
for row in df_pl.iter_rows():  # çok yavaş
    ...

# ✅ Vectorized operation
result = df_pl.select(pl.col("gelir") * 1.18)  # hızlı
```

> **Senior Notu:** 2026'da "Pandas vs Polars" tartışması "NumPy vs Pandas" tartışmasını andırıyor — geçiş kademeli, her ikisi de hayatta kalacak. Pratikte: yeni projede Polars, eski büyük kod tabanında Pandas. Polars'ın `polars.interchange` modülü sayesinde ikisi arasında veri aktarımı kolaylaştı:
> ```python
> # Polars → Pandas (gerekirse)
> df_pandas = df_polars.to_pandas()
> # Pandas → Polars
> df_polars = pl.from_pandas(df_pandas)
> ```
> Her iki kütüphaneyi de bilen DS 2026'da avantajlı.

> **Sektör Notu (2026):** Polars 1.0 Haziran 2024'te çıktı ve API stabilitesi garantilendi. Hugging Face, Ruff, Pydantic gibi Rust tabanlı araçların yaygınlaşmasıyla birlikte Polars de data stack'in standart parçası olmaya yaklaşıyor. DuckDB ile de çok iyi entegre olur: DuckDB SQL sorgusu → Polars DataFrame doğrudan döner.

### DuckDB — SQL ile Yerel Analitik

```python
import duckdb
import pandas as pd

# 1. Pandas DataFrame üzerinde doğrudan SQL
df = pd.read_csv("orders.csv")

result = duckdb.sql("""
    SELECT user_id, SUM(amount) as total, COUNT(*) as n_orders
    FROM df                     -- Pandas df'i doğrudan tablo gibi kullan
    WHERE amount > 0
    GROUP BY user_id
    ORDER BY total DESC
""").df()  # Sonucu pandas DataFrame'e çevir

# 2. Parquet dosyalarını doğrudan sorgula (belleğe yüklemeden!)
result = duckdb.sql("""
    SELECT user_id, SUM(amount) as total, COUNT(*) as n_orders
    FROM 'data/*.parquet'
    WHERE amount > 0
    GROUP BY user_id
    ORDER BY total DESC
""").df()
```

### DuckDB ile Cohort Analizi (Parquet)

```python
import duckdb

# Parquet dosyalarından cohort retention — belleğe sığmayan veri için ideal
cohort = duckdb.sql("""
    WITH first_order AS (
        SELECT user_id,
               DATE_TRUNC('month', order_date) AS cohort_month
        FROM 'data/orders_*.parquet'
        GROUP BY user_id
    ),
    monthly_activity AS (
        SELECT o.user_id,
               f.cohort_month,
               DATE_TRUNC('month', o.order_date) AS activity_month
        FROM 'data/orders_*.parquet' o
        JOIN first_order f USING (user_id)
    )
    SELECT
        cohort_month,
        DATE_DIFF('month', cohort_month, activity_month) AS period,
        COUNT(DISTINCT user_id) AS active_users
    FROM monthly_activity
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()

print(cohort.head(20))
```

### DuckDB vs Pandas — Ne Zaman Hangisi?

| Kriter | Pandas | DuckDB |
|--------|--------|--------|
| Veri boyutu <1 GB | Yeterli | Gereksiz |
| Veri boyutu 1–50 GB | Bellek sorunu | Parquet streaming ile rahat |
| SQL bilen ekip | `.groupby` öğrenmeli | SQL ile doğrudan çalışır |
| Karmaşık join + window | Pandas kodu uzar | SQL daha okunur |
| Notebook prototipi | Standart | `duckdb.sql("SELECT * FROM df")` ile kolay geçiş |

> **Senior Notu:** Pandas veri >5 GB olduğunda ya Polars'a geç ya da DuckDB kullan. DuckDB'nin en güçlü yanı: Parquet dosyalarını belleğe yüklemeden sorgulayabilmesi. Ölçek gerektirirse Spark (bkz. `katman-H-buyuk-veri.md`). Ama önce örnekle çalış — çoğu "büyük veri" problemi aslında küçük örnekle test edilebilir. Data pipeline entegrasyonu için bkz. `katman-E-data-pipeline.md`.

---

## A.4 İstatistik ve Olasılık

### Merkezi Limit Teoremi (CLT)

### Sezgisel Açıklama

Orijinal dağılım ne olursa olsun (çarpık, bimodal, uniform), yeterince büyük bir örneklemden yeterince çok ortalama aldığında bu ortalamalar **normal dağılır**.

Bu neden önemli? Çoğu istatistik testi normal dağılım varsayımına dayanır. CLT bu varsayımı pratik olarak geçerli kılar.

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

distributions = {
    "Üstel": lambda n: np.random.exponential(1, n),
    "Düzgün": lambda n: np.random.uniform(0, 10, n),
    "Bimodal": lambda n: np.concatenate([
        np.random.normal(2, 0.5, n//2),
        np.random.normal(8, 0.5, n//2)
    ])
}

for i, (name, dist_fn) in enumerate(distributions.items()):
    # Orijinal dağılım
    sample = dist_fn(10000)
    axes[0, i].hist(sample, bins=60, density=True, color="steelblue")
    axes[0, i].set_title(f"{name} Dağılım")

    # Örneklem ortalamaları dağılımı (CLT)
    means = [np.mean(dist_fn(50)) for _ in range(5000)]
    axes[1, i].hist(means, bins=60, density=True, color="coral")

    # Normal üst üste koy
    mu, sigma = np.mean(means), np.std(means)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    axes[1, i].plot(x, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2),
                    color="darkred", lw=2)
    axes[1, i].set_title(f"CLT: n=50 Ortalamaları")

plt.tight_layout()
plt.savefig("clt_demonstration.png", dpi=100)
```

### Güven Aralığı (CI)

```python
import numpy as np
from scipy import stats

def parametric_ci(data, alpha=0.05):
    """Parametrik CI (normal varsayım)."""
    n = len(data)
    mu = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    return mu - t_crit * se, mu + t_crit * se

def bootstrap_ci(data, stat_fn=np.mean, n_boot=10_000, alpha=0.05):
    """Bootstrap CI — dağılım varsayımı yok."""
    boots = [stat_fn(np.random.choice(data, size=len(data), replace=True))
             for _ in range(n_boot)]
    return np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])

# Sağa çarpık gelir verisi ile karşılaştır
np.random.seed(42)
income_data = np.random.lognormal(mean=np.log(500), sigma=0.8, size=200)

p_ci = parametric_ci(income_data)
b_ci = bootstrap_ci(income_data)

print(f"Parametrik CI: ({p_ci[0]:.0f}, {p_ci[1]:.0f})")
print(f"Bootstrap CI:  ({b_ci[0]:.0f}, {b_ci[1]:.0f})")
# Çarpık verida bootstrap daha doğru (parametrik varsayım kırılıyor)
```

> **Senior Notu:** Gelir, sipariş tutarı gibi çarpık metrikler için bootstrap CI kullan. Parametrik CI güven aralığını daraltır (yanıltıcı kesinlik). p-value yanında her zaman effect size + CI ver.

### Hipotez Testleri

```python
from scipy import stats
import numpy as np

np.random.seed(42)

# İki grup: kontrol vs test
control = np.random.normal(50, 10, 500)
treatment = np.random.normal(53, 10, 500)

# 1. İki örneklem t-testi
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")

# 2. Effect size (Cohen's d)
pooled_std = np.sqrt((np.std(control)**2 + np.std(treatment)**2) / 2)
cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")  # 0.2=küçük, 0.5=orta, 0.8=büyük

# 3. Pratik anlamlılık
relative_lift = (np.mean(treatment) - np.mean(control)) / np.mean(control)
print(f"Uplift: {relative_lift:.1%}")

# 4. Permutation test (varsayımdan bağımsız)
observed_diff = np.mean(treatment) - np.mean(control)
combined = np.concatenate([control, treatment])

n_permutations = 10_000
null_diffs = []
for _ in range(n_permutations):
    np.random.shuffle(combined)
    perm_diff = np.mean(combined[:len(treatment)]) - np.mean(combined[len(treatment):])
    null_diffs.append(perm_diff)

p_perm = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
print(f"Permutation p-value: {p_perm:.4f}")
```

---

## A.5 SQL — Analitik Derinlik

### Sezgisel Açıklama

SQL'de hedef "sorgu yazmak" değil, **doğru metriği doğru tanımlamak**. Çoğu DS hatası SQL seviyesinde başlar: yanlış join, yanlış cohort tanımı, eksik null yönetimi.

### Pencere Fonksiyonları (Window Functions)

```sql
-- Temel sözdizimi
<aggregate/rank_fn>() OVER (
  PARTITION BY <grup>
  ORDER BY <sıralama>
  ROWS/RANGE BETWEEN ...
)

-- Sıralama
SELECT
  user_id,
  amount,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS row_num,
  RANK()       OVER (PARTITION BY user_id ORDER BY amount DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS dense_rnk
FROM orders;

-- Her kullanıcının en son siparişi
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date DESC) AS rn
  FROM orders
)
SELECT * FROM ranked WHERE rn = 1;

-- Kayan ortalama (7 günlük)
SELECT
  order_date,
  SUM(amount) AS daily_revenue,
  AVG(SUM(amount)) OVER (
    ORDER BY order_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_avg
FROM orders
GROUP BY order_date;
```

### Cohort Retention

```sql
WITH first_order AS (
  SELECT user_id, MIN(DATE_TRUNC('month', order_ts)) AS cohort_month
  FROM orders
  GROUP BY user_id
),
orders_with_cohort AS (
  SELECT
    o.user_id,
    f.cohort_month,
    DATE_TRUNC('month', o.order_ts) AS order_month
  FROM orders o
  JOIN first_order f USING (user_id)
),
retention AS (
  SELECT
    cohort_month,
    DATEDIFF('month', cohort_month, order_month) AS period,
    COUNT(DISTINCT user_id) AS n_users
  FROM orders_with_cohort
  GROUP BY 1, 2
),
cohort_size AS (
  SELECT cohort_month, n_users AS cohort_n
  FROM retention
  WHERE period = 0
)
SELECT
  r.cohort_month,
  r.period,
  r.n_users,
  cs.cohort_n,
  ROUND(100.0 * r.n_users / cs.cohort_n, 1) AS retention_pct
FROM retention r
JOIN cohort_size cs USING (cohort_month)
ORDER BY 1, 2;
```

### Sessionization

```sql
-- 30 dk hareketsizlik = yeni session
WITH events_with_prev AS (
  SELECT
    user_id, event_ts,
    LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts
  FROM events
),
session_breaks AS (
  SELECT *,
    CASE
      WHEN DATEDIFF('minute', prev_ts, event_ts) > 30 OR prev_ts IS NULL
      THEN 1 ELSE 0
    END AS is_new_session
  FROM events_with_prev
),
session_ids AS (
  SELECT *,
    SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_ts) AS session_num
  FROM session_breaks
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

> **Senior Notu:** DS mülakatlarının büyük bölümü window function. DataLemur ve StrataScratch'te en az 20 soru çöz. Sorgu performansı için: `EXPLAIN ANALYZE` ile planı oku, `SELECT *` kullanma, date sütunlarında partition pruning yap.

---

## A.6 Veri Görselleştirme ve Hikâyeleştirme

### Grafik Seçim Rehberi

| Soru | Grafik |
|------|--------|
| Dağılım nasıl? | histogram + KDE |
| Uç değer var mı? | boxplot, violin plot |
| İki değişken ilişkisi? | scatter + trend çizgisi |
| Kategori karşılaştırması? | barplot (sıralı!) |
| Zaman trendi? | lineplot + rolling average |
| Korelasyon matrisi? | heatmap (annotated) |
| Bileşim (oran)? | stacked bar (pie değil) |

### Kod Örneği

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Senior kalitesinde grafik
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

np.random.seed(42)
n = 500
df = pd.DataFrame({
    "amount": np.random.lognormal(5.5, 0.8, n),
    "segment": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
    "churn": np.random.binomial(1, 0.1, n)
})

# 1. Dağılım: histogram + KDE
axes[0].hist(df["amount"], bins=40, density=True, alpha=0.6, color="steelblue")
from scipy.stats import gaussian_kde
kde = gaussian_kde(df["amount"])
x_range = np.linspace(df["amount"].min(), df["amount"].max(), 200)
axes[0].plot(x_range, kde(x_range), color="darkblue", lw=2)
axes[0].set_title("Sipariş Tutarı Dağılımı\n(Log-normal, sağa çarpık)")
axes[0].set_xlabel("Tutar (TL)")

# 2. Segment karşılaştırması: violin + box overlay
parts = axes[1].violinplot([df[df["segment"] == s]["amount"].values for s in ["A", "B", "C"]],
                            positions=[0, 1, 2], showmedians=True)
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["A", "B", "C"])
axes[1].set_title("Segment Bazlı Tutarlar")

# 3. Zaman trendi
df["date"] = pd.date_range("2024-01-01", periods=n, freq="D").repeat(1)[:n]
monthly = df.set_index("date").resample("W")["amount"].mean()
axes[2].plot(monthly.index, monthly.values, color="steelblue", lw=2)
axes[2].fill_between(monthly.index, monthly.values, alpha=0.15, color="steelblue")
axes[2].set_title("Haftalık Ortalama Tutar")

plt.suptitle("Müşteri Analizi — Q1 2024", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("customer_analysis.png", dpi=150, bbox_inches="tight")
```

### Görsel Yalanlar ve Tuzaklar

- **Y ekseni sıfırdan başlamıyor:** Küçük farkı büyük gösterir — kandırmaca
- **Çok fazla kategori:** >7 kategori → grupla veya renkli bar chart
- **Çift eksen (dual axis):** Yanıltıcı ilişki hissi yaratır — kaçın
- **3D grafik:** Neredeyse her zaman gereksiz ve yanıltıcı
- **Pie chart:** >3 dilimde karşılaştırma zorlaşır → stacked bar tercih et
- **Renk körlüğü:** `viridis`, `cividis`, `okabe-ito` palette kullan

### Executive Summary Şablonu

```markdown
## Analiz Özeti — [Tarih]

**Bulgu:** [Tek cümle, sayı içeren]

**Kanıt:**
- [Grafik 1 yorumu — sayısal]
- [Grafik 2 yorumu]
- [Risk veya kısıt]

**Öneri:** [Aksiyon + beklenen etki + güven aralığı]

**Sonraki Adım:** [Kim yapacak, ne zaman]
```

> **Senior Notu:** "1 slayt = 1 mesaj" kuralı. Başlık = sonuç ("X %Y arttı"), grafik = kanıt, bullet = risk ve öneri. Teknik detay yedek slayta.

---

## A.8 Data Validation — Veri Kalite Kontrolü

### Sezgisel Açıklama

"Garbage in, garbage out" — modelin girdisi çöpse, çıktısı da çöptür. Ne kadar sofistike bir model kullanırsan kullan, eğer giren veri bozuksa sonuç anlamsızdır. Data validation, verinin beklenen şemaya, tiplere, aralıklara ve iş kurallarına uyup uymadığını **otomatik** olarak kontrol eder. Manuel göz gezdirme ölçeklenmez; validation kodu pipeline'ın parçası olmalı.

### Pandera ile DataFrame Schema Doğrulama

Pandera, Pandas (ve Polars) DataFrame'leri için tip kontrolü, değer aralığı ve null kontrolünü deklaratif olarak tanımlamana olanak sağlar.

```python
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Schema tanımı — veri nasıl görünmeli?
order_schema = DataFrameSchema(
    columns={
        "user_id":    Column(int, Check.gt(0), nullable=False),
        "amount":     Column(float, [
                          Check.ge(0),              # Negatif tutar olamaz
                          Check.le(100_000),         # Mantıklı üst sınır
                      ], nullable=False),
        "order_date": Column("datetime64[ns]", nullable=False),
        "country":    Column(str, Check.isin(["TR", "US", "DE", "UK", "FR"]),
                         nullable=False),
        "status":     Column(str, Check.isin(["completed", "pending", "cancelled"]),
                         nullable=True),
    },
    # DataFrame seviyesinde kontroller
    checks=[
        Check(lambda df: df["order_date"].max() <= pd.Timestamp.now(),
              error="Gelecek tarihli sipariş olamaz!"),
    ],
    coerce=True,  # Tipleri otomatik dönüştürmeyi dene
)

# Kullanım
df = pd.read_csv("orders.csv", parse_dates=["order_date"])

try:
    validated_df = order_schema.validate(df, lazy=True)  # lazy=True: tüm hataları topla
    print("✓ Veri doğrulandı!")
except pa.errors.SchemaErrors as err:
    print(f"✗ {len(err.failure_cases)} doğrulama hatası bulundu:")
    print(err.failure_cases)  # Hangi satır, hangi sütun, ne beklendi?
```

```python
# Pandera ile class-based API (daha okunabilir, büyük projelerde tercih edilir)
from pandera import DataFrameModel, Field
import pandera as pa

class OrderSchema(pa.DataFrameModel):
    user_id:    int   = Field(gt=0, nullable=False)
    amount:     float = Field(ge=0, le=100_000, nullable=False)
    order_date: pa.DateTime = Field(nullable=False)
    country:    str   = Field(isin=["TR", "US", "DE", "UK", "FR"])
    status:     str   = Field(isin=["completed", "pending", "cancelled"], nullable=True)

    class Config:
        coerce = True
        strict = True  # Tanımlanmayan sütun varsa hata ver

# Fonksiyon dekoratörü ile otomatik validation
@pa.check_types
def process_orders(df: pa.typing.DataFrame[OrderSchema]) -> pd.DataFrame:
    """Giren veri otomatik olarak schema'ya karşı doğrulanır."""
    return df.groupby("country")["amount"].sum().reset_index()
```

### Great Expectations ile Data Quality Kontrolleri

Great Expectations (GX), daha büyük ölçekli data pipeline'lar için kullanılan bir veri kalitesi framework'üdür. Her bir "Expectation" bir iş kuralını tanımlar.

```python
import great_expectations as gx

# Context oluştur
context = gx.get_context()

# Pandas DataFrame'i data source olarak ekle
data_source = context.data_sources.add_pandas("orders_source")
data_asset = data_source.add_dataframe_asset(name="orders")

# Expectation suite tanımla
suite = context.suites.add(
    gx.ExpectationSuite(name="orders_quality_checks")
)

# Expectation'ları ekle — her biri bir iş kuralı
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="amount", min_value=0, max_value=100_000
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeInSet(
        column="status", value_set=["completed", "pending", "cancelled"]
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="order_date",
        min_value="2020-01-01",
        max_value="2026-12-31"
    )
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="order_id")
)

# Validation çalıştır
batch = data_asset.add_batch_definition_whole_dataframe("full_batch")
results = batch.validate(suite)

# Sonuçları incele
if results.success:
    print("✓ Tüm kalite kontrolleri geçti!")
else:
    for result in results.results:
        if not result.success:
            print(f"✗ BAŞARISIZ: {result.expectation_config.type}")
            print(f"  Detay: {result.result}")
```

### Pandera vs Great Expectations — Hangisi Ne Zaman?

| Kriter | Pandera | Great Expectations |
|--------|---------|-------------------|
| Kullanım kolaylığı | Basit, hızlı başlangıç | Daha fazla kurulum |
| Notebook prototipi | İdeal | Ağır kalabilir |
| Production pipeline | Orta ölçek | Enterprise seviye |
| Data docs / raporlama | Yok | Otomatik HTML rapor |
| Polars desteği | Doğrudan | Sınırlı |

> **Senior Notu:** Data validation pipeline'ın **ilk** adımı olmalı, son değil. `read_csv()` → `validate()` → işle sırası doğru sıradır. Bozuk veriyi downstream'de yakalamak 10× daha pahalıdır. CI/CD'ye validation testi ekle: veri değiştiğinde otomatik kontrol. Bkz. `katman-E-data-pipeline.md` — pipeline orkestrasyon ile validation entegrasyonu.

---

## A.9 EDA (Exploratory Data Analysis) Çerçevesi

### Sezgisel Açıklama

EDA yapma sırası: önce veri kalitesi, sonra dağılımlar, sonra ilişkiler, son olarak segment analizi. Her adımda "Bu beklediğim mi?" sor.

```python
import pandas as pd
import numpy as np
import missingno as msno

def eda_raporu(df: pd.DataFrame, target_col: str = None):
    """Standart EDA raporu."""
    print("=" * 60)
    print(f"Şekil: {df.shape}")
    print(f"\nVeri tipleri:\n{df.dtypes.value_counts()}")

    # Eksik değer analizi
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\nEksik değer oranları:\n{missing}")
    else:
        print("\nEksik değer yok")

    # Sayısal özet
    print(f"\nSayısal özet:\n{df.describe().round(2)}")

    # Kardinalite
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print(f"\nKategorik sütunlar — kardinalite:")
        for c in cat_cols:
            print(f"  {c}: {df[c].nunique()} benzersiz")

    # Target analizi
    if target_col and target_col in df.columns:
        print(f"\nTarget ({target_col}) dağılımı:")
        print(df[target_col].value_counts(normalize=True).round(3))

    return missing

# Outlier tespiti
def outlier_report(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """IQR tabanlı outlier raporu."""
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    reports = []
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        reports.append({
            "col": col,
            "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "lower": lower, "upper": upper,
            "n_outliers": n_outliers,
            "outlier_pct": n_outliers / len(df)
        })
    return pd.DataFrame(reports)
```

---

---

## A.10 Alıştırma Soruları

Aşağıdaki soruları çözmeden Katman B'ye geçme. Her soru gerçek iş senaryolarından türetilmiştir.

### Soru 1 — Pandas Tuzağı: Sessiz Hata

Aşağıdaki kodda ne yanlış? Hatayı bul ve düzelt.

```python
df = pd.read_csv("orders.csv")
high_value = df[df["amount"] > 1000]
high_value["segment"] = "premium"      # Bu satırda ne olur?
print(df["segment"].value_counts())    # Beklenen sonucu verir mi?
```

**İpucu:** `SettingWithCopyWarning` — `.copy()` ve `.loc` arasındaki fark nedir?

### Soru 2 — SQL Window Function: Kullanıcı Bazında Sıralama

Bir `orders` tablosunda her kullanıcının en yüksek tutarlı 3 siparişini getiren bir SQL sorgusu yaz. Eşit tutarlı siparişlerde `order_date`'i erken olanı tercih et. `ROW_NUMBER`, `RANK` ve `DENSE_RANK` arasındaki farkı açıkla — bu soruda hangisi doğru seçim?

### Soru 3 — Veri Temizleme Pipeline'ı

Bir CSV dosyasında şu sorunlar var:
- `price` sütununda "N/A", "null", "-" gibi string değerler
- `email` sütununda bazı değerler büyük harf, bazıları küçük harf
- `date` sütununda "2024-01-15", "15/01/2024", "Jan 15, 2024" gibi farklı formatlar

Bu üç sorunu çözen bir `clean_dataframe(df)` fonksiyonu yaz. Pandera ile temizlenmiş verinin şemasını doğrula.

### Soru 4 — DuckDB vs Pandas Performans Karşılaştırması

10 milyon satırlık bir Parquet dosyasında şu analizi hem Pandas hem DuckDB ile yap:
- Aylık cohort bazında kullanıcı sayısı
- Her cohort'un 1., 2., 3. ay retention oranı

Her iki çözümü `%%timeit` ile ölç. Hangi yaklaşım daha hızlı ve neden?

### Soru 5 — Bootstrap vs Parametrik CI

Sağa çarpık bir gelir verisinde (log-normal dağılım) 100 örneklem al. Hem parametrik (t-dağılımı) hem bootstrap CI hesapla. Hangisi medyana daha yakın? Neden? Sonuçları bir grafik ile göster.

### Soru 6 — Polars Lazy Evaluation

Aşağıdaki Pandas kodunu Polars lazy API'ye çevir. `group_by` (v0.20+ API) kullan. Lazy evaluation'ın query optimizer'ı hangi adımları birleştirir?

```python
df = pd.read_csv("large_orders.csv")
df = df[df["amount"] > 0]
df["year"] = pd.to_datetime(df["order_date"]).dt.year
result = df.groupby(["year", "country"]).agg(
    total=("amount", "sum"),
    avg_order=("amount", "mean"),
    n_users=("user_id", "nunique")
).reset_index().sort_values("total", ascending=False)
```

### Soru 7 — Data Validation Senaryosu

Bir ML pipeline'ında model eğitim verisi her gece güncelleniyor. Geçen hafta `age` sütununda negatif değerler ve `salary` sütununda NaN'ler girdi, model sessizce bozuk tahminler üretti. Pandera ile bu durumu önleyen bir validation katmanı tasarla. Validation başarısız olduğunda pipeline'ı durduran bir mekanizma ekle.

---

## Çapraz Referanslar

| Konu | İlgili Katman | Dosya |
|------|--------------|-------|
| Büyük veri işleme (Spark, Dask) | Katman H | `katman-H-buyuk-veri.md` |
| Data pipeline orkestrasyon (Airflow, Prefect) | Katman E | `katman-E-mlops.md` |
| Data validation + pipeline entegrasyonu | Katman E | `katman-E-mlops.md` |
| Feature engineering (ileri seviye) | Katman B | `katman-B-klasik-ml.md` |
| DuckDB/Polars ile büyük ölçek analitik | Katman H | `katman-H-buyuk-veri.md` |
| A/B testi istatistik detayları | Katman C | `katman-C-deney-nedensellik.md` |

---

## Katman A Kontrol Listesi

Katman A'yı tamamlamadan Katman B'ye geçme:

- [ ] Python: OOP, type hints, pytest, logging biliyorum
- [ ] Pandas: SettingWithCopyWarning anlıyorum, .loc kullanıyorum
- [ ] Pandas: groupby+agg, merge+validate, zaman serisi özellikleri
- [ ] Polars ve DuckDB ile en az 1 örnek yaptım
- [ ] Polars: `group_by` (v0.20+) API değişikliğini biliyorum
- [ ] DuckDB: Pandas DataFrame ve Parquet üzerinde SQL sorgusu yazdım
- [ ] SQL: 10+ window function sorusu çözdüm
- [ ] SQL: cohort retention + funnel sorgusu yazdım
- [ ] İstatistik: Bootstrap CI vs parametrik CI farkını biliyorum
- [ ] İstatistik: p-value'nun ne olmadığını açıklayabilirim
- [ ] Data validation: Pandera ile schema doğrulama yaptım
- [ ] EDA: eksik değer, outlier, dağılım, korelasyon analizi yapabilirim
- [ ] Alıştırma sorularının en az 5'ini tamamladım
- [ ] Proje-0 tamamlandı (analysis.ipynb + queries.sql + README.md)

---

<div class="nav-footer">
  <span><a href="#file_katman_0_matematik">← Önceki: Katman 0 — Matematik</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_B_klasik_ml">Sonraki: Katman B — Klasik ML →</a></span>
</div>
