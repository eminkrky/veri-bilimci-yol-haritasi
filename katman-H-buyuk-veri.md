# Katman H — Büyük Veri ve Dağıtık Hesaplama

> Bu katmanda ne öğrenilir: Tek makine sınırlarını aşmak. Apache Spark, Dask, Polars, cloud veri ambarları ve dağıtık hesaplama paradigmaları.
>
> Süre: 1–2 hafta. Büyük veri gerektiren rollar için (data-intensive DS, platformlar).
>
> **Ne zaman bu katmanı öğren:** Pandas yavaşladığında veya veri >5–10 GB olduğunda.


<div class="prereq-box">
<strong>Önkoşul:</strong> <strong>Katman A</strong> (Pandas, SQL) ve <strong>Katman E</strong> (MLOps) tamamlanmış olmalı. Büyük veri araçları gereksiz karmaşıklık ekler — önce küçük veriyle ustalaş.
</div>

---

## H.1 Ne Zaman Büyük Veri Gerekir?

### Sezgisel Açıklama

"Büyük veri" ezber etmek için değil, doğru zamanda doğru araç seçmek için öğrenilir.

```
Veri boyutu?
  < 1 GB   → Pandas + DuckDB yeterli
  1–10 GB  → Polars (lazy, Rust) + DuckDB
  10–100 GB→ Dask veya PySpark (lokal cluster)
  > 100 GB → Apache Spark + cloud (Databricks, EMR)
  Streaming → Kafka + Flink/Spark Structured Streaming
```

**Kural:** Önce örnekle çalış. Çoğu "büyük veri" problemi, akıllı örnekleme + Parquet ile küçülür.

```python
# Önce boyutu anla
import os
import pandas as pd

# Dosya boyutu
size_gb = os.path.getsize("data.csv") / 1024**3
print(f"Dosya boyutu: {size_gb:.1f} GB")

# Örnek ile incele
df_sample = pd.read_csv("data.csv", nrows=10_000)
print(f"Tahmini satır sayısı: {size_gb / (df_sample.memory_usage().sum() / 1024**3):.0f}")

# Parquet'e çevir (5–10× küçülür)
df_sample.to_parquet("data_sample.parquet", compression="snappy")
```

---

## H.2 Polars — Hızlı Tek Makine

### Sezgisel Açıklama

Polars, Rust ile yazılmış, lazy evaluation destekleyen bir DataFrame kütüphanesi. Pandas'tan 10–100× hızlı. Büyük veriyi Spark'a gerek kalmadan işler.

2026 itibarıyla 100M satıra kadar Polars tek makineyi yeterince kullanır.

### Kod Örneği

```python
import polars as pl

# Lazy API — query optimizer devreye girer
result = (
    pl.scan_parquet("data/*.parquet")      # Lazy okuma
    .filter(pl.col("amount") > 0)
    .filter(pl.col("date") >= pl.lit("2024-01-01").cast(pl.Date))
    .group_by("user_id")
    .agg([
        pl.col("amount").sum().alias("total_spend"),
        pl.col("amount").count().alias("n_orders"),
        pl.col("amount").mean().alias("avg_order"),
        pl.col("date").max().alias("last_order_date"),
    ])
    .sort("total_spend", descending=True)
    .collect()                              # Hesaplamayı başlat
)

# Pandas'tan Polars'a geçiş
import pandas as pd

# Pandas DataFrame → Polars
df_pl = pl.from_pandas(df_pandas)

# Polars → Pandas (gerektiğinde)
df_pandas = df_pl.to_pandas()

# Polars ile zaman serisi
df = pl.scan_csv("time_series.csv")
result = (
    df
    .with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    ])
    .with_columns([
        pl.col("amount").rolling_mean(window_size=7, min_periods=1)
            .over("user_id").alias("rolling_7d_mean"),
        pl.col("amount").shift(1).over("user_id").alias("prev_amount"),
    ])
    .collect()
)

# Benchmark: Polars vs Pandas
import time

# Polars
start = time.time()
result_pl = (
    pl.scan_csv("large_data.csv")
    .group_by("category")
    .agg(pl.col("amount").sum())
    .collect()
)
t_polars = time.time() - start

# Pandas
start = time.time()
df_pd = pd.read_csv("large_data.csv")
result_pd = df_pd.groupby("category")["amount"].sum()
t_pandas = time.time() - start

print(f"Polars: {t_polars:.2f}s | Pandas: {t_pandas:.2f}s | Fark: {t_pandas/t_polars:.0f}×")
```

> **Senior Notu:** Polars lazy API ile columnar processing + parallelism + query optimization aynı anda çalışır. `scan_*` fonksiyonları lazy, `read_*` eager. Büyük veride her zaman lazy başla.

---

## H.3 DuckDB — SQL ile Analitik

### Sezgisel Açıklama

DuckDB = "SQLite for analytics". Parquet, CSV, Arrow doğrudan sorgular. Python, R, CLI veya Wasm ile çalışır. Kurulum yok, veritabanı yok.

```python
import duckdb

# Parquet dosyaları üzerinde doğrudan SQL
result = duckdb.sql("""
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        COUNT(DISTINCT user_id) AS n_users,
        SUM(amount) AS total_revenue,
        AVG(amount) AS avg_order
    FROM 'data/orders/*.parquet'
    WHERE order_date >= '2024-01-01'
      AND amount > 0
    GROUP BY 1
    ORDER BY 1
""").df()  # pandas DataFrame'e çevir

# Polars DataFrame üzerinde SQL
import polars as pl

df = pl.scan_parquet("orders.parquet").collect()
result = duckdb.sql("SELECT user_id, SUM(amount) FROM df GROUP BY 1").pl()

# S3 üzerindeki Parquet (boto3 gerekmez!)
duckdb.sql("""
    INSTALL httpfs;
    LOAD httpfs;
    SET s3_region='eu-central-1';
    SELECT COUNT(*) FROM 's3://bucket/prefix/*.parquet'
""")

# Window functions (hızlı!)
cohort_result = duckdb.sql("""
    WITH first_order AS (
        SELECT user_id, MIN(DATE_TRUNC('month', order_date)) AS cohort_month
        FROM 'orders.parquet'
        GROUP BY user_id
    )
    SELECT
        cohort_month,
        DATE_DIFF('month', cohort_month, DATE_TRUNC('month', order_date)) AS period,
        COUNT(DISTINCT o.user_id) AS n_users
    FROM 'orders.parquet' o
    JOIN first_order f USING (user_id)
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
```

---

## H.4 Apache Spark — Dağıtık İşleme

### Sezgisel Açıklama

Spark = verinin birden fazla makineye dağıtılarak işlenmesi. 100GB+ veri veya streaming için standart.

Temel kavramlar:
- **RDD:** Temel veri yapısı (artık DataFrame API tercih edilir)
- **DataFrame/Dataset:** Pandas benzeri ama dağıtık
- **Lazy evaluation:** Action çağrılana kadar hesaplama yapılmaz
- **Transformation vs Action:** filter/map lazy; count/collect eager

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# SparkSession oluştur
spark = (SparkSession.builder
         .appName("DS-Analytics")
         .config("spark.executor.memory", "4g")
         .config("spark.driver.memory", "2g")
         .getOrCreate())

# Veri oku (S3, HDFS, lokal)
df = spark.read.parquet("s3://bucket/orders/")
print(f"Satır sayısı: {df.count()}")
print(f"Şema: {df.printSchema()}")

# Transformation'lar (lazy)
df_filtered = df.filter(F.col("amount") > 0)
df_enriched = df_filtered.withColumn("year_month",
    F.date_trunc("month", F.col("order_date")))

# Aggregation
monthly_stats = (
    df_enriched
    .groupBy("year_month", "country")
    .agg(
        F.count("order_id").alias("n_orders"),
        F.sum("amount").alias("total_revenue"),
        F.avg("amount").alias("avg_order"),
        F.countDistinct("user_id").alias("n_users"),
    )
    .orderBy("year_month", "country")
)

# Action — hesaplamayı tetikle
monthly_stats.show(20)
monthly_stats.write.parquet("s3://bucket/output/monthly_stats/", mode="overwrite")

# Window functions
window_spec = Window.partitionBy("user_id").orderBy("order_date")

df_with_features = df.withColumns({
    "prev_amount": F.lag("amount", 1).over(window_spec),
    "row_num": F.row_number().over(window_spec),
    "cumulative_spend": F.sum("amount").over(window_spec),
    "rolling_7d_avg": F.avg("amount").over(
        window_spec.rowsBetween(-6, 0)
    ),
})

# MLlib ile dağıtık ML
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Feature assembler
assembler = VectorAssembler(
    inputCols=["n_orders", "total_spend", "days_since_last_order"],
    outputCol="features"
)

# Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Model
lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="churn",
    maxIter=100
)

# Pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# Değerlendirme
evaluator = BinaryClassificationEvaluator(labelCol="churn")
auc = evaluator.evaluate(model.transform(test_df))
print(f"Test AUC: {auc:.4f}")

spark.stop()
```

### Spark Performans İpuçları

```python
# 1. Partition sayısı: veri boyutu / 200MB = partition sayısı
df = df.repartition(20)  # Çok küçük partition birleştir

# 2. Broadcast join: küçük tablo (<100MB) → her worker'a kopyala
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_lookup_df), "key")

# 3. Cache: çok kez kullanılacak DataFrame
df_filtered.cache()
df_filtered.count()  # Cache tetikle

# 4. Persist: farklı storage level
from pyspark.storagelevel import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# 5. Avoid UDF (Python UDF yavaş — Pandas UDF daha iyi)
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np

@pandas_udf("float")
def custom_feature(amount: pd.Series) -> pd.Series:
    return np.log1p(amount)  # Vektörize → hızlı

df = df.withColumn("log_amount", custom_feature(F.col("amount")))
```

---

## H.5 Cloud Veri Ambarları

### Google BigQuery

```sql
-- BigQuery özel sözdizimi
-- Partition + clustering → maliyet kontrolü
CREATE TABLE `project.dataset.orders`
PARTITION BY DATE(order_date)
CLUSTER BY country, user_id
AS SELECT * FROM raw_orders;

-- UNNEST → nested/repeated field açma
SELECT
    user_id,
    item.product_id,
    item.quantity,
    item.price
FROM orders,
UNNEST(order_items) AS item;

-- ARRAY_AGG → alt tablo sıkıştırma
SELECT
    user_id,
    ARRAY_AGG(STRUCT(order_date, amount) ORDER BY order_date) AS order_history
FROM orders
GROUP BY user_id;

-- Approx functions → sampling ile hızlı analiz
SELECT APPROX_COUNT_DISTINCT(user_id) AS approx_unique_users
FROM events;

-- BigQuery ML
CREATE OR REPLACE MODEL `dataset.churn_model`
OPTIONS(
    model_type='logistic_reg',
    input_label_cols=['churn'],
    auto_class_weights=TRUE
) AS
SELECT * FROM training_data;
```

### Databricks ve Delta Lake

```python
# Delta Lake — ACID transaction + time travel
from delta.tables import DeltaTable

# Delta tablo oluştur
spark.range(0, 100).write.format("delta").save("/delta/orders")

# Upsert (Merge)
delta_table = DeltaTable.forPath(spark, "/delta/orders")
delta_table.alias("existing").merge(
    new_data.alias("updates"),
    "existing.order_id = updates.order_id"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

# Time travel
df_yesterday = spark.read.format("delta").option("versionAsOf", 5).load("/delta/orders")
df_7days_ago = spark.read.format("delta").option("timestampAsOf", "2024-03-15").load("/delta/orders")

# Schema evolution
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

# Streaming ile real-time güncelleme
stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "broker:9092")
    .option("subscribe", "order_events")
    .load()
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", "/checkpoint/orders")
    .start("/delta/orders")
)
```

### Apache Iceberg — Vendor-Agnostic Lakehouse

2026 itibarıyla Apache Iceberg, Delta Lake'in en güçlü alternatifi. Multi-cloud ve vendor-agnostic yapısı sayesinde Snowflake, BigQuery, Spark, Flink, Trino ve Athena gibi birçok engine ile çalışır.

#### Delta Lake vs Iceberg Karşılaştırma

| Özellik | Delta Lake | Apache Iceberg |
|---------|-----------|----------------|
| ACID transactions | Evet | Evet |
| Time travel | Evet | Evet |
| Schema evolution | Evet (auto-merge) | Evet (add/rename/drop/reorder) |
| Partition evolution | Sınırlı | Güçlü (hidden partitioning) |
| Engine desteği | Spark-ağırlıklı | Multi-engine (Spark, Flink, Trino, Presto) |
| Vendor bağımlılığı | Databricks ekosistemi | Vendor-agnostic |
| Metadata yönetimi | `_delta_log` (JSON + Parquet) | Manifest dosyaları (snapshot-based) |
| Update-heavy workload | Güçlü | İyi |
| Büyük partition sayısı | İyi | Çok güçlü (partition pruning) |
| Topluluk/ekosistem | Databricks destekli | Apache Foundation, geniş topluluk |

#### Iceberg Temel Kullanım (Spark ile)

```python
from pyspark.sql import SparkSession

# Iceberg destekli SparkSession
spark = (SparkSession.builder
         .appName("Iceberg-Demo")
         .config("spark.jars.packages",
                 "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.1")
         .config("spark.sql.catalog.my_catalog",
                 "org.apache.iceberg.spark.SparkCatalog")
         .config("spark.sql.catalog.my_catalog.type", "hadoop")
         .config("spark.sql.catalog.my_catalog.warehouse", "s3://bucket/warehouse")
         .getOrCreate())

# Iceberg tablo oluştur
spark.sql("""
    CREATE TABLE my_catalog.db.orders (
        order_id    BIGINT,
        user_id     BIGINT,
        amount      DOUBLE,
        order_date  DATE
    )
    USING iceberg
    PARTITIONED BY (month(order_date))
""")

# Veri yaz
df.writeTo("my_catalog.db.orders").append()

# Time travel — snapshot ile
spark.read.option("snapshot-id", 123456789).table("my_catalog.db.orders")

# Schema evolution — kolon ekle (mevcut veri etkilenmez)
spark.sql("ALTER TABLE my_catalog.db.orders ADD COLUMN category STRING")

# Partition evolution — yeniden yazma gerekmez!
spark.sql("ALTER TABLE my_catalog.db.orders ADD PARTITION FIELD bucket(16, user_id)")
```

#### PyIceberg ile Python-Native Erişim

```python
from pyiceberg.catalog import load_catalog

catalog = load_catalog("my_catalog", **{
    "type": "rest",
    "uri": "http://localhost:8181"
})

# Tablo yükle
table = catalog.load_table("db.orders")

# Arrow ile oku
arrow_table = table.scan(
    row_filter="amount > 100",
    selected_fields=("order_id", "user_id", "amount"),
).to_arrow()

# Pandas'a çevir
df = arrow_table.to_pandas()
```

#### Ne Zaman Hangisi?

- **Delta Lake tercih et:** Databricks ekosistemindeysen, Spark-ağırlıklı iş yükün varsa, streaming-heavy senaryolarda
- **Iceberg tercih et:** Multi-cloud / vendor-agnostic istiyorsan, çok sayıda partition varsa, Snowflake/Trino/Athena ile çalışıyorsan
- **UniForm:** Databricks'in UniForm özelliği Delta tablolarını Iceberg client'ları ile okunabilir kılıyor — her iki dünyayı köprülüyor

---

## H.6 Dask — Pandas'ı Ölçekle

```python
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

# Dask Client — lokal cluster
client = Client(n_workers=4, threads_per_worker=2, memory_limit="4GB")
print(client.dashboard_link)  # Tarayıcıda izle

# Pandas gibi ama lazy ve paralel
df = dd.read_parquet("data/*.parquet")  # Lazy
print(df.dtypes)  # Metadata anında

# Aggregation
result = (
    df
    .groupby("user_id")["amount"]
    .agg(["sum", "mean", "count"])
    .compute()  # Eager — hesaplamayı başlat
)

# Map-partitions — her partisyona custom fonksiyon
def feature_engineering(partition: pd.DataFrame) -> pd.DataFrame:
    partition["log_amount"] = np.log1p(partition["amount"])
    return partition

df_featured = df.map_partitions(feature_engineering)

# Büyük array işlemleri
X_big = da.from_array(np.random.randn(10_000_000, 50), chunks=(100_000, 50))
result = da.dot(X_big, X_big.T[:50]).compute()

client.close()
```

### Dask-ML ile Dağıtık Model Eğitimi

```python
from dask_ml.model_selection import IncrementalSearchCV, HyperbandSearchCV
from dask_ml.wrappers import ParallelPostFit
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import dask.dataframe as dd

# --- IncrementalSearchCV: Büyük veri üzerinde aşamalı hyperparameter tuning ---
# SGDClassifier gibi partial_fit destekleyen modeller ile çalışır
from scipy.stats import uniform, loguniform

param_dist = {
    "alpha": loguniform(1e-5, 1e-1),
    "l1_ratio": uniform(0, 1),
}

model = SGDClassifier(loss="log_loss", penalty="elasticnet")

search = IncrementalSearchCV(
    model,
    param_dist,
    n_initial_parameters=20,   # Başlangıçta 20 kombinasyon dene
    max_iter=100,
    random_state=42,
)

# Dask array ile fit — veri parça parça worker'lara gider
search.fit(X_dask, y_dask, classes=[0, 1])
print(f"En iyi parametreler: {search.best_params_}")
print(f"En iyi skor: {search.best_score_:.4f}")

# --- ParallelPostFit: Eğitilmiş modeli paralel predict/transform ---
# Eğitim küçük veri, tahmin büyük veri senaryoları için
from sklearn.ensemble import RandomForestClassifier

# Küçük veri ile eğit
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf.fit(X_train, y_train)

# Büyük veri üzerinde paralel tahmin
parallel_clf = ParallelPostFit(clf)
predictions = parallel_clf.predict(X_big_dask)       # Dask array
probas = parallel_clf.predict_proba(X_big_dask)       # Paralel çalışır
```

### Dask + RAPIDS (GPU Hızlandırma)

NVIDIA RAPIDS, Dask ile entegre çalışarak GPU üzerinde DataFrame ve ML işlemlerini hızlandırır.

```python
# cuDF: GPU-hızlandırılmış DataFrame (Pandas API uyumlu)
import cudf
import dask_cudf

# GPU üzerinde Parquet oku
gdf = dask_cudf.read_parquet("data/*.parquet")

# Pandas/Dask API'si ile çalış — ama GPU'da
result = (
    gdf
    .groupby("user_id")["amount"]
    .agg(["sum", "mean"])
    .compute()  # GPU'da hesapla
)

# cuML: GPU-hızlandırılmış ML (scikit-learn API uyumlu)
# from cuml.ensemble import RandomForestClassifier as cuRF
# model = cuRF(n_estimators=100)
# model.fit(X_gpu, y_gpu)
```

> **Senior Notu:** RAPIDS, NVIDIA GPU ve CUDA gerektirir. Cloud ortamlarında (AWS g4dn, GCP A100) kullanılabilir. Lokal geliştirmede cuDF yerine Polars tercih et; production GPU workload'larında RAPIDS değerlendir.

### Araç Seçim Rehberi

| Senaryo | Araç |
|---------|------|
| <5GB, Python | Pandas |
| 5–100GB, tek makine | Polars (lazy) + DuckDB |
| SQL tabanlı analiz, Parquet | DuckDB |
| >100GB, dağıtık | PySpark |
| Pandas kodu ölçekle | Dask |
| ML workload dağıt | Ray |
| Streaming | Kafka + Flink |
| Cloud analiz, serverless | BigQuery, Snowflake |
| Lakehouse, ACID | Delta Lake, Apache Iceberg |

---

## H.6.1 Ray — Dağıtık ML Platformu

### Sezgisel Açıklama

Ray, Python fonksiyonlarını dağıtık çalıştırmak için tasarlanmış bir framework. Dask veri işleme odaklıyken, Ray ML workload'ları (eğitim, tuning, inference) için optimize. OpenAI, ChatGPT eğitiminde Ray kullanıyor.

### Ray Temel Kullanım

```python
import ray

ray.init()  # Lokal cluster başlat (veya ray://cluster-address)

# Herhangi bir fonksiyonu dağıtık çalıştır
@ray.remote
def process_partition(partition_id):
    """Her partition bağımsız işlenir."""
    import pandas as pd
    df = pd.read_parquet(f"data/part_{partition_id}.parquet")
    return df.groupby("category")["amount"].sum()

# 100 partition paralel işle
futures = [process_partition.remote(i) for i in range(100)]
results = ray.get(futures)  # Tüm sonuçları topla
```

### Ray Tune ile Hyperparameter Tuning

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_model(config):
    """Her deneme bir hyperparameter kombinasyonu dener."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import cross_val_score

    X, y = load_breast_cancer(return_X_y=True)

    model = GradientBoostingClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    tune.report({"auc": np.mean(scores)})

# ASHA scheduler: kötü denemeleri erken durdurur → kaynak tasarrufu
scheduler = ASHAScheduler(
    metric="auc",
    mode="max",
    max_t=100,
    grace_period=10,
)

# Arama alanı tanımla
search_space = {
    "n_estimators": tune.choice([50, 100, 200, 500]),
    "max_depth": tune.randint(2, 8),
    "learning_rate": tune.loguniform(1e-3, 0.3),
    "subsample": tune.uniform(0.6, 1.0),
}

# Tune çalıştır — N deneme paralel
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=50,          # 50 farklı kombinasyon
    scheduler=scheduler,
    resources_per_trial={"cpu": 2},
)

print(f"En iyi config: {analysis.best_config}")
print(f"En iyi AUC: {analysis.best_result['auc']:.4f}")
```

### Ray Data ile Büyük Veri İşleme

```python
import ray

# Ray Data: streaming veri pipeline'ı
ds = ray.data.read_parquet("s3://bucket/data/")

# Map batches — her batch üzerinde işlem
def feature_engineering(batch):
    import numpy as np
    batch["log_amount"] = np.log1p(batch["amount"])
    batch["amount_squared"] = batch["amount"] ** 2
    return batch

ds_featured = ds.map_batches(feature_engineering, batch_format="pandas")

# Model inference pipeline
def predict_batch(batch):
    import joblib
    model = joblib.load("model.pkl")      # Her worker modeli yükler
    batch["prediction"] = model.predict(batch[["log_amount", "amount_squared"]])
    return batch

ds_predictions = ds_featured.map_batches(predict_batch, batch_format="pandas")
ds_predictions.write_parquet("s3://bucket/predictions/")
```

### Ray vs Dask Karşılaştırma

| Özellik | Ray | Dask |
|---------|-----|------|
| Odak | ML workload (train, tune, serve) | Veri işleme (DataFrame, Array) |
| API | Low-level (remote functions) + high-level libs | Pandas/NumPy benzeri |
| GPU desteği | Güçlü (Ray Train + PyTorch) | RAPIDS ile |
| Hyperparameter tuning | Ray Tune (ASHA, PBT, Bayesian) | Dask-ML (sınırlı) |
| Model serving | Ray Serve | Yok (harici araç gerekir) |
| Pandas entegrasyonu | Dolaylı (Ray Data) | Doğrudan (Dask DataFrame) |
| Ne zaman kullan? | Dağıtık ML eğitimi, tuning, inference | Büyük DataFrame/Array işleme |

> **Kural:** Veri işleme ağırlıklıysa Dask, ML eğitimi/tuning ağırlıklıysa Ray tercih et. İkisi birlikte de kullanılabilir.

---

## H.7 Streaming ML — Gerçek Zamanlı Veri İşleme

### Sezgisel Açıklama — Batch vs Streaming

```
Batch işleme:
  Veri birikir → Toplu işle → Sonuç (saatler/günler sonra)
  Örnek: Günlük satış raporu, haftalık churn modeli

Streaming işleme:
  Veri gelir → Anında işle → Sonuç (saniyeler/dakikalar içinde)
  Örnek: Fraud detection, gerçek zamanlı öneri, anlık anomali tespiti
```

Fark basit: **batch** "dün ne oldu?" sorusunu yanıtlar, **streaming** "şu an ne oluyor?" sorusunu yanıtlar.

### Spark Structured Streaming — Window ve Watermark

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType

spark = SparkSession.builder.appName("StreamingDemo").getOrCreate()

# Kafka'dan streaming okuma
raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "broker:9092")
    .option("subscribe", "order_events")
    .option("startingOffsets", "latest")
    .load()
)

# JSON parse
schema = (StructType()
    .add("order_id", StringType())
    .add("user_id", StringType())
    .add("amount", DoubleType())
    .add("event_time", TimestampType()))

orders = (
    raw_stream
    .selectExpr("CAST(value AS STRING) as json_str")
    .select(F.from_json("json_str", schema).alias("data"))
    .select("data.*")
)

# --- Watermark + Window Aggregation ---
# Watermark: 10 dakikaya kadar geç gelen veriye izin ver
# Window: 5 dakikalık pencereler, 1 dakika kayma (sliding window)
windowed_stats = (
    orders
    .withWatermark("event_time", "10 minutes")  # Geç veri toleransı
    .groupBy(
        F.window("event_time", "5 minutes", "1 minute"),  # 5dk pencere, 1dk slide
        "user_id"
    )
    .agg(
        F.count("order_id").alias("order_count"),
        F.sum("amount").alias("total_amount"),
        F.avg("amount").alias("avg_amount"),
    )
)

# Konsola yaz (debug için)
query = (
    windowed_stats.writeStream
    .outputMode("update")           # Güncellenen pencereleri yaz
    .format("console")
    .option("truncate", False)
    .trigger(processingTime="30 seconds")  # 30 saniyede bir tetikle
    .start()
)

# Production'da: Delta Lake veya Kafka'ya yaz
# .format("delta").option("checkpointLocation", "/checkpoint/orders")
# .start("/delta/streaming_orders")
```

### River — Online (Streaming) Machine Learning

River, veri noktalarını tek tek işleyen online ML kütüphanesi. Model her yeni veri geldiğinde güncellenir — tüm veriyi bellekte tutmak gerekmez.

```python
from river import tree, metrics, stream, preprocessing, compose
import csv

# --- Online Classification: HoeffdingTreeClassifier ---
# Hoeffding Tree: streaming için tasarlanmış karar ağacı
# Veriyi tek geçişte (single-pass) öğrenir

model = compose.Pipeline(
    preprocessing.StandardScaler(),
    tree.HoeffdingTreeClassifier(grace_period=100, max_depth=10)
)

metric = metrics.Accuracy()
roc_auc = metrics.ROCAUC()

# Veriyi satır satır işle — batch yükleme yok
n_samples = 0
for x, y in stream.iter_csv("transactions.csv", target="is_fraud",
                             converters={"amount": float, "hour": int}):
    # 1. Tahmin yap (model henüz bu veriyi görmedi)
    y_pred = model.predict_one(x)

    # 2. Metriği güncelle
    metric.update(y, y_pred)
    roc_auc.update(y, model.predict_proba_one(x))

    # 3. Modeli güncelle (online learning)
    model.learn_one(x, y)

    n_samples += 1
    if n_samples % 10_000 == 0:
        print(f"[{n_samples:>8}] Accuracy: {metric.get():.4f} | ROC-AUC: {roc_auc.get():.4f}")

# Sonuç: model sürekli öğreniyor, bellek sabit kalıyor
print(f"Final Accuracy: {metric.get():.4f}")
print(f"Final ROC-AUC: {roc_auc.get():.4f}")
```

### Kafka → Spark → Model Inference Akış Şeması

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Kaynak    │     │  Spark Structured│     │   Model          │
│  Sistemler  │────>│  Streaming       │────>│   Inference      │
│  (App, IoT) │     │  (Parse + Enrich)│     │  (predict_batch) │
└─────────────┘     └──────────────────┘     └──────────────────┘
       │                     │                        │
       v                     v                        v
  ┌─────────┐        ┌────────────┐           ┌────────────┐
  │  Kafka  │        │  Watermark │           │   Sonuç    │
  │  Topic  │        │  + Window  │           │  (Delta /  │
  │         │        │  Agg       │           │   Kafka /  │
  └─────────┘        └────────────┘           │   DB)      │
                                              └────────────┘

Akış:
1. Kaynak sistemler event'leri Kafka topic'e yazar
2. Spark Structured Streaming Kafka'dan okur, JSON parse eder
3. Watermark ile geç veri yönetilir, window ile aggregation yapılır
4. Eğitilmiş model (MLflow'dan yüklenen) batch inference uygular
5. Sonuçlar Delta Lake'e (analiz) veya Kafka topic'e (aksiyon) yazılır
```

### "Ne Zaman Streaming Gerekir?" Karar Ağacı

```
Verinin yaşı önemli mi?
  │
  ├── Hayır, günlük/haftalık yeterli → BATCH (çoğu DS problemi)
  │
  └── Evet, dakikalar/saniyeler içinde sonuç lazım
       │
       ├── Sadece aggregation (count, sum, avg)?
       │    → Spark Structured Streaming + window
       │
       ├── Model sürekli güncellenmeli mi?
       │    ├── Evet → River (online learning)
       │    └── Hayır → Spark Streaming + önceden eğitilmiş model
       │
       └── Çok düşük latency (<100ms)?
            → Kafka Streams veya Flink (Spark değil)
```

> **Senior Notu:** Çoğu veri bilimi problemi batch ile çözülür. Streaming karmaşıklık ekler: state management, exactly-once semantics, checkpoint yönetimi, hata kurtarma. "Gerçek zamanlı" talebi geldiğinde ilk soru şu olmalı: "Gerçekten saniyeler içinde mi gerekiyor, yoksa saatlik batch yeterli mi?" Çoğu zaman micro-batch (5–15 dakika) yeterlidir.

---

## H.8 Credentials ve Kimlik Bilgileri Yönetimi

Büyük veri araçları genellikle cloud kaynaklarına erişim gerektirir. Kimlik bilgilerini güvenli yönetmek kritik.

### AWS S3 Erişimi — boto3 + credentials

```python
# 1. AWS credentials dosyası oluştur
# ~/.aws/credentials dosyası:
# [default]
# aws_access_key_id = AKIAIOSFODNN7EXAMPLE
# aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# 2. Python ile S3 erişimi
import boto3

# Credentials otomatik olarak ~/.aws/credentials'dan okunur
s3 = boto3.client("s3")

# Dosya listele
response = s3.list_objects_v2(Bucket="my-data-bucket", Prefix="data/")
for obj in response.get("Contents", []):
    print(obj["Key"], f"{obj['Size'] / 1024**2:.1f} MB")

# Parquet indir ve oku
s3.download_file("my-data-bucket", "data/orders.parquet", "/tmp/orders.parquet")

# Pandas ile doğrudan S3'ten oku (s3fs gerekir)
import pandas as pd
df = pd.read_parquet("s3://my-data-bucket/data/orders.parquet")
```

### GCP BigQuery — Service Account JSON

```python
# 1. GCP Console → IAM → Service Accounts → Key oluştur (JSON)
# İndirilen dosya: service-account-key.json

# 2. Environment variable ile ayarla
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# 3. Python ile BigQuery erişimi
from google.cloud import bigquery

# Credentials otomatik okunur (GOOGLE_APPLICATION_CREDENTIALS env var)
client = bigquery.Client(project="my-project-id")

query = """
    SELECT user_id, SUM(amount) as total
    FROM `my-project.dataset.orders`
    GROUP BY user_id
    ORDER BY total DESC
    LIMIT 100
"""

df = client.query(query).to_dataframe()
print(df.head())

# Alternatif: credentials dosyasını doğrudan belirt
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    "service-account-key.json",
    scopes=["https://www.googleapis.com/auth/bigquery"]
)
client = bigquery.Client(credentials=credentials, project="my-project-id")
```

### .env Dosyası ile Güvenli Credentials Yönetimi

```python
# 1. .env dosyası oluştur (proje kök dizininde)
# .env dosyası içeriği:
# AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
# AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# DATABASE_URL=postgresql://user:pass@host:5432/dbname
# KAFKA_BOOTSTRAP_SERVERS=broker1:9092,broker2:9092

# 2. python-dotenv ile yükle
from dotenv import load_dotenv
import os

load_dotenv()  # .env dosyasını oku

# Artık environment variable olarak erişilebilir
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
db_url = os.getenv("DATABASE_URL")
kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

# 3. Kod içinde kullan
import boto3
s3 = boto3.client("s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
```

### .gitignore Uyarısı

```gitignore
# .gitignore dosyasına MUTLAKA ekle:
.env
*.pem
*.key
service-account-key.json
credentials.json
.aws/
```

> **UYARI — Gizli Bilgileri Git'e Commitleme!**
> - `.env`, credentials dosyaları ve API key'leri ASLA Git repository'sine commitlenme.
> - Yanlışlıkla committlediysen: `git filter-branch` veya `BFG Repo-Cleaner` ile geçmişten temizle.
> - Daha iyisi: credential'ları hiç dosyaya yazma — AWS IAM Role, GCP Workload Identity gibi role-based erişim kullan.
> - CI/CD'de: GitHub Secrets, GitLab CI Variables veya HashiCorp Vault kullan.
> - Bkz. **Katman E** (MLOps) — production credential yönetimi ve secret rotation.

---

## H.9 Alıştırmalar

### Soru 1 — Araç Seçimi (Kavramsal)

Aşağıdaki senaryoların her biri için en uygun aracı seçin ve nedenini açıklayın:

a) 500 MB CSV dosyasında basit group-by analiz
b) 50 GB Parquet dosyalarında SQL ile cohort analizi
c) 200 GB event verisi üzerinde günlük feature engineering (Spark cluster mevcut)
d) E-ticaret sitesinde gerçek zamanlı fraud detection (100ms altı latency)
e) 10 farklı hyperparameter kombinasyonunu 8 GPU üzerinde paralel deneme

### Soru 2 — Streaming vs Batch Karar (Uygulama)

Bir e-ticaret şirketi aşağıdaki raporları istiyor. Hangisi batch, hangisi streaming olmalı? Her biri için teknolojiyi ve yenileme sıklığını belirtin:

a) Günlük gelir raporu
b) Anlık stok uyarısı (ürün 10'un altına düştüğünde)
c) Haftalık müşteri segmentasyonu
d) Gerçek zamanlı ürün önerisi
e) Aylık churn tahmin modeli yeniden eğitimi

### Soru 3 — Delta Lake vs Iceberg (Pratik)

Şirketiniz yeni bir data lakehouse kuracak. Aşağıdaki bilgilerle Delta Lake mi, Iceberg mi tercih edersiniz?

- 3 farklı cloud provider kullanılıyor (AWS, GCP, Azure)
- Analiz araçları: Spark, Trino, Athena
- Veri: 500+ partition ile bölümlenmiş 10 TB event verisi
- Mevcut ekip Databricks deneyimi yok

Kararınızı gerekçesiyle açıklayın. Hangi senaryoda kararınız değişirdi?

### Soru 4 — Dask Pipeline (Kod)

20 GB'lık bir CSV dosyanız var. Aşağıdaki pipeline'ı Dask ile yazın:
1. Veriyi lazy oku
2. `amount > 0` filtrele
3. `user_id` bazında `amount` toplamı ve ortalamasını hesapla
4. Sonucu Parquet olarak kaydet

Bonus: Aynı pipeline'ı Polars lazy API ile de yazın ve farkları tartışın.

---

## Çapraz Referanslar

- **Katman E (MLOps ve Production):** Data pipeline'ları, feature store entegrasyonu, model serving ve CI/CD. Streaming pipeline'lar production'a alınırken Katman E'deki monitoring ve logging pratikleri kritik.
- **Katman F (Sistem Tasarımı):** Feature store mimarisi, büyük ölçekli sistem tasarımı ve ML system design interview soruları. H katmanındaki araçlar (Spark, Kafka, Delta Lake) F katmanındaki sistem tasarımı bileşenleri olarak kullanılır.
- **Katman A (Python):** Pandas, NumPy temelleri — H katmanındaki araçlar bu temellerin ölçeklenmiş halleri.
- **Katman C (Makine Öğrenimi):** Model eğitimi ve değerlendirme — Ray Tune ve Dask-ML ile dağıtık hale getirme.

---

## Sektör Notu — Büyük Veri 2026

Ekosistem değişimi:

- **Polars + DuckDB** kombinasyonu "kurum içi küçük Databricks" gibi çalışıyor. 10 GB veri için Spark kurmak overkill.

- **Databricks** ML platformu olarak büyüyor: Delta Lake + MLflow + Feature Store + Serverless SQL = bir arada. UniForm özelliği ile Delta tablolarını Iceberg client'ları ile okumak mümkün.

- **Apache Iceberg** Delta Lake ile başa baş rekabet ediyor. Multi-cloud, vendor-agnostic — Snowflake, BigQuery, Spark, Trino, Athena hepsi destekliyor. Partition evolution özelliği büyük avantaj.

- **DuckDB** benchmark sonuçları: 100M satır aggregation'da Spark ve Dask'tan 5–15× hızlı (tek makine). Cluster kurma maliyeti olmadan.

- **Ray** ML dağıtımı için standart haline geldi: OpenAI ChatGPT eğitiminde Ray kullanıyor. Ray, PyTorch Foundation'a katıldı — enterprise adoption hızlanıyor.

- **River (online ML)** streaming ML için olgunlaşıyor. Ancak çoğu production streaming ML hala micro-batch + retrain yaklaşımı ile çalışıyor.

---

## Katman H Kontrol Listesi

- [ ] Polars lazy API ile gerçek bir analiz yaptım
- [ ] DuckDB ile Parquet üzerinde SQL sorgusu çalıştırdım
- [ ] Spark temel kavramlarını biliyorum (transformation vs action, lazy)
- [ ] Spark DataFrame ile aggregation + window function yazdım
- [ ] Delta Lake time travel ve upsert anladım
- [ ] Apache Iceberg ve Delta Lake farkını biliyorum
- [ ] BigQuery'de partition + clustering farkını biliyorum
- [ ] Dask-ML ile dağıtık model eğitimi veya paralel tahmin denedim
- [ ] Ray Tune ile hyperparameter tuning denedim
- [ ] Spark Structured Streaming veya River ile streaming kavramını anladım
- [ ] Cloud credentials (AWS/GCP) güvenli şekilde yönetmeyi biliyorum
- [ ] .env + .gitignore ile gizli bilgi yönetimi yapabiliyorum
- [ ] "Ne zaman hangi araç" ve "batch vs streaming" sorularını yanıtlayabilirim

---

<div class="nav-footer">
  <span><a href="#file_katman_G_senior_davranislar">← Önceki: Katman G — Senior Davranışlar</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_projeler">Sonraki: Portföy Projeleri →</a></span>
</div>
