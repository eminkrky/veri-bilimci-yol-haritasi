# Katman 0 — Matematik Temelleri (DS için)

> Bu katmanda ne öğrenilir: Lineer cebir, kalkülüs ve olasılık teorisinin **sezgisel anlamı**. Hedef ezber değil, "ML algoritmaları neden böyle davranıyor?" sorusunu yanıtlayabilmek.
>
> Süre: 3–5 gün yoğun çalışma, sonrasında her katmanda derinleşir.


<div class="prereq-box">
<strong>Önkoşul:</strong> Lise matematik yeterli. Bu katman makine öğrenmesindeki matematiksel sezgiyi kurmak için tasarlanmıştır.
</div>

### Notasyon Kuralları

Bu dosyada tutarlı olarak aşağıdaki notasyon kullanılmaktadır:

| Sembol | Anlam | Örnek |
|--------|-------|-------|
| Küçük harf kalın / oklu | Vektör | **v**, **w**, **x**, **y** |
| Büyük harf kalın | Matris | **A**, **X**, **W**, **Σ** |
| Küçük harf normal (italik) | Skalar | α, λ, σ, n, d |
| λᵢ | i. özdeğer | λ₁ ≥ λ₂ ≥ ... ≥ λ_d |
| σᵢ | i. tekil değer | σ₁ ≥ σ₂ ≥ ... ≥ 0 |
| ∇L(**w**) | Loss'un **w**'ye göre gradyanı | Vektör-değerli türev |

---

## 0.1 Lineer Cebir

### Sezgisel Açıklama

Lineer cebiri anlamak için şu soruyu sor: "Bir veri noktasını nasıl temsil ederim?"

Cevap: **vektör**. Bir kullanıcının 10 farklı özelliği (yaş, harcama, son_login_günü, vb.) varsa, o kullanıcı 10 boyutlu uzayda bir nokta.

Lineer cebir bize o uzayda nasıl hareket edeceğimizi, ölçeği değiştireceğimizi, yönleri bulacağımızı öğretir.

### Vektörler

- Vektör: sayıların sıralı listesi → `v = [3, 1, -2]`
- **İç çarpım (dot product):** `a · b = a₁b₁ + a₂b₂ + ... = ||a|| ||b|| cos(θ)`
  - Sezgi: iki vektörün ne kadar aynı yönde olduğunu ölçer
  - θ = 0° → iç çarpım maksimum (aynı yön)
  - θ = 90° → iç çarpım = 0 (dikgen, ilişkisiz)
- **Kosinüs benzerliği:** `cos_sim = (a · b) / (||a|| · ||b||)` → NLP/RecSys'te temel araç

**DS Bağlantısı:**
- Her veri satırı = bir vektör (feature uzayında nokta)
- Embedding vektörleri (kelimeler, kullanıcılar, ürünler)
- Model tahminleri: `ŷ = w · x` (ağırlık vektörü ile feature vektörünün iç çarpımı)

### Matematik Detayı

```
İç çarpım: a · b = Σᵢ aᵢ bᵢ

Norm (uzunluk): ||a|| = √(a · a) = √(Σᵢ aᵢ²)

Kosinüs benzerliği: sim(a, b) = (a · b) / (||a|| · ||b||)  ∈ [-1, 1]
```

### Kod Örneği

```python
import numpy as np

# İki kullanıcı feature vektörü
user1 = np.array([5, 3, 0, 1])   # [alışveriş, film izleme, spor, müzik]
user2 = np.array([4, 2, 1, 2])

# İç çarpım
dot = np.dot(user1, user2)

# Kosinüs benzerliği
cos_sim = dot / (np.linalg.norm(user1) * np.linalg.norm(user2))
print(f"Benzerlik: {cos_sim:.3f}")  # 1'e yakın = benzer kullanıcılar

# Embedding benzerlik matrisi (RecSys temeli)
users = np.random.randn(100, 32)  # 100 kullanıcı, 32 boyutlu embedding
items = np.random.randn(500, 32)  # 500 ürün
scores = users @ items.T          # (100, 500) benzerlik matrisi
```

> **Senior Notu:** Yüksek boyutlu uzaylarda (>100d) kosinüs benzerliği yorumlamak zorlaşır. "Hubness" problemi: bazı noktalar her şeye yakın görünür. Approximate Nearest Neighbor (FAISS, ScaNN) bunu pratikte aşar.

---

### Matrisler

### Sezgisel Açıklama

Matris = satırlar halinde vektörler. 1000 kullanıcı ve 50 feature varsa, veri matrisin boyutu (1000, 50).

Matris çarpımı lineer dönüşümdür: girdiyi çarparak yeni bir uzaya taşırsın. Neural network'lerin her katmanı budur — giriş vektörünü ağırlık matrisiyle çarpar, sonra aktivasyon uygular.

### Matematik Detayı

```
A (m×n) × B (n×p) = C (m×p)

Cᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ

Lineer regresyon normal denklemi:
  y = Xw  →  w = (XᵀX)⁻¹ Xᵀy

  [Burada X (n×d) veri matrisi, y (n,) hedef vektörü, w (d,) ağırlıklar]
```

### Kod Örneği

```python
# Lineer regresyon: normal denklem vs gradient descent kıyası
np.random.seed(42)
n, d = 200, 3
X = np.random.randn(n, d)
true_w = np.array([2.0, -1.5, 0.8])
y = X @ true_w + np.random.randn(n) * 0.3

# Normal denklem (küçük veri için, büyük veride pahalı)
w_normal = np.linalg.inv(X.T @ X) @ X.T @ y

# Gradient descent
w_gd = np.zeros(d)
lr = 0.01
for _ in range(500):
    grad = -2 * X.T @ (y - X @ w_gd) / n
    w_gd -= lr * grad

print(f"Gerçek w: {true_w}")
print(f"Normal denklem: {w_normal.round(3)}")
print(f"Gradient descent: {w_gd.round(3)}")
```

> **Senior Notu:** Normal denklem O(d³) zaman alır — d büyüdükçe gradient descent tercih et. Büyük veri için stochastic gradient descent (SGD) veya mini-batch kullan.

---

### Özdeğer ve Özvektör (Eigenvalue/Eigenvector)

### Sezgisel Açıklama

`Av = λv` — **A** matrisi **v** vektörünü sadece ölçeklendiriyor (yönünü değiştirmiyor). λ büyükse o yön en fazla varyansı taşıyor.

Düşün: verinin en "yayılgan" olduğu yön → en büyük özdeğere ait özvektör. PCA tam bunu yapar.

### Matematik Detayı

```
Kovaryans matrisi: Σ = (1/n) XᵀX  (merkezlenmiş X için)

PCA: Σ vᵢ = λᵢ vᵢ
  λ₁ ≥ λ₂ ≥ ... ≥ λd  (büyükten küçüğe özdeğerler)
  v₁, v₂, ...          (karşılık gelen özvektörler = principal components)

İlk k bileşen açıklanan varyans oranı:
  Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ᵈ λᵢ
```

### Kod Örneği

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 2D veri oluştur (çapraz yayılmış)
np.random.seed(42)
angle = np.pi / 4
cov = [[2, 1.5], [1.5, 1.5]]
data = np.random.multivariate_normal([0, 0], cov, 200)

# PCA ile boyut azaltma
pca = PCA(n_components=2)
pca.fit(data)
print("Açıklanan varyans oranları:", pca.explained_variance_ratio_)
print("Principal components (özvektörler):")
print(pca.components_)

# Scree plot
plt.bar(range(1, 3), pca.explained_variance_ratio_)
plt.xlabel("Bileşen")
plt.ylabel("Açıklanan Varyans Oranı")
plt.title("PCA Scree Plot")
plt.show()
```

> **Senior Notu:** PCA'yı yorumlarken dikkat: özellikler ölçeklenmemişse büyük ölçekteki özellikler baskın çıkar. Önce `StandardScaler` uygula. Ayrıca PCA doğrusal ilişkileri bulur — doğrusal olmayan yapılar için t-SNE veya UMAP.
>
> **Çapraz Referans:** PCA'nın pratikte nasıl uygulandığı ve boyut indirgeme stratejileri için bkz. **Katman B — PCA ve Boyut İndirgeme**. Eigenvalue decomposition ile PCA arasındaki doğrudan bağlantı aşağıdaki SVD bölümünde açıklanmaktadır.

---

### Tekil Değer Ayrışımı (SVD — Singular Value Decomposition)

### Sezgisel Açıklama

SVD'yi şöyle düşün: **her matrisi 3 basit dönüşüme ayırma**. Herhangi bir **A** matrisi (kare olmak zorunda değil) şu üç adıma ayrılabilir:

1. **Döndürme** (**Vᵀ**): girdi uzayını döndür
2. **Ölçekleme** (**Σ**): eksenleri farklı oranlarda uzat/sıkıştır
3. **Döndürme** (**U**): çıktı uzayını döndür

Eigenvalue decomposition sadece kare ve simetrik matrislere uygulanabilirken, SVD **her** matrise uygulanabilir. Bu onu çok daha genel ve güçlü kılar.

### Matematik Detayı

```
A = UΣVᵀ

A  : (m × n) herhangi bir matris
U  : (m × m) sol tekil vektörler (ortonormal — UᵀU = I)
Σ  : (m × n) köşegen matris, tekil değerler σ₁ ≥ σ₂ ≥ ... ≥ 0
Vᵀ : (n × n) sağ tekil vektörler (ortonormal — VᵀV = I)

Rank-k yaklaşımı (boyut indirgeme):
  A ≈ Aₖ = Uₖ Σₖ Vₖᵀ
  → İlk k tekil değer ve karşılık gelen vektörlerle en iyi rank-k yaklaşımı
  → Frobenius normunda optimal (Eckart–Young teoremi)

PCA ile bağlantı:
  Kovaryans matrisi Σ_cov = (1/n) XᵀX ise,
  X = UΣVᵀ  →  XᵀX = VΣ²Vᵀ
  → Sağ tekil vektörler (V) = PCA özvektörleri (principal components)
  → σᵢ² / n = λᵢ (özdeğerler)
```

### Kod Örneği

```python
import numpy as np
import matplotlib.pyplot as plt

# --- SVD temel kullanım ---
np.random.seed(42)
A = np.random.randn(5, 3)

U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U shape: {U.shape}")    # (5, 3)
print(f"s (tekil değerler): {s.round(3)}")
print(f"Vt shape: {Vt.shape}")  # (3, 3)

# Geri çarpımla doğrula: A ≈ U @ diag(s) @ Vt
A_reconstructed = U @ np.diag(s) @ Vt
print(f"Yeniden oluşturma hatası: {np.linalg.norm(A - A_reconstructed):.2e}")

# --- Boyut indirgeme ile görselleştirme ---
# Yüksek boyutlu veriyi SVD ile 2D'ye indirge
np.random.seed(42)
n_samples = 200
# 3 küme oluştur (10 boyutlu uzayda)
cluster1 = np.random.randn(n_samples // 3, 10) + np.array([3]*10)
cluster2 = np.random.randn(n_samples // 3, 10) + np.array([-3]*10)
cluster3 = np.random.randn(n_samples - 2*(n_samples//3), 10)
X = np.vstack([cluster1, cluster2, cluster3])
labels = np.array([0]*(n_samples//3) + [1]*(n_samples//3) + [2]*(n_samples - 2*(n_samples//3)))

# SVD ile 2D'ye indir
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_2d = U[:, :2] * s[:2]  # İlk 2 bileşen

plt.figure(figsize=(8, 6))
for label in [0, 1, 2]:
    mask = labels == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Küme {label}", alpha=0.6)
plt.xlabel("1. Tekil Bileşen")
plt.ylabel("2. Tekil Bileşen")
plt.title("SVD ile 10D → 2D Boyut İndirgeme")
plt.legend()
plt.show()

# Açıklanan varyans oranı
explained_var = s**2 / np.sum(s**2)
print(f"İlk 2 bileşenin açıkladığı varyans: {explained_var[:2].sum():.1%}")
```

**DS'de SVD kullanım alanları:**
- **Boyut indirgeme:** PCA'nın temelinde SVD var — yüksek boyutlu veriyi düşük boyuta indir
- **Gürültü temizleme:** Küçük tekil değerleri sıfırlayarak sinyal/gürültü ayrımı yap
- **Matrix completion:** Netflix Prize'daki gibi eksik değer tahmini (collaborative filtering)
- **Doğal dil işleme:** LSA (Latent Semantic Analysis) — terim-belge matrisine SVD uygula
- **Resim sıkıştırma:** Rank-k yaklaşımı ile az veriyle görsel temsil (alıştırmalar bölümünde)

> **Senior Notu:** Büyük matrislerde tam SVD pahalıdır — O(mn·min(m,n)). Pratikte `scipy.sparse.linalg.svds` (truncated SVD) veya `sklearn.decomposition.TruncatedSVD` ile sadece ilk k bileşeni hesapla. Sparse matrisler (NLP bag-of-words gibi) için bu zorunlu.
>
> **Çapraz Referans:** SVD'nin PCA ile ilişkisi → **Katman B — PCA ve Boyut İndirgeme**. Matrix completion uygulamaları → **Katman B — Öneri Sistemleri**.

---

## 0.2 Kalkülüs ve Optimizasyon

### Sezgisel Açıklama

Kalkülüsü ML'de neden öğreniyoruz? Çünkü "modelin en iyi parametreleri nedir?" sorusu bir optimizasyon problemi. Optimizasyon = loss fonksiyonunu minimize et = türevler hesapla = gradyanı takip et.

Analoji: Dağda gözleri kapalı en derin noktayı bulmak istiyorsun. Her adımda ayağının altındaki eğimi (türev) hissedip o yöne iniyorsun. Bu gradient descent.

### Türev (Derivative)

### Matematik Detayı

```
f'(x) = lim_{h→0} [f(x+h) - f(x)] / h

Zincir kuralı:
  d/dx [f(g(x))] = f'(g(x)) · g'(x)

Önemli türevler:
  d/dx [xⁿ]     = n·xⁿ⁻¹
  d/dx [eˣ]     = eˣ
  d/dx [ln(x)]  = 1/x
  d/dx [sigmoid(x)] = sigmoid(x) · (1 - sigmoid(x))
```

### Kod Örneği

```python
# Sigmoid ve türevi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

x = np.linspace(-6, 6, 100)

# Sayısal türev ile doğrula
h = 1e-7
numerical_deriv = (sigmoid(x + h) - sigmoid(x)) / h
analytical_deriv = sigmoid_derivative(x)

print(f"Max hata: {np.max(np.abs(numerical_deriv - analytical_deriv)):.2e}")
# Çok küçük olmalı (~1e-7 seviyesinde)
```

---

### Gradient Descent

### Sezgisel Açıklama

Loss fonksiyonunun "en dik iniş" yönünde adım at. Gradient (çok değişkenli türev) sana en hızlı artış yönünü söyler. Tersi = en hızlı iniş yönü.

```
w ← w - α · ∇L(w)

α = learning rate (adım büyüklüğü)
∇L(w) = loss'un w'ye göre gradyanı
```

### Kod Örneği

```python
# Gradient descent ile lineer regresyon — tam örnek
np.random.seed(42)
n = 300
X = np.column_stack([np.ones(n), np.random.randn(n, 2)])  # bias dahil
true_w = np.array([1.0, 3.0, -2.0])
y = X @ true_w + np.random.randn(n) * 0.5

w = np.zeros(3)
lr = 0.01
losses = []

for epoch in range(300):
    # Forward pass: tahmin
    y_hat = X @ w

    # Loss: MSE
    loss = np.mean((y - y_hat) ** 2)
    losses.append(loss)

    # Backward pass: gradient
    grad = -2 * X.T @ (y - y_hat) / n
    w -= lr * grad

print(f"Öğrenilen w: {w.round(3)}")  # [~1.0, ~3.0, ~-2.0]

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss Eğrisi")
plt.show()
```

> **Senior Notu:** Sabit learning rate hassas. Pratikte learning rate scheduling kullan (cosine annealing, warmup). Batch normalization da training'i stabilize eder. PyTorch'ta `torch.optim.AdamW` ile `torch.optim.lr_scheduler.CosineAnnealingLR` kombinasyonu 2026 itibarıyla standart.
>
> **Çapraz Referans:** Gradient descent'in derin ağlardaki uygulaması için bkz. **Katman D — Backpropagation ve Gradient Akışı**. Regularization ile gradient ilişkisi için bkz. **Katman B — Regularization (L1/L2)**.

---

### Convexity (Dışbükeylik)

### Sezgisel Açıklama

**Kase şeklinde vs dağlık arazi — neden önemli?**

Bir optimizasyon problemini düşün: kayıp fonksiyonunun yüzeyinde en düşük noktayı arıyorsun.

- **Convex (dışbükey) fonksiyon** = kase şeklinde. Nereye koyarsan koy topu, hep aynı en derin noktaya yuvarlanır. **Tek bir global minimum** var, yerel minimuma takılma riski yok.
- **Non-convex fonksiyon** = dağlık arazi. Birçok çukur (yerel minimum), tepeler (yerel maksimum) ve düzlükler (plato). Gradient descent başlangıç noktasına göre farklı çukurlara düşebilir.

### Matematik Detayı

```
Convex fonksiyon tanımı:
  f(λ**x** + (1-λ)**y**) ≤ λf(**x**) + (1-λ)f(**y**)    ∀ λ ∈ [0,1]
  → Herhangi iki nokta arasındaki doğru parçası fonksiyonun üstünde kalır.

İkinci türev testi:
  f''(x) ≥ 0  (tek değişken) → convex
  ∇²f(**x**) ⪰ 0  (Hessian pozitif yarı-tanımlı) → convex

Yerel minimum vs global minimum:
  Convex fonksiyon: her yerel minimum = global minimum  ✓
  Non-convex fonksiyon: yerel minimum ≠ global minimum  ✗
```

### Convex vs Non-convex Loss Fonksiyonları

| Özellik | Convex Loss | Non-convex Loss |
|---------|-------------|-----------------|
| Örnek | MSE (lineer regresyon), Hinge loss (SVM) | Derin ağ loss fonksiyonları |
| Garanti | Global minimuma yakınsama | Yakınsama garantisi yok |
| Yerel minimum | Yok (tek minimum) | Çok sayıda |
| Çözüm yöntemi | Gradient descent yeterli | SGD + momentum + lr scheduling |

**Neden MSE convex ama DL loss'ları genelde non-convex?**

- MSE + lineer model: `L(**w**) = ||**y** - **Xw**||²` → **w** cinsinden ikinci derece polinom → kase şekli → convex
- Derin ağlar: aktivasyon fonksiyonları (ReLU, sigmoid) katmanlı çarpımlarla non-lineerlik yaratır → loss yüzeyi dalgalı ve karmaşık
- Ama pratikte: yüksek boyutlu DL loss yüzeylerinde "kötü" yerel minimumlar nadirdir (çoğu yerel minimum global minimuma yakın değerdedir — Li et al., 2018)

### Kod Örneği

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 1. Convex fonksiyon: MSE loss yüzeyi ---
w1_range = np.linspace(-3, 5, 100)
w2_range = np.linspace(-3, 5, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Basit MSE: L(w1, w2) = (w1 - 1)² + (w2 - 2)² (kase şekli)
Loss_convex = (W1 - 1)**2 + (W2 - 2)**2

axes[0].contour(W1, W2, Loss_convex, levels=30, cmap='viridis')
axes[0].plot(1, 2, 'r*', markersize=15, label='Global minimum')
axes[0].set_title("Convex Loss (MSE) — Kase Şekli")
axes[0].set_xlabel("w₁")
axes[0].set_ylabel("w₂")
axes[0].legend()

# --- 2. Non-convex fonksiyon: birden fazla minimum ---
x = np.linspace(-4, 4, 300)
y_nonconvex = x**4 - 5*x**2 + 4 + 0.5*x  # İki yerel minimum

axes[1].plot(x, y_nonconvex, 'b-', linewidth=2)
axes[1].set_title("Non-convex Loss — Dağlık Arazi")
axes[1].set_xlabel("w")
axes[1].set_ylabel("Loss")

# Yerel ve global minimumları işaretle
from scipy.optimize import minimize_scalar
res1 = minimize_scalar(lambda x: x**4 - 5*x**2 + 4 + 0.5*x, bounds=(-3, 0), method='bounded')
res2 = minimize_scalar(lambda x: x**4 - 5*x**2 + 4 + 0.5*x, bounds=(0, 3), method='bounded')
f = lambda x: x**4 - 5*x**2 + 4 + 0.5*x

if f(res1.x) < f(res2.x):
    axes[1].plot(res1.x, f(res1.x), 'r*', markersize=15, label='Global minimum')
    axes[1].plot(res2.x, f(res2.x), 'yo', markersize=10, label='Yerel minimum')
else:
    axes[1].plot(res2.x, f(res2.x), 'r*', markersize=15, label='Global minimum')
    axes[1].plot(res1.x, f(res1.x), 'yo', markersize=10, label='Yerel minimum')

axes[1].legend()
plt.tight_layout()
plt.savefig("convexity_comparison.png", dpi=100)
plt.show()
```

> **Senior Notu:** Non-convex optimizasyonda başarının sırrı: iyi initialization (Xavier/He), adaptive optimizer (Adam/AdamW), ve learning rate warmup. Ayrıca loss landscape visualization araçları (Li et al., "Visualizing the Loss Landscape of Neural Nets") model mimarisinin eğitilebilirliğini anlamada çok faydalı.
>
> **Çapraz Referans:** Regularization teknikleri (L1/L2) loss fonksiyonunun şeklini nasıl değiştirir → **Katman B — Regularization**. Backpropagation sırasında gradient sorunları (vanishing/exploding) → **Katman D — Gradient Akışı**.

---

### Backpropagation (Zincir Kuralı Uygulaması)

### Sezgisel Açıklama

Derin ağda her katmanın katkısını nasıl ölçeriz? Çıktıdan başlayıp zincir kuralıyla her katmanın gradyanını hesaplarız.

Forward pass: girdi → sonuç hesapla
Backward pass: çıktıdan girdi katmanına gradyanları zincirle ilet

```
L = loss(y_hat, y)
y_hat = f(z)        dL/dz = dL/dy_hat · dy_hat/dz
z = Wx + b          dL/dW = dL/dz · x.T
                    dL/db = dL/dz
```

### Kod Örneği

```python
# 2 katmanlı mini ağda backprop — PyTorch olmadan
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

# Forward pass
np.random.seed(0)
X_input = np.random.randn(5, 3)      # 5 örnek, 3 feature
W1 = np.random.randn(3, 4) * 0.1    # 1. katman ağırlıkları
W2 = np.random.randn(4, 1) * 0.1    # 2. katman ağırlıkları
y_true = np.random.randn(5, 1)

z1 = X_input @ W1          # (5, 4)
a1 = relu(z1)              # (5, 4)
z2 = a1 @ W2               # (5, 1)
loss = np.mean((z2 - y_true) ** 2)

# Backward pass
dL_dz2 = 2 * (z2 - y_true) / len(X_input)  # (5, 1)
dL_dW2 = a1.T @ dL_dz2                       # (4, 1)
dL_da1 = dL_dz2 @ W2.T                       # (5, 4)
dL_dz1 = dL_da1 * relu_grad(z1)              # (5, 4)
dL_dW1 = X_input.T @ dL_dz1                  # (3, 4)

print(f"dL/dW2 shape: {dL_dW2.shape}")
print(f"dL/dW1 shape: {dL_dW1.shape}")
```

> **Çapraz Referans:** Backpropagation'ın derin ağlardaki tüm detayları, vanishing/exploding gradient sorunları ve çözümleri → **Katman D — Backpropagation ve Gradient Akışı**.

---

## 0.3 Olasılık Teorisi

### Sezgisel Açıklama

Olasılık, belirsizliği sayılaştırmanın dili. ML'de her şey olasılıksal: model "bu kullanıcı %73 ihtimalle churn" der. Bu sayı ne anlama geliyor? Gerçekten kalibre mi? İşte olasılık teorisi bu soruları yanıtlar.

---

### Temel Kavramlar

### Matematik Detayı

```
Rastgele değişken X: olası sonuçları sayı ile eşleştiren fonksiyon

Discrete: PMF — P(X = x)
Continuous: PDF — f(x), integral = 1

Beklenen değer: E[X] = Σ x·P(X=x)  (discrete)
              : E[X] = ∫ x·f(x) dx  (continuous)

Varyans: Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²

Kovaryans: Cov(X, Y) = E[(X - μx)(Y - μy)]
Korelasyon: ρ = Cov(X,Y) / (σx · σy)  ∈ [-1, 1]
```

---

### Önemli Dağılımlar (DS Sezgisi)

| Dağılım | DS Kullanımı | Parametre |
|---------|-------------|-----------|
| Bernoulli(p) | Tek click/churn kararı | p ∈ [0,1] |
| Binomial(n, p) | A/B test sayımları | n (deneme), p (olasılık) |
| Normal N(μ, σ²) | CLT ile ortaya çıkar, t-testlerin temeli | μ (ortalama), σ² (varyans) |
| Poisson(λ) | Web trafiği, sipariş/saat sayısı | λ (ortalama oran) |
| Exponential(λ) | Olaylar arası süre, destek yanıt süresi | λ (oran) |
| Beta(α, β) | Bayesian A/B test için olasılık parametresi | α, β (şekil) |
| Log-normal | Gelir verisi (çarpık, pozitif) | μ, σ (log ölçeğinde) |

### Kod Örneği

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# 1. Bernoulli — churn
p_churn = 0.07
bernoulli = stats.bernoulli(p_churn)
print(f"Churn E[X]={bernoulli.mean():.2f}, Var={bernoulli.var():.4f}")

# 2. Binomial — A/B test
n_users, p_convert = 1000, 0.05
binom = stats.binom(n_users, p_convert)
axes[0, 0].bar(range(30, 80), binom.pmf(range(30, 80)))
axes[0, 0].set_title("Binomial: Dönüşüm Sayısı (n=1000, p=0.05)")

# 3. Normal — CLT simülasyonu
samples = [np.mean(np.random.exponential(1, 30)) for _ in range(10000)]
axes[0, 1].hist(samples, bins=60, density=True)
axes[0, 1].set_title("CLT: Üstel Dağılımdan Örneklem Ortalamaları")

# 4. Poisson — web trafiği
lam = 15
poisson = stats.poisson(lam)
x = range(0, 35)
axes[0, 2].bar(x, poisson.pmf(list(x)))
axes[0, 2].set_title(f"Poisson(λ={lam}): Saatlik İstek Sayısı")

# 5. Beta — Bayesian A/B prior → posterior
alpha_prior, beta_prior = 1, 1     # uniform prior
successes, failures = 47, 953      # 47 dönüşüm / 1000 deneme
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures
x = np.linspace(0, 0.15, 300)
axes[1, 0].plot(x, stats.beta(1, 1).pdf(x), label="Prior", linestyle="--")
axes[1, 0].plot(x, stats.beta(alpha_post, beta_post).pdf(x), label="Posterior")
axes[1, 0].legend()
axes[1, 0].set_title("Beta: Prior → Posterior")

# 6. Log-normal — gelir verisi
log_income = np.random.lognormal(mean=np.log(500), sigma=0.8, size=1000)
axes[1, 1].hist(log_income, bins=50, density=True)
axes[1, 1].set_title("Log-normal: Gelir Dağılımı")

# 7. Exponential — destek bileti yanıt süresi
exp_times = np.random.exponential(scale=30, size=1000)
axes[1, 2].hist(exp_times, bins=50, density=True)
axes[1, 2].set_title("Üstel: Destek Yanıt Süresi (dk)")

plt.tight_layout()
plt.savefig("distributions.png", dpi=100)
```

---

### Bayes Teoremi

### Matematik Detayı

```
P(A|B) = P(B|A) · P(A) / P(B)

Bayesian terminoloji:
  Prior:    P(A)     — Veri gelmeden önceki inanç
  Likelihood: P(B|A) — Bu veriyi bu hipotez altında görme olasılığı
  Posterior: P(A|B)  — Veri geldikten sonra güncellenen inanç
  Marginal: P(B)     — Normalleştirme sabiti

Posterior ∝ Prior × Likelihood
```

### Kod Örneği

```python
# Spam filtresi — Naive Bayes mantığı
# P(spam | "para" kelimesi) = ?

p_spam = 0.3          # Prior: gelen mailin %30'u spam
p_para_given_spam = 0.8    # Spam maillerde "para" %80 görünür
p_para_given_not_spam = 0.1  # Normal maillerde %10

# Bayes teoremi
p_para = p_para_given_spam * p_spam + p_para_given_not_spam * (1 - p_spam)
p_spam_given_para = (p_para_given_spam * p_spam) / p_para

print(f"P(spam | 'para'): {p_spam_given_para:.1%}")
# %77.4 çıkmalı → mail spam olma ihtimali yüksek

# Bayesian A/B test: Beta-Binomial
import numpy as np
from scipy import stats

# Kontrol grubu: 1000 kullanıcı, 47 dönüşüm
# Test grubu: 1000 kullanıcı, 58 dönüşüm

alpha_c, beta_c = 1 + 47, 1 + 953
alpha_t, beta_t = 1 + 58, 1 + 942

# Simülasyon ile P(test > control)
n_sim = 100_000
control_samples = stats.beta(alpha_c, beta_c).rvs(n_sim)
test_samples = stats.beta(alpha_t, beta_t).rvs(n_sim)

p_test_better = np.mean(test_samples > control_samples)
print(f"P(test > control) = {p_test_better:.1%}")
```

> **Senior Notu:** Bayesian A/B test yorumlaması daha sezgisel: "Test grubunun kontrol grubunu geçme olasılığı %89." Bu, frequentist p-value'dan ("H₀ altında bu kadar uç veri görme olasılığı") çok daha doğrudan iş kararı için kullanılabilir.

---

## 0.4 Bilgi Teorisi Temeli (İsteğe Bağlı)

### Entropi ve Bilgi Kazanımı

ML'de entropi ağaç modellerinde (decision tree, random forest) bölme kriteri olarak kullanılır.

```
Entropi: H(X) = -Σ P(xᵢ) · log₂(P(xᵢ))
  → Belirsizliğin ölçüsü. Tüm sınıflar eşit → maksimum entropi.
  → Tek sınıf → 0 entropi.

Bilgi Kazanımı (Information Gain):
  IG(Y, X) = H(Y) - H(Y|X)
  → X bölmesinden önce ve sonra entropi farkı.

KL-Divergence (iki dağılım arası mesafe):
  KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
  → Asimetrik! KL(P||Q) ≠ KL(Q||P)
  → VAE loss fonksiyonunun bir parçası (ELBO)

Cross-Entropy Loss:
  L = -Σᵢ yᵢ · log(ŷᵢ)
  → Sınıflandırma modelleri için standart loss
  → Bayesian bakışla: KL-divergence minimizasyonu
```

### Kod Örneği

```python
import numpy as np

def entropy(p_array):
    """p_array: olasılıklar listesi, toplam 1"""
    p = np.array(p_array)
    p = p[p > 0]  # log(0) tanımsız
    return -np.sum(p * np.log2(p))

# Sınıf dağılımları
pure_node = [1.0, 0.0]          # Sadece bir sınıf
balanced = [0.5, 0.5]           # Eşit dağılım
imbalanced = [0.9, 0.1]         # Dengesiz

print(f"Saf node (1.0, 0.0): H = {entropy(pure_node):.3f} bits")
print(f"Dengeli (0.5, 0.5): H = {entropy(balanced):.3f} bits")
print(f"Dengesiz (0.9, 0.1): H = {entropy(imbalanced):.3f} bits")

# Cross-entropy loss — binary sınıflandırma
def cross_entropy_loss(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y = np.array([1, 0, 1, 1, 0])
y_hat_good = np.array([0.9, 0.1, 0.8, 0.85, 0.15])
y_hat_bad = np.array([0.6, 0.4, 0.55, 0.6, 0.45])

print(f"\nİyi model loss: {cross_entropy_loss(y, y_hat_good):.3f}")
print(f"Kötü model loss: {cross_entropy_loss(y, y_hat_bad):.3f}")
```

---

## Özet ve Pratik Ödevler

Bu katmanda tamamlanması gereken pratikler:

1. **Lineer cebir:**
   - [ ] NumPy ile matris çarpımı + transpoz + eigendecomposition
   - [ ] Kosinüs benzerliği hesapla, 2 vektör üzerinde yorumla
   - [ ] PCA ile 2D veriyi görselleştir (açıklanan varyans plot)
   - [ ] SVD ile matris ayrıştırma ve geri çarpımla doğrulama

2. **Kalkülüs:**
   - [ ] Sigmoid türevini analitik + sayısal hesapla, doğrula
   - [ ] Gradient descent ile lineer regresyon (sıfırdan, NumPy ile)
   - [ ] Learning rate değiştirip yakınsama farkını gözlemle
   - [ ] Convex vs non-convex fonksiyon görselleştirmesi

3. **Olasılık:**
   - [ ] 6 farklı dağılımı görselleştir (yukarıdaki kod)
   - [ ] Bayes teoremi spam filtresi hesabı
   - [ ] Bayesian A/B test: P(test > control) simülasyonu

---

### Ek Alıştırmalar (Uygulamalı)

#### Alıştırma 1: SVD ile Resim Sıkıştırma (Rank-k Yaklaşımı)

```python
import numpy as np
import matplotlib.pyplot as plt

# Gri tonlama resim oluştur (veya kendi resminizi yükleyin)
# from PIL import Image
# img = np.array(Image.open("foto.jpg").convert("L"), dtype=float)

# Örnek: rastgele yapay resim (gerçek resimle deneyin!)
np.random.seed(42)
img = np.random.randn(200, 300)
# Yapısal desen ekle (sıkıştırmayı anlamlı kılmak için)
for i in range(200):
    for j in range(300):
        img[i, j] += 50 * np.sin(i/20) * np.cos(j/30)

U, s, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ranks = [5, 20, 50, len(s)]

for ax, k in zip(axes, ranks):
    # Rank-k yaklaşımı
    img_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    compression_ratio = (k * (200 + 300 + 1)) / (200 * 300)
    ax.imshow(img_approx, cmap='gray')
    ax.set_title(f"Rank-{k}\n({compression_ratio:.1%} veri)")
    ax.axis('off')

plt.suptitle("SVD ile Resim Sıkıştırma: Rank-k Yaklaşımı", fontsize=14)
plt.tight_layout()
plt.show()

# Tekil değerlerin düşüşünü göster
plt.figure(figsize=(8, 4))
plt.plot(s, 'b-')
plt.xlabel("Tekil Değer İndeksi")
plt.ylabel("σᵢ (Tekil Değer)")
plt.title("Tekil Değer Spektrumu — Hızlı Düşüş = İyi Sıkıştırılabilirlik")
plt.yscale('log')
plt.show()
```

#### Alıştırma 2: Gradient Descent'te Learning Rate Scheduler Etkisi

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200
X = np.column_stack([np.ones(n), np.random.randn(n)])
true_w = np.array([2.0, -3.0])
y = X @ true_w + np.random.randn(n) * 0.5

def train_gd(X, y, lr_fn, n_epochs=200):
    """lr_fn: epoch → learning rate döndüren fonksiyon"""
    w = np.zeros(X.shape[1])
    losses, lrs = [], []
    for epoch in range(n_epochs):
        lr = lr_fn(epoch)
        y_hat = X @ w
        loss = np.mean((y - y_hat) ** 2)
        grad = -2 * X.T @ (y - y_hat) / len(y)
        w -= lr * grad
        losses.append(loss)
        lrs.append(lr)
    return losses, lrs, w

# Farklı scheduler'lar
n_epochs = 200
schedulers = {
    "Sabit (lr=0.01)": lambda ep: 0.01,
    "Step Decay (her 50 epoch ½)": lambda ep: 0.05 * (0.5 ** (ep // 50)),
    "Cosine Annealing": lambda ep: 0.05 * (1 + np.cos(np.pi * ep / n_epochs)) / 2,
    "Warmup + Decay": lambda ep: min(0.05 * (ep + 1) / 20, 0.05 * (1 - ep / n_epochs)),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, lr_fn in schedulers.items():
    losses, lrs, w_final = train_gd(X, y, lr_fn, n_epochs)
    axes[0].plot(losses, label=name)
    axes[1].plot(lrs, label=name)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Loss Eğrileri — Farklı LR Scheduler'lar")
axes[0].legend(fontsize=8)
axes[0].set_yscale('log')

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Learning Rate")
axes[1].set_title("Learning Rate Değişimi")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("lr_scheduler_comparison.png", dpi=100)
plt.show()
```

> **Çapraz Referans:** Learning rate scheduling'in derin öğrenmedeki önemi → **Katman D — Training Stratejileri**.

#### Alıştırma 3: Log-Likelihood ile MLE Hesaplama (Normal Dağılım)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Gerçek parametrelerle veri üret
np.random.seed(42)
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, size=100)

# --- Analitik MLE ---
mu_mle = np.mean(data)
sigma_mle = np.std(data)  # MLE: n ile böl (n-1 değil)
print(f"Gerçek: μ={true_mu}, σ={true_sigma}")
print(f"MLE:    μ̂={mu_mle:.3f}, σ̂={sigma_mle:.3f}")

# --- Log-likelihood yüzeyi görselleştirme ---
mu_range = np.linspace(3, 7, 100)
sigma_range = np.linspace(1, 3.5, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)

log_lik = np.zeros_like(MU)
for i in range(len(sigma_range)):
    for j in range(len(mu_range)):
        log_lik[i, j] = np.sum(stats.norm.logpdf(data, MU[i, j], SIGMA[i, j]))

plt.figure(figsize=(8, 6))
plt.contourf(MU, SIGMA, log_lik, levels=50, cmap='RdYlBu_r')
plt.colorbar(label="Log-Likelihood")
plt.plot(mu_mle, sigma_mle, 'r*', markersize=15, label=f"MLE: μ̂={mu_mle:.2f}, σ̂={sigma_mle:.2f}")
plt.xlabel("μ")
plt.ylabel("σ")
plt.title("Normal Dağılım Log-Likelihood Yüzeyi")
plt.legend()
plt.show()

# Not: Log-likelihood formülü
# log L(μ, σ | x) = -n/2 · log(2π) - n·log(σ) - Σᵢ(xᵢ - μ)² / (2σ²)
```

#### Alıştırma 4: Information Gain ile Decision Tree Split Hesaplama

```python
import numpy as np

def entropy(labels):
    """Bir düğümdeki etiketlerin entropisini hesapla"""
    n = len(labels)
    if n == 0:
        return 0
    counts = np.bincount(labels)
    probs = counts[counts > 0] / n
    return -np.sum(probs * np.log2(probs))

def information_gain(parent_labels, left_labels, right_labels):
    """Bölme sonrası bilgi kazanımını hesapla"""
    n = len(parent_labels)
    H_parent = entropy(parent_labels)
    H_children = (len(left_labels) / n * entropy(left_labels) +
                  len(right_labels) / n * entropy(right_labels))
    return H_parent - H_children

# Örnek: müşteri churn tahmini
# Özellikler: aylık harcama (TL), sözleşme süresi (ay)
# Etiketler: 0 = kaldı, 1 = churn
labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1])
harcama = np.array([200, 350, 500, 150, 600, 80, 120, 90, 450, 60, 300, 400, 100, 250, 70])

print(f"Kök düğüm entropi: {entropy(labels):.4f} bits")
print()

# Farklı eşik değerlerini dene
thresholds = [100, 150, 200, 300, 400]
for t in thresholds:
    left_mask = harcama <= t
    right_mask = ~left_mask
    ig = information_gain(labels, labels[left_mask], labels[right_mask])
    print(f"Eşik={t:>3d} TL | Sol: {labels[left_mask]} | Sağ: {labels[right_mask]}")
    print(f"           IG = {ig:.4f} bits")
    print()

# En iyi split'i bul
best_ig, best_t = 0, 0
for t in range(50, 650, 10):
    left_mask = harcama <= t
    if left_mask.sum() == 0 or (~left_mask).sum() == 0:
        continue
    ig = information_gain(labels, labels[left_mask], labels[~left_mask])
    if ig > best_ig:
        best_ig, best_t = ig, t

print(f"En iyi split: harcama ≤ {best_t} TL (IG = {best_ig:.4f} bits)")
```

#### Alıştırma 5: Eigenvalue Decomposition ile PCA Bağlantısı

```python
import numpy as np
np.random.seed(42)

# 2D veri oluştur (korelasyonlu)
n = 500
cov_matrix = np.array([[3.0, 1.5],
                        [1.5, 1.0]])
X = np.random.multivariate_normal([0, 0], cov_matrix, n)

# --- Yöntem 1: Kovaryans matrisine eigenvalue decomposition ---
X_centered = X - X.mean(axis=0)
cov_empirical = (X_centered.T @ X_centered) / (n - 1)

eigenvalues, eigenvectors = np.linalg.eigh(cov_empirical)
# eigh küçükten büyüğe sıralar, tersine çevir
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("=== Yöntem 1: Eigenvalue Decomposition ===")
print(f"Özdeğerler (λ): {eigenvalues.round(4)}")
print(f"Özvektörler (principal components):\n{eigenvectors.round(4)}")

# --- Yöntem 2: SVD ile aynı sonuca ulaş ---
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
eigenvalues_from_svd = s**2 / (n - 1)

print("\n=== Yöntem 2: SVD ===")
print(f"SVD'den özdeğerler (σ²/(n-1)): {eigenvalues_from_svd.round(4)}")
print(f"SVD'den özvektörler (Vᵀ satırları):\n{Vt.round(4)}")

# --- Yöntem 3: sklearn PCA ---
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

print("\n=== Yöntem 3: sklearn PCA ===")
print(f"Açıklanan varyans: {pca.explained_variance_.round(4)}")
print(f"Bileşenler:\n{pca.components_.round(4)}")

# Üç yöntemin de aynı sonucu verdiğini doğrula
print("\n=== Doğrulama ===")
print(f"Eigen vs SVD özdeğer farkı: {np.max(np.abs(eigenvalues - eigenvalues_from_svd)):.2e}")
print(f"Açıklanan varyans oranı: {eigenvalues / eigenvalues.sum()}")
print("→ PCA = Kovaryans matrisinin eigendecomposition'ı = Veri matrisinin SVD'si")
```

> **Çapraz Referans:** PCA'nın pratikte büyük veri setlerinde kullanımı ve boyut seçimi stratejileri → **Katman B — PCA ve Boyut İndirgeme**.

### Sektör Notu

2026 itibarıyla "matematiği atlayabilirim" algısı değişiyor. LLM entegrasyonu artsa da şirketler senior pozisyonlarda gradient, loss fonksiyonu, dağılım ve istatistik anlayışını hâlâ sorguluyor. MIT ve Stanford araştırmaları "matematiksel sezgisi güçlü DS'ler üretim sorunlarını 3x daha hızlı teşhis ediyor" sonucuna ulaştı. Bu katmanı ezberlemek değil, sezgi geliştirmek için kullan.

---

## Katman 0 Kontrol Listesi

- [ ] Vektör iç çarpımı ve kosinüs benzerliği hesaplayabildim
- [ ] Matris çarpımını elle ve NumPy ile yaptım
- [ ] SVD'yi sezgisel olarak açıklayabilirim (döndürme + ölçekleme + döndürme)
- [ ] SVD ile boyut indirgeme ve resim sıkıştırma uyguladım
- [ ] PCA'yı sezgisel olarak açıklayabilirim (varyansı maksimize eden yön)
- [ ] Eigenvalue decomposition, SVD ve PCA arasındaki bağlantıyı kurabiliyorum
- [ ] Gradient descent'i sıfırdan NumPy ile kodladım
- [ ] Convex vs non-convex loss fonksiyonları arasındaki farkı açıklayabilirim
- [ ] Sigmoid türevini analitik olarak türettim
- [ ] Learning rate scheduler etkisini görselleştirdim
- [ ] 6 temel dağılımı ve DS kullanım alanlarını sayabilirim
- [ ] Bayes teoremini bir örnekle açıkladım (spam, test sonucu, vb.)
- [ ] MLE hesabını log-likelihood ile yaptım
- [ ] Cross-entropy loss'un binary sınıflandırmada nasıl çalıştığını bilirim
- [ ] Entropi ve bilgi kazanımı (karar ağacı split kriteri) sezgimi anlattım
- [ ] Information gain ile en iyi split noktasını hesapladım

---

<div class="nav-footer">
  <span><a href="#file_01_yetkinlik_matrisi">← Önceki: Yetkinlik Matrisi</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_A_temeller">Sonraki: Katman A — Temeller →</a></span>
</div>
