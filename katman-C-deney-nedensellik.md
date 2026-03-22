# Katman C — Deney Tasarımı, A/B Testi ve Nedensellik

> Bu katmanda ne öğrenilir: Doğru deney tasarımı, istatistiksel güç, A/A test ile platform doğrulama, CUPED ile varyans azaltma, sequential testing, network effects ve gözlemsel veriden causal inference.
>
> Süre: 1–3 hafta. FAANG ve growth odaklı şirketlerde bu katman en çok ayrım yaratan beceri.
>
> **Çapraz referanslar:** Katman B'de SHAP değerleri ile nedensellik farkı ele alınır (SHAP ≠ causal). Katman F'te A/B test platformu tasarımı ve altyapı bileşenleri detaylı işlenir. `mulakat.md` dosyasında deney tasarımı mülakat soruları bulunur.


<div class="prereq-box">
<strong>Önkoşul:</strong> <strong>Katman A</strong> (istatistik bölümü) ve <strong>Katman B</strong> (model değerlendirme) tamamlanmış olmalı.
</div>

---

## C.1 A/B Test Tasarımının İskeleti

### Sezgisel Açıklama

A/B test bir bilimsel deney. Farkı görmenin tek yolu önceden plan yapmak: ne ölçeceğini, nasıl ölçeceğini ve hangi eşikte karar vereceğini **deneyi başlatmadan** belirle.

"Sonuçlara bak, iyi çıkınca yayınla" yaklaşımı p-hacking ve yanlış pozitif üretir.

### Deney Tasarım Şablonu (Senior Standardı)

```markdown
## Deney: [İsim]
**Tarih:** 2024-XX-XX  |  **Sahip:** Emin  |  **Durum:** Tasarım

### Problem
Neden bu deney yapılıyor? (İş bağlamı)

### Hipotez
"[Değişiklik], [primary metric]'i [yön] yönde değiştirir,
çünkü [mekanizma]."

### Tasarım
- **Randomization unit:** user_id (session değil — interference)
- **Sample size:** [her kol için]
- **Süre:** [hafta] (hafta içi + sonu dengesi için en az 1 tam hafta)
- **Trafik:** %[X] treatment, %[Y] control

### Metrikler
- **Primary metric (1!):** kullanıcı başına gelir (ARPU)
- **Guardrail metrics:** latency p99, refund oranı, app crash rate
- **Secondary metrics (bilgi amaçlı):** CTR, sepet büyüklüğü

### Minimum Detectable Effect (MDE)
- Mevcut ARPU: 45 TL
- İş için anlamlı minimum etki: +%2 (0.9 TL)
- Bu etki büyüklüğünü yakalamak için gereken n: [hesap]

### Analiz Planı (önceden yaz!)
- İstatistiksel test: iki örneklem t-testi + bootstrap CI
- Alpha: 0.05 (two-sided)
- Power: 0.80
- CUPED: var (pre-period geliri covariate olarak)

### Durmak İçin Kriterler
- Süre doldu → analizi yap
- Guardrail ihlali → derhal dur
- SRM tespiti → dur, incelemi yap
```

---

## C.2 Power Analizi ve MDE

### Sezgisel Açıklama

Power = gerçek bir etki varsa onu yakalama olasılığı. Yetersiz power = etkiyi göremezsin, "deney sonuçsuz" dersin ama aslında etki vardı.

Analoji: Hastanede test. Power düşükse hastalığı sağlıklı kişide görememe riski artar.

### Matematik Detayı

```
n = [(z_{α/2} + z_β) / δ]² · 2σ²

z_{α/2}: 0.05 için 1.96 (two-sided)
z_β:    0.80 power için 0.84
δ: tespit etmek istediğin etki (MDE)
σ²: metrik varyansı

Cohen's d = δ / σ  (standartlaştırılmış etki büyüklüğü)
```

### Kod Örneği

```python
from statsmodels.stats.power import NormalIndPower, TTestIndPower
import numpy as np
import matplotlib.pyplot as plt

# Gerekli örnek boyutu — farklı MDE'ler için
analysis = NormalIndPower()

# Senaryo: ARPU = 45 TL, std = 30 TL
mu = 45
sigma = 30
mde_values = [0.5, 1.0, 2.0, 3.0, 4.5]  # TL cinsinden

print("MDE (TL) | MDE (%) | Cohen's d | n per arm")
print("-" * 50)
for mde in mde_values:
    d = mde / sigma
    n = analysis.solve_power(effect_size=d, alpha=0.05, power=0.80,
                              alternative="two-sided")
    print(f"{mde:8.1f} | {mde/mu:6.1%}  | {d:9.3f} | {int(np.ceil(n)):>9,}")

# Power grafiği — farklı n'ler için
sample_sizes = np.arange(100, 5000, 100)
mde = 1.5  # Sabit MDE = 1.5 TL
d = mde / sigma

powers = [analysis.solve_power(effect_size=d, alpha=0.05, nobs1=n,
                                 alternative="two-sided") for n in sample_sizes]

plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, powers, lw=2)
plt.axhline(0.80, color="red", linestyle="--", label="Hedef power (%80)")
plt.axvline(sample_sizes[np.argmax(np.array(powers) >= 0.80)],
            color="green", linestyle="--", label="Min gerekli n")
plt.xlabel("Kol başına örnek sayısı")
plt.ylabel("İstatistiksel güç")
plt.title(f"Power Analizi (MDE={mde} TL, σ={sigma} TL)")
plt.legend()
plt.grid(True, alpha=0.3)

# Sample Ratio Mismatch (SRM) kontrolü
def check_srm(n_control: int, n_treatment: int, expected_ratio: float = 0.5,
               alpha: float = 0.01) -> bool:
    """Randomizasyon bozulmuş mu?"""
    from scipy import stats
    total = n_control + n_treatment
    expected_control = total * (1 - expected_ratio)
    expected_treatment = total * expected_ratio

    chi2_stat, p_value = stats.chisquare(
        [n_control, n_treatment],
        [expected_control, expected_treatment]
    )
    print(f"SRM testi: χ²={chi2_stat:.3f}, p={p_value:.4f}")
    if p_value < alpha:
        print("⚠️  SRM tespit edildi! Randomizasyonu kontrol et.")
        return True
    return False

check_srm(n_control=4850, n_treatment=5150)  # 10000 kullanıcı, %50/%50 hedef
```

---

## C.3 A/A Test — Platform Doğrulama

### Sezgisel Açıklama

A/A test, iki gruba da **aynı deneyimi** verip istatistiksel test uygular. Amaç: "Sistemin kendisi yanlış pozitif üretiyor mu?" sorusunu yanıtlamak.

Düşün: Bir tartıyı kullanmaya başlamadan önce sıfır noktasını kontrol edersin. A/A test, deney platformunun "sıfır noktası" kontrolüdür. Randomizasyon düzgün çalışıyor mu? Metrik pipeline doğru mu? Logging kaybı var mı?

A/A testte iki grup arasında **gerçek bir fark yok**. Dolayısıyla α=0.05 kullanıyorsak, simülasyonların yaklaşık %5'inde "anlamlı" sonuç beklenir. Bundan fazlaysa platform hatalı.

### A/A Test Simülasyonu

```python
import numpy as np
from scipy import stats

def aa_test_simulation(n_iterations=1000, n_per_group=5000,
                        mu=45, sigma=30, alpha=0.05):
    """
    A/A test simülasyonu: iki özdeş grup arası t-test.
    Beklenen yanlış pozitif oranı ≈ alpha (%5).

    Eğer yanlış pozitif oranı %5'ten anlamlı şekilde farklıysa,
    istatistiksel altyapıda sorun var demektir.
    """
    false_positive_count = 0
    p_values = []

    for _ in range(n_iterations):
        # İki grup DA AYNI dağılımdan çekiliyor (gerçek fark = 0)
        group_a = np.random.normal(mu, sigma, n_per_group)
        group_b = np.random.normal(mu, sigma, n_per_group)

        _, p = stats.ttest_ind(group_a, group_b)
        p_values.append(p)

        if p < alpha:
            false_positive_count += 1

    fp_rate = false_positive_count / n_iterations

    print(f"A/A Test Simülasyonu ({n_iterations} iterasyon)")
    print(f"{'='*50}")
    print(f"Gözlenen yanlış pozitif oranı: {fp_rate:.3f} ({fp_rate*100:.1f}%)")
    print(f"Beklenen yanlış pozitif oranı: {alpha:.3f} ({alpha*100:.1f}%)")

    # p-value dağılımı uniform olmalı
    # KS testi: p-value'lar Uniform(0,1)'den mi geliyor?
    ks_stat, ks_p = stats.kstest(p_values, 'uniform')
    print(f"\nKS testi (p-value uniformluğu): stat={ks_stat:.4f}, p={ks_p:.4f}")
    if ks_p < 0.05:
        print("⚠️  p-value dağılımı uniform değil — istatistiksel altyapı sorunlu!")
    else:
        print("✓ p-value dağılımı uniform — altyapı sağlıklı.")

    # Yanlış pozitif oranının beklentiden sapması (binomial CI)
    se = np.sqrt(alpha * (1 - alpha) / n_iterations)
    ci_low, ci_high = alpha - 2*se, alpha + 2*se
    if ci_low <= fp_rate <= ci_high:
        print(f"✓ FP oranı beklenen aralıkta [{ci_low:.3f}, {ci_high:.3f}]")
    else:
        print(f"⚠️  FP oranı beklenen aralık dışında [{ci_low:.3f}, {ci_high:.3f}]")

    return p_values, fp_rate

np.random.seed(42)
p_vals, fp = aa_test_simulation()
```

### A/A Test + SRM Kontrolü

```python
def aa_test_with_srm(n_iterations=1000, n_total=10000,
                      split_ratio=0.5, alpha=0.05):
    """
    A/A test + SRM kontrolü birlikte.
    Hem metrik tutarlılığını hem randomizasyon bütünlüğünü test eder.
    """
    from scipy import stats

    fp_count = 0
    srm_count = 0

    for _ in range(n_iterations):
        # Gerçek hayatta randomizasyon kusurlu olabilir
        # Simülasyonda %50/%50 olması beklenir
        n_a = np.random.binomial(n_total, split_ratio)
        n_b = n_total - n_a

        # SRM kontrolü (chi-squared)
        expected_a = n_total * split_ratio
        expected_b = n_total * (1 - split_ratio)
        chi2, srm_p = stats.chisquare([n_a, n_b], [expected_a, expected_b])
        if srm_p < 0.01:  # SRM için daha sıkı alpha
            srm_count += 1

        # Metrik testi
        group_a = np.random.normal(45, 30, n_a)
        group_b = np.random.normal(45, 30, n_b)
        _, p = stats.ttest_ind(group_a, group_b)
        if p < alpha:
            fp_count += 1

    print(f"Yanlış pozitif oranı: {fp_count/n_iterations:.3f}")
    print(f"SRM alarm oranı: {srm_count/n_iterations:.3f}")
    print(f"(SRM alarm oranı da ~%1 civarında olmalı, alpha_srm=0.01)")

np.random.seed(42)
aa_test_with_srm()
```

> **Senior Notu:** A/A testinden önce A/B test güvenilir değildir. Yeni bir deney platformu kurduysan, metrik pipeline değiştirdiysen veya randomizasyon mantığında güncelleme yaptıysan, mutlaka A/A test koş. Büyük platformlarda (Statsig, Eppo, GrowthBook) bu sürekli ve otomatik çalışır. A/A testte SRM hatası görüyorsan, logging asimetrisi, bot filtreleme farkı veya hash fonksiyonu sorununa bak.

---

## C.4 CUPED (Varyans Azaltma)

### Sezgisel Açıklama

CUPED = Controlled-experiment Using Pre-Experiment Data. Deney öncesi metrik bilgisini kullanarak gürültüyü azalt → aynı n ile daha dar CI → ya daha az kullanıcı gerekir ya da aynı kullanıcıyla daha küçük etki tespiti.

Microsoft Research'ün 2013 makalesinden. Bugün tüm büyük platformlarda standart (Airbnb, Netflix, Google).

### Matematik Detayı

```
Y_cuped = Y_post - θ · X_pre

θ = Cov(Y_post, X_pre) / Var(X_pre)  ← OLS katsayısı

Var(Y_cuped) = Var(Y_post) · (1 - ρ²)

ρ = korelasyon(Y_post, X_pre)

Örnek: ρ = 0.7 → Var %51 azalır → CI %29 daralır
```

### Kod Örneği

```python
import numpy as np
from scipy import stats

def cuped_analysis(control_post, treatment_post,
                   control_pre, treatment_pre,
                   alpha=0.05):
    """
    CUPED ile A/B test analizi.

    Returns:
        dict: original_pvalue, cuped_pvalue, var_reduction, ci_narrowing
    """
    # Orijinal t-testi
    t_orig, p_orig = stats.ttest_ind(treatment_post, control_post)
    se_orig = np.sqrt(np.var(control_post)/len(control_post) +
                       np.var(treatment_post)/len(treatment_post))

    # CUPED düzeltmesi
    # θ her kol için ayrı hesaplanır (leakage önleme)
    all_post = np.concatenate([control_post, treatment_post])
    all_pre = np.concatenate([control_pre, treatment_pre])

    theta = np.cov(all_post, all_pre)[0, 1] / np.var(all_pre)

    control_cuped = control_post - theta * (control_pre - np.mean(all_pre))
    treatment_cuped = treatment_post - theta * (treatment_pre - np.mean(all_pre))

    t_cuped, p_cuped = stats.ttest_ind(treatment_cuped, control_cuped)
    se_cuped = np.sqrt(np.var(control_cuped)/len(control_cuped) +
                        np.var(treatment_cuped)/len(treatment_cuped))

    # Sonuçlar
    var_reduction = 1 - se_cuped**2 / se_orig**2
    ci_narrowing = 1 - se_cuped / se_orig

    print(f"Orijinal p-value: {p_orig:.4f}")
    print(f"CUPED p-value:    {p_cuped:.4f}")
    print(f"Varyans azalması: {var_reduction:.1%}")
    print(f"CI daralması:     {ci_narrowing:.1%}")

    # Pre-post korelasyon
    rho = np.corrcoef(all_post, all_pre)[0, 1]
    print(f"Pre-post korelasyon (ρ): {rho:.3f}")

    return dict(p_orig=p_orig, p_cuped=p_cuped,
                var_reduction=var_reduction, ci_narrowing=ci_narrowing)

# Simülasyon
np.random.seed(42)
n = 5000
true_effect = 1.0  # 1 TL etki

# Pre-period (önceki hafta geliri)
control_pre = np.random.normal(45, 25, n)
treatment_pre = np.random.normal(45, 25, n)

# Post-period (korelasyonlu + treatment etkisi)
control_post = 0.7 * control_pre + np.random.normal(0, 18, n)
treatment_post = 0.7 * treatment_pre + true_effect + np.random.normal(0, 18, n)

results = cuped_analysis(control_post, treatment_post, control_pre, treatment_pre)
```

> **Senior Notu:** CUPED en iyi çalışır pre-post korelasyon yüksekken (ρ > 0.5). Yeni kullanıcılar için pre-period verisi olmadığında uygulanamaz. Covariate olarak aynı metriğin önceki değeri ideal, farklı metrik de kullanılabilir (ama yorumlama dikkatli).

---

## C.5 Sequential Testing (Peeking Sorunu)

### Sezgisel Açıklama

Deney başladıktan sonra her gün bakıp "anlamlı çıktı mı?" dersen, zamanla yanlış pozitif oranı artar. Kural: ya sabit horizon (sonunda bir kez bak), ya önceden tasarlanmış sequential test.

### Peeking Simülasyonu

```python
import numpy as np

def simulate_peeking_error(n_simulations=10000, n_days=14,
                            users_per_day=500, alpha=0.05):
    """
    Deney boyunca her gün bakmanın yanlış pozitif oranına etkisi.
    H₀ doğru (gerçek etki yok) senaryosu.
    """
    from scipy import stats

    false_positives_peeking = 0
    false_positives_fixed = 0

    for _ in range(n_simulations):
        # H₀ doğru — her iki kolda aynı dağılım
        control_data = []
        treatment_data = []
        found_by_peeking = False

        for day in range(1, n_days + 1):
            control_data.extend(np.random.normal(50, 25, users_per_day))
            treatment_data.extend(np.random.normal(50, 25, users_per_day))

            # Her gün bak (peeking)
            if not found_by_peeking:
                _, p = stats.ttest_ind(treatment_data, control_data)
                if p < alpha:
                    found_by_peeking = True
                    false_positives_peeking += 1

        # Sadece sonda bak (sabit horizon)
        _, p_final = stats.ttest_ind(treatment_data, control_data)
        if p_final < alpha:
            false_positives_fixed += 1

    fp_peeking = false_positives_peeking / n_simulations
    fp_fixed = false_positives_fixed / n_simulations

    print(f"Her gün bakan (peeking): %{fp_peeking*100:.1f} yanlış pozitif")
    print(f"Sadece sonda bakan:     %{fp_fixed*100:.1f} yanlış pozitif")
    print(f"Hedef (alpha={alpha}):  %{alpha*100:.1f}")
    print(f"\n⚠️  Peeking yanlış pozitifi {fp_peeking/fp_fixed:.1f}× artırıyor!")

simulate_peeking_error()
```

### SPRT — Sequential Probability Ratio Test

Wald'ın 1945'teki SPRT'si, sequential testing'in temelini oluşturur. Her yeni gözlemde "yeterli kanıt var mı?" sorusunu sorar ve üç karar verir: H₀ reddet, H₁ reddet veya devam et.

```
Log-likelihood ratio:
  Λ_n = Σᵢ log[ f₁(xᵢ) / f₀(xᵢ) ]

Karar sınırları:
  A = log[(1-β) / α]   → Λ_n ≥ A ise H₀ reddet (treatment etkili)
  B = log[β / (1-α)]   → Λ_n ≤ B ise H₁ reddet (etki yok)
  B < Λ_n < A ise → veri toplamaya devam et
```

```python
import numpy as np
from scipy import stats

def sprt_normal(control_data, treatment_data,
                delta=0.5, sigma=1.0, alpha=0.05, beta=0.20):
    """
    SPRT (Sequential Probability Ratio Test) — Normal dağılım varsayımı.

    Her gözlem çiftinde log-likelihood ratio güncellenir.
    Sınırlara ulaşıldığında karar verilir.

    Args:
        control_data: Control grubu gözlemleri (sıralı)
        treatment_data: Treatment grubu gözlemleri (sıralı)
        delta: H₁ altında beklenen etki büyüklüğü
        sigma: Bilinen standart sapma
        alpha: Tip-I hata oranı
        beta: Tip-II hata oranı (1-power)

    Returns:
        dict: karar, durma noktası, log-likelihood ratio geçmişi
    """
    # Karar sınırları
    A = np.log((1 - beta) / alpha)      # H₀ reddet sınırı (üst)
    B = np.log(beta / (1 - alpha))       # H₁ reddet sınırı (alt)

    n = min(len(control_data), len(treatment_data))
    llr = 0  # Log-likelihood ratio
    llr_history = [0]

    for i in range(n):
        # Her gözlem çifti için fark
        diff = treatment_data[i] - control_data[i]

        # Log-likelihood ratio güncellemesi (normal varsayım)
        # H₁: diff ~ N(delta, 2σ²)  vs  H₀: diff ~ N(0, 2σ²)
        llr += (delta * diff - delta**2 / 2) / (2 * sigma**2)
        llr_history.append(llr)

        if llr >= A:
            print(f"KARAR: H₀ reddedildi (treatment etkili)")
            print(f"  Gözlem #{i+1}'de duruldu (toplam n'nin {(i+1)/n:.0%}'i)")
            print(f"  Log-LR: {llr:.3f} >= A={A:.3f}")
            return {"decision": "reject_H0", "stop_at": i+1,
                    "llr_history": llr_history}

        if llr <= B:
            print(f"KARAR: H₁ reddedildi (etki yok)")
            print(f"  Gözlem #{i+1}'de duruldu")
            print(f"  Log-LR: {llr:.3f} <= B={B:.3f}")
            return {"decision": "reject_H1", "stop_at": i+1,
                    "llr_history": llr_history}

    print(f"Kararsız: {n} gözlemde sınırlara ulaşılamadı")
    return {"decision": "inconclusive", "stop_at": n,
            "llr_history": llr_history}

# Simülasyon: gerçek etki var
np.random.seed(42)
n = 5000
control = np.random.normal(50, 10, n)
treatment = np.random.normal(50.5, 10, n)  # 0.5 birimlik etki

result = sprt_normal(control, treatment, delta=0.5, sigma=10)
```

### O'Brien-Fleming Sınırları (Group Sequential Test)

O'Brien-Fleming yöntemi, önceden planlanmış ara analiz noktalarında (interim analyses) test yapmayı sağlar. Alpha bütçesini harcama fonksiyonu ile dağıtır: erken bakışlarda çok sıkı, son bakışta standart alpha'ya yakın.

```python
import numpy as np
from scipy import stats

def obrien_fleming_bounds(n_analyses=5, overall_alpha=0.05):
    """
    O'Brien-Fleming sınırlarını hesapla.

    n_analyses ara analiz sayısı (eşit aralıklı varsayım).
    Her ara analizde kullanılacak z-sınırı ve nominal alpha döndürür.

    O'Brien-Fleming'de z-sınırı: z_k = z_final / sqrt(k/K)
    Bu, erken bakışlarda çok yüksek (sıkı), son bakışta düşük (gevşek) sınır verir.
    """
    # Son analizdeki z-sınırı ≈ genel z_alpha (hafif ayarlanmış)
    z_final = stats.norm.ppf(1 - overall_alpha / 2)

    print(f"O'Brien-Fleming Sınırları ({n_analyses} ara analiz)")
    print(f"{'='*60}")
    print(f"{'Analiz':>8} | {'Bilgi %':>10} | {'z-sınırı':>10} | {'Nominal α':>12}")
    print(f"{'-'*60}")

    bounds = []
    for k in range(1, n_analyses + 1):
        info_fraction = k / n_analyses
        # O'Brien-Fleming: z_k = z_final / sqrt(info_fraction)
        z_k = z_final / np.sqrt(info_fraction)
        nominal_alpha_k = 2 * (1 - stats.norm.cdf(z_k))

        bounds.append({
            "analysis": k,
            "info_fraction": info_fraction,
            "z_bound": z_k,
            "nominal_alpha": nominal_alpha_k
        })

        print(f"{k:>8} | {info_fraction:>9.0%} | {z_k:>10.3f} | {nominal_alpha_k:>12.6f}")

    print(f"\nYorum: İlk bakışta nominal α = {bounds[0]['nominal_alpha']:.6f} (çok sıkı),")
    print(f"       Son bakışta nominal α = {bounds[-1]['nominal_alpha']:.6f} (standart α'ya yakın)")

    return bounds

bounds = obrien_fleming_bounds(n_analyses=5)
```

### Always-Valid Confidence Intervals ve Anytime p-value

Johari et al. (2017, Operations Research 2022) tarafından önerilen "always-valid" yaklaşımda, CI'lar herhangi bir durma zamanında geçerlidir. Klasik CI'dan farkı: genişliği sabit değil, veri birikiminin karekökü ile daralır ama her noktada geçerli kalır.

2026 itibarıyla bu yaklaşım endüstri standardı haline geldi: Statsig ve GrowthBook gibi platformlar mSPRT tabanlı always-valid CI'ları default olarak sunuyor. Son araştırmalar (MDPI Mathematics, 2025) iki oran karşılaştırması için fixed-width sequential confidence intervals (FWCI) yöntemlerini öneriyor — log-olasılık farkı ve log-odds ratio için iki ayrı prosedür tanımlanıyor ve Monte Carlo simülasyonlarıyla etkinlikleri doğrulanıyor.

**Neden önemli:** Sabit horizonlu testte CI sadece deney sonunda geçerli. Peeking yapıldığında CI'ın kapsama olasılığı nominal %95'in altına düşer. Always-valid CI ise her bakışta %95 kapsama garantisi verir — bunun bedeli daha geniş CI'lardır, ama karar anında geçerlilik korunur.

```python
import numpy as np
from scipy import stats

def always_valid_ci(data_stream, alpha=0.05, rho=1.0):
    """
    Always-valid (anytime-valid) confidence interval hesaplayıcı.

    Mixture sequential probability ratio (mSPRT) tabanlı.
    Her t anında geçerli CI döndürür — peeking sorunu yok.

    Args:
        data_stream: Sıralı gözlemler (treatment - control farkları)
        alpha: Genel anlamlılık düzeyi
        rho: mSPRT mixing parametresi (varyansa oranla prior genişliği)

    Returns:
        list of tuples: (n, lower_ci, upper_ci, anytime_p_value)
    """
    results = []
    cumsum = 0
    cumsum_sq = 0

    for t, x in enumerate(data_stream, 1):
        cumsum += x
        cumsum_sq += x**2

        # Örneklem ortalaması ve varyansı
        x_bar = cumsum / t
        if t > 1:
            s2 = (cumsum_sq - t * x_bar**2) / (t - 1)
        else:
            s2 = 1.0  # İlk gözlemde varsayılan

        # mSPRT log-likelihood ratio (normal mixture)
        # V_t = t * s2 (varyans bileşeni)
        # Lambda = (t * x_bar)^2 / (2 * (rho + t * s2))
        V_t = rho + t * s2
        lambda_t = (t * x_bar)**2 / (2 * V_t)

        # Anytime p-value
        anytime_p = min(1.0, np.exp(-lambda_t))

        # Always-valid CI
        # Genişlik: sqrt(2 * V_t * log(1/alpha) / t^2)
        width = np.sqrt(2 * V_t * np.log(1 / alpha)) / t
        ci_lower = x_bar - width
        ci_upper = x_bar + width

        results.append((t, ci_lower, ci_upper, anytime_p))

    return results

# Demo: gerçek etki = 0.3 olan veri akışı
np.random.seed(42)
n = 2000
diffs = np.random.normal(0.3, 5.0, n)  # treatment - control farkları

cis = always_valid_ci(diffs, alpha=0.05)

# Her 200 gözlemde rapor
print(f"{'n':>6} | {'CI alt':>10} | {'CI üst':>10} | {'Anytime p':>12} | {'0 CI dışında?':>15}")
print("-" * 65)
for t, lo, hi, p in cis:
    if t % 200 == 0 or t == 1:
        outside = "Evet (anlamlı)" if lo > 0 or hi < 0 else "Hayır"
        print(f"{t:>6} | {lo:>10.4f} | {hi:>10.4f} | {p:>12.6f} | {outside:>15}")
```

### Düzeltilmiş vs Düzeltilmemiş Karşılaştırma

```python
import numpy as np
from scipy import stats

def compare_corrections(n_simulations=5000, n_days=10,
                         users_per_day=300, alpha=0.05):
    """
    Peeking senaryosunda:
    1. Düzeltilmemiş (naive p-value, her gün bak)
    2. Bonferroni düzeltmeli (alpha/n_looks)
    3. O'Brien-Fleming düzeltmeli

    Tüm senaryolarda H₀ doğru (gerçek etki yok).
    """
    fp_naive = 0
    fp_bonferroni = 0
    fp_obf = 0

    # O'Brien-Fleming sınırları
    z_final = stats.norm.ppf(1 - alpha / 2)

    for _ in range(n_simulations):
        control = []
        treatment = []
        found_naive = False
        found_bonf = False
        found_obf = False

        for day in range(1, n_days + 1):
            control.extend(np.random.normal(50, 25, users_per_day))
            treatment.extend(np.random.normal(50, 25, users_per_day))

            t_stat, p = stats.ttest_ind(treatment, control)
            z = abs(t_stat)

            # 1. Naive: sabit alpha
            if not found_naive and p < alpha:
                found_naive = True

            # 2. Bonferroni: alpha / n_looks
            if not found_bonf and p < alpha / n_days:
                found_bonf = True

            # 3. O'Brien-Fleming: z sınırı bilgi fraksiyonuna göre
            info_frac = day / n_days
            z_bound = z_final / np.sqrt(info_frac)
            if not found_obf and z > z_bound:
                found_obf = True

        fp_naive += found_naive
        fp_bonferroni += found_bonf
        fp_obf += found_obf

    print(f"Yanlış Pozitif Oranları (H₀ doğru, {n_simulations} simülasyon)")
    print(f"{'='*50}")
    print(f"Düzeltilmemiş (naive):    {fp_naive/n_simulations:.3f}")
    print(f"Bonferroni düzeltmeli:    {fp_bonferroni/n_simulations:.3f}")
    print(f"O'Brien-Fleming:          {fp_obf/n_simulations:.3f}")
    print(f"Hedef (alpha={alpha}):    {alpha:.3f}")
    print(f"\nYorum: Naive yaklaşım alpha'yı şişirir. OBF, alpha'yı korurken")
    print(f"       Bonferroni'den daha az muhafazakar (daha yüksek power).")

np.random.seed(42)
compare_corrections()
```

### Çözümler

```python
# 1. Bayesian monitoring (peeking sorunu yok)
def bayesian_ab_monitor(control_successes, control_trials,
                          treat_successes, treat_trials,
                          n_sim=50000):
    """Bayesian A/B: her zaman bakabilir, posterior yorumlanır."""
    from scipy import stats

    # Posterior parametreleri (Beta-Binomial)
    alpha_c = 1 + control_successes
    beta_c = 1 + (control_trials - control_successes)
    alpha_t = 1 + treat_successes
    beta_t = 1 + (treat_trials - treat_successes)

    # Simülasyon ile P(treat > control)
    control_samples = stats.beta(alpha_c, beta_c).rvs(n_sim)
    treat_samples = stats.beta(alpha_t, beta_t).rvs(n_sim)

    prob_treat_better = np.mean(treat_samples > control_samples)
    expected_lift = np.mean(treat_samples / control_samples - 1)

    print(f"P(treatment > control): {prob_treat_better:.1%}")
    print(f"Beklenen lift: {expected_lift:.2%}")
    return prob_treat_better
```

> **Senior Notu:** 2026 itibarıyla sequential testing artık "isteğe bağlı ileri konu" değil, her A/B test platformunun temel bileşeni. Statsig ve Eppo always-valid CI'ları (mSPRT tabanlı) default kullanıyor. Sabit horizon testler hala geçerli ama "3. günde baktım, anlamlı çıktı, yayınladım" kabul edilmiyor. Eğer platformun sequential test desteklemiyorsa, en azından O'Brien-Fleming sınırları ile ara analiz planla.

---

## C.6 Karıştırıcı Etkiler ve Network Effects

### Temel Sorunlar

```
Network effects (interference):
  Kullanıcılar birbirini etkiliyor → treatment spillover
  → Uber Eats, sosyal ağ, iki taraflı pazar
  → Standart A/B geçersiz (SUTVA varsayımı kırılıyor)

Çözümler:
  1. Cluster randomization: coğrafi bölge veya grafik topluluk bazında atama
  2. Switchback test: zaman bazlı (1 saat treatment, 1 saat control)
  3. Bipartite graph cluster (LinkedIn yaklaşımı)

Novelty effect:
  Kullanıcılar yeni özelliği meraktan deniyor → abartılı kısa vadeli etki
  Çözüm: uzun deney (4+ hafta), "long-run" kullanıcı analizi

Non-compliance:
  Control grubundaki kullanıcı bazen treatment özelliğini görüyor
  ITT (Intent-to-treat) vs As-treated analizi
  → ITT: daha muhafazakar, ama gerçekçi
```

### Heterogeneous Treatment Effects

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def heterogeneous_treatment_analysis(df: pd.DataFrame,
                                       treatment_col: str,
                                       outcome_col: str,
                                       segment_cols: list):
    """
    Segment bazlı treatment etkisi analizi.
    HTE (Heterogeneous Treatment Effects) araştırması.
    """
    from scipy import stats

    print("=" * 60)
    print("HTE Analizi — Segment Bazlı Treatment Etkisi")
    print("=" * 60)

    # Global etki
    control = df[df[treatment_col] == 0][outcome_col]
    treatment = df[df[treatment_col] == 1][outcome_col]
    global_effect = treatment.mean() - control.mean()
    t, p = stats.ttest_ind(treatment, control)
    print(f"\nGlobal etki: {global_effect:.3f} (p={p:.4f})")

    # Segment bazlı
    for seg_col in segment_cols:
        print(f"\n--- {seg_col} ---")
        for seg_val in df[seg_col].unique():
            seg = df[df[seg_col] == seg_val]
            c = seg[seg[treatment_col] == 0][outcome_col]
            t_seg = seg[seg[treatment_col] == 1][outcome_col]
            if len(c) > 30 and len(t_seg) > 30:
                effect = t_seg.mean() - c.mean()
                _, p_seg = stats.ttest_ind(t_seg, c)
                print(f"  {seg_val}: etki={effect:.3f}, p={p_seg:.4f}, n={len(seg)}")
```

---

## C.7 Causal Inference — Gözlemsel Veri

### Sezgisel Açıklama

A/B test mümkün değilse (etik, teknik, maliyet) gözlemsel veriden nedensellik çıkarılabilir — ama varsayımlar dikkatli olmalı.

"Kampanya alan kullanıcılar daha çok aldı" → Nedensel mi? Belki zaten satın alma eğilimli kullanıcılar kampanyayı aldı (confounding).

> **Çapraz referans:** Katman B'de SHAP ile feature importance hesaplandığında, SHAP değerleri korelasyon gösterir ama nedensellik değildir. Buradaki yöntemler (DiD, RDD, IV, Synthetic Control) nedensellik iddiası kurmaya çalışır — ama her biri farklı varsayımlar gerektirir. SHAP "bu feature tahmine katkı yapıyor" der; causal inference "bu müdahale sonucu değiştirdi" der.

### Difference-in-Differences (DiD)

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def did_estimate(df: pd.DataFrame,
                  outcome: str,
                  treatment_col: str,
                  post_col: str,
                  covariates: list = None):
    """
    DiD tahmini — parallel trends varsayımı altında.

    Model: y = β₀ + β₁*treatment + β₂*post + β₃*(treatment×post) + covariates
    β₃ = DiD tahmini (nedensel etki tahmini)
    """
    formula = f"{outcome} ~ {treatment_col} * {post_col}"
    if covariates:
        formula += " + " + " + ".join(covariates)

    model = smf.ols(formula, data=df).fit()
    print(model.summary())

    # DiD katsayısı
    did_coef = model.params[f"{treatment_col}:{post_col}"]
    did_se = model.bse[f"{treatment_col}:{post_col}"]
    print(f"\nDiD tahmini: {did_coef:.3f} ± {did_se:.3f}")
    print(f"95% CI: [{did_coef - 1.96*did_se:.3f}, {did_coef + 1.96*did_se:.3f}]")
    return model

# Propensity Score Matching
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def propensity_score_matching(df, treatment_col, covariates, outcome_col,
                               caliper=0.05):
    """Propensity score matching ile ATE tahmini."""
    X = df[covariates]
    T = df[treatment_col]

    # Propensity score tahmini
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ps_model = LogisticRegression(max_iter=500)
    ps_model.fit(X_scaled, T)

    df = df.copy()
    df["propensity_score"] = ps_model.predict_proba(X_scaled)[:, 1]

    # Caliper matching (greedy)
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()

    matched_pairs = []
    control_used = set()

    for _, t_row in treated.iterrows():
        # Caliper içindeki en yakın control
        diffs = abs(control[~control.index.isin(control_used)]["propensity_score"]
                    - t_row["propensity_score"])
        if len(diffs) > 0 and diffs.min() < caliper:
            best_match = diffs.idxmin()
            matched_pairs.append((t_row.name, best_match))
            control_used.add(best_match)

    if not matched_pairs:
        print("Eşleşme bulunamadı. Caliper'ı artır.")
        return None

    t_ids = [p[0] for p in matched_pairs]
    c_ids = [p[1] for p in matched_pairs]

    ate = (df.loc[t_ids, outcome_col].mean() -
           df.loc[c_ids, outcome_col].mean())
    print(f"Eşleşme sayısı: {len(matched_pairs)}")
    print(f"ATE tahmini: {ate:.3f}")
    return ate
```

### Synthetic Control Method (Sentetik Kontrol)

**Sezgisel açıklama:** Bir şehirde yeni vergi uygulandı. Etkisini ölçmek istiyorsun ama A/B test yapamazsın (tüm şehri etkiliyor). Çözüm: Diğer şehirlerin ağırlıklı ortalamasından "sentetik" bir kontrol şehri oluştur ki, müdahale öncesi dönemde gerçek şehirle aynı trendi göstersin. Müdahale sonrası fark = nedensel etki tahmini.

Analoji: İkiz kardeşin yok ama birçok arkadaşının özelliklerini karıştırarak "sanal ikizini" oluşturabilirsin. Sonra "ikizin yapmasaydı ne olurdu?" sorusunu yanıtlarsın.

> **Python ekosistemi (2026):** `SyntheticControlMethods` (PyPI), `pysyncon` ve Microsoft'un `SparseSC` paketi hazır implementasyonlar sunar. Aşağıdaki numpy implementasyonu kavramsal anlayış içindir; production'da bu paketler tercih edilebilir.

```python
import numpy as np
from scipy.optimize import minimize

def synthetic_control(treated_pre, treated_post, donor_pre, donor_post):
    """
    Basit Synthetic Control implementasyonu (numpy).

    Args:
        treated_pre: Tedavi edilen birimin müdahale öncesi sonuçları (T_pre,)
        treated_post: Tedavi edilen birimin müdahale sonrası sonuçları (T_post,)
        donor_pre: Donor havuzu müdahale öncesi (T_pre, J) — J donor birim
        donor_post: Donor havuzu müdahale sonrası (T_post, J)

    Returns:
        dict: ağırlıklar, sentetik kontrol, tahmini etki
    """
    J = donor_pre.shape[1]  # donor sayısı

    # Amaç: min ||treated_pre - donor_pre @ w||²
    # Kısıt: w >= 0, sum(w) = 1 (konveks kombinasyon)
    def objective(w):
        synthetic = donor_pre @ w
        return np.sum((treated_pre - synthetic) ** 2)

    # Başlangıç: eşit ağırlık
    w0 = np.ones(J) / J

    # Kısıtlar
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * J

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    w_star = result.x

    # Sentetik kontrol serisi
    synthetic_pre = donor_pre @ w_star
    synthetic_post = donor_post @ w_star

    # Tahmini nedensel etki (gap)
    gap = treated_post - synthetic_post

    # Pre-period fit kalitesi (RMSPE)
    pre_rmspe = np.sqrt(np.mean((treated_pre - synthetic_pre) ** 2))

    print(f"Synthetic Control Sonuçları")
    print(f"{'='*50}")
    print(f"Donor ağırlıkları (> 0.01):")
    for j in range(J):
        if w_star[j] > 0.01:
            print(f"  Donor {j}: {w_star[j]:.3f}")
    print(f"\nPre-period RMSPE: {pre_rmspe:.4f}")
    print(f"Ortalama post-period etki: {np.mean(gap):.3f}")
    print(f"Post-period etki (son gözlem): {gap[-1]:.3f}")

    return {"weights": w_star, "synthetic_post": synthetic_post,
            "gap": gap, "pre_rmspe": pre_rmspe}

# Simülasyon: 1 tedavi birimi, 10 donor, müdahale sonrası etki = 5
np.random.seed(42)
T_pre, T_post, J = 20, 10, 10

# Donor birimleri: trend + birim sabit etki
base_trend = np.cumsum(np.random.normal(0.5, 0.3, T_pre + T_post))
donor_data = np.column_stack([
    base_trend + np.random.normal(0, 1, T_pre + T_post) + np.random.uniform(-5, 5)
    for _ in range(J)
])

# Tedavi birimi: donor 2 ve 5'in ağırlıklı ortalamasına yakın + etki
true_weights = np.zeros(J)
true_weights[2] = 0.6
true_weights[5] = 0.4
treated_data = donor_data @ true_weights + np.random.normal(0, 0.5, T_pre + T_post)
# Müdahale sonrası etki ekle
treated_data[T_pre:] += 5.0

result = synthetic_control(
    treated_pre=treated_data[:T_pre],
    treated_post=treated_data[T_pre:],
    donor_pre=donor_data[:T_pre],
    donor_post=donor_data[T_pre:]
)
```

### Regression Discontinuity Design (RDD)

**Sezgisel açıklama:** Bir burs programı sınav puanı 70 ve üzeri olan öğrencilere veriliyor. 69 puan alan ile 71 puan alan öğrenci neredeyse aynı kişi — ama biri burs alıyor, diğeri almıyor. Bu "eşik" etrafındaki fark, bursun nedensel etkisini verir.

RDD'nin gücü: Eşik etrafında doğal bir "yarı-randomizasyon" oluşur. Tam eşiğe yakın kişiler arasında confounding minimal olur.

```
Y = α + β₁·X + β₂·T + β₃·(X·T) + ε

X: running variable (sınav puanı)
T: treatment göstergesi (T = 1 eğer X ≥ cutoff)
β₂: nedensel etki tahmini (eşikteki sıçrama)
```

```python
import numpy as np
import statsmodels.api as sm

def rdd_analysis(running_var, outcome, cutoff, bandwidth=None):
    """
    Regression Discontinuity Design — local linear regression.

    Args:
        running_var: Atama değişkeni (sınav puanı, gelir eşiği vb.)
        outcome: Sonuç değişkeni
        cutoff: Eşik değeri
        bandwidth: Pencere genişliği (None ise otomatik: IQR/2)

    Returns:
        dict: tahmini etki, standart hata, p-value
    """
    running_var = np.array(running_var)
    outcome = np.array(outcome)

    # Treatment atama: running_var >= cutoff
    treatment = (running_var >= cutoff).astype(float)

    # Bandwidth belirleme (basit kural)
    if bandwidth is None:
        bandwidth = np.percentile(np.abs(running_var - cutoff), 50)

    # Eşik etrafında pencere
    mask = np.abs(running_var - cutoff) <= bandwidth
    X_local = running_var[mask] - cutoff  # Merkezle
    T_local = treatment[mask]
    Y_local = outcome[mask]

    print(f"RDD Analizi")
    print(f"{'='*50}")
    print(f"Cutoff: {cutoff}, Bandwidth: {bandwidth:.2f}")
    print(f"Pencere içi gözlem: {mask.sum()} / {len(running_var)}")
    print(f"  Sol (control): {(~treatment[mask].astype(bool)).sum()}")
    print(f"  Sağ (treatment): {treatment[mask].astype(bool).sum()}")

    # Local linear regression: Y = a + b1*X + b2*T + b3*X*T
    X_design = np.column_stack([
        np.ones(len(X_local)),
        X_local,
        T_local,
        X_local * T_local
    ])

    model = sm.OLS(Y_local, X_design).fit()

    # β₂ = treatment etkisi (eşikteki sıçrama)
    effect = model.params[2]
    se = model.bse[2]
    p_val = model.pvalues[2]

    print(f"\nTahmini nedensel etki (eşikteki sıçrama): {effect:.3f}")
    print(f"Standart hata: {se:.3f}")
    print(f"p-value: {p_val:.4f}")
    print(f"95% CI: [{effect - 1.96*se:.3f}, {effect + 1.96*se:.3f}]")

    return {"effect": effect, "se": se, "p_value": p_val}

# Simülasyon: burs programı, cutoff = 70
np.random.seed(42)
n = 2000
exam_scores = np.random.uniform(40, 100, n)
treatment = (exam_scores >= 70).astype(float)

# Sonuç: doğal trend + treatment etkisi (5 birim)
outcome = (
    0.5 * exam_scores              # Puan arttıkça zaten gelir artıyor
    + 5.0 * treatment              # Bursun gerçek etkisi = 5
    + np.random.normal(0, 8, n)    # Gürültü
)

result = rdd_analysis(exam_scores, outcome, cutoff=70, bandwidth=10)
```

### Instrumental Variables (IV) — Araç Değişkenler

**Sezgisel açıklama:** Eğitimin gelire etkisini ölçmek istiyorsun. Ama "zeki insanlar hem daha çok eğitim alır hem daha çok kazanır" (confounding). Çözüm: Bir "araç değişken" bul — eğitimi etkileyen ama geliri **yalnızca eğitim yoluyla** etkileyen bir değişken.

Analoji: Doğum yılının çeyreği (Q1, Q2...) okula başlama yaşını etkiler → daha uzun eğitim → gelir. Ama doğum çeyreği doğrudan geliri etkilemez (yalnızca eğitim yoluyla). Bu durumda doğum çeyreği bir "araç değişken" (instrument) olur.

```
IV gereksinimleri (üç koşul):
1. Relevance:  Z → X  (araç, treatment'ı etkiler)
2. Exclusion:  Z → Y yalnızca X yoluyla (doğrudan etki yok)
3. Independence: Z ⊥ U  (araç, gizli confounder'dan bağımsız)
```

```python
import numpy as np
from linearmodels.iv import IV2SLS
import pandas as pd

def iv_analysis_example():
    """
    Instrumental Variables (2SLS) — Eğitim-Gelir örneği.

    Araç değişken: Üniversiteye yakınlık (proximity)
    → Yakın olan daha çok eğitim alır (relevance)
    → Ama yakınlık doğrudan geliri etkilemez (exclusion)
    """
    np.random.seed(42)
    n = 5000

    # Gizli confounder: yetenek (gözlemlenmiyor)
    ability = np.random.normal(0, 1, n)

    # Araç değişken: üniversiteye yakınlık (km)
    proximity = np.random.uniform(0, 100, n)

    # Eğitim yılı (treatment): yetenek + yakınlık + gürültü
    education = 12 + 0.8 * ability - 0.03 * proximity + np.random.normal(0, 1, n)

    # Gelir (outcome): eğitimin GERÇEK etkisi = 2.0, yetenek confound
    true_effect = 2.0
    income = 20 + true_effect * education + 3.0 * ability + np.random.normal(0, 5, n)

    df = pd.DataFrame({
        "income": income,
        "education": education,
        "proximity": proximity,
        "ability": ability  # Normalde gözlemlenmez — karşılaştırma için
    })

    # 1. Naive OLS (confounded — ability gözlemlenmiyor)
    import statsmodels.formula.api as smf
    ols = smf.ols("income ~ education", data=df).fit()
    print("Naive OLS (confounded):")
    print(f"  Eğitim katsayısı: {ols.params['education']:.3f}")
    print(f"  (Gerçek etki: {true_effect}, OLS yukarı sapmalı çünkü ability confound)")

    # 2. IV-2SLS (proximity araç değişken)
    iv_model = IV2SLS.from_formula("income ~ 1 + [education ~ proximity]", data=df)
    iv_result = iv_model.fit()
    print(f"\nIV-2SLS (araç: proximity):")
    print(f"  Eğitim katsayısı: {iv_result.params['education']:.3f}")
    print(f"  Standart hata: {iv_result.std_errors['education']:.3f}")
    print(f"  (Gerçek etkiye ({true_effect}) daha yakın)")

    # İlk aşama F-istatistiği (araç gücü kontrolü)
    first_stage = smf.ols("education ~ proximity", data=df).fit()
    print(f"\nİlk aşama F-istatistiği: {first_stage.fvalue:.1f}")
    print(f"  (F > 10 gerekli — zayıf araç sorunu için)")

    return iv_result

iv_result = iv_analysis_example()
```

**Alternatif: `statsmodels` ile IV-2SLS**

`linearmodels` yerine `statsmodels` ile de IV tahmini yapılabilir. Daha az bağımlılık gerektirdiği için hafif projelerde tercih edilebilir:

```python
import numpy as np
import pandas as pd
from statsmodels.sandbox.regression.gmm import IV2SLS as SM_IV2SLS

def iv_with_statsmodels():
    """
    statsmodels IV2SLS alternatifi.
    linearmodels kurulu değilse veya minimal bağımlılık isteniyorsa kullan.
    """
    np.random.seed(42)
    n = 5000

    # Aynı veri üretim süreci
    ability = np.random.normal(0, 1, n)
    proximity = np.random.uniform(0, 100, n)
    education = 12 + 0.8 * ability - 0.03 * proximity + np.random.normal(0, 1, n)
    income = 20 + 2.0 * education + 3.0 * ability + np.random.normal(0, 5, n)

    # statsmodels IV2SLS
    # endog = Y, exog = [const, X_endogenous], instrument = [const, Z]
    from statsmodels.api import add_constant
    endog = income
    exog = add_constant(education)           # [1, education]
    instrument = add_constant(proximity)     # [1, proximity]

    iv_mod = SM_IV2SLS(endog, exog, instrument).fit()
    print("statsmodels IV-2SLS sonuçları:")
    print(f"  Eğitim katsayısı: {iv_mod.params[1]:.3f}")
    print(f"  Standart hata:    {iv_mod.bse[1]:.3f}")
    print(iv_mod.summary())

    return iv_mod

# iv_with_statsmodels()  # Çalıştırmak için yorum kaldır
```

> **Not:** `statsmodels.sandbox.regression.gmm.IV2SLS` sandbox modülünde yer alır ve API'si `linearmodels` kadar ergonomik değildir. Production kalitesinde IV analizi için `linearmodels` tercih edilir; ancak mülakat veya hızlı prototip için `statsmodels` yeterlidir.

### Front-Door Criterion (Ön Kapı Kriteri)

**Kavramsal açıklama:** Backdoor adjustment (propensity score, DiD vb.) gözlenmeyen confounder varsa çalışmaz. Front-door criterion farklı bir strateji sunar: treatment'tan outcome'a giden **ara mekanizmayı** (mediator) kullanarak nedensel etkiyi tanımla.

```
Yapı:
  X → M → Y     (X treatment, M mekanizma, Y outcome)
  U → X, U → Y  (U gözlenmeyen confounder)

Örnek:
  Sigara (X) → Akciğer katranı (M) → Kanser (Y)
  Genetik yatkınlık (U) → hem sigara içme eğilimi hem kanser riski

Front-door şartları:
  1. X → M yolunu M tamamen aracılık eder (X, Y'ye M olmadan etkilemez)
  2. X → M yolu confound edilmemiş
  3. M → Y confound edilmiş olabilir ama X koşullanarak düzeltilebilir

Front-door formülü:
  P(Y | do(X)) = Σ_m P(M=m | X) · Σ_x' P(Y | M=m, X=x') · P(X=x')
```

Bu kriter pratikte nadiren uygulanır çünkü uygun bir mediator bulmak zordur. Ama kavramsal olarak önemlidir: "Gözlenmeyen confounder olsa bile, doğru mekanizmayı tanımlayabilirsen nedensellik kurabilirsin."

> **Senior Notu:** Front-door criterion teorik bir araç. Mülakatta sorulduğunda kavramsal olarak açıklayabilmek yeterli. Uygulamada DiD, RDD ve IV çok daha yaygın.

### DoWhy ile Causal Inference

```python
import dowhy
from dowhy import CausalModel
import pandas as pd

# Nedensel model oluştur
model = CausalModel(
    data=df,
    treatment="received_discount",    # Müdahale
    outcome="purchased",              # Sonuç
    common_causes=["age", "past_purchases", "device_type"],  # Confounders
    instruments=["random_email_variant"],  # Araç değişken (eğer varsa)
)

# Grafik görselleştir
model.view_model()

# Nedensel etkiyi tanımla (ID)
identified = model.identify_effect(proceed_when_unidentifiable=True)

# Tahmin et
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.propensity_score_matching"
)
print(f"Causal effect: {estimate.value:.4f}")

# Refutation (varsayım sağlamlık testi)
refutation = model.refute_estimate(
    identified, estimate,
    method_name="random_common_cause"
)
print(refutation)
```

> **Senior Notu:** DoWhy'da `refute_estimate` kritik. "Eğer gizli bir confounder eklesem sonuç değişir mi?" sorusunu yanıtlar. Değişiyorsa nedensel iddia zayıf. Her causal analysis raporuna refutation ekle.

### C.7.6 Causal Forest — Heterogeneous Treatment Effects (HTE)

**Sezgisel:** "Tedavi etkisi herkes için aynı mı? Hayır — 25 yaş kadın vs 55 yaş erkek için farklı."

Causal Forest (Athey & Wager, 2019) her alt grup için ayrı treatment effect tahmin eder. Tek bir ortalama ATE yerine, her gözlem için ayrı bir CATE (Conditional Average Treatment Effect) üretir. Bu sayede "Bu kampanya hangi kullanıcı segmenti için karlı?" gibi soruları yanıtlamak mümkün olur.

```python
# pip install econml
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# Sentetik veri: promosyon etkisi yaşa göre değişiyor
X = np.random.randn(n, 3)  # özellikler: yaş, geçmiş alım, segment
T = (X[:, 0] + np.random.randn(n) > 0).astype(float)  # treatment (promosyon)
# Genç kullanıcılar (X[:,0] > 0) promosyona daha duyarlı
tau_true = 2 + 3 * (X[:, 0] > 0)  # HTE: yaşa göre farklı etki
Y = tau_true * T + X[:, 0] * 0.5 + np.random.randn(n)  # outcome

# Causal Forest eğitimi
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=LassoCV(cv=5),
    n_estimators=200,
    min_samples_leaf=20,
    random_state=42,
    cv=5
)
cf.fit(Y, T, X=X)

# Treatment effect tahminleri
tau_hat = cf.effect(X)
print(f"Ortalama tahmini etki: {tau_hat.mean():.3f} (gerçek: {tau_true.mean():.3f})")
print(f"Yüksek etki grubu (X0>0): {tau_hat[X[:,0]>0].mean():.3f}")
print(f"Düşük etki grubu (X0≤0): {tau_hat[X[:,0]<=0].mean():.3f}")

# Güven aralıkları
lb, ub = cf.effect_interval(X, alpha=0.1)  # %90 CI
print(f"\nOrtalama CI genişliği: {(ub - lb).mean():.3f}")
```

> **Senior Notu:** Causal Forest'ı A/B test sonrası analiz için kullan: "Bu kullanıcı segmenti için kampanya karlı mıydı?" HTE yüksek varyansa sahip olabilir — büyük örneklem (n>5000) gerekir. EconML'in `const_marginal_effect` ve policy learning API'si üretim kalitesinde değerlendirme sağlar.

---

## C.8 Multi-Armed Bandit (MAB)

### Sezgisel Açıklama

A/B test: önceden belirlenen süre, sonunda kazananı seç. Regret yüksek.

Bandit: Her gün performansa göre trafik yeniden dağıt. Regret düşük ama keşif-exploit dengesi kritik.

```python
import numpy as np
import matplotlib.pyplot as plt

class ThompsonSampling:
    """
    Thompson Sampling — Bayesian bandit.
    Beta-Binomial ile: her arm için posterior örnekle, en yüksekle oyna.
    """
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # başarılar + 1
        self.beta = np.ones(n_arms)   # başarısızlıklar + 1

    def select(self) -> int:
        """Hangi arm'ı seç?"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm: int, reward: int):
        """Reward gözlemle, posterior'u güncelle."""
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

# Simülasyon
np.random.seed(42)
true_rates = [0.05, 0.07, 0.10, 0.06]  # 4 arm, gerçek CTR
n_rounds = 10000

ts = ThompsonSampling(n_arms=4)
ucb_rewards = []
ts_rewards = []

for round_n in range(n_rounds):
    arm = ts.select()
    reward = np.random.binomial(1, true_rates[arm])
    ts.update(arm, reward)
    ts_rewards.append(reward)

print(f"Thompson Sampling kümülatif ödül: {sum(ts_rewards)}")
print(f"Optimal kol sürekli seçilseydi: {n_rounds * max(true_rates):.0f}")
print(f"Regret: {n_rounds * max(true_rates) - sum(ts_rewards):.0f}")

# Kol seçim dağılımı
arm_counts = [ts.alpha[i] + ts.beta[i] - 2 for i in range(4)]
print(f"\nKol seçim sayıları: {arm_counts}")
```

```python
# UCB (Upper Confidence Bound) — güven aralığı tabanlı exploration
class UCB1:
    """
    UCB1 — Optimism in the Face of Uncertainty.
    Az denenen arm'a güven bonusu ver.
    """
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)    # Her arm kaç kez seçildi?
        self.values = np.zeros(n_arms)    # Ortalama ödül

    def select(self, t: int) -> int:
        # İlk turda her arm'ı bir kez dene
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        # UCB skoru: ortalama + keşif bonusu
        ucb_scores = self.values + np.sqrt(2 * np.log(t + 1) / self.counts)
        return np.argmax(ucb_scores)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

# Karşılaştırma: Thompson vs UCB
np.random.seed(42)
ucb = UCB1(n_arms=4)
ucb_rewards = []

for t, _ in enumerate(range(n_rounds)):
    arm = ucb.select(t)
    reward = np.random.binomial(1, true_rates[arm])
    ucb.update(arm, reward)
    ucb_rewards.append(reward)

print(f"UCB1 kümülatif ödül:          {sum(ucb_rewards)}")
print(f"Thompson Sampling kümülatif:  {sum(ts_rewards)}")
print(f"Optimal (oracle):             {n_rounds * max(true_rates):.0f}")
```

### Bandit Algoritmaları Karşılaştırması

| Algoritma | Exploration | Production Kolaylığı | Ne Zaman? |
|-----------|------------|---------------------|-----------|
| Epsilon-greedy | Rastgele ε oranda | ✓ Basit | Başlangıç prototipi |
| UCB1 | Güven aralığı | ✓ Deterministik | Stabil ortam |
| Thompson Sampling | Bayesian posterior | ✓ Genellikle en iyi | Genel amaç |
| LinUCB | Bağlam + güven | ✗ Daha karmaşık | Kişiselleştirme |

### LinUCB — Contextual Bandit (Bağlamsal Bandit)

**Sezgisel:** Standart banditler her kullanıcıya aynı kararı verir. LinUCB ise kullanıcı özelliklerini (bağlamı) dikkate alır: genç bir kullanıcıya farklı içerik, yaşlı bir kullanıcıya farklı içerik önerir. Haber önerisi, e-ticaret banner seçimi, kişisel kampanya gibi durumlarda uygundur.

```python
# LinUCB — Contextual Bandit (Bağlamsal Bandit)
# Kullanıcı özelliklerini (bağlamı) dikkate alır
class LinUCB:
    """
    LinUCB — haber/içerik önerisi gibi bağlamsal kararlar için.
    Her arm için ayrı lineer model tutar.
    """
    def __init__(self, n_arms: int, d: int, alpha: float = 1.0):
        self.alpha = alpha          # Exploration katsayısı
        self.A = [np.eye(d) for _ in range(n_arms)]       # d×d matris
        self.b = [np.zeros(d) for _ in range(n_arms)]     # d-boyutlu vektör

    def select(self, context: np.ndarray) -> int:
        """Context = kullanıcı özellikleri (ör. yaş, geçmiş tıklamalar)"""
        scores = []
        for arm in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            # UCB skoru: beklenti + keşif bonusu
            ucb = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            scores.append(ucb)
        return np.argmax(scores)

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

# Kullanım örneği: haber önerisi
np.random.seed(42)
d = 5  # kullanıcı feature boyutu (yaş, platform, geçmiş kategoriler...)
linucb = LinUCB(n_arms=4, d=d, alpha=0.5)

for _ in range(1000):
    user_context = np.random.randn(d)  # gerçekte: kullanıcı feature vektörü
    arm = linucb.select(user_context)
    reward = np.random.binomial(1, true_rates[arm])
    linucb.update(arm, user_context, reward)

print("LinUCB eğitimi tamamlandı — bağlamsal öneri hazır")
```

> **Senior Notu:** Bandit vs A/B test seçimi: Bandit regret'i minimize eder ama kontrol edilmesi zordur. A/B test temiz bir istatistiksel anlama sahipken bandit "winner" belirsizdir. Production'da: (1) Bandit'i exploration-only bölgelerde kullan (yeni içerik, cold-start), (2) İş metrikleri üzerinde A/B test yap; (3) Her zaman offline replay evaluation ile simüle et.

---

## C.9 Sektör Notu — A/B Testing 2026

2026 itibarıyla A/B testing alanındaki gelişmeler:

- **CUPED standartlaştı:** Statsig, Eppo, Optimizely gibi platformlar CUPED'i default uyguluyor. Araştırmalar CUPED + trimmed mean kombinasyonunun çarpık metriklerde ek %15–20 varyans azalması sağladığını gösteriyor.

- **Sequential testing yaygınlaştı:** Sürekli izleme artık hata kabul edilmiyor. Anytime-valid confidence intervals (mSPRT tabanlı) endüstri standardı haline geldi. O'Brien-Fleming sınırları ile group sequential testler ara analiz planlaması için yaygın kullanılıyor.

- **AI-powered experimentation:** ML modelleri deney önceliklendirme ve segment bazlı analizi otomatikleştiriyor.

- **Causal inference önem kazandı:** Doğal deneyler, DiD, synthetic control ve RDD metodları büyük platform şirketlerinde standart hale geldi (Lyft, DoorDash, Airbnb blog yazıları bu gelişimi belgeler).

- **A/A test otomasyonu:** Büyük platformlar sürekli A/A test koşarak platform sağlığını izliyor. SRM kontrolü, p-value uniformluk testi ve metrik pipeline doğrulaması otomatik çalışıyor.

- **Platform tools:** GrowthBook (açık kaynak), Statsig, Eppo → küçük-orta ölçekli şirketler için sıfırdan kurmak yerine bu platformları kullan.

> **Çapraz referans:** Katman F'te A/B test platformu tasarımı (feature flagging, trafik yönlendirme, metrik pipeline) detaylı ele alınır. `mulakat.md` dosyasında deney tasarımı mülakat soruları ve beklenen yanıt çerçeveleri bulunur.

---

## C.10 Alıştırma Soruları — Senaryo Bazlı

Bu sorular gerçek mülakat ve iş senaryolarını yansıtır. Her soruyu yanıtlarken hangi kavramları kullandığını açıkça belirt.

### Soru 1: Anlamlılık mı, İş Değeri mi?

> Bir e-ticaret sitesinde conversion rate %2.00'den %2.10'a çıktı. Product manager "Harika, yayınlayalım!" diyor. Bu sonuç gerçekten anlamlı mı?

**Düşünme çerçevesi:**
- Power analizi yap: Bu farkı (MDE = 0.10 pp = %5 relative) tespit etmek için kaç kullanıcı gerekli?
- σ = sqrt(p(1-p)) ≈ 0.14 (Bernoulli metrik), Cohen's h ≈ 0.007 → n ≈ 250,000+ kullanıcı/kol
- Örneklem yeterli mi? Güven aralığı 0'ı kapsıyor mu?
- İş değeri: %5 relative lift yıllık ne kadar gelir demek? MDE'yi iş bağlamıyla birlikte değerlendir.

### Soru 2: Segmentlerde Zıt Yönde Sonuç

> A/B test sonuçları: Genel etki = +%3 (p=0.02). Ama "mobil" segmentte etki = -%2 (p=0.04), "desktop" segmentte etki = +%8 (p=0.001). Mobil kullanıcılar toplamın %70'i. Ne yaparsın?

**Düşünme çerçevesi:**
- Simpson's paradox: Segment bazlı ve genel sonuçlar çelişebilir.
- HTE (Heterogeneous Treatment Effects) analizi yap — segment bazlı güven aralıkları kontrol et.
- Çoklu karşılaştırma düzeltmesi (Bonferroni/BH) uyguladın mı? 2 segment test = 2 karşılaştırma.
- Segment bazlı analiz önceden planlandı mı yoksa post-hoc mu? Post-hoc ise keşifsel olarak raporla, doğrulama deneyi planla.
- İş kararı: Mobil ve desktop için farklı deneyim sunulabilir mi?

### Soru 3: Randomizasyon Yapılamıyor

> Yönetim yeni fiyatlandırma modelini tüm Avrupa'da aynı anda uyguladı. Kontrol grubu yok. Fiyat değişikliğinin gelire etkisini nasıl ölçersin?

**Düşünme çerçevesi:**
- A/B test yapılamıyor → gözlemsel causal inference yöntemleri:
  - **DiD:** Avrupa (treatment) vs benzer pazarlar (ABD, Asya) müdahale öncesi ve sonrası. Parallel trends varsayımını kontrol et.
  - **Synthetic Control:** Avrupa'nın sentetik versiyonunu diğer bölgelerin ağırlıklı ortalamasından oluştur.
  - **Interrupted Time Series:** Sadece Avrupa verisiyle müdahale öncesi trendi modelleyip sonrasını tahmin et, farkı ölç.
- Her yöntemin varsayımlarını açıkça belirt (parallel trends, no spillover vb.).

### Soru 4: SRM — Sample Ratio Mismatch

> A/B testinde %50/%50 trafik böldüğün halde, test grubunda %30 daha fazla kullanıcı var (treatment: 6500, control: 5000). Sorun ne?

**Düşünme çerçevesi:**
- Chi-squared SRM testi: Bu oran sapması rastgele olamayacak kadar büyük (p ≈ 0).
- **Olası nedenler:**
  - Bot/crawler filtreleme asimetrisi (treatment sayfası farklı bot davranışı tetikliyor)
  - Redirect veya page load farkı (treatment yavaşsa kullanıcılar düşer ama loglanmaz)
  - Hash fonksiyonu sorunlu (kullanıcı ID'lerin hash dağılımı düzgün değil)
  - Trigger condition asimetrisi (treatment özelliği sadece belirli kullanıcılarda aktif)
- **Karar:** SRM varsa deneyi **durdur ve nedenini araştır**. SRM ile yapılan analiz güvenilir değildir. Sorunu çöz, A/A test ile doğrula, sonra yeniden başlat.

### Soru 5: Peeking Sorunu

> Deneyi 7 gün planladın. 3. günde p-value = 0.03 (anlamlı). 7. günde p-value = 0.08 (anlamlı değil). Ne oldu? Hangisine güvenirsin?

**Düşünme çerçevesi:**
- Klasik peeking sorunu: Her bakışta yanlış pozitif riski birikir. 3. gündeki "anlamlılık" muhtemelen tesadüf.
- 7. gün planlanmış süre ise, 7. gün sonucuna güven.
- **Neden değişti?** Erken gözlemler volatil, etki tahmini stabilize olmamış. Küçük n ile büyük efekt tahmini → p küçük, n artınca efekt tahmini gerçeğe yaklaşır ve küçülür.
- **Doğru yaklaşım:** Sequential testing kullan (SPRT veya O'Brien-Fleming). Eğer kullanmadıysan, sadece planlanan bitiş noktasındaki sonuca bak.
- **Karar:** 7. gün sonucu "anlamlı değil" → power yeterli miydi? Belki etki var ama tespit için yeterli n yok (underpowered). Power analizi tekrar yap.

### Soru 6: A/A Test Anomalisi

> Yeni deney platformunu devreye aldın. A/A test koştun: 1000 simülasyonda yanlış pozitif oranı %12 çıktı (beklenen %5). p-value dağılımı uniform değil, düşük p-value'larda yığılma var. Sorun nerede olabilir?

**Düşünme çerçevesi:**
- %12 yanlış pozitif, beklenen %5'in çok üzerinde — platform güvenilir değil.
- **Olası nedenler:**
  - Randomizasyon hash fonksiyonu düzgün dağılım üretmiyor (kullanıcı ID yapısı hash'i boğuyor).
  - Metrik pipeline'da logging asimetrisi: bir grup daha fazla event kaydediyor.
  - Deney birimi (randomization unit) ile analiz birimi (analysis unit) uyumsuz — örneğin session bazlı randomizasyon, user bazlı analiz.
  - Bot/crawler filtreleme iki gruba farklı uygulanıyor.
- **Çözüm:** Sorunu izole et (randomizasyon mı, metrik mi, logging mi?), düzelt, A/A testi tekrarla. Platform %5 FP oranını tutturana kadar A/B test sonuçlarına güvenme.

### Soru 7: CUPED Ne Zaman İşe Yaramaz?

> Yeni kullanıcılar için bir onboarding deneyimi test ediyorsun. CUPED uygulamak istiyorsun ama varyans azalması neredeyse sıfır. Neden?

**Düşünme çerçevesi:**
- CUPED, pre-period verisi ile post-period arasındaki korelasyona (ρ) dayanır. Varyans azalması = 1 - ρ².
- Yeni kullanıcıların pre-period verisi yok (ilk kez platforma geliyorlar) → ρ ≈ 0 → CUPED etkisiz.
- **Alternatifler:**
  - Kayıt anında toplanan covariate'ler kullan (device type, referral source, coğrafya) — stratified randomization veya post-stratification.
  - Pre-period metrik yerine proxy metrik dene (örneğin ilk 24 saat aktivite → sonraki hafta gelir tahmini).
  - Daha uzun deney süresi veya daha büyük örneklem ile power'ı telafi et.

### Soru 8: Instrumental Variable Geçerliliği

> Bir araştırmada "hava durumu" araç değişken olarak kullanılarak mağaza ziyaretinin satışa etkisi ölçülmek isteniyor: Kötü hava → daha az mağaza ziyareti → daha az satış. Bu geçerli bir IV mi?

**Düşünme çerçevesi:**
- **Relevance (Z → X):** Kötü hava mağaza ziyaretini azaltır — muhtemelen sağlanır.
- **Exclusion (Z → Y yalnızca X yoluyla):** Hava durumu satışları başka yollarla etkiler mi? Kötü havada online satışlar artar, mevsimsel tüketim kalıpları değişir, bazı ürün kategorileri hava durumuna doğrudan duyarlıdır (şemsiye, mont). → **Exclusion restriction muhtemelen ihlal ediliyor.**
- **Independence (Z ⊥ U):** Hava durumu gizli confounder'larla ilişkili olabilir (mevsim → tatil sezonu → hem hava hem satış).
- **Karar:** Bu IV zayıf. Exclusion restriction'ı savunmak zor. Daha iyi bir IV ara veya farklı causal inference yöntemi (DiD, RDD) kullan.

---

## Katman C Kontrol Listesi

- [ ] Deney tasarım şablonunu bir gerçek (veya simülasyon) senaryosuna uyguladım
- [ ] Power analizi ve MDE hesabı yaptım
- [ ] A/A test simülasyonu koşup yanlış pozitif oranını doğruladım
- [ ] A/A test + SRM kontrolünü birlikte uyguladım
- [ ] CUPED ile CI daralmasını simülasyonla gösterdim
- [ ] Peeking simülasyonu yaptım ve yanlış pozitif artışını gözlemledim
- [ ] SPRT veya O'Brien-Fleming ile sequential test uyguladım
- [ ] Always-valid CI kavramını ve anytime p-value'yu anlıyorum
- [ ] SRM kontrolü nasıl yapılır biliyorum
- [ ] Network effects nedir, nasıl tespit edilir açıklayabilirim
- [ ] DiD veya propensity score matching ile en az 1 analiz yaptım
- [ ] Synthetic Control veya RDD ile gözlemsel analiz yaptım
- [ ] IV (Instrumental Variables) kavramını ve 2SLS'i açıklayabilirim
- [ ] Front-door criterion'u kavramsal olarak açıklayabilirim
- [ ] DoWhy ile bir causal model kurdum
- [ ] Thompson Sampling bandit simülasyonu yazdım
- [ ] Senaryo bazlı alıştırma sorularını yanıtladım (8 soru)
- [ ] IV geçerlilik koşullarını (relevance, exclusion, independence) bir örnek üzerinde değerlendirdim
- [ ] statsmodels IV2SLS alternatifini denedim
- [ ] Proje-2 (A/B Test Paketi) tamamlandı

---

<div class="nav-footer">
  <span><a href="#file_katman_B_klasik_ml">← Önceki: Katman B — Klasik ML</a></span>
  <span><a href="#toc">↑ İçindekiler</a></span>
  <span><a href="#file_katman_D_derin_ogrenme">Sonraki: Katman D — Derin Öğrenme →</a></span>
</div>
