#!/usr/bin/env python3
"""
Veri Bilimci Yol Haritası — MD → PDF dönüştürücü
Kullanım: python3 build_pdf.py
"""

import os
import re
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Kitap sırası (meta dosyalar hariç)
FILES = [
    "README.md",
    "00-uygulama-sirasi.md",
    "01-yetkinlik-matrisi.md",
    "katman-0-matematik.md",
    "katman-A-temeller.md",
    "katman-B-klasik-ml.md",
    "katman-C-deney-nedensellik.md",
    "katman-D-derin-ogrenme.md",
    "katman-E-mlops.md",
    "katman-F-sistem-tasarimi.md",
    "katman-G-senior-davranislar.md",
    "katman-H-buyuk-veri.md",
    "projeler.md",
    "mulakat.md",
    "kaynaklar.md",
]

# Bölüm meta verileri
CHAPTER_META = {
    "README.md": {
        "num": None,
        "title": "Giriş ve Genel Bakış",
        "short": "Giriş",
        "desc": "Rehberin amacı, yapısı ve okuyucu profillerine göre önerilen öğrenme yolları.",
        "time": "30 dakika",
        "difficulty": "Giriş",
        "prereq": None,
        "is_separator": True,
        "book_label": None, "book_num": None, "category": "front",
        "topics": ["Rehberin amacı ve kapsamı", "Okuyucu profilleri ve öğrenme yolları", "Dosya yapısı ve navigasyon"],
    },
    "00-uygulama-sirasi.md": {
        "num": None,
        "title": "Uygulama Sırası",
        "short": "Uygulama Planı",
        "desc": "8 aşamalı sıralı öğrenme planı; her aşamanın teslim edilebilir çıktısı.",
        "time": "1 saat okuma",
        "difficulty": "Giriş",
        "prereq": None,
        "is_separator": True,
        "book_label": None, "book_num": None, "category": "front",
        "topics": ["8 aşamalı öğrenme planı", "Her aşamanın teslim edilebilir çıktıları", "Haftalık çalışma ritmi ve zaman planı"],
    },
    "01-yetkinlik-matrisi.md": {
        "num": None,
        "title": "Yetkinlik Matrisi",
        "short": "Yetkinlik Matrisi",
        "desc": "Senior veri bilimci tanımı; teknik ve davranışsal yetkinlikler haritası.",
        "time": "1 saat okuma",
        "difficulty": "Giriş",
        "prereq": None,
        "is_separator": True,
        "book_label": None, "book_num": None, "category": "front",
        "topics": ["Junior / Mid / Senior karşılaştırması", "Teknik ve davranışsal yetkinlikler", "Kendini değerlendirme rehberi"],
    },
    "katman-0-matematik.md": {
        "num": "0",
        "title": "Matematik Temelleri",
        "short": "Bölüm 1 — Matematik",
        "desc": "Lineer cebir, kalkülüs ve olasılık teorisinin makine öğrenmesi perspektifinden sezgisel anlatımı.",
        "time": "3–5 gün",
        "difficulty": "Temel",
        "prereq": "Lise matematik yeterli",
        "is_separator": True,
        "book_label": "Bölüm 1", "book_num": 1, "category": "chapter",
        "topics": ["Lineer cebir: vektörler, matrisler, SVD", "Kalkülüs: gradyan, zincir kuralı, optimizasyon", "Olasılık: Bayes teoremi, dağılımlar"],
    },
    "katman-A-temeller.md": {
        "num": "A",
        "title": "Temeller",
        "short": "Bölüm 2 — Temeller",
        "desc": "Python/Pandas, analitik SQL, istatistik temeli ve görselleştirme; günlük DS araç takımı.",
        "time": "3–4 hafta",
        "difficulty": "Temel",
        "prereq": "Bölüm 1 (Matematik)",
        "is_separator": True,
        "book_label": "Bölüm 2", "book_num": 2, "category": "chapter",
        "topics": ["Python ve Pandas ile veri manipülasyonu", "Analitik SQL: pencere fonksiyonları, CTE", "Temel istatistik ve hipotez testleri", "Veri görselleştirme ve EDA"],
    },
    "katman-B-klasik-ml.md": {
        "num": "B",
        "title": "Klasik Makine Öğrenmesi",
        "short": "Bölüm 3 — Klasik ML",
        "desc": "Doğrusal modeller, karar ağaçları, boosting ve model değerlendirme; teoriden üretime.",
        "time": "3–4 hafta",
        "difficulty": "Orta",
        "prereq": "Bölüm 2 (Temeller)",
        "is_separator": True,
        "book_label": "Bölüm 3", "book_num": 3, "category": "chapter",
        "topics": ["Problem framing ve metrik seçimi", "Karar ağaçları ve ensemble yöntemleri", "XGBoost/LightGBM ile modelleme", "SHAP ile açıklanabilirlik", "Model kalibrasyon ve değerlendirme"],
    },
    "katman-C-deney-nedensellik.md": {
        "num": "C",
        "title": "Deney Tasarımı ve Nedensellik",
        "short": "Bölüm 4 — Deney/Nedensellik",
        "desc": "A/B test tasarımı, güç analizi, CUPED ve nedensel çıkarım; korelasyondan nedenselliğe.",
        "time": "2–3 hafta",
        "difficulty": "Orta-İleri",
        "prereq": "Bölüm 2 (İstatistik), Bölüm 3",
        "is_separator": True,
        "book_label": "Bölüm 4", "book_num": 4, "category": "chapter",
        "topics": ["A/B test tasarımı ve güç analizi", "CUPED ile varyans azaltma", "Nedensel çıkarım: DiD, IV, RDD", "Çoklu test düzeltmeleri"],
    },
    "katman-D-derin-ogrenme.md": {
        "num": "D",
        "title": "Derin Öğrenme",
        "short": "Bölüm 5 — Derin Öğrenme",
        "desc": "Sinir ağları, NLP, bilgisayarlı görü, öneri sistemleri ve LLM/RAG mimarileri.",
        "time": "4–6 hafta",
        "difficulty": "İleri",
        "prereq": "Bölüm 1 (Matematik), Bölüm 3",
        "is_separator": True,
        "book_label": "Bölüm 5", "book_num": 5, "category": "chapter",
        "topics": ["Sinir ağları ve backpropagation", "NLP: Transformers, BERT, fine-tuning", "Bilgisayarlı görü ve transfer learning", "LLM ve RAG mimarileri", "Öneri sistemleri"],
    },
    "katman-E-mlops.md": {
        "num": "E",
        "title": "MLOps",
        "short": "Bölüm 6 — MLOps",
        "desc": "Model paketleme, servis, izleme, drift tespiti ve CI/CD; açık kaynak araç odaklı.",
        "time": "2–3 hafta",
        "difficulty": "İleri",
        "prereq": "Bölüm 3 veya 5 (çalışan bir model şart)",
        "is_separator": True,
        "book_label": "Bölüm 6", "book_num": 6, "category": "chapter",
        "topics": ["Model paketleme ve Docker", "FastAPI ile model servisi", "MLflow ile deney takibi", "Data drift izleme (Evidently)", "CI/CD ve LLMOps"],
    },
    "katman-F-sistem-tasarimi.md": {
        "num": "F",
        "title": "ML Sistem Tasarımı",
        "short": "Bölüm 7 — Sistem Tasarımı",
        "desc": "Online/offline servis mimarileri, feature store, ölçekleme ve maliyet optimizasyonu.",
        "time": "2–3 hafta",
        "difficulty": "İleri",
        "prereq": "Bölüm 6 (MLOps)",
        "is_separator": True,
        "book_label": "Bölüm 7", "book_num": 7, "category": "chapter",
        "topics": ["Online vs offline serving mimarileri", "Feature store tasarımı", "Ölçekleme ve maliyet optimizasyonu", "ML sistem tasarım senaryoları"],
    },
    "katman-G-senior-davranislar.md": {
        "num": "G",
        "title": "Senior Davranışlar",
        "short": "Bölüm 8 — Senior",
        "desc": "Etki odaklı çalışma, teknik liderlik, dokümantasyon ve kariyer stratejisi.",
        "time": "1–2 hafta",
        "difficulty": "Senior",
        "prereq": "Tüm teknik bölümler (1–7)",
        "is_separator": True,
        "book_label": "Bölüm 8", "book_num": 8, "category": "chapter",
        "topics": ["Etki odaklı çalışma ve ICE framework", "Stakeholder iletişimi", "OKR ve STAR çerçeveleri", "Teknik borç yönetimi", "Dokümantasyon ve mentorluk"],
    },
    "katman-H-buyuk-veri.md": {
        "num": "H",
        "title": "Büyük Veri",
        "short": "Bölüm 9 — Büyük Veri",
        "desc": "Spark, Dask ve dağıtık hesaplama; veriyi tek makineye sığmadığında ne yapılır?",
        "time": "2–3 hafta",
        "difficulty": "İleri",
        "prereq": "Bölüm 2 (Pandas, SQL), Bölüm 6 (MLOps)",
        "is_separator": True,
        "book_label": "Bölüm 9", "book_num": 9, "category": "chapter",
        "topics": ["Ne zaman büyük veri gerekir?", "Apache Spark ile dağıtık hesaplama", "Polars ve DuckDB", "Delta Lake ve veri gölleri", "Streaming ve gerçek zamanlı işleme"],
    },
    "projeler.md": {
        "num": None,
        "title": "Portföy Projeleri",
        "short": "Ek A — Projeler",
        "desc": "Proje-0'dan Proje-7'ye sekiz portfolyo projesi; iş vitrini oluşturma rehberi.",
        "time": "Her proje: 1–2 hafta",
        "difficulty": "Uygulama",
        "prereq": None,
        "is_separator": True,
        "book_label": "Ek A", "book_num": "A", "category": "appendix",
        "topics": ["8 portfolyo projesi (Proje-0 → Proje-7)", "Teslimat standartları ve kod yapısı", "Portföy sunumu stratejileri"],
    },
    "mulakat.md": {
        "num": None,
        "title": "Mülakat Hazırlığı",
        "short": "Ek B — Mülakat",
        "desc": "SQL, ML, case study ve davranışsal mülakat soruları; çözümlü örnekler.",
        "time": "1–2 hafta",
        "difficulty": "Uygulama",
        "prereq": None,
        "is_separator": True,
        "book_label": "Ek B", "book_num": "B", "category": "appendix",
        "topics": ["SQL mülakat soruları (30+ soru)", "ML case study örnekleri", "Sistem tasarım senaryoları", "Davranışsal mülakat hazırlığı"],
    },
    "kaynaklar.md": {
        "num": None,
        "title": "Kaynaklar",
        "short": "Ek C — Kaynaklar",
        "desc": "Ücretsiz ve temel kaynak listesi; kitaplar, kurslar, araçlar ve topluluklar.",
        "time": "Referans",
        "difficulty": "Referans",
        "prereq": None,
        "is_separator": True,
        "book_label": "Ek C", "book_num": "C", "category": "appendix",
        "topics": ["Ücretsiz kurslar ve kitaplar", "Araçlar ve frameworkler", "Topluluklar ve newsletterlar"],
    },
}

MD_EXTENSIONS = [
    "fenced_code",
    "codehilite",
    "tables",
    "toc",
    "attr_list",
    "def_list",
    "admonition",
    "sane_lists",
]

MD_EXTENSION_CONFIGS = {
    "codehilite": {
        "linenums": False,
        "guess_lang": False,
        "css_class": "highlight",
    },
    "toc": {
        "permalink": False,
    },
}

CSS_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ════════════════════════════════════════
   SAYFA DÜZENI
   ════════════════════════════════════════ */

@page {
    size: A4;
    margin: 2.8cm 2.2cm 2.8cm 2.8cm;

    @top-left {
        content: "Veri Bilimci Yol Haritası";
        font-family: 'Inter', sans-serif;
        font-size: 7.5pt;
        color: #b0bcd0;
        letter-spacing: 0.04em;
    }
    @top-right {
        content: string(chapter-title);
        font-family: 'Inter', sans-serif;
        font-size: 7.5pt;
        color: #8898aa;
        font-style: italic;
    }
    @top-center {
        content: "";
        border-bottom: 0.5pt solid #e8edf3;
        width: 100%;
        display: block;
    }
    @bottom-right {
        content: counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 8.5pt;
        color: #64748b;
        font-weight: 500;
    }
    @bottom-left {
        content: string(chapter-title);
        font-family: 'Inter', sans-serif;
        font-size: 7.5pt;
        color: #b0bcd0;
    }
    @bottom-center { content: none; }
}

/* Kapak sayfası — tam sayfa, header/footer yok */
@page cover-page {
    size: A4;
    margin: 0;
    @top-left    { content: none; }
    @top-right   { content: none; }
    @top-center  { content: none; }
    @bottom-left { content: none; }
    @bottom-right { content: none; }
    @bottom-center { content: none; }
}

/* Bölüm kapak sayfası — tam sayfa, header/footer yok */
@page chapter-sep-page {
    size: A4;
    margin: 0;
    @top-left    { content: none; }
    @top-right   { content: none; }
    @top-center  { content: none; }
    @bottom-left { content: none; }
    @bottom-right { content: none; }
    @bottom-center { content: none; }
}

/* Ön madde — Roman rakamı */
@page front-matter {
    size: A4;
    margin: 2.8cm 2.2cm 2.8cm 2.8cm;
    @bottom-right {
        content: counter(page, lower-roman);
        font-family: 'Inter', sans-serif;
        font-size: 8.5pt;
        color: #94a3b8;
    }
    @bottom-left   { content: none; }
    @top-left      { content: none; }
    @top-right     { content: none; }
    @top-center    { content: none; }
    @bottom-center { content: none; }
}

.front-matter { page: front-matter; }

/* ════════════════════════════════════════
   TEMEL STİLLER
   ════════════════════════════════════════ */

* { box-sizing: border-box; }

body {
    font-family: 'Inter', 'DejaVu Sans', sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1e2432;
    background: #fff;
    -webkit-font-smoothing: antialiased;
}

/* ════════════════════════════════════════
   KAPAK SAYFASI
   ════════════════════════════════════════ */

.cover {
    page: cover-page;
    width: 21cm;
    min-height: 29.7cm;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    background: linear-gradient(150deg, #0a1628 0%, #0f3460 40%, #1a1a2e 100%);
    color: white;
    padding: 3cm 3cm;
    position: relative;
}

.cover-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 6px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
}

.cover h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.3em;
    border: none;
    line-height: 1.15;
    letter-spacing: -0.02em;
    string-set: none;
}

.cover .subtitle {
    font-size: 13pt;
    color: #7dd3fc;
    margin-bottom: 2.5em;
    font-weight: 300;
    letter-spacing: 0.02em;
}

.cover .cover-topics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
    justify-content: center;
    margin-bottom: 3em;
}

.cover .topic-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 0.3em 0.9em;
    border-radius: 20px;
    font-size: 9pt;
    color: #94a3b8;
    letter-spacing: 0.03em;
}

.cover .cover-divider {
    width: 3cm;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    margin: 0 auto 2.5em;
}

.cover .meta {
    font-size: 9pt;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ════════════════════════════════════════
   İÇİNDEKİLER
   ════════════════════════════════════════ */

.toc-page {
    page: front-matter;
    page-break-before: always;
    page-break-after: always;
    padding-top: 0.3cm;
}

.toc-page h2 {
    font-size: 22pt;
    font-weight: 700;
    color: #0f3460;
    border: none;
    padding-bottom: 0.4em;
    margin-bottom: 0.3em;
    string-set: none;
    letter-spacing: -0.02em;
}

.toc-rule {
    height: 3px;
    background: linear-gradient(90deg, #0f3460, #3b82f6, transparent);
    margin-bottom: 1.8em;
    border: none;
}

.toc-section-label {
    font-size: 7.5pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #94a3b8;
    margin: 1.4em 0 0.5em 0;
    padding-bottom: 0.3em;
    border-bottom: 1px solid #f1f5f9;
}

.toc-entry {
    display: flex;
    align-items: baseline;
    margin-bottom: 0.5em;
    font-size: 10.5pt;
}

.toc-entry .t-num {
    font-weight: 600;
    color: #1e40af;
    min-width: 5.5em;
    flex-shrink: 0;
    font-size: 9.5pt;
    letter-spacing: 0.02em;
}

.toc-entry .t-title { flex-grow: 1; }

.toc-entry .t-title a {
    color: #1e2432;
    text-decoration: none;
}

.toc-entry .t-dots {
    flex: 1;
    border-bottom: 1px dotted #d1d9e6;
    margin: 0 0.7em;
    min-width: 0.5em;
    align-self: flex-end;
    margin-bottom: 4px;
}

a.t-pageref {
    color: #64748b;
    font-weight: 500;
    font-size: 9.5pt;
    text-decoration: none;
    min-width: 2em;
    text-align: right;
    flex-shrink: 0;
}

a.t-pageref::after {
    content: target-counter(attr(href), page);
}

/* TOC alt girişleri */
.toc-subentry {
    display: flex;
    align-items: baseline;
    margin-bottom: 0.2em;
    font-size: 8.5pt;
    padding-left: 5.5em;
    color: #64748b;
}

.toc-subentry .t-title { flex-grow: 1; }

.toc-subentry .t-title a {
    color: #64748b;
    text-decoration: none;
}

.toc-subentry .t-dots {
    flex: 1;
    border-bottom: 1px dotted #e2e8f0;
    margin: 0 0.5em;
    min-width: 0.3em;
    align-self: flex-end;
    margin-bottom: 3px;
}

.toc-subentry a.t-pageref {
    color: #94a3b8;
    font-size: 8pt;
}

/* ════════════════════════════════════════
   BÖLÜM KAPAK SAYFASI
   ════════════════════════════════════════ */

.chapter-separator {
    page: chapter-sep-page;
    page-break-before: always;
    page-break-after: always;
    width: 21cm;
    min-height: 29.7cm;
    margin: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 3cm 3.5cm 4cm 3.5cm;
    background: linear-gradient(155deg, #080f1e 0%, #0d2248 35%, #0f3460 70%, #1a1a3e 100%);
    color: white;
    position: relative;
    overflow: hidden;
}

.chapter-separator::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
}

.chapter-separator::after {
    content: "";
    position: absolute;
    bottom: -2cm;
    right: -2cm;
    width: 12cm;
    height: 12cm;
    border: 2px solid rgba(255,255,255,0.03);
    border-radius: 50%;
}

.chapter-separator .ch-num {
    font-size: 120pt;
    font-weight: 800;
    color: rgba(255,255,255,0.05);
    line-height: 1;
    position: absolute;
    top: 1.5cm;
    right: 2cm;
    letter-spacing: -0.05em;
}

.chapter-separator .ch-label {
    font-size: 9pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #60a5fa;
    margin-bottom: 0.8em;
}

.chapter-separator .ch-title {
    font-size: 28pt;
    font-weight: 700;
    color: #f1f5f9;
    border: none;
    line-height: 1.15;
    margin: 0 0 0.7em 0;
    letter-spacing: -0.02em;
    string-set: chapter-title content();
    max-width: 14cm;
}

.chapter-separator .ch-desc {
    font-size: 11pt;
    color: #94a3b8;
    max-width: 13cm;
    line-height: 1.7;
    margin-bottom: 2em;
    font-weight: 300;
}

.chapter-separator .ch-divider {
    width: 2.5cm;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, transparent);
    margin-bottom: 1.5em;
}

.chapter-separator .ch-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
}

.chapter-separator .ch-meta span {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 0.3em 1em;
    border-radius: 20px;
    font-size: 8.5pt;
    color: #7dd3fc;
    letter-spacing: 0.02em;
}

/* Ek bölüm kapak varyantı — mor tonları */
.chapter-separator.appendix-sep {
    background: linear-gradient(155deg, #1a1025 0%, #2d1b4e 35%, #3b2070 70%, #1a1a3e 100%);
}
.chapter-separator.appendix-sep::before {
    background: linear-gradient(90deg, #8b5cf6, #a78bfa, #c4b5fd);
}
.chapter-separator.appendix-sep .ch-label { color: #a78bfa; }
.chapter-separator.appendix-sep .ch-meta span { color: #c4b5fd; }

/* Ön madde kapak varyantı */
.chapter-separator.front-sep {
    background: linear-gradient(155deg, #0a1628 0%, #162a50 35%, #1a3a5c 70%, #1a2a4e 100%);
}

/* "Bu Bölümde" kutusu */
.ch-preview {
    margin-top: 2em;
    padding: 1em 1.5em;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    max-width: 13cm;
}

.ch-preview-label {
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #60a5fa;
    margin-bottom: 0.5em;
    font-weight: 600;
}

.ch-preview ul {
    color: #94a3b8;
    font-size: 9.5pt;
    margin: 0;
    padding-left: 0;
    list-style: none;
}

.ch-preview li {
    margin-bottom: 0.3em;
    padding-left: 1em;
}

.ch-preview li::before {
    content: "›";
    color: #60a5fa;
    margin-right: 0.5em;
    font-weight: bold;
}

/* ════════════════════════════════════════
   BÖLÜM İÇERİĞİ
   ════════════════════════════════════════ */

.chapter-start {
    page-break-before: always;
}

/* ════════════════════════════════════════
   BAŞLIKLAR
   ════════════════════════════════════════ */

h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #0f3460;
    margin-top: 0;
    margin-bottom: 1em;
    page-break-after: avoid;
    letter-spacing: -0.02em;
    line-height: 1.2;
    string-set: chapter-title content();
    padding-bottom: 0.4em;
    border-bottom: 3px solid #0f3460;
}

h2 {
    font-size: 14.5pt;
    font-weight: 600;
    color: #1e40af;
    margin-top: 1.6em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
    page-break-before: avoid;
    letter-spacing: -0.01em;
    padding-bottom: 0.25em;
    border-bottom: 1.5px solid #dbeafe;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #1d4ed8;
    margin-top: 1.3em;
    margin-bottom: 0.45em;
    page-break-after: avoid;
    page-break-before: avoid;
}

h4 {
    font-size: 9.5pt;
    font-weight: 600;
    color: #2563eb;
    margin-top: 1.1em;
    margin-bottom: 0.35em;
    page-break-after: avoid;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ════════════════════════════════════════
   PARAGRAFLAR
   ════════════════════════════════════════ */

p {
    margin: 0 0 0.75em 0;
    text-align: justify;
    hyphens: auto;
}

/* ════════════════════════════════════════
   KOD BLOKLARI
   ════════════════════════════════════════ */

pre {
    background: #111827;
    color: #e2e8f0;
    border-radius: 8px;
    padding: 1.1em 1.3em;
    font-family: 'JetBrains Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 8.2pt;
    line-height: 1.6;
    margin: 1em 0 1.2em;
    border-top: 3px solid #3b82f6;
    border-bottom: 1px solid #1e293b;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
}

code {
    font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace;
    font-size: 8.3pt;
    background: #eff6ff;
    color: #1e40af;
    padding: 0.15em 0.45em;
    border-radius: 4px;
    border: 1px solid #bfdbfe;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
    border: none;
}

td code {
    font-size: 7pt;
    padding: 0.08em 0.25em;
}

/* Syntax highlighting */
.highlight { background: #111827 !important; border-radius: 8px; }
.highlight .k,
.highlight .kn,
.highlight .kr { color: #f472b6; }
.highlight .s,
.highlight .s1,
.highlight .s2 { color: #86efac; }
.highlight .c,
.highlight .c1,
.highlight .cm { color: #6b7280; font-style: italic; }
.highlight .n  { color: #e2e8f0; }
.highlight .o  { color: #f9a8d4; }
.highlight .mi,
.highlight .mf { color: #93c5fd; }
.highlight .nb { color: #fbbf24; }
.highlight .nf { color: #a78bfa; }
.highlight .nc { color: #fb923c; }
.highlight .nn { color: #fb923c; }
.highlight .p  { color: #e2e8f0; }
.highlight .bp { color: #fbbf24; }

/* ════════════════════════════════════════
   TABLOLAR
   ════════════════════════════════════════ */

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5pt;
    margin: 1em 0 1.2em;
    page-break-inside: auto;
}

thead tr {
    background: linear-gradient(135deg, #1e3a8a, #1e40af);
}

th {
    color: white;
    padding: 0.55em 0.9em;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.03em;
    font-size: 9pt;
}

td {
    padding: 0.42em 0.9em;
    border-bottom: 1px solid #e8edf5;
    vertical-align: top;
    line-height: 1.55;
}

tr:nth-child(even) td { background: #f8faff; }
tr:hover td { background: #f0f4ff; }

tbody tr:last-child td { border-bottom: 2px solid #dbeafe; }

/* ════════════════════════════════════════
   ALINTILA / NOTLAR
   ════════════════════════════════════════ */

blockquote {
    border-left: 4px solid #f59e0b;
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    margin: 1.1em 0;
    padding: 0.8em 1.1em;
    border-radius: 0 8px 8px 0;
    color: #78350f;
    font-size: 9.5pt;
    page-break-inside: avoid;
    box-shadow: inset 0 0 0 1px rgba(245,158,11,0.15);
}

blockquote p {
    margin: 0;
    text-align: left;
}

/* ════════════════════════════════════════
   LİSTELER
   ════════════════════════════════════════ */

ul, ol {
    margin: 0.3em 0 0.8em 0;
    padding-left: 1.7em;
}

li {
    margin-bottom: 0.3em;
    line-height: 1.65;
}

li > ul, li > ol { margin: 0.2em 0 0.3em; }

/* ════════════════════════════════════════
   AYIRICILAR
   ════════════════════════════════════════ */

hr {
    border: none;
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #dbeafe 20%, #dbeafe 80%, transparent);
    margin: 1.8em 0;
}

/* ════════════════════════════════════════
   BAĞLANTILAR
   ════════════════════════════════════════ */

a {
    color: #1d4ed8;
    text-decoration: none;
}

/* ════════════════════════════════════════
   ÖN KOŞUL KUTUSU
   ════════════════════════════════════════ */

.prereq-box {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid #0284c7;
    padding: 0.8em 1.1em;
    border-radius: 0 8px 8px 0;
    margin: 0 0 1.5em 0;
    font-size: 9.5pt;
    color: #0c4a6e;
    page-break-inside: avoid;
    box-shadow: inset 0 0 0 1px rgba(2,132,199,0.12);
}

.prereq-box strong { color: #0369a1; }

/* ════════════════════════════════════════
   NAVİGASYON FOOTER
   ════════════════════════════════════════ */

.nav-footer {
    margin-top: 1.5em;
    padding: 0.9em 1.2em;
    background: #f8faff;
    border: 1px solid #e2eaf5;
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 8.5pt;
    color: #64748b;
    page-break-inside: avoid;
    page-break-before: avoid;
}

.nav-footer a {
    color: #1d4ed8;
    text-decoration: none;
    font-weight: 500;
}

.nav-footer span:nth-child(2) {
    font-size: 7.5pt;
    letter-spacing: 0.05em;
    color: #94a3b8;
}

/* ════════════════════════════════════════
   ÖZEL CALLOUT KUTULARI
   ════════════════════════════════════════ */

.industry-quote {
    background: linear-gradient(135deg, #f0f4ff, #e8eef8);
    border-left: 4px solid #6366f1;
    margin: 1.3em 0;
    padding: 1em 1.3em;
    border-radius: 0 8px 8px 0;
    page-break-inside: avoid;
    font-style: italic;
    color: #312e81;
    font-size: 10pt;
    line-height: 1.65;
}

.industry-quote .quote-author {
    font-style: normal;
    font-weight: 600;
    font-size: 8.5pt;
    color: #4338ca;
    margin-top: 0.5em;
    display: block;
}

.learning-objectives {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left: 4px solid #059669;
    padding: 0.9em 1.2em;
    border-radius: 0 8px 8px 0;
    margin: 0 0 1.5em 0;
    font-size: 9.5pt;
    color: #064e3b;
    page-break-inside: avoid;
}

.learning-objectives strong {
    display: block;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4em;
    color: #047857;
}

.learning-objectives ul {
    margin: 0;
    padding-left: 1.3em;
}

.learning-objectives li {
    margin-bottom: 0.2em;
}

.whats-next {
    background: linear-gradient(135deg, #faf5ff, #ede9fe);
    border-left: 4px solid #7c3aed;
    padding: 0.9em 1.2em;
    border-radius: 0 8px 8px 0;
    margin: 1.5em 0 1em 0;
    font-size: 9.5pt;
    color: #4c1d95;
    page-break-inside: avoid;
}

.whats-next strong {
    display: block;
    font-size: 9pt;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4em;
    color: #6d28d9;
}

.interview-tip {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border-left: 4px solid #e11d48;
    padding: 0.7em 1.1em;
    border-radius: 0 8px 8px 0;
    margin: 1em 0;
    font-size: 9pt;
    color: #881337;
    page-break-inside: avoid;
}

.interview-tip strong {
    color: #be123c;
}

.portfolio-idea {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 4px solid #d97706;
    padding: 0.7em 1.1em;
    border-radius: 0 8px 8px 0;
    margin: 1em 0;
    font-size: 9pt;
    color: #78350f;
    page-break-inside: avoid;
}

.portfolio-idea strong {
    color: #b45309;
}

/* ════════════════════════════════════════
   YARDIMCI
   ════════════════════════════════════════ */

.page-break { page-break-after: always; }

.file-label {
    display: block;
    font-size: 7.5pt;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3em;
}
"""


# ════════════════════════════════════════
#   HEADING REMAP — Katman harflerini bölüm numaralarına çevir
# ════════════════════════════════════════

HEADING_REMAP = {"0": "1", "A": "2", "B": "3", "C": "4", "D": "5", "E": "6", "F": "7", "G": "8", "H": "9"}


def remap_headings(html_text, fname):
    """Remap katman headings to book chapter numbers: A.1 → 2.1, B.3 → 3.3 etc."""
    meta = CHAPTER_META.get(fname, {})
    old_prefix = meta.get("num")
    new_num = meta.get("book_num")
    if old_prefix and new_num and old_prefix in HEADING_REMAP:
        html_text = re.sub(
            rf'(<h[234][^>]*>)\s*{re.escape(old_prefix)}\.',
            rf'\g<1>{new_num}.',
            html_text,
        )
    return html_text


def anchor_id(fname):
    return "file_" + fname.replace(".md", "").replace("-", "_").replace(".", "_")


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def md_to_html(text):
    md = markdown.Markdown(
        extensions=MD_EXTENSIONS,
        extension_configs=MD_EXTENSION_CONFIGS,
    )
    return md.convert(text)


def build_cover():
    return """
<div class="cover" id="cover-page">
  <div class="cover-accent"></div>
  <h1>Veri Bilimci<br>Yol Haritası</h1>
  <div class="subtitle">0 → Senior · Kapsamlı Türkçe Öğrenme Rehberi</div>
  <div class="cover-divider"></div>
  <div class="cover-topics">
    <span class="topic-pill">Matematik</span>
    <span class="topic-pill">İstatistik</span>
    <span class="topic-pill">Klasik ML</span>
    <span class="topic-pill">Derin Öğrenme</span>
    <span class="topic-pill">NLP / LLM</span>
    <span class="topic-pill">MLOps</span>
    <span class="topic-pill">Sistem Tasarımı</span>
    <span class="topic-pill">Büyük Veri</span>
  </div>
  <div class="meta">MART 2026</div>
</div>
"""


def build_toc():
    rows = []

    # Ön madde bölümü
    rows.append('<div class="toc-section-label">Ön Madde</div>')
    front_files = ["README.md", "00-uygulama-sirasi.md", "01-yetkinlik-matrisi.md"]
    for fname in front_files:
        meta = CHAPTER_META.get(fname, {})
        aid = anchor_id(fname)
        title = meta.get("title", fname)
        rows.append(f"""
  <div class="toc-entry">
    <span class="t-num"></span>
    <span class="t-title"><a href="#{aid}">{title}</a></span>
    <span class="t-dots"></span>
    <a class="t-pageref" href="#{aid}"></a>
  </div>""")

    # Bölümler
    rows.append('<div class="toc-section-label" style="margin-top:1.2em;">Bölümler</div>')
    katman_files = [f for f in FILES if f.startswith("katman-")]
    for fname in katman_files:
        meta = CHAPTER_META.get(fname, {})
        aid = anchor_id(fname)
        book_label = meta.get("book_label", "")
        title = meta.get("title", fname)
        rows.append(f"""
  <div class="toc-entry">
    <span class="t-num">{book_label}</span>
    <span class="t-title"><a href="#{aid}">{title}</a></span>
    <span class="t-dots"></span>
    <a class="t-pageref" href="#{aid}"></a>
  </div>""")
        # h2 alt girişleri
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            content = read_file(fpath)
            for line in content.split("\n"):
                if line.startswith("## ") and not line.startswith("###"):
                    h2_title = line[3:].strip()
                    # Heading remap for TOC display
                    old_prefix = meta.get("num")
                    new_num = meta.get("book_num")
                    if old_prefix and new_num and h2_title.startswith(f"{old_prefix}."):
                        h2_title = f"{new_num}.{h2_title[len(old_prefix)+1:]}"
                    rows.append(f"""
  <div class="toc-subentry">
    <span class="t-title"><a href="#{aid}">{h2_title}</a></span>
    <span class="t-dots"></span>
    <a class="t-pageref" href="#{aid}"></a>
  </div>""")

    # Ekler bölümü
    rows.append('<div class="toc-section-label" style="margin-top:1.2em;">Ekler</div>')
    appendix_files = ["projeler.md", "mulakat.md", "kaynaklar.md"]
    for fname in appendix_files:
        meta = CHAPTER_META.get(fname, {})
        aid = anchor_id(fname)
        book_label = meta.get("book_label", "")
        title = meta.get("title", fname)
        rows.append(f"""
  <div class="toc-entry">
    <span class="t-num">{book_label}</span>
    <span class="t-title"><a href="#{aid}">{title}</a></span>
    <span class="t-dots"></span>
    <a class="t-pageref" href="#{aid}"></a>
  </div>""")

    return f"""
<div class="front-matter toc-page" id="toc">
  <h2>İçindekiler</h2>
  <hr class="toc-rule">
  {"".join(rows)}
</div>
"""


def build_chapter_separator(fname):
    meta = CHAPTER_META.get(fname, {})
    num = meta.get("num", "")
    book_num = meta.get("book_num")
    book_label = meta.get("book_label")
    title = meta.get("title", fname)
    desc = meta.get("desc", "")
    time_est = meta.get("time", "")
    difficulty = meta.get("difficulty", "")
    prereq = meta.get("prereq", "")
    category = meta.get("category", "chapter")
    topics = meta.get("topics", [])
    aid = anchor_id(fname)

    # Ghost number
    ghost = str(book_num) if book_num else ""
    ghost_html = f'<div class="ch-num">{ghost}</div>' if ghost else ""

    # Label
    if book_label:
        label_html = f'<div class="ch-label">{book_label}</div>'
    elif category == "front":
        label_html = '<div class="ch-label">Ön Madde</div>'
    else:
        label_html = '<div class="ch-label">Bölüm</div>'

    # Category CSS class
    sep_class = "chapter-separator"
    if category == "appendix":
        sep_class += " appendix-sep"
    elif category == "front":
        sep_class += " front-sep"

    # Meta pills
    meta_items = []
    if time_est:
        meta_items.append(f'<span>Süre: {time_est}</span>')
    if difficulty:
        meta_items.append(f'<span>Seviye: {difficulty}</span>')
    if prereq:
        meta_items.append(f'<span>Önkoşul: {prereq}</span>')
    meta_html = f'<div class="ch-meta">{"".join(meta_items)}</div>' if meta_items else ""

    # Topics preview
    topics_html = ""
    if topics:
        items = "".join(f"<li>{t}</li>" for t in topics)
        topics_html = f"""<div class="ch-preview">
      <div class="ch-preview-label">Bu Bölümde</div>
      <ul>{items}</ul>
    </div>"""

    return f"""
<div class="{sep_class}" id="{aid}">
  {ghost_html}
  {label_html}
  <h2 class="ch-title">{title}</h2>
  <div class="ch-divider"></div>
  <p class="ch-desc">{desc}</p>
  {meta_html}
  {topics_html}
</div>
"""


def main():
    print("PDF oluşturuluyor...")

    html_parts = []

    # 1. Kapak
    html_parts.append(build_cover())

    # 2. İçindekiler
    html_parts.append(build_toc())

    for i, fname in enumerate(FILES):
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  UYARI: {fname} bulunamadı, atlanıyor.")
            continue

        print(f"  [{i+1}/{len(FILES)}] {fname} işleniyor...")
        meta = CHAPTER_META.get(fname, {})
        category = meta.get("category", "chapter")
        aid = anchor_id(fname)
        text = read_file(fpath)
        body_html = md_to_html(text)
        body_html = remap_headings(body_html, fname)

        # Bölüm kapak sayfası
        if meta.get("is_separator"):
            html_parts.append(build_chapter_separator(fname))

        # Sayfa sınıfı
        page_cls = "chapter-start front-matter" if category == "front" else "chapter-start"
        content_id = f"{aid}_content" if meta.get("is_separator") else aid
        html_parts.append(f'<div class="{page_cls}" id="{content_id}">{body_html}</div>')

    full_html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="author" content="Veri Bilimci Yol Haritası">
<meta name="description" content="Türkçe kapsamlı veri bilimi öğrenme rehberi — 0'dan Senior'a">
<meta name="keywords" content="veri bilimi, makine öğrenmesi, MLOps, Python, SQL, derin öğrenme">
<title>Veri Bilimci Yol Haritası</title>
</head>
<body>
{"".join(html_parts)}
</body>
</html>"""

    output_path = os.path.join(BASE_DIR, "veri-bilimci-yol-haritasi.pdf")

    font_config = FontConfiguration()
    css = CSS(string=CSS_STYLE, font_config=font_config)

    print("  WeasyPrint ile PDF render ediliyor (bu biraz sürebilir)...")
    html_doc = HTML(string=full_html, base_url=BASE_DIR)
    html_doc.write_pdf(output_path, stylesheets=[css], font_config=font_config)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nTamamlandı! → {output_path}")
    print(f"Dosya boyutu: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
