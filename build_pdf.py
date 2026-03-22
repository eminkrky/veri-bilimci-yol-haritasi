#!/usr/bin/env python3
"""
Veri Bilimci Yol Haritası — MD → PDF dönüştürücü
Kullanım: python3 build_pdf.py
"""

import os
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
        "is_separator": False,
    },
    "00-uygulama-sirasi.md": {
        "num": None,
        "title": "Uygulama Sırası",
        "short": "Uygulama Planı",
        "desc": "8 aşamalı sıralı öğrenme planı; her aşamanın teslim edilebilir çıktısı.",
        "time": "1 saat okuma",
        "difficulty": "Giriş",
        "prereq": None,
        "is_separator": False,
    },
    "01-yetkinlik-matrisi.md": {
        "num": None,
        "title": "Yetkinlik Matrisi",
        "short": "Yetkinlik Matrisi",
        "desc": "Senior veri bilimci tanımı; teknik ve davranışsal yetkinlikler haritası.",
        "time": "1 saat okuma",
        "difficulty": "Giriş",
        "prereq": None,
        "is_separator": False,
    },
    "katman-0-matematik.md": {
        "num": "0",
        "title": "Matematik Temelleri",
        "short": "Katman 0 — Matematik",
        "desc": "Lineer cebir, kalkülüs ve olasılık teorisinin makine öğrenmesi perspektifinden sezgisel anlatımı.",
        "time": "3–5 gün",
        "difficulty": "Temel",
        "prereq": "Lise matematik yeterli",
        "is_separator": True,
    },
    "katman-A-temeller.md": {
        "num": "A",
        "title": "Temeller",
        "short": "Katman A — Temeller",
        "desc": "Python/Pandas, analitik SQL, istatistik temeli ve görselleştirme; günlük DS araç takımı.",
        "time": "3–4 hafta",
        "difficulty": "Temel",
        "prereq": "Katman 0 (Matematik)",
        "is_separator": True,
    },
    "katman-B-klasik-ml.md": {
        "num": "B",
        "title": "Klasik Makine Öğrenmesi",
        "short": "Katman B — Klasik ML",
        "desc": "Doğrusal modeller, karar ağaçları, boosting ve model değerlendirme; teoriden üretime.",
        "time": "3–4 hafta",
        "difficulty": "Orta",
        "prereq": "Katman A (Temeller)",
        "is_separator": True,
    },
    "katman-C-deney-nedensellik.md": {
        "num": "C",
        "title": "Deney Tasarımı ve Nedensellik",
        "short": "Katman C — Deney/Nedensellik",
        "desc": "A/B test tasarımı, güç analizi, CUPED ve nedensel çıkarım; korelasyondan nedenselliğe.",
        "time": "2–3 hafta",
        "difficulty": "Orta-İleri",
        "prereq": "Katman A (İstatistik), Katman B",
        "is_separator": True,
    },
    "katman-D-derin-ogrenme.md": {
        "num": "D",
        "title": "Derin Öğrenme",
        "short": "Katman D — Derin Öğrenme",
        "desc": "Sinir ağları, NLP, bilgisayarlı görü, öneri sistemleri ve LLM/RAG mimarileri.",
        "time": "4–6 hafta",
        "difficulty": "İleri",
        "prereq": "Katman 0 (Matematik), Katman B",
        "is_separator": True,
    },
    "katman-E-mlops.md": {
        "num": "E",
        "title": "MLOps",
        "short": "Katman E — MLOps",
        "desc": "Model paketleme, servis, izleme, drift tespiti ve CI/CD; açık kaynak araç odaklı.",
        "time": "2–3 hafta",
        "difficulty": "İleri",
        "prereq": "Katman B veya D (çalışan bir model şart)",
        "is_separator": True,
    },
    "katman-F-sistem-tasarimi.md": {
        "num": "F",
        "title": "ML Sistem Tasarımı",
        "short": "Katman F — Sistem Tasarımı",
        "desc": "Online/offline servis mimarileri, feature store, ölçekleme ve maliyet optimizasyonu.",
        "time": "2–3 hafta",
        "difficulty": "İleri",
        "prereq": "Katman E (MLOps)",
        "is_separator": True,
    },
    "katman-G-senior-davranislar.md": {
        "num": "G",
        "title": "Senior Davranışlar",
        "short": "Katman G — Senior",
        "desc": "Etki odaklı çalışma, teknik liderlik, dokümantasyon ve kariyer stratejisi.",
        "time": "1–2 hafta",
        "difficulty": "Senior",
        "prereq": "Tüm teknik katmanlar (A–F)",
        "is_separator": True,
    },
    "katman-H-buyuk-veri.md": {
        "num": "H",
        "title": "Büyük Veri",
        "short": "Katman H — Büyük Veri",
        "desc": "Spark, Dask ve dağıtık hesaplama; veriyi tek makineye sığmadığında ne yapılır?",
        "time": "2–3 hafta",
        "difficulty": "İleri",
        "prereq": "Katman A (Pandas, SQL), Katman E (MLOps)",
        "is_separator": True,
    },
    "projeler.md": {
        "num": None,
        "title": "Portföy Projeleri",
        "short": "Projeler",
        "desc": "Proje-0'dan Proje-7'ye sekiz portfolyo projesi; iş vitrini oluşturma rehberi.",
        "time": "Her proje: 1–2 hafta",
        "difficulty": "Uygulama",
        "prereq": None,
        "is_separator": False,
    },
    "mulakat.md": {
        "num": None,
        "title": "Mülakat Hazırlığı",
        "short": "Mülakat",
        "desc": "SQL, ML, case study ve davranışsal mülakat soruları; çözümlü örnekler.",
        "time": "1–2 hafta",
        "difficulty": "Uygulama",
        "prereq": None,
        "is_separator": False,
    },
    "kaynaklar.md": {
        "num": None,
        "title": "Kaynaklar",
        "short": "Kaynaklar",
        "desc": "Ücretsiz ve temel kaynak listesi; kitaplar, kurslar, araçlar ve topluluklar.",
        "time": "Referans",
        "difficulty": "Referans",
        "prereq": None,
        "is_separator": False,
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
    "nl2br",
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Sayfa düzeni ─── */
@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2.5cm;

    @top-left {
        content: "Veri Bilimci Yol Haritası";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #c0c8d8;
    }
    @top-right {
        content: string(chapter-title);
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #888;
    }
    @bottom-left {
        content: string(chapter-title);
        font-family: 'Inter', sans-serif;
        font-size: 7.5pt;
        color: #aab;
    }
    @bottom-right {
        content: counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #64748b;
        font-weight: 500;
    }
    @bottom-center { content: none; }
}

/* İlk sayfa (kapak) — header/footer yok */
@page :first {
    @bottom-left   { content: none; }
    @bottom-right  { content: none; }
    @top-left      { content: none; }
    @top-right     { content: none; }
}

/* Ön madde sayfaları — Roman rakamı */
@page front-matter {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2.5cm;
    @bottom-right {
        content: counter(page, lower-roman);
        font-family: 'Inter', sans-serif;
        font-size: 8.5pt;
        color: #94a3b8;
    }
    @bottom-left  { content: none; }
    @top-left     { content: none; }
    @top-right    { content: none; }
    @bottom-center { content: none; }
}

.front-matter {
    page: front-matter;
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', 'DejaVu Sans', sans-serif;
    font-size: 10.5pt;
    line-height: 1.7;
    color: #1a1a2e;
    background: #fff;
}

/* ─── Kapak sayfası ─── */
.cover {
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 24cm;
    text-align: center;
    background: linear-gradient(135deg, #0f3460 0%, #16213e 60%, #0f3460 100%);
    color: white;
    padding: 3cm;
    margin: -2.5cm -2cm -2.5cm -2.5cm;
}

.cover h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.5em;
    border: none;
    line-height: 1.2;
    string-set: none;
}

.cover .subtitle {
    font-size: 14pt;
    color: #94a3b8;
    margin-bottom: 2em;
}

.cover .meta {
    font-size: 10pt;
    color: #64748b;
    margin-top: 3em;
}

/* ─── İçindekiler sayfası ─── */
.toc-page {
    page: front-matter;
    page-break-after: always;
    padding-top: 0.5cm;
}

.toc-page h2 {
    font-size: 20pt;
    font-weight: 700;
    color: #0f3460;
    border-bottom: 3px solid #0f3460;
    padding-bottom: 0.4em;
    margin-bottom: 1.5em;
    string-set: none;
}

.toc-section-label {
    font-size: 8pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin: 1.2em 0 0.4em 0;
}

.toc-entry {
    display: flex;
    align-items: baseline;
    margin-bottom: 0.55em;
    font-size: 10.5pt;
    gap: 0;
}

.toc-entry .t-num {
    font-weight: 600;
    color: #1e40af;
    min-width: 5.5em;
    flex-shrink: 0;
}

.toc-entry .t-title a {
    color: #1a1a2e;
    text-decoration: none;
}

.toc-entry .t-title {
    flex-grow: 1;
}

.toc-entry .t-dots {
    flex: 1;
    border-bottom: 1px dotted #cbd5e1;
    margin: 0 0.6em;
    min-width: 0.5em;
    align-self: flex-end;
    margin-bottom: 4px;
}

a.t-pageref {
    color: #64748b;
    font-weight: 500;
    font-size: 10pt;
    text-decoration: none;
    min-width: 2em;
    text-align: right;
    flex-shrink: 0;
}

a.t-pageref::after {
    content: target-counter(attr(href), page);
}

/* ─── Bölüm kapak sayfası ─── */
.chapter-separator {
    page-break-before: always;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    min-height: 24cm;
    padding: 3cm 3cm 3.5cm 3cm;
    background: linear-gradient(160deg, #0f3460 0%, #1a2f58 45%, #16213e 100%);
    color: white;
    margin: -2.5cm -2cm -2.5cm -2.5cm;
}

.chapter-separator .ch-num {
    font-size: 96pt;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.08);
    line-height: 1;
    margin-bottom: -0.15em;
    font-variant-numeric: tabular-nums;
}

.chapter-separator .ch-title {
    font-size: 26pt;
    font-weight: 700;
    color: #e2e8f0;
    border: none;
    line-height: 1.2;
    margin-bottom: 0.6em;
    margin-top: 0;
    string-set: chapter-title content();
}

.chapter-separator .ch-desc {
    font-size: 11pt;
    color: #94a3b8;
    max-width: 75%;
    line-height: 1.7;
    margin-bottom: 2em;
}

.chapter-separator .ch-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6em;
    font-size: 9pt;
    color: #64748b;
}

.chapter-separator .ch-meta span {
    background: rgba(255, 255, 255, 0.07);
    padding: 0.3em 0.9em;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* ─── Bölüm başlangıcı ─── */
.chapter-start {
    page-break-before: always;
}

/* ─── Başlıklar ─── */
h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #0f3460;
    border-bottom: 3px solid #0f3460;
    padding-bottom: 0.3em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
    page-break-after: avoid;
    string-set: chapter-title content();
}

h2 {
    font-size: 14pt;
    font-weight: 600;
    color: #1e40af;
    border-bottom: 1px solid #bfdbfe;
    padding-bottom: 0.2em;
    margin-top: 1.4em;
    margin-bottom: 0.6em;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #1d4ed8;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: 600;
    color: #2563eb;
    margin-top: 1em;
    margin-bottom: 0.4em;
    page-break-after: avoid;
}

p {
    margin: 0.5em 0 0.8em 0;
    text-align: justify;
    orphans: 3;
    widows: 3;
}

/* ─── Kod blokları ─── */
pre {
    background: #0d1117;
    color: #e6edf3;
    border-radius: 6px;
    padding: 1em 1.2em;
    font-family: 'JetBrains Mono', 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 8.5pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
    margin: 0.8em 0;
    border-left: 4px solid #1e40af;
}

code {
    font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace;
    font-size: 8.5pt;
    background: #f1f5f9;
    color: #0f172a;
    padding: 0.15em 0.4em;
    border-radius: 3px;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
    font-size: inherit;
}

/* ─── Syntax highlighting ─── */
.highlight { background: #0d1117 !important; }
.highlight .k  { color: #ff7b72; }
.highlight .kn { color: #ff7b72; }
.highlight .s  { color: #a5d6ff; }
.highlight .s1 { color: #a5d6ff; }
.highlight .s2 { color: #a5d6ff; }
.highlight .c1 { color: #8b949e; font-style: italic; }
.highlight .n  { color: #e6edf3; }
.highlight .o  { color: #ff7b72; }
.highlight .mi { color: #79c0ff; }
.highlight .mf { color: #79c0ff; }
.highlight .nb { color: #ffa657; }
.highlight .nf { color: #d2a8ff; }
.highlight .nc { color: #ffa657; }
.highlight .nn { color: #ffa657; }
.highlight .p  { color: #e6edf3; }

/* ─── Tablolar ─── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5pt;
    margin: 1em 0;
    page-break-inside: avoid;
}

th {
    background: #1e40af;
    color: white;
    padding: 0.5em 0.8em;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 0.4em 0.8em;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

tr:nth-child(even) td {
    background: #f8fafc;
}

/* ─── Blockquote ─── */
blockquote {
    border-left: 4px solid #f59e0b;
    background: #fffbeb;
    margin: 1em 0;
    padding: 0.7em 1em;
    border-radius: 0 6px 6px 0;
    color: #78350f;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

blockquote p {
    margin: 0;
    text-align: left;
}

/* ─── Listeler ─── */
ul, ol {
    margin: 0.5em 0 0.8em 0;
    padding-left: 1.8em;
}

li {
    margin-bottom: 0.25em;
}

/* ─── Yatay çizgi ─── */
hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 1.5em 0;
}

/* ─── Bağlantılar ─── */
a {
    color: #1d4ed8;
    text-decoration: none;
}

/* ─── Sayfa kırma ─── */
.page-break {
    page-break-after: always;
}

/* ─── Önkoşul kutusu ─── */
.prereq-box {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    padding: 0.7em 1em;
    border-radius: 0 6px 6px 0;
    margin: 0 0 1.5em 0;
    font-size: 9.5pt;
    color: #0c4a6e;
    page-break-inside: avoid;
}

.prereq-box strong {
    color: #0369a1;
}

/* ─── Navigasyon footer ─── */
.nav-footer {
    margin-top: 2.5em;
    padding-top: 1em;
    border-top: 2px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 9pt;
    color: #64748b;
    page-break-inside: avoid;
}

.nav-footer a {
    color: #1d4ed8;
    text-decoration: none;
}

/* ─── Dosya etiketi ─── */
.file-label {
    display: block;
    font-size: 8pt;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3em;
}
"""


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
  <h1>Veri Bilimci<br>Yol Haritası</h1>
  <div class="subtitle">0 → Senior Kapsamlı Öğrenme Rehberi</div>
  <p style="color:#94a3b8; font-size:11pt;">
    Matematik · İstatistik · Makine Öğrenmesi<br>
    Derin Öğrenme · MLOps · Sistem Tasarımı
  </p>
  <div class="meta">
    Mart 2026
  </div>
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

    # Katmanlar bölümü
    rows.append('<div class="toc-section-label" style="margin-top:1.2em;">Öğrenme Katmanları</div>')
    katman_files = [f for f in FILES if f.startswith("katman-")]
    for fname in katman_files:
        meta = CHAPTER_META.get(fname, {})
        aid = anchor_id(fname)
        num = meta.get("num", "")
        label = f"Katman {num}" if num else ""
        title = meta.get("title", fname)
        time_est = meta.get("time", "")
        rows.append(f"""
  <div class="toc-entry">
    <span class="t-num">{label}</span>
    <span class="t-title"><a href="#{aid}">{title}</a></span>
    <span class="t-dots"></span>
    <a class="t-pageref" href="#{aid}"></a>
  </div>""")

    # Ekler bölümü
    rows.append('<div class="toc-section-label" style="margin-top:1.2em;">Ekler</div>')
    extra_files = ["projeler.md", "mulakat.md", "kaynaklar.md"]
    for fname in extra_files:
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

    return f'<div class="front-matter toc-page" id="toc"><h2>İçindekiler</h2>{"".join(rows)}</div>'


def build_chapter_separator(fname):
    meta = CHAPTER_META.get(fname, {})
    aid = anchor_id(fname)
    num = meta.get("num", "")
    title = meta.get("title", "")
    desc = meta.get("desc", "")
    time_est = meta.get("time", "")
    diff = meta.get("difficulty", "")
    prereq = meta.get("prereq", "")

    num_html = f'<div class="ch-num">{num}</div>' if num else ""
    prereq_html = f'<span>Önkoşul: {prereq}</span>' if prereq else ""

    return f"""
<div class="chapter-separator" id="{aid}">
  {num_html}
  <h2 class="ch-title">{title}</h2>
  <p class="ch-desc">{desc}</p>
  <div class="ch-meta">
    <span>Süre: {time_est}</span>
    <span>Seviye: {diff}</span>
    {prereq_html}
  </div>
</div>"""


def main():
    print("PDF oluşturuluyor...")

    html_parts = []

    # 1. Kapak
    html_parts.append(build_cover())

    # 2. İçindekiler
    html_parts.append(build_toc())

    FRONT_MATTER_FILES = {"README.md", "00-uygulama-sirasi.md", "01-yetkinlik-matrisi.md"}

    for i, fname in enumerate(FILES):
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  UYARI: {fname} bulunamadı, atlanıyor.")
            continue

        print(f"  [{i+1}/{len(FILES)}] {fname} işleniyor...")
        meta = CHAPTER_META.get(fname, {})
        aid = anchor_id(fname)
        text = read_file(fpath)
        body_html = md_to_html(text)

        # Katman dosyaları için ayırıcı kapak sayfası
        if meta.get("is_separator"):
            html_parts.append(build_chapter_separator(fname))

        # Sayfa sınıfı
        page_cls = "chapter-start front-matter" if fname in FRONT_MATTER_FILES else "chapter-start"
        html_parts.append(f'<div class="{page_cls}" id="{aid}">{body_html}</div>')

    full_html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
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
