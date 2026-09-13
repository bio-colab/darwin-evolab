# Hardened Self-Calibrating PDF-to-RTF Lab
### Universal Invariant Core + Evolutive Calibration Engine

> **Locked Scope**: English PDFs generated exclusively by Microsoft Word $\rightarrow$ Structurally, Semantically, and Formatted RTF 1.9 documents.
>
> **Out of Scope**: Arabic / Bidi scripts, Scanned documents / OCR, Non-Word generators (LibreOffice, LaTeX, etc.).
>
> **Fidelity Guarantee**: 100.0% Text Integrity on Genuine Microsoft Word PDFs, $\ge 98.9\%$ Multi-Gate Equivalence Score.

---

## 1. Architectural Decision: Why RTF and Not DOCX? (لماذا RTF وليس DOCX؟)

A fundamental architectural choice of this project is emitting **Microsoft Rich Text Format (RTF 1.9)** rather than Office Open XML (**DOCX**):

### أ. تنسيق نصّي مسطّح مقابل أرشيف مضغوط معقد (Flat Text vs. Multi-File Zip Container)
- **DOCX** ليس ملفاً واحداً بسيطاً، بل هو حاوية مضغوطة (`zip container` وفق معيار OPC) تضم بنية شجرية من ملفات XML المتعددة والمترابطة (`word/document.xml`, `word/styles.xml`, `word/_rels/document.xml.rels`, `[Content_Types].xml`, إلخ) مع شبكة معقدة من مراجع العلاقات الداخلية (`r:id`). توليده يدوياً بصرامة وبدون مكتبات خارجية يتطلب إدارة معقدة للأرشيف وللمخططات وللمراجع المتبادلة.
- **RTF** في المقابل هو **تنسيق نصّي مسطّح (Flat plain-text)** تحكمه كلمات تحكم صريحة ومباشرة (`control words` مثل `\b`, `\par`, `\trowd`, `\cell`). هذا يجعل توليده برمجياً من التمثيل الوسيط (`Document IR`) وتحليله أسهل بكثير، وأقل عرضة للأخطاء العشوائية، وممكناً بالاعتماد الحصري على مكتبة بايثون القياسية (Pure Python Standard Library — صفر حزم خارجية).

### ب. قابلية التدقيق والمقارنة النصية الفورية (Auditability & Plain-Text Diffing)
- في بيئة التجارب التطورية والمعايرة التلقائية (Evolutionary Calibration)، نحتاج لمقارنة مخرجات التجارب والمتغيرات وفحص أثر كل طفرة أو سياسة بسرعة وبدقة.
- ملفات الـ RTF يمكن مقارنتها مباشرة بأدوات المقارنة النصية القياسية (`diff`), بينما يتطلب DOCX فك الضغط ومقارنة أشجار XML المتفرقة، مما يعقد المراقبة والتحقق الجنائي التلقائي.

### ج. الملاءمة المثالية لمنطق "النواة الثابتة + السياسات القابلة للتعديل"
- المبدأ الحاكم للمشروع يفصل بدقة بين:
  1. **نواة هندسية قطعية وثابتة (Invariant Core)**: التمثيل الوسيط (`ir.py`)، باعث RTF الصافي (`rtf_emitter.py`)، محلل RTF (`rtf_parser.py`)، ومحكم البوابات الرياضي (`equivalence.py`).
  2. **سطح استكشافي توافقي (Tunable Heuristic Policies)**: عتبات المسافات الفاصلة بين الفقرات، معايير كشف الجداول، وحزم مطابقة الخطوط.
- تنسيق RTF يتيح الحفاظ على هذا الفصل الهندسي الأنيق بأقل قدر من التعقيد المعرفي والكودي، مما يجعله الخيار الأمثل والمحكم لبناء أساس متين يمكن الاعتماد عليه.

---

## 2. System Architecture (المعمارية العامة)

```
┌────────────────────────────────────────────────────────┐
│               Source PDF (Microsoft Word)              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼ (pdf_extractor.py via PyMuPDF)
   Raw PDF Layout & Fonts (Word PostScript Subset Stripping: ^[A-Z]{6}\+)
                            │
                            ▼ (Tunable Policy: calibrated_word_profile.json)
┌────────────────────────────────────────────────────────┐
│                 Canonical Document IR                  │ <── Invariant Core
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼ (rtf_emitter.py — Pure Python Standard Library)
┌────────────────────────────────────────────────────────┐
│                   RTF 1.9 Document                     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼ (rtf_parser.py — Pure Python Standard Library)
┌────────────────────────────────────────────────────────┐
│                  Reconstructed IR                      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│          Multi-Gate Equivalence Verifier (equivalence.py)              │
│  - Gate 1: Text Integrity (Exact Token Match == 1.0, Diff Reporting)   │
│  - Gate 2: Structure Integrity (Paragraphs, Alignments, Physical Gaps) │
│  - Gate 2.5: Table Oracle (Precision / Recall on Matrix Grid)          │
│  - Gate 3: Formatting Integrity (Canonical Families, Sizes, Styles)    │
│  - Gate 4: Visual / Geometry Oracle (Calibrated Density, Noise Floor)  │
│  * Lexicographic Cascading Penalty: Text Failure Suppresses Formatting │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Peer-Review Hardening & Genuine MS Word Holdout Suite

استجابةً للتدقيق النقدي الصارم من مراجعة الأقران المستقلة، خضع النظام لعملية تحصين وتمتين شاملة:

1. **القضاء على التحقق الذاتي التركيبي (Eliminating Synthetic Self-Fulfillment)**:
   - بدلاً من الاعتماد على مستندات اصطناعية مولدة ومختبرة بذات المحرك، أُنشئت حزمة هولداوت مستقلة تماماً مولدة من **تطبيق Microsoft Word الرسمي (Office 16.0)** عبر أتمتة COM على نظام Windows.
   - تم استخراج شجرة الحقيقة المرجعية (Ground-Truth Reference IR) مباشرة من نموذج كائنات Word الداخلي (`doc.Paragraphs`, `doc.Tables`, `doc.PageSetup`) دون أي تدخل من PyMuPDF.
2. **الفصل الصارم بين بيانات التدريب والتقييم (Strict Train/Holdout Separation)**:
   - تدريب ومعايرة مصفوفة MAP-Elites يتم حصراً على حزمة التطوير (`corpus.py`).
   - التقييم النهائي والحكم يتم على حزمة هولداوت مايكروسوفت وورد الحقيقية والمستقلة (`word_holdout.py`).
3. **تفعيل البوابة الرابعة وتكامل البوابات الخمس (Full Multi-Gate Verification)**:
   - تفعيل `Gate 4: Visual / Geometry Oracle` لحساب كثافة الكتل وتدفق الصفحات بنظام أوزان خماسي:
     $$\text{Text: } 0.35, \quad \text{Structure: } 0.20, \quad \text{Table: } 0.20, \quad \text{Formatting: } 0.15, \quad \text{Visual: } 0.10$$
   - إضافة أرضية ضوضاء معايرة ($\epsilon = 0.05$) لامتصاص الفروق الدقيقة في تصيير الخطوط.
4. **استخراج التباعد الحقيقي ومحاذاة الجداول**:
   - تفعيل البعد الجينومي `line_spacing_round_pt` لقياس المسافات الرأسية الحقيقية بين الأسطر والفقرات.
   - حفظ المسافات الفاصلة قبل الجداول في `space_after_pt` للفقرة السابقة.
   - دعم محلل RTF لمحاذاة صفوف الجداول (`\trql`, `\trqc`, `\trqr`).
   - مطابقة الفجوة المادية بين الكتل المتتابعة ($\Delta y_{i-1 \to i}$) لتعويض الاندماج التلقائي لهوامش Word.

---

## 4. Empirical Benchmark Audit Results (النتائج التجريبية المدققة)

### أ. حزمة هولداوت مايكروسوفت وورد الحقيقية (Genuine MS Word Holdout Suite)
المصدر: [`reports/pdf2rtf_real_word_holdout_benchmark.json`](../../reports/pdf2rtf_real_word_holdout_benchmark.json)

| المستند الحقيقي (Word Document) | سلامة النص (Gate 1) | بنية المستند (Gate 2) | مصفوفة الجداول (Gate 2.5) | التنسيق والخطوط (Gate 3) | الهندسة البصرية (Gate 4) | الدرجة المركبة | زمن المعالجة | النتيجة |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **word_academic_paper** | **100.00%** | 97.50% | 100.00% | 99.38% | 100.00% | **99.41%** | 18.1 ms | **PASSED** ✅ |
| **word_financial_report** | **100.00%** | 95.62% | 100.00% | 100.00% | 100.00% | **99.12%** | 26.2 ms | **PASSED** ✅ |
| **word_executive_letter** | **100.00%** | 97.19% | 100.00% | 99.29% | 100.00% | **99.33%** | 14.8 ms | **PASSED** ✅ |
| **word_styled_article** | **100.00%** | 94.58% | 100.00% | 92.17% | 100.00% | **97.74%** | 10.7 ms | **PASSED** ✅ |
| **المتوسط الإجمالي (Holdout)** | **100.00%** | **96.22%** | **100.00%** | **97.71%** | **100.00%** | **98.90%** | **17.4 ms** | **100% ALL PASS** ✅ |

### ب. حزمة التطوير التركيبية (Synthetic Dev Suite)
المصدر: [`reports/pdf2rtf_golden_benchmark.json`](../../reports/pdf2rtf_golden_benchmark.json)

| المستند النمطي (Synthetic Archetype) | سلامة النص (Gate 1) | بنية المستند (Gate 2) | مصفوفة الجداول (Gate 2.5) | التنسيق والخطوط (Gate 3) | الهندسة البصرية (Gate 4) | الدرجة المركبة | زمن المعالجة | النتيجة |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **article_standard** | **100.00%** | 100.00% | 100.00% | 95.83% | 100.00% | **99.37%** | 14.0 ms | **PASSED** ✅ |
| **financial_table** | **100.00%** | 100.00% | 100.00% | 95.83% | 100.00% | **99.37%** | 17.0 ms | **PASSED** ✅ |
| **executive_summary** | **100.00%** | 100.00% | 100.00% | 97.92% | 100.00% | **99.69%** | 8.6 ms | **PASSED** ✅ |
| **bulleted_memo** | **100.00%** | 100.00% | 100.00% | 98.33% | 100.00% | **99.75%** | 9.6 ms | **PASSED** ✅ |
| **subset_font_stress** | **100.00%** | 100.00% | 100.00% | 97.22% | 100.00% | **99.58%** | 7.8 ms | **PASSED** ✅ |
| **المتوسط الإجمالي (Dev)** | **100.00%** | **100.00%** | **100.00%** | **97.03%** | **100.00%** | **99.55%** | **11.4 ms** | **100% ALL PASS** ✅ |

---

## 5. How to Run (طريقة التشغيل والتحقق)

### تشغيل مشغل المعايير الشامل (Benchmark Runner)
```bash
# تشغيل كلا المعيارين (حزمة التطوير + حزمة هولداوت Word الحقيقية):
python -m experimental.pdf2rtf.benchmark --all

# تشغيل حزمة هولداوت مايكروسوفت وورد الحقيقية فقط:
python -m experimental.pdf2rtf.benchmark --holdout

# تشغيل حزمة التطوير التركيبية فقط:
python -m experimental.pdf2rtf.benchmark --dev
```

### تشغيل المعايرة التطورية (MAP-Elites Calibration)
```bash
python -m experimental.pdf2rtf.calibrate
```

### تشغيل حزمة الاختبارات المؤتمتة (Pytest)
```bash
# تشغيل جميع اختبارات وحدة مختبر pdf2rtf (44 اختباراً):
pytest experimental/pdf2rtf/tests/ -v

# تشغيل كامل اختبارات المستودع (740 اختباراً):
pytest tests/ experimental/ -q
```

---

## 6. Project Roadmap Status (حالة مراحل المشروع)

- **Phase 1: Deterministic Foundation & Invariant Core** [COMPLETED ✅]
  - Document IR, RTF Emitter, RTF Parser, PDF Extractor (subset prefix stripping).
- **Phase 2: Equivalence Oracles & Multi-Gate Verifier** [COMPLETED ✅]
  - 5-Gate verification engine with cascading penalty.
- **Phase 3: Evolab Integration & Heuristic Policy Tuning** [COMPLETED ✅]
  - Parameterized `ProfileGenome` (7 dimensions), `PDF2RTFAdapter` extending `DomainAdapter`, MAP-Elites calibration.
- **Phase 4: Hardening & Golden Corpus Benchmark** [COMPLETED ✅]
  - 5 deterministic document archetypes, multi-document evaluation harness.
- **Peer-Review Hardening & Genuine Word Holdout Evaluation** [COMPLETED ✅]
  - Native MS Word COM holdout corpus, strict train/holdout split, 100% text integrity, 98.90% holdout fidelity.
- **Phase 5: Self-Contained CLI & Open-Source Packaging** [NEXT 🚀]
  - Standalone CLI `pdf2rtf convert`, distribution packaging, pip install readiness.
