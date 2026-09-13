# Hardened Self-Calibrating PDF-to-RTF Lab
### Universal Invariant Core + Evolutive Calibration Engine

> **Locked Scope**: English PDFs generated exclusively by Microsoft Word $\rightarrow$ Structurally, Semantically, and Formatted RTF 1.9 documents.
>
> **Out of Scope**: Arabic / Bidi scripts, Scanned documents / OCR, Non-Word generators (LibreOffice, LaTeX, etc.).

---

## 1. Architectural Decision: Why RTF and Not DOCX? (لماذا RTF وليس DOCX؟)

A fundamental architectural choice of this project is emitting **Microsoft Rich Text Format (RTF 1.9)** rather than Office Open XML (**DOCX**):

### أ. تنسيق نصّي مسطّح مقابل أرشيف مضغوط معقد (Flat Text vs. Multi-File Zip Container)
- **DOCX** ليس ملفاً واحداً بسيطاً، بل هو حاوية مضغوطة (`zip container` وفق معيار OPC) تضم بنية شجرية من ملفات XML المتعددة والمترابطة (`word/document.xml`, `word/styles.xml`, `word/_rels/document.xml.rels`, `[Content_Types].xml`, إلخ) مع شبكة معقدة من مراجع العلاقات الداخلية (`r:id`). توليده يدوياً بصرامة وبدون مكتبات خارجية يتطلب إدارة معقدة للأرشيف وللمخططات وللمراجع المتبادلة.
- **RTF** في المقابل هو **تنسيق نصّي مسطّح (Flat plain-text)** تحكمه كلمات تحكم صريحة ومباشرة (`control words` مثل `\b`, `\par`, `\trowd`, `\cell`). هذا يجعل توليده برمجياً من التمثيل الوسيط (`Document IR`) وتحليله أسهل بكثير، وأقل عرضة للأخطاء العشوائية، وممكناً بالاعتماد الحصري على مكتبة بايثون القياسية (Pure Python Standard Library).

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
┌────────────────────────┐
│  Source PDF (MS Word)  │
└───────────┬────────────┘
            │
            ▼ (pdf_extractor.py via fitz)
   Raw PDF Spans & Fonts (Subset stripped: ^[A-Z]{6}\+)
            │
            ▼ (Tunable Heuristic Policies: profile.json)
┌────────────────────────┐
│  Canonical Document IR │ <─── intermediate representation
└───────────┬────────────┘
            │
            ▼ (rtf_emitter.py - Pure Python RTF 1.9)
┌────────────────────────┐
│      RTF Document      │
└───────────┬────────────┘
            │
            ▼ (rtf_parser.py - Pure Python RTF AST)
┌────────────────────────┐
│   Reconstructed IR     │
└───────────┬────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│     Multi-Gate Equivalence Verifier (equivalence.py)    │
│  - Gate 1: Text Integrity (Exact Token Match == 1.0)    │
│  - Gate 2: Structure Integrity (Paragraphs, Blocks)     │
│  - Gate 2.5: Table Oracle (Precision / Recall on Grid)  │
│  - Gate 3: Formatting Integrity (Fonts, Sizes, Styles)  │
│  - Gate 4: Visual / Geometry Diff Oracle (Calibrated)   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Project Roadmap

- **Phase 1: Deterministic Foundation & Invariant Core** [COMPLETED]
  - Document IR, RTF Emitter, RTF Parser, PDF Extractor (subset prefix stripping).
- **Phase 2: Equivalence Oracles & Multi-Gate Verifier** [ACTIVE]
  - Multi-gate verification engine with cascading penalty.
- **Phase 3: Evolab Integration & Heuristic Policy Tuning**
  - Parameterized `profile.json`, `PDF2RTFAdapter` extending `DomainAdapter`, MAP-Elites calibration.
- **Phase 4: Hardening & Golden Corpus Benchmark**
  - Pre-registered Word-generated PDF test corpus, regression gates.
- **Phase 5: Self-Contained CLI & Open-Source Packaging**
  - Standalone CLI `pdf2rtf convert`, benchmark harness, and packaging.
