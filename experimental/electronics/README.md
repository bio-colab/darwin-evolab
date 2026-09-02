# مسار الإلكترونيات التجريبي

مختبر تقييم دوائر فوق محرك البحث. المحرك يختار مرشحاً؛ هذا المجلد يقيّمه. ليس محاكي سيليكون ولا طبقة عتاد.

الإصدار المحيط 0.5.0. النقطة الوحيدة مع النواة: `bridge.py` / `scenarios.py` و`--genome electronics`.

## ماذا يعمل

**رقمي 74HC.** كتالوج TI، نتليست لوحة، جدول حقيقة، تأخير من **نقاط ورقة منشورة** (لا استيفاء حراري):

- اسمي: 4.5 V / 25 °C
- بطيء: 4.5 V / 85 °C
- سريع: 6.0 V / 25 °C
- حمل: CL = 15 pF و 50 pF (`delay_ns_cl15` / `cl50`)

حاجز صلاحية كهربائية قبل المحاكاة (سائق→حمل، لا VCC/GND، لا سائق مزدوج).  
نصف جامع معروف (86+08): لياقة ≈ 90.6 وholdout ناجح وZ3 مكافئ إن وُجدت المكتبة.  
`exp1` تحقق من ذلك النتليست + طبقة صورية.  
الجيل الأول في مشهد `half_adder` يزرع بذرة من `functions_needed` عبر `seed_for_controller`؛ الباقي عشوائي.

**تماثلي.** `AnalogSizingEvaluator` و`CircuitConfigEvaluator` على `circuits/`. ngspice إن وُجد في PATH؛ وإلا `analytical_fallback`. Oracle: `physical_claim` فقط مع ngspice. بلا ngspice `passed_holdout=False` حتى لو ارتفعت الدرجة.

**أثر زمني.** `run_transient_file` → `TransientArtifact` → `measure_transient`. بلا ngspice الأثر فارغ؛ لا تُخترع موجة. `measure_transient` يقيس على الأساس الزمني الحقيقي (خطوات ngspice تكيفية: استيفاء خطي عند العبورات، واجب وزن-زمني) — لا يفترض شبكة عينات منتظمة. `parse_tran_table` يفصل صفحات أعمدة `.print` المتناوبة حسب توقيع عنوانها؛ دمج كل الصفوف المتساوية العرض يخلط إشارات مختلفة في عمود واحد.

**عقد تشغيل.** JSON عبر `spec_schema.parse_request` فقط. النص الحر يُرفض.

## الأرشيف التراكمي (موصول بمسار التشغيل)

`archive.py` موصول الآن في `prepare_electronics_run` — كل المشاهد (CLI `run.py evolve`، الاختبارات، الـprobes) تمر منه. كل تقييم يخزَّن في `data/archive.db` (append-only: fingerprint, scenario, fitness, passed_holdout, tool_used)، والجينوم الذي قُيِّم سابقاً لنفس السيناريو يُخدَم من الأرشيف بدل إعادة التقييم.

**مفاتيح العزل:** مفتاح السيناريو يتضمن بصمة مواصفة المقيّم (`half_adder:spec-3f57a2a223`) — تغيير المواصفة (حدود أسرع، أهداف مختلفة، جدول حقيقة جديد) يفتح مفتاحاً جديداً فلا يقرأ الكاش صفوفاً قديمة من مواصفة مغايرة. أعمدة الإثبات (`passed_holdout`/`tool_used`) تُسجَّل عبر `EvidenceCapture` فلا يضيع طبقة الدليل (الخطر T2)، و`lookup` يفضّل أعلى طبقة أداة دائماً.

**العقد:** المحرك يستخدم `__call__` (الواعي بالكاش)؛ `.evaluate()` يظل تقييماً خاماً كاملاً بالـartifacts للفحص والاختبارات — الكاش يخدم مسار اللياقة فقط.

**مقابض البيئة:**
| المتغير | الافتراضي | المعنى |
|---|---|---|
| `EVOLAB_ARCHIVE` | `1` | `0` يعيد المقيّم الخام بلا أرشفة |
| `EVOLAB_ARCHIVE_DB` | `data/archive.db` | مسار sqlite بديل (للعزل) |
| `EVOLAB_ARCHIVE_MIN_RANK` | `0` | أدنى طبقة تُخدَم من الكاش (`2` = مؤكَّد بـngspice فقط) |
| `EVOLAB_ARCHIVE_SEED` | `0` | عدد النخب المحقونة في ذيل الجماعة (الفتحة 0 — بذرة المشهد — لا تُمسّ، والمكررات تُستبعد) |

**قياس فعلي:** `run.py evolve --genome electronics --scenario half_adder -g 4 -p 6 --seed 7` مرتان: الأولى 7 إصابات/17 إعادة تقييم، الثانية **24 إصابة/0 إعادة تقييم** بنفس النتيجة 90.5778 — وCLI يطبع سطر `Archive:` بالإحصاءات.

اختبارات الربط: `tests/test_archive_wiring.py` (6 اختبارات).

## ماذا لا يعمل / لا يُدّعى

- لا عتاد، لا PEX، لا FPU.
- ~~`--scenario analog_sizing` من CLI ينهار اليوم~~ **أُصلح** في الالتزام `7ac0e71`: صار `cli.py` يشتق `genome_size` من طول جينوم السكان (`len(pop[0].genome)`) بدل افتراض المحرك. مُتحقق منه 2026-08-30: `--scenario analog_sizing -g 4 -p 6` و`--scenario ptm180nm_opamp -g 2 -p 4` يكملان بلا أخطاء (PASS/TARGET_MISSED).
- البحث العشوائي لا يكتشف half adder من صفر خلال أجيال قليلة.
- الـ fallback ليس قياساً مخبرياً.
- ngspice بلا مقاييس قابلة للتحليل يعيد `tool_used=ngspice_no_metrics` وأصفاراً — لا أرقام افتراضية تُلقَّب بقياسات.
- `CircuitConfigEvaluator` يوسم كل قيمة بمصدرها في `metric_source` (`measured_ngspice` / `analytical_model` / `datasheet_formula` / `model_estimate`)، والمفاتيح غير القابلة للاشتقاق تبقى في `unmeasured_specs` بعقوبة لياقة — بلا "تحقيق مواصفة بالتخمين".

## التشغيل

من جذر المستودع:

```bash
PYTHONPATH=src:. python run.py evolve --genome electronics --scenario half_adder -g 4 -p 6
PYTHONPATH=src:. python -m pytest experimental/electronics/tests
PYTHONPATH=src:. python -c "from experimental.electronics.experiments.exp1_datasheet_74xx_synthesis import run_half_adder_synthesis_lab; print(run_half_adder_synthesis_lab()['fitness_score'])"
```

المشاهد: `half_adder`, `full_adder`, `analog_sizing` (API لا CLI حتى يُصلح الطول)، ودوائر `circuits/` (`bjt_ce_amp`, `chargepump`, `comparator_simple`, `ptm180nm_opamp`, `timer_555_astable`).

## التخطيط

```
bridge.py / scenarios.py   باب التشغيل
oracle.py                  اتفاق أدوات + امتثال مواصفة + منع ادّعاء الـ fallback
spec_schema.py             مخطط طلب المستخدم
proposal.py                اقتراح محدود بلا LLM
instruments/               راسم مثالي على أثر جاهز
models/                    نتليست، صلاحية، Z3 اختياري، جسر ngspice
evaluators/                رقمي + تماثلي
components/                كتالوج 74HC + أرشيف مواصفات
circuits/                  نتليست + config
experiments/               exp1 تحقق، exp2 مقارنة، capability_probe
tests/                     40 اختبار انحدار
```

مرجع المحاكاة: `models/ngspice_bridge.py`. `simulators/` إعادة تصدير.

## العقود

- الرقمي: `CircuitNetlistGenome` + `FitnessResult`.
- التحجيم: `FloatGenome`.
- الأداة والخطأ و`physical_claim` في `artifacts`.

سجل تجارب الهزاز و`.tran` ومقارنة البحث: `reports/LAB_NOTES.md`.

## حالة تجربة NE555 (مغلقة)

**الحالة:** 🟢 مغلقة - معتمدة كـ Benchmark من المستوى A (هجين)

**الخلاصة:** تم اعتماد النهج الهجين (Level A) لتقييم دوائر NE555:
- التردد ودورة العمل: معادلات تحليلية من Datasheet
- الجهد: تحقق عبر ngspice باستخدام rc_sanity.cir
- physical_claim: False (صريح)
- benchmark_level: "A_hybrid_fast"

السبب: فشل النموذج السلوكي في ngspice-39 بسبب عدم دعم دوال if()/limit() في الحلقات المغلقة.

راجع التقرير الكامل: [reports/555_final_closure_report.md](reports/555_final_closure_report.md)

## تحديث: إعادة فتح المستوى B (النموذج السلوكي يعمل فعلاً)

التشخيص أعلاه كان خاطئاً. ngspice-44.2 يدعم الحلقات المغلقة جيداً؛ السلوكي كان معطلاً بعيوب توصيف وتحليل مستقلة عن المحرك، أُصلحت كلها والتحقق يغطيها:

**عيوب الدائرة `astable_behavioral.cir` (أصلية):**
1. `Rpullup_set`/`Rpullup_reset` موصولتان بالأرضي `0` بدل `vcc` — مخرجات المقارنات لا ترتفع أبداً فيعلق مزلاج SR بحالة Q=HIGH للأبد (صفر عبورات على كامل `.tran`).
2. `Qdisch 7 0 discharge_ctrl 0` — ترتيب أطراف BJT خاطئ (القاعدة على الأرضي) فلن يتوصّل التفريغ حتى لو صح المزلاج.
3. مفتاحا المقارنات (SW بـ Ron=1Ω↔1Meg) يسببان `Timestep too small` عند عبور العتبة — استُبدلا بمصادر سلوكية سيغمويدية متصلة `Bcmp1/Bcmp2` (لا قفزة موصيلة في المصفوفة) مع `.options method=gear`.
4. `.tran 0.01u 10m` مع 7 إشارات مطبوعة ≈ 161MB من stdout لكل تقييم — خُفّضت إلى `.tran 5u 50m` (نافذة 50ms تكشف 5 دورات عند هدف 100Hz).

**عيوب القياس:**
5. ngspice يطبع جداول `.print` العريضة كصفحات أعمدة متناوبة (61 صفاً من v(2)/v(out)/v(q) ثم v(q_bar)/v(7)/v(set) لنفس الأزمنة)؛ المحلّل القديم كان يدمج كل الصفوف المتساوية العرض فيتلوث عمود v(out) بعينات v(7) — أصلح `parse_tran_table` بالتقسيم حسب توقيع العنوان.
6. `measure_transient` كان يقدّر `fs` من أول 31 فترة ثم يعامل الموجة كشبكة منتظمة؛ مع الخطوات التكيفية هذا ضخّم التردد ×1.6 وحرّف الواجب — أُعيد كتابته بوعي زمني (استيفاء خطي، واجب وزن-زمني).
7. المقيّم كان يقرأ أول إشارة مطبوعة (عقدة المكثف `v(2)`) بدل الخرج — أصبح `signal="v(out)"` صريحاً.

**التحقق بعد الإصلاح** (مقيّم `Timer555TrueTransientEvaluator` حقيقي، physical_claim=True):
- 10k/10k/100n: مقاس 476.46Hz مقابل صيغة Datasheet 480Hz (خطأ 0.74%)، واجب 0.676 مقابل 0.667، ثقة 0.9996.
- هدف 100Hz (2k/50k/141n): مقاس 99.09Hz (خطأ 1.03%)، درجة 0.1686 — تدرّج لياقة حقيقي.
- 2k/2k/20n: مقاس 12.5kHz مقابل 12kHz، واجب 0.6677 (خطأ 0.16%).
- `run_555_tran_e2e.py` يكتمل الآن: GA كامل على قياس ngspice بأفضل لياقة 0.195 (كان كل تقييم يعيد 0.0 بعد مهلة 10s).
- زمن التقييم: ~0.3-1s (كان 24s مهلة فاشلة). اختبار انحدار جديد: `tests/test_behavioral_555_oscillates.py`.

## الرخصة

MIT كالمستودع.
