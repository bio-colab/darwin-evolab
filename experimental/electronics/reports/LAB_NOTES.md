# سجل تجارب المسار الإلكتروني — استئناف لاحق

آخر تحديث: 2026-08-29. ليس دليل تسويق. أرقام من تشغيل حي.

## أين نقف

ثبتت سلسلة فيزيائية:

`R,C → ngspice .tran → TransientArtifact → measure_waveform → لياقة مستهدفة`

لم يثبت تفوق GA على العشوائي. اللياقة القديمة (عتبة 100 Hz / 3 V) كانت سهلة فأخفت ذلك.

## الملفات

| الملف | الدور |
| :--- | :--- |
| `circuits/timer_555_astable/astable_tran.cir` | هزاز BJT لـ `.tran` (نتليست 555 القديم لا يتذبذب) |
| `evaluators/tran_evaluator.py` | `Timer555TranEvaluator` (ngspice) و`Synthetic555Evaluator` (قانون 555 + موجة نظيفة) |
| `instruments/oscilloscope.py` | قياس + `cycles_used` + `frequency_confidence` |
| `models/ngspice_bridge.py` | `run_transient_file` / `parse_tran_table` (جداول متعددة الصفحات، حقن SI بلا وحدة قديمة) |
| `experiments/run_555_tran_e2e.py` | تشغيل GA صغير؛ يقرأ `EVOLAB_NGSPICE` |
| `reports/555_tran_e2e.json` | أول E2E (رقم 100 مضلّل) |
| `reports/555_ga_audit.json` | الفائز كان الفرد المزروع 5k/5k/50n |
| `reports/555_targeted_compare.json` | لياقة مستهدفة + مسح C1 + GA مقابل عشوائي (شريحة) |
| `reports/555_feasibility.json` | ثقة القياس + قابلية الهدف على قانون 555 |

الثنائي: ngspice 44.2. لا يُنفَّذ من `artifacts/` (fuse). ضعه في `/opt/ngspice-root/usr/bin/ngspice` أو `/tmp/...` ومرّر المسار. الحزمة: `http://ftp.debian.org/debian/pool/main/n/ngspice/ngspice_44.2+ds-1_amd64.deb`

## اللياقة المعتمدة الآن (لا ترجع للعتبة)

هدف معلن: 1000 Hz ±1٪، 5 Vpp ±2٪، duty 50٪ ±2٪.

درجة = قرب لوغاريتمي للتردد × قرب Vpp × قرب duty × جودة تذبذب × `max(confidence, 0.05)`.

الثقة من عدد الدورات والـ jitter. لا تُعدَّل لتجميل GA.

## نتائج يجب عدم نسيانها

1. `10k/10k/100n` → ~13.8 Hz، Vpp ~4.97، crossings 5، ~21k عينة، ngspice حقيقي، درجة عتبة قديمة ~61.
2. `5k/5k/50n` → ~1711 Hz، Vpp ~5.72، duty ~0.22، crossings 22. كانت 100 على العتبة الرخوة، وصارت **0.055** على الهدف.
3. تدقيق GA pop=4 gens=3: الـ100 جاء من تقييم الفرد الرابع المزروع لا من بحث. نفس المرشح لكل البذور. عشوائي بنفس 12 تقييماً وصل 100 أيضاً.
4. بعد تصحيح اللياقة (3 بذور × 12 تقييماً، تهيئة عشوائية): وسيط GA **0.63** مقابل عشوائي **6.48**. لا تفوق.
5. مسح C1 عند R=5k غير رتيب (30n → 80 kHz مقاس، 200n → 66 Hz / duty 0.54). جزء منه فيزياء، جزء أثر قصير.
6. موجة مربعة 1 kHz اصطناعية: 18 دورة، ثقة 1.0.
7. قانون 555: `duty = (R1+R2)/(R1+2R2)` ≥ **0.51** داخل حدود 2k–50k. هدف 50٪±2٪ شبه غير قابل. 80 عيّنة اصطناعية: أفضل 15.8، صفر فوق 20.

## معياران لا تخلطهما

- **A — Synthetic555Evaluator:** هل البحث يعمل على موجة نظيفة؟
- **B — Timer555TranEvaluator:** هل يعمل على أثر ngspice المشوّش؟

إن تفوّق GA على A وفشل على B فالمشكلة قياس/فيزياء لا المحرك وحده.

## بروتوكول التكملة (بهذا الترتيب)

1. إمّا تخفيف duty المستهدف إلى 0.60–0.67 (555 متوازن) أو الإبقاء على 0.50 مع الإعلان أن المسألة شبه مستحيلة.
2. Random N=100 على A ثم مدرج درجات. إن ندر >20 فميزانية 12 كانت عبثاً.
3. مسح قابلية أخشن (شبكة أو LHS) على A أولاً لمعرفة سقف الدرجة الممكن.
4. نفس الميزانية A: GA مقابل عشوائي، تهيئة عشوائية مستقلة، بلا فرد مزروع.
5. نفس الميزانية على B إن ثبت مسار ngspice في الجلسة.
6. بعدها فقط: GA + `proposal.py` ثم مقترح لغوي فوق نفس المخطط.

لا تخلط تجربة استعادة حل مزروع مع بحث أعمى.

## أوامر استئناف

```bash
# ثنائي (مرة لكل بيئة)
dpkg-deb -x ngspice_44.2+ds-1_amd64.deb /opt/ngspice-root
export EVOLAB_NGSPICE=/opt/ngspice-root/usr/bin/ngspice

PYTHONPATH=src:. python -m experimental.electronics.experiments.run_555_tran_e2e
PYTHONPATH=src:. python -m pytest experimental/electronics/tests
```

**تحديث 2026-08-30:** مشكلة `analog_sizing` أُصلحت في الالتزام `7ac0e71` — صار `cli.py` يمرر `genome_size=len(pop[0].genome)` للمحرك بدل الافتراض. مُتحقق من الأمرين `analog_sizing` و`ptm180nm_opamp` على CLI بلا انهيار.

## حكم مختبر

فيزياء القياس: تعمل.  
تصميم التجربة القديمة: لا يصلح لإثبات تطور.  
الخطوة التالية ليست تحسين GA ولا LLM؛ هي قابلية الهدف ثم جودة التردد المقيس ثم مقارنة متساوية الميزانية.
