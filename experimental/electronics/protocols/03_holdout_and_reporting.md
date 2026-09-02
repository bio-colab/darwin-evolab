# Holdout والتقارير

- الرقمي: الزوايا المنشورة 4.5 V/25 °C و4.5 V/85 °C و6.0 V/25 °C. لا تُضبط الأرقام على زاوية الـ holdout.
- التماثلي: `passed_holdout` فيزيائي فقط إذا `tool_used=ngspice`. الـ fallback يمنع `physical_claim`.
- كل تقرير تجربة يذكر `tool_used`. لا تُكتب موجة إن لم يوجد `.tran`.
