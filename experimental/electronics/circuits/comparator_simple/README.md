# Comparator Simple - BJT Differential Pair

**TI LM339-inspired** 4-layer ground truth: datasheet Vos<5mV, tpd<1.3us + diff-pair formula + SPICE + holdout Vcc 4.5-5.5V.

**Design vars:** R1,R2 (output divider), RC (collector), Vref (2.5V nom). **Specs:** vout_high>3.5V, vout_low<1.0V (TTL compatible).

**Cost:** 0s via analytical fallback, ~0.1s with ngspice (BJT model QMOD, no PDK).
