# التحقق

1. مواصفة: زوايا `published_corners` و`ElectricalLimits`.
2. مرجع منطقي: `IndependentDigitalVerifier`.
3. محاكاة: `BreadboardCircuit` رقمياً؛ ngspice إن وُجد تماثلياً. المسار: `models/ngspice_bridge.py`.
4. Holdout رقمي على 4.5 V/85 °C المنشور. تماثلياً لا ادّعاء بلا ngspice.
