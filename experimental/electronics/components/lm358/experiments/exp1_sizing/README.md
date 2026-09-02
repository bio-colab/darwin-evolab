# LM358 Exp1 - Sizing from Datasheet Ground Truth

**Source:** `../source/SLOS068AB_LM358B.pdf.txt` (SHA256 64C2626E... of the transcription — original TI SLOS068AB PDF not bundled).

**Targets (from extracted_specs.json experiment_targets):**
- GBW >=1.0 MHz (typ 1.2), Vos <=3.0mV, Iq <=460uA @5V, PM >=45deg, SR >=0.3 V/us

**Design vars:** from `circuits/ptm180nm_opamp/config.json` (19 vars) proxy for LM358.

**Protocol:** `protocols/01` + `02` + `03` - Oracle merges analytical and SPICE.

**Run:** `python -m experimental.electronics.components.lm358.experiments.exp1_sizing.run`
