# Security Policy

## Supported Versions

We release patches and security advisories for the following versions of **Darwin-Evolab**:

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| 0.6.x   | :white_check_mark: |
| < 0.6.0 | :x:                |

---

## Security Architecture & Invariants

Darwin-Evolab executes automated code modifications, program repair benchmarks, and synthesis loops. To protect host systems from adversarial or runaway candidate programs, the runtime implements:

1. **Hermetic Subprocess Sandboxing (`evolab.sandbox`)**:
   - Subprocess creation interception (`subprocess.Popen`, `subprocess.run`, `os.system`, `os.popen`).
   - Binary interface restriction (disabling untrusted dynamic loads via `ctypes.CDLL`).
   - Virtual filesystem jail preventing writes outside ephemeral scratch directories.
   - Resource execution quotas with bounded time limits (`timeout_seconds`) and CPU quotas.

2. **Cryptographic Provenance & Attestation (`evolab.provenance`)**:
   - Continuous runtime attestation reporting SHA-256 binary digests and source Merkle tree roots.
   - Tamper-evident evidence capsules (`.evobundle`) with HMAC-SHA256 and SHA-256 integrity seals.

---

## Reporting a Vulnerability

If you discover a security vulnerability, sandbox escape, or integrity flaw in Darwin-Evolab:

1. **Do NOT report security vulnerabilities through public GitHub issues or discussions.**
2. Send an email to the security response team at: **`security@evolab.dev`** or open a [Private Vulnerability Report](https://github.com/bio-colab/darwin-evolab/security/advisories/new) on GitHub.
3. Please include:
   - A detailed description of the vulnerability.
   - Minimal reproducible proof-of-concept (PoC) script or scenario.
   - Expected vs. actual behavior and potential security impact.

### Response Timeline
- **Initial Acknowledgment**: Within 48 hours.
- **Triage & Reproduction**: Within 5 business days.
- **Patch & Advisory Publication**: Coordinated with reporter under standard responsible disclosure policies.
