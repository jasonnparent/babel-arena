# Babel Arena v3: Integrated Analysis Framework

## Overview

Three complementary analyses to distinguish **real learning** from **brittle memorization**:

```
INPUT GEOMETRY
      ↓
┌─────────────────────────────────────────────────┐
│                 BABEL ARENA v3                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  1️⃣  BASELINE COMPARISON                       │
│     Learned LLM vs Deterministic PCA/Hash      │
│     → Does learning beat geometry hashing?      │
│                                                 │
│  2️⃣  REPRESENTATION DISENTANGLEMENT            │
│     Which geometric properties → which tokens?  │
│     → Can we interpret the encoding?            │
│                                                 │
│  3️⃣  ADVERSARIAL ROBUSTNESS                    │
│     Graceful degradation vs Collapse?           │
│     → Is representation distributed or brittle?│
│                                                 │
└─────────────────────────────────────────────────┘
      ↓
OUTPUT: Real learning? Distributed? Interpretable?
```

---

## Component 1: Representation Disentanglement

### What It Measures

**Maps geometric properties to token patterns** across all trials.

| Geometric Property | What It Captures | Example Mapping |
|---|---|---|
| **Cluster Separation** | How distinct are point groups? | Low→TOKEN_01, High→TOKEN_03 |
| **Symmetry Score** | How radially symmetric? | Asymmetric→TOKEN_02, Symmetric→TOKEN_05 |
| **Outlier Fraction** | % of points > 2σ from center | Few outliers→TOKEN_04, Many→TOKEN_07 |
| **Density Uniformity** | How evenly distributed? | Clustered→TOKEN_06, Uniform→TOKEN_00 |
| **Topology Class** | Shape type (blob/clusters/sparse) | Blob→TOKEN_02, Clusters→TOKEN_04 |

### Key Metrics

```python
Consistency:  How focused is token distribution on top tokens?
              High (0.8+) = strong property→token mapping
              Low (<0.3) = random token allocation

Separation:   How different are tokens across property bins?
              High (0.7+) = different properties use different tokens
              Low (<0.3) = all properties scrambled together

Disentanglement Score = Consistency × Separation
                        0.0-0.1: No structure (pure noise)
                        0.1-0.3: Weak structure (mostly memorization)
                        0.3-0.6: Moderate structure (some learning)
                        0.6+:    Strong structure (real encoding)
```

### Interpretation

**✅ STRONG DISENTANGLEMENT (score > 0.4):**
- Tokens reliably encode specific geometric properties
- Same property consistently maps to same token(s)
- Protocol is interpretable: "TOKEN_02 = high cluster separation"
- → **Evidence of real learned structure**

**⚠️ WEAK DISENTANGLEMENT (score 0.2-0.4):**
- Some consistent patterns, but mostly noise
- Token allocation partially driven by geometry, partially random
- → **Suggests superficial learning or memorization**

**❌ NO DISENTANGLEMENT (score < 0.2):**
- Tokens are arbitrary across trials
- No correlation between geometry and token use
- → **Pure noise: tokens encode nothing about geometry**

---

## Component 2: Adversarial Robustness Testing

### What It Tests

**How does the protocol degrade under perturbation?**

#### Perturbation Types

```
1. TOKEN SCRAMBLING
   Swap positions of tokens in message
   Low severity: shuffle 10% of positions
   High severity: shuffle 70% of positions
   → Tests: How sensitive is decoding to token order?

2. GEOMETRY NOISE
   Add Gaussian noise to point coordinates
   Low severity: σ = 1
   High severity: σ = 7
   → Tests: Encoder robustness to input corruption

3. OUTLIER INJECTION
   Replace points with random far-away coordinates
   Low severity: 10% of points become outliers
   High severity: 40% replacement
   → Tests: Does encoder collapse on unexpected geometry?

4. SCALING DISTORTION
   Non-uniform scaling: multiply coordinates by random factors
   Low severity: ×(1 ± 0.1)
   High severity: ×(1 ± 0.5)
   → Tests: Is encoding invariant or fragile?
```

### Failure Modes

```
ERROR AMPLIFICATION = Perturbed Error / Baseline Error

< 1.1x : ROBUST
        Protocol largely unaffected
        Error increases < 10% even at high severity
        → Strong distributed representation

1.1x - 2.0x : GRACEFUL DEGRADATION
              Error scales proportionally with perturbation
              No sudden jumps or non-linearities
              → Represents partially distributed encoding

> 2.0x : CATASTROPHIC COLLAPSE
         Small perturbation → large error spike
         Suggests brittle shortcut learning
         → Single-mode or memorized solution
```

### Interpretation

**✅ GRACEFUL DEGRADATION:**
- Error increases smoothly: 0.1 severity → 1.1x error, 0.5 severity → 1.5x error
- Protocol continues to function at high perturbations
- → **Distributed representation, not memorized**

**❌ CATASTROPHIC COLLAPSE:**
- Linear scaling breaks down: 0.3 severity → 1.05x error, 0.4 severity → 3.2x error
- Small changes trigger massive failures
- → **Brittle shortcut: probably memorized specific token→geometry pairs**

---

## Component 3: Baseline Comparison

### What It Measures

Each trial computes two errors:
1. **Learned Error**: LLM encodes → LLM decodes
2. **Baseline Error**: Deterministic PCA+hash encodes → Deterministic decodes

### Interpretation

```
Improvement = Baseline Error - Learned Error

CASE 1: Improvement > 0.05 (5% gain)
        ✅ LLMs beat deterministic geometry hashing
        → Learning is real

CASE 2: Improvement ≈ 0 (within noise)
        ⚠️ Parity: LLMs match or slightly beat hashing
        → Learning is weak or purely memorization

CASE 3: Improvement < -0.05 (LLM worse)
        ❌ Deterministic baseline superior
        → "Learning" is actually harmful distortion
```

**Why this matters:**
- Baseline encodes pure geometry (PCA) with no LLM bias
- If LLMs can't beat it, they're not learning structure
- If LLMs beat it + high MI + graceful degradation → real learning

---

## Reading the Output Files

### `babel_master_results.csv`

```
phase, error, baseline_error, mutual_information, token_entropy, valid
WARMUP, 0.145, 0.203, 0.34, 0.62, True
BASELINE, 0.121, 0.198, 0.41, 0.58, True
...
```

**Key columns:**
- `error - baseline_error` > 0 → LLM advantage
- `mutual_information` > 0.3 → tokens carry structure
- `token_entropy` > 0.5 → vocab well-used

### `disentanglement_scores.csv`

```
property, consistency, separation, disentanglement_score, n_bins
cluster_separation, 0.64, 0.58, 0.371, 4
symmetry_score, 0.51, 0.42, 0.214, 3
outlier_fraction, 0.58, 0.65, 0.377, 3
...
```

**Interpretation:**
- High scores: interpretable encoding
- Low scores: random token allocation

### `adversarial_robustness.csv`

```
perturbation_type, severity, error_amplification, failure_mode
token_scrambling, 0.1, 1.05, robust
token_scrambling, 0.5, 1.32, graceful
noise, 0.3, 1.15, graceful
noise, 0.7, 2.8, catastrophic
outlier_injection, 0.2, 1.22, graceful
...
```

**What to look for:**
- Linear growth in amplification → graceful (✅)
- Sudden jumps → catastrophic (❌)
- Max amplification < 1.5 across all perturbations → robust (✅)

---

## Complete Diagnostic Matrix

| Signal | Strong Learning | Weak/Memorization | Pure Noise |
|---|---|---|---|
| **Baseline Comparison** | LLM beats by >5% | Parity | Worse |
| **Mutual Information** | MI > 0.35, corr < -0.5 | 0.2 < MI < 0.3 | MI < 0.15 |
| **Disentanglement** | Score > 0.4 | 0.2-0.3 | < 0.1 |
| **Robustness** | Graceful + linear | Some graceful, some jumps | Catastrophic |
| **Token Entropy** | > 0.6 (vocab used) | 0.3-0.5 (vocab underused) | < 0.2 (stuck) |

---

## How to Run

```bash
# Full integrated analysis
python babel_integrated_v3.py

# Or individual components:
python babel_arena_v2.py        # Baseline + MI + train/test
python babel_disentanglement.py # Deep disentanglement analysis
```

## Expected Runtime

- Warmup: ~30s (8 trials)
- Baseline: ~60s (16 trials)
- Disentanglement: Automated on collected trials
- Adversarial: ~90s (27 perturbation tests)
- **Total: ~3-4 minutes**

---

## Interpretation Workflow

1. **Run full analysis** → generates 3 CSV files
2. **Check baseline comparison**
   - If LLM < baseline: abandon, tokens are noise
   - If LLM ≈ baseline: proceed to disentanglement
   - If LLM > baseline: strong signal, check robustness
3. **Examine disentanglement scores**
   - Score > 0.4? Properties encode consistently
   - Score < 0.2? Token allocation is random
4. **Inspect adversarial results**
   - Max amplification < 2x? Distributed representation
   - Max > 3x? Brittle shortcut learning
5. **Cross-check with MI correlation**
   - Corr < -0.5 + high MI? Tokens encode geometry
   - Corr > 0 + low MI? Noise alignment

---

## Example Conclusion Paths

### Scenario A: Strong Learning ✅

```
✓ LLM beats baseline by 8%
✓ Disentanglement: 0.52 (cluster_sep), 0.48 (symmetry), 0.44 (topology)
✓ MI correlation: -0.68 (high MI → low error)
✓ Robustness: max amplification 1.35x (graceful)

→ CONCLUSION: Real emergent structure
   Tokens encode interpretable geometric properties
   Protocol is robust and generalizable
```

### Scenario B: Memorization ⚠️

```
✓ LLM beats baseline by 2% (marginal)
⚠ Disentanglement: 0.18 (cluster_sep), 0.12 (symmetry), 0.08 (topology)
⚠ MI correlation: 0.05 (independent of error)
✗ Robustness: max amplification 4.2x (catastrophic on token scrambling)

→ CONCLUSION: Brittle memorization
   Tokens are arbitrary; small perturbations cause collapse
   Protocol does not learn generalizable structure
```

### Scenario C: Noise ❌

```
✗ LLM worse than baseline by 3%
✗ Disentanglement: 0.05 (all properties)
✗ MI correlation: 0.01 (random)
✗ Robustness: max amplification 6.1x (catastrophic)
✗ Token entropy: 0.15 (stuck on 1-2 tokens)

→ CONCLUSION: Pure noise
   LLMs failed to learn anything interpretable
   Tokens are decoupled from geometry
   Protocol is completely brittle
```

---

## Next Steps / Extensions

1. **Interpretability**: Extract rules from disentanglement
   - "TOKEN_02 + TOKEN_05 = high cluster separation + symmetric"
   
2. **Generalization testing**: Apply learned protocol to unseen geometry families
   - Train on blobs, test on line arrangements
   
3. **Information bottleneck**: Measure minimum vocab size needed
   - Can 4 tokens achieve same error as 8?

4. **Cross-model transfer**: Use strong model examples to teach weak model
   - Does disentanglement enable zero-shot transfer?
