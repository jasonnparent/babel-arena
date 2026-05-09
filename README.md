# Babel Arena

A rigorous experimental framework for analyzing emergent communication protocols between language models. Distinguishes real learned structure from brittle memorization through baseline comparison, mutual information analysis, representation disentanglement, and adversarial robustness testing.

## Overview

Babel Arena tests whether two LLMs can develop a meaningful communication protocol to transmit 3D point cloud geometry using a constrained vocabulary. The framework goes beyond success/failure metrics to answer the critical question:

**Is the learned communication real or just memorized shortcuts?**

## Core Analyses

### 1. Baseline Comparison

Compares learned LLM performance against a deterministic baseline encoder:
- Deterministic encoder uses PCA + quantization + geometric hashing
- No neural network, no learning—pure geometry compression
- If LLMs beat this baseline, learning is real
- If LLMs match or lose, communication is noise-aligned

**Output:** Improvement scores showing LLM advantage over geometry-only encoding

### 2. Mutual Information Analysis

Measures information flow between geometry and token patterns:
- High MI: tokens encode structural information about geometry
- Low MI: tokens are noise-aligned despite decent reconstruction
- Correlation analysis: Does MI predict reconstruction quality?

**Output:** Per-trial MI values, correlation coefficients indicating signal strength

### 3. Representation Disentanglement

Maps which geometric properties correspond to which token patterns:

**Properties extracted:**
- Cluster separation (how distinct are point groups?)
- Symmetry score (how radially balanced?)
- Outlier fraction (proportion of anomalous points)
- Density uniformity (evenly distributed vs clustered?)
- Topology class (blob vs multi-cluster vs sparse)

**Disentanglement Score = Consistency × Separation**
- 0.6+: Strong structure (interpretable encoding)
- 0.3-0.6: Moderate structure (partial learning)
- 0.1-0.3: Weak structure (mostly memorization)
- <0.1: No structure (pure noise)

**Interpretation:**
- High scores indicate same properties consistently use same tokens
- Different properties use distinct token sets
- Protocol is interpretable: "TOKEN_02 = high cluster separation"

### 4. Adversarial Robustness Testing

Systematically degrades protocol under perturbations:

**Perturbation types:**
- Token scrambling: shuffle message order
- Geometry noise: add Gaussian corruption
- Outlier injection: replace points with far-away coordinates
- Scaling distortion: non-uniform stretching

**Failure modes:**
- Robust (<1.1x amplification): protocol tolerates perturbations
- Graceful degradation (1.1-2.0x): error scales proportionally with severity
- Catastrophic collapse (>2.0x): small perturbations trigger massive failures

**Interpretation:**
- Graceful degradation indicates distributed representation
- Catastrophic collapse suggests brittle shortcut learning

## Train/Test Split

Strict separation prevents example leakage:
- 70% of examples used in prompts during trials
- 30% held out for validation
- No test examples appear in encoding/decoding prompts
- Prevents memorization of test set patterns

## File Structure

```
babel_arena_v2.py              Core framework with baseline, MI, train/test split
babel_disentanglement.py       Representation disentanglement analysis
babel_integrated_v3.py         Complete end-to-end pipeline
ANALYSIS_GUIDE.md              Comprehensive interpretation guide
```

## Output Files

Running `babel_integrated_v3.py` generates three CSV files:

### babel_master_results.csv
All trials with computed metrics:
- error: reconstruction error
- baseline_error: deterministic encoder error
- mutual_information: I(Geometry; Tokens)
- token_entropy: Shannon entropy of token distribution
- token_count: number of tokens used
- valid: successful reconstruction
- is_test: held-out example flag

### disentanglement_scores.csv
Per-property analysis:
- consistency: how focused token distribution is on top tokens
- separation: how different tokens are across property bins
- disentanglement_score: product (0-1 scale)
- n_bins: number of property value bins

### adversarial_robustness.csv
Perturbation test results:
- perturbation_type: token_scrambling, noise, outlier_injection, scaling
- severity: perturbation intensity (0-1)
- error_amplification: perturbed_error / baseline_error
- failure_mode: robust, graceful, catastrophic
- valid_reconstructions: number of successful trials

## Interpretation Guide

### Strong Learning

- LLM beats baseline by >5%
- Disentanglement score > 0.4
- MI correlation < -0.5 (high MI predicts low error)
- Max adversarial amplification < 1.5x

**Conclusion:** Real emergent structure, distributed representation

### Weak Learning / Memorization

- LLM beats baseline by <3% (marginal)
- Disentanglement score 0.1-0.3
- MI correlation near 0 (independent of error)
- Max amplification 2-4x (some graceful, some jumps)

**Conclusion:** Brittle memorization, poor generalization

### Pure Noise

- LLM worse than baseline
- Disentanglement score < 0.1
- MI correlation ~0
- Max amplification > 4x (catastrophic)

**Conclusion:** No learning occurred

## Usage

### Full Analysis

```bash
python babel_integrated_v3.py
```

Runs complete pipeline: warmup, baseline measurement, disentanglement analysis, adversarial robustness testing. Generates three CSV output files.

Expected runtime: 3-4 minutes

### Individual Components

```bash
# Baseline + MI + train/test split
python babel_arena_v2.py

# Disentanglement analysis only
python babel_disentanglement.py
```

## Configuration

Edit CONFIG dict in source files to customize:

```python
CONFIG = {
    "strong_model": "nemotron3:33b",  # Encoder model
    "weak_model": "llama3.2",          # Decoder model (swap phases)
}
```

Requires Ollama with specified models installed:

```bash
ollama pull nemotron3:33b
ollama pull llama3.2
```

## Key Classes

### RepresentationDisentangler
Extracts geometric properties and correlates with token patterns:
- `extract_properties()`: analyzes point cloud geometry
- `record_trial()`: maps trial to geometric bins
- `analyze_disentanglement()`: computes consistency/separation
- `generate_report()`: human-readable summary

### AdversarialRobustnessTester
Systematically perturbs protocol and measures degradation:
- `measure_baseline()`: clean performance
- `perturb_tokens()`: scramble message
- `perturb_geometry()`: noise/outliers/scaling
- `run_full_robustness_suite()`: complete test sequence

### ExampleSplitter
Enforces train/test separation:
- `add_example()`: auto-assigns to train/test
- `get_train_examples()`: returns only training pool
- `validate_separation()`: confirms no leakage

### DeterministicEncoder
Baseline geometry compression:
- `fit()`: learn PCA on training geometries
- `encode()`: project to PCA space, quantize, hash to tokens
- `decode()`: invert hash, inverse transform to 3D

## Geometric Properties

Each trial extracts:

- **cluster_count**: number of distinct clusters detected
- **cluster_separation**: inter-cluster distance / within-cluster spread
- **symmetry_score**: 0-1, how radially symmetric around centroid
- **outlier_fraction**: proportion of points > 2 sigma from mean
- **density_uniformity**: 0-1, uniformity of local point density
- **topology_class**: blob, clusters, line, or sparse
- **convex_hull_ratio**: volume occupied / convex hull volume
- **dimensionality**: intrinsic dimensionality (0-1 scale)

## References

Framework implements:
- Procrustes analysis for rigid alignment error
- Hierarchical clustering for property extraction
- Mutual information as signal-vs-noise detector
- Adversarial perturbation for robustness evaluation
- Disentanglement metrics from interpretability literature

## Future Directions

1. Cross-model transfer: use strong model examples to teach weak model
2. Generalization testing: train on blob geometry, test on line arrangements
3. Information bottleneck: measure minimum vocab needed
4. Rule extraction: derive human-interpretable encoding rules
5. Evolutionary analysis: track protocol emergence over communication rounds

## Author

Created with Raccoon AI (https://raccoonai.tech)

Co-Authored-By: ACE <ace@raccoonai.tech>
