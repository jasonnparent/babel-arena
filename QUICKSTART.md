# Babel Arena - Quick Start Guide

## What You Have

Five core files for analyzing emergent LLM communication protocols:

1. **babel_arena_v2.py** - Core framework (baseline encoder + MI + train/test split)
2. **babel_disentanglement.py** - Representation disentanglement analysis
3. **babel_integrated_v3.py** - Complete end-to-end pipeline (recommended)
4. **README.md** - Full documentation
5. **ANALYSIS_GUIDE.md** - Detailed interpretation guide

## Installation

### Prerequisites

- Python 3.8+
- Ollama (https://ollama.ai)
- Two LLMs installed via Ollama

### Setup

```bash
# Install required Python packages
pip install numpy pandas scipy scikit-learn

# Pull LLMs (if not already installed)
ollama pull nemotron3:33b
ollama pull llama3.2
```

## Running the Analysis

### Full Pipeline (Recommended)

```bash
python babel_integrated_v3.py
```

This runs:
- Warmup phase (calibration)
- Baseline measurements
- Representation disentanglement analysis
- Adversarial robustness testing

**Output:** Three CSV files
- `babel_master_results.csv` - All trial metrics
- `disentanglement_scores.csv` - Property-token mappings
- `adversarial_robustness.csv` - Perturbation test results

**Runtime:** ~3-4 minutes

### Individual Components

```bash
# Baseline + MI + train/test
python babel_arena_v2.py

# Disentanglement analysis only
python babel_disentanglement.py
```

## Configuration

Edit CONFIG in any Python file to customize:

```python
CONFIG = {
    "strong_model": "nemotron3:33b",  # Encoder
    "weak_model": "llama3.2",          # Decoder
}
```

## Understanding Results

### Key Metrics (from babel_master_results.csv)

- **error**: Reconstruction error (lower is better)
- **baseline_error**: Deterministic encoder error (comparison point)
- **mutual_information**: Signal vs noise (0-1, higher = more signal)
- **token_entropy**: Vocabulary usage (0-1, higher = better use)

### Interpretation Quick Reference

**Strong Learning:**
- error < baseline_error by >5%
- mutual_information > 0.35
- disentanglement_score > 0.4

**Weak Learning/Memorization:**
- error ≈ baseline_error (within 3%)
- mutual_information < 0.25
- disentanglement_score < 0.2

**No Learning:**
- error > baseline_error
- mutual_information < 0.15
- disentanglement_score < 0.1

## Push to GitHub

```bash
# Initialize git (if starting fresh)
git init
git add .
git commit -m "Initial commit: Babel Arena framework"

# Add remote and push
git remote add origin https://github.com/YOUR-USERNAME/babel-arena.git
git branch -M main
git push -u origin main
```

## File Descriptions

### babel_integrated_v3.py

Complete analysis pipeline combining all three approaches. Recommended entry point.

**Classes:**
- RepresentationDisentangler: Extract geometric properties
- AdversarialRobustnessTester: Perturbation testing
- BabelArena: Core communication protocol framework

### babel_arena_v2.py

Baseline encoder comparison + mutual information analysis.

**Classes:**
- DeterministicEncoder: PCA + quantization + hashing
- MutualInformationAnalyzer: Signal detection
- ExampleSplitter: Train/test separation

### babel_disentanglement.py

Deep representation analysis - maps geometry properties to tokens.

**Extracts:**
- Cluster separation
- Symmetry scores
- Outlier fractions
- Density uniformity
- Topology classification

## Troubleshooting

### "Model not found" error

Install the model:
```bash
ollama pull nemotron3:33b
ollama pull llama3.2
```

### Ollama connection refused

Start Ollama:
```bash
ollama serve
```

(Run in a separate terminal)

### Low performance / no valid trials

- Increase warmup rounds in code (default: 5)
- Use stronger models (e.g., mistral, neural-chat)
- Lower vocab_size (default: 8) to make task easier

## Next Steps

1. Run full pipeline: `python babel_integrated_v3.py`
2. Examine CSV outputs
3. Read ANALYSIS_GUIDE.md for detailed interpretation
4. Modify CONFIG to test different model pairs
5. Push results to GitHub

## Support

See README.md for full documentation and interpretation guide.

---

Created with Raccoon AI (https://raccoonai.tech)
