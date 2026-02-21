# AlphaKnit 🧶 - v6.6-F The Blind Discovery Engine

AlphaKnit is a research-grade AI system that translates 3D point clouds into knitting / amigurumi crochet patterns. This version (v6.6-F) introduces **The Blind Discovery Engine (Falsifiable Edition)**, transforming the system into a robust scientific falsification machine.

## Architecture

```
3D Point Cloud (N×3)
        │
   PointNetEncoder          ← multi-scale: max-pool + avg-pool + Angular Positional Encoding
        │
   KnittingTransformer       ← Encoder-decoder with Sequential Factorized prediction heads
        │
   Discovery Engine (v6.6-F) ← Scientific Falsification (Causal Interventions, Null Controls)
        │
   Stitch Tuple Sequence     ← (type, p1_offset, p2_offset)
        │
   KnittingCompiler          ← Validates topology & builds stitch graph
```

## v6.6-F Scientific Features

- **Causal Falsification**: The `InterventionEngine` injects perturbations into hypothesized causal layers. Survival under random noise but failure under causal intervention validates the phenomenon.
- **Null Emergence Suite (Placebo Controls)**: Automated scientific controls using **Random Labels**, **Noise Inputs**, and **Geometry Nulls** (randomized weights/shuffled layers) to ensure discovery is not a representational artifact.
- **Observer Purity**: The `HiddenProbePool` rotates multiple orthogonal probe sets, combined with **Measurement Dropout**, to prevent the optimizer from "learning the instruments."
- **Failure Monitor**: Automated hypothesis rejection if null controls outperform real runs or if discovery fails the **Persistence Window** check.
- **Causal Reality Mapping**: New visualization script (`visualize_causal.py`) to trace hypothesis survivalCurves across training epochs.

## Watchtower Observatory Telemetry

- **Latent Phase Portraits**: Online PCA trajectory visualization of structural embeddings.
- **Phase Lag Monitoring**: Real-time optimizer alignment tracking to detect "Explosions of Choice".
- **Topology Tension Field (TTF)**: Passive bias encouraging structural organization through edge-density penalties.

## Project Structure

```
src/alphaknit/
├── scientific.py       # [NEW] InterventionEngine, NullEmergenceSuite, HypothesisEngine
├── visualize_causal.py # [NEW] Causal Reality Mapping visualization
├── research.py         # Phase Lag, Latent Portraits, HiddenProbePool
├── metrics.py          # Logit Margin, TTF Loss, PIB (Probe Influence Bounds)
├── model.py            # PointNetEncoder + Factorized KnittingTransformer
├── train.py            # Falsifiable Training Loop (Scientific Guards integration)
├── inference.py        # AlphaKnitPredictor — wraps model + compiler
├── compiler.py         # KnittingCompiler — validates stitch sequences
├── simulator.py        # ForwardSimulator — reconstruct mesh from stitch graph
├── tokenizer.py        # Vocabulary & Edge-Action tokenization
├── knitting_dataset.py # WebDataset-optimized loader
└── config.py           # Shared constants
```

## Installation

```bash
pip install -r requirements_pc.txt  # Optimized for local PC (CUDA-ready)
```

## Running

### Training with Scientific Controls

AlphaKnit v6.6-F supports multiple scientific control modes:

```powershell
# Standard Training (Real Mode)
.\run_pc.bat

# Scientific Control (Geometry Null)
$env:AK_NULL_MODE="geometry_null"; .\run_pc.bat
```

### Visualization and Verification

To visualize the causal survival curve after training:

```powershell
python src/alphaknit/visualize_causal.py --history checkpoints/training_history_v6.6.json
```

## Phase Strategy Evolution

| Version | Focus | Key Technology |
|---|---|---|
| **v4.0** | Stability | Selective Reset + Shock LR |
| **v5.0** | Automation | State-aware Curriculum (PhaseDetector) |
| **v6.0** | Research | Watchtower Observatory (Passive Telemetry) |
| **v6.6-F** | Falsification | Blind Discovery Engine (Causal Guards) |

## Tests

```bash
python -m pytest tests/
```
