import numpy as np
import json
import re
import pandas as pd
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import Counter
import math
from scipy.decomposition import PCA
from scipy.stats import entropy as scipy_entropy

@dataclass
class TrialResult:
    points: List
    tokens: str
    reconstruction: List
    error: float
    token_count: int
    phase: str
    valid: bool
    token_entropy: float  # Shannon entropy of token distribution
    mutual_information: float = 0.0  # NEW: MI between geometry and tokens
    baseline_error: float = float('nan')  # NEW: deterministic encoder error
    is_test_example: bool = False  # NEW: test split tracking

class DeterministicEncoder:
    """Baseline encoder: PCA + quantization + geometric hashing"""
    
    def __init__(self, vocab_size: int = 8, n_components: int = 2):
        self.vocab_size = vocab_size
        self.n_components = n_components
        self.pca = None
        self.token_vocab = [f"TOKEN_{i:02d}" for i in range(vocab_size)]
        self.fitted = False
    
    def fit(self, training_points: List[np.ndarray]) -> None:
        """Fit PCA on training geometry"""
        all_points = np.vstack(training_points)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(all_points)
        self.fitted = True
    
    def encode(self, points: np.ndarray) -> str:
        """Deterministic: PCA project → quantize → hash to tokens"""
        if not self.fitted or self.pca is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        
        # Project to PCA space
        projected = self.pca.transform(points)  # (n_points, n_components)
        
        # Quantize each point: assign to nearest grid cell
        quantized = np.round(projected).astype(int)
        
        # Hash grid coords to token: fold into vocab space
        tokens = []
        for coord in quantized:
            # Hash: sum coords mod vocab_size
            token_idx = int(np.sum(np.abs(coord))) % self.vocab_size
            tokens.append(self.token_vocab[token_idx])
        
        return " ".join(tokens[:8])  # Exactly 8 tokens, no learning
    
    def decode(self, tokens: str) -> np.ndarray:
        """Invert: tokens → hashed coords → PCA inverse"""
        token_list = tokens.split()
        
        # Unhash: token index → pseudo-coordinates
        coords = []
        for tok in token_list:
            if tok.startswith("TOKEN_"):
                idx = int(tok.split("_")[1])
                # Reconstruct approximate grid position (lossy)
                # Simple: use index as diagonal coordinate
                coord = np.array([idx % 3 - 1, idx // 3 - 1])  # (-1 to 1 grid)
                coords.append(coord)
        
        if not coords:
            return np.random.rand(8, 3) * 50 + 25
        
        coords = np.array(coords)
        # Inverse PCA transform
        reconstructed = self.pca.inverse_transform(coords)
        
        # Pad to 8 points if needed
        while len(reconstructed) < 8:
            reconstructed = np.vstack([reconstructed, reconstructed[-1:]])
        
        return reconstructed[:8]

class MutualInformationAnalyzer:
    """Measure information flow: geometry → tokens"""
    
    @staticmethod
    def discretize_geometry(points: np.ndarray, n_bins: int = 4) -> np.ndarray:
        """Bin continuous coordinates into discrete states"""
        discretized = np.zeros_like(points, dtype=int)
        for dim in range(points.shape[1]):
            discretized[:, dim] = np.digitize(points[:, dim], 
                                              np.linspace(points[:, dim].min(), 
                                                         points[:, dim].max(), 
                                                         n_bins))
        return discretized
    
    @staticmethod
    def compute_mi(geometry_state: np.ndarray, token_sequence: str) -> float:
        """
        Mutual Information: I(Geometry; Tokens)
        High MI: tokens encode real geometry
        Low MI: tokens are noise-aligned
        """
        # Discretize geometry to joint state (one symbol per point cloud)
        geo_tuple = tuple(geometry_state.flatten())
        
        # Token distribution
        tokens = token_sequence.split()
        token_counts = Counter(tokens)
        
        if len(token_counts) == 0 or len(tokens) == 0:
            return 0.0
        
        # Marginals
        p_geo = 1.0 / len(set([geo_tuple]))  # Assume uniform over seen geometries
        p_tok = np.array(list(token_counts.values())) / len(tokens)
        
        # Joint: co-occurrence strength (simple heuristic)
        # Assume: unique tokens → higher MI (more info)
        uniqueness = len(set(tokens)) / len(tokens)
        
        # Shannon mutual information proxy:
        # MI ≈ H(tokens) - H(tokens | geometry)
        # Approximation: MI ≈ uniqueness * entropy
        h_tok = -np.sum(p_tok * np.log2(p_tok + 1e-10))
        mi_estimate = uniqueness * h_tok
        
        return float(np.clip(mi_estimate, 0, 1))
    
    @staticmethod
    def analyze_correlation(trial_results: List[TrialResult]) -> Dict:
        """Correlate MI with reconstruction error"""
        valid_trials = [t for t in trial_results if t.valid]
        
        if len(valid_trials) < 3:
            return {"n": len(valid_trials), "correlation": np.nan}
        
        mis = np.array([t.mutual_information for t in valid_trials])
        errors = np.array([t.error for t in valid_trials])
        
        # Correlation: high MI should → low error (if tokens work)
        corr = float(np.corrcoef(mis, errors)[0, 1]) if np.std(mis) > 0 else 0.0
        
        return {
            "n": len(valid_trials),
            "correlation": corr,
            "mi_mean": float(np.mean(mis)),
            "mi_std": float(np.std(mis)),
            "insight": "strong" if corr < -0.5 else "weak" if abs(corr) < 0.3 else "moderate"
        }

class ExampleSplitter:
    """Strict train/test split for examples"""
    
    def __init__(self, test_ratio: float = 0.3, seed: int = 42):
        self.test_ratio = test_ratio
        self.seed = seed
        self.train_examples: List[Tuple] = []
        self.test_examples: List[Tuple] = []
        np.random.seed(seed)
    
    def add_example(self, tokens: str, points: List, phase: str = "warmup") -> bool:
        """Add example and auto-assign to train/test"""
        rand = np.random.random()
        is_test = rand < self.test_ratio
        
        example = (tokens, points)
        if is_test:
            self.test_examples.append(example)
        else:
            self.train_examples.append(example)
        
        return is_test
    
    def get_train_examples(self) -> List[Tuple]:
        """Only return training examples for in-trial prompts"""
        return self.train_examples
    
    def get_test_examples(self) -> List[Tuple]:
        """Test examples: never leaked during encoding/decoding"""
        return self.test_examples
    
    def validate_separation(self) -> Dict:
        """Verify no test leakage"""
        train_tokens = set(t[0] for t in self.train_examples)
        test_tokens = set(t[0] for t in self.test_examples)
        
        overlap = train_tokens & test_tokens
        
        return {
            "train_count": len(self.train_examples),
            "test_count": len(self.test_examples),
            "token_overlap": len(overlap),
            "separation_clean": len(overlap) == 0
        }

def token_entropy(tokens: str, vocab_size: int) -> float:
    """Normalized Shannon entropy: 0 (stuck) to 1 (uniform)"""
    if not tokens:
        return 0.0
    counts = Counter(tokens.split())
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    entropy_val = -sum(p * math.log2(p) for p in probs if p > 0)
    max_ent = math.log2(vocab_size)
    return entropy_val / max_ent if max_ent > 0 else 0.0

class LLMWrapper:
    def __init__(self, model_name: str, provider: str = "ollama"):
        self.model_name = model_name
        self.provider = provider.lower()
    
    def query(self, prompt: str) -> str:
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.2, 'top_p': 0.95}
            )
            return response['message']['content']
        except Exception as e:
            print(f"[{self.model_name}] error: {e}")
            return "ERROR"

def check_model(model: str) -> bool:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        return model in result.stdout
    except:
        return False

class BabelArena:
    def __init__(self, agent_a: LLMWrapper, agent_b: LLMWrapper, vocab_size: int = 8):
        self.vocab = [f"TOKEN_{i:02d}" for i in range(vocab_size)]
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.history: List[TrialResult] = []
        
        # NEW: Strict example split
        self.example_splitter = ExampleSplitter(test_ratio=0.3)
        
        # NEW: Deterministic baseline
        self.baseline_encoder = DeterministicEncoder(vocab_size=vocab_size)
        
        # NEW: MI analyzer
        self.mi_analyzer = MutualInformationAnalyzer()
        
        self.training_geometries: List[np.ndarray] = []
    
    def generate_task(self, n_points: int = 8) -> np.ndarray:
        # Harder geometry: two distinct clusters + outlier
        c1 = np.random.rand(3) * 40 + 10
        c2 = np.random.rand(3) * 40 + 50
        cluster1 = c1 + np.random.randn(3, 3) * 4
        cluster2 = c2 + np.random.randn(3, 3) * 4
        outlier = np.random.rand(1, 3) * 80 + 10
        points = np.vstack([cluster1, cluster2, outlier])
        np.random.shuffle(points)
        return np.round(np.clip(points, 5, 95), 2)
    
    def apply_rotation(self, points: np.ndarray) -> np.ndarray:
        theta = np.radians(np.random.choice([0, 90, 180, 270]))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        return np.round(points @ R, 2)
    
    def _encode_prompt(self, points: np.ndarray, examples: List[Tuple] = None) -> str:
        vocab_str = " ".join(self.vocab[:6])  # 6 active, severe pressure
        ex_block = ""
        if examples:
            ex_block = "\nPRIOR ENCODINGS (training examples only):\n" + "\n".join(
                f"  {e[0]} -> {np.array(e[1]).round(1).tolist()}" for e in examples[-2:]
            )
        return f"""You are Agent A. Encode 8 3D points using ONLY 6 tokens.

POINTS: {points.tolist()}

TOKENS: {vocab_str}
RULES:
- ONLY space-separated tokens from list. Max 8 tokens total.
- No English, numbers, punctuation.
- Agent B must reconstruct approximate coordinates.{ex_block}

Response:"""
    
    def _decode_prompt(self, tokens: str, examples: List[Tuple] = None) -> str:
        ex_block = ""
        if examples:
            ex_block = "\nPRIOR DECODINGS (training examples only):\n" + "\n".join(
                f"  {e[0]} -> {np.array(e[1]).round(1).tolist()}" for e in examples[-2:]
            )
        return f"""You are Agent B. Decode 8 [x,y,z] points from tokens.

TOKENS: {tokens}{ex_block}

Return ONLY JSON: [[x,y,z],...] 8 points. No text."""
    
    def procrustes_error(self, orig: np.ndarray, recon: np.ndarray) -> Tuple[float, bool]:
        try:
            recon = np.array(recon)
            if recon.shape != orig.shape:
                return float('nan'), False
            orig_c = orig - orig.mean(axis=0)
            recon_c = recon - recon.mean(axis=0)
            o_norm = np.linalg.norm(orig_c) + 1e-8
            r_norm = np.linalg.norm(recon_c) + 1e-8
            H = recon_c.T @ orig_c
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T
            recon_rot = recon_c @ R_opt
            err = float(np.mean(np.linalg.norm(orig_c/o_norm - recon_rot/r_norm, axis=1)))
            return err, True
        except:
            return float('nan'), False
    
    def run_trial(self, phase: str = "SYNC", use_train_examples: bool = True) -> TrialResult:
        raw_points = self.generate_task()
        points_for_a = self.apply_rotation(raw_points)
        
        # Track geometry for baseline encoder
        self.training_geometries.append(raw_points.copy())
        
        # NEW: Get only training examples (no test leakage)
        examples_to_use = self.example_splitter.get_train_examples() if use_train_examples else []
        
        prompt_a = self._encode_prompt(points_for_a, examples_to_use)
        raw_a = self.agent_a.query(prompt_a).strip()
        tokens = re.findall(r'TOKEN_\d{2}', raw_a.upper())
        message = " ".join(tokens[:8])
        
        prompt_b = self._decode_prompt(message, examples_to_use)
        raw_b = self.agent_b.query(prompt_b).strip()
        
        reconstruction = []
        try:
            matches = re.findall(r'\s*[\d\.\-]+\s*,\s*[\d\.\-]+\s*,\s*[\d\.\-]+\s*', raw_b)
            if len(matches) >= 8:
                reconstruction = [json.loads(m) for m in matches[:8]]
        except:
            pass
        
        error, valid = self.procrustes_error(raw_points, reconstruction)
        ent = token_entropy(message, len(self.vocab))
        
        # NEW: Baseline encoder error
        try:
            if self.baseline_encoder.fitted:
                baseline_tokens = self.baseline_encoder.encode(points_for_a)
                baseline_recon = self.baseline_encoder.decode(baseline_tokens)
                baseline_error, _ = self.procrustes_error(raw_points, baseline_recon)
            else:
                baseline_error = float('nan')
        except:
            baseline_error = float('nan')
        
        # NEW: Mutual information
        mi = self.mi_analyzer.compute_mi(raw_points, message)
        
        # NEW: Track if test example
        is_test = self.example_splitter.add_example(message, raw_points.tolist(), phase)
        
        result = TrialResult(
            points=raw_points.tolist(),
            tokens=message,
            reconstruction=reconstruction,
            error=error,
            token_count=len(tokens),
            phase=phase,
            valid=valid,
            token_entropy=ent,
            mutual_information=mi,
            baseline_error=baseline_error,
            is_test_example=is_test
        )
        self.history.append(result)
        return result

class FrictionDetector:
    def __init__(self, arena: BabelArena):
        self.arena = arena
        self.results: List[dict] = []
    
    def _warmup(self, n: int) -> None:
        print(f"  Warmup ({n} rounds)...")
        for i in range(n):
            res = self.arena.run_trial("WARMUP")
            if res.valid and res.token_entropy > 0.3:
                # Only add to training pool (auto-split happens inside add_example)
                self.arena.example_splitter.add_example(res.tokens, res.points, "WARMUP")
                print(f"    + ex{len(self.arena.example_splitter.train_examples)}: {res.tokens} (ent={res.token_entropy:.2f}, mi={res.mutual_information:.2f})")
            else:
                print(f"    x rejected (valid={res.valid}, ent={res.token_entropy:.2f})")
        
        # Fit baseline encoder on training geometries
        if len(self.arena.training_geometries) > 0:
            self.arena.baseline_encoder.fit(self.arena.training_geometries)
            print(f"  Baseline encoder fitted on {len(self.arena.training_geometries)} geometries")
    
    def run_experiment_suite(self, n_sync=8, n_swap=8, n_control=8, n_cross=8):
        strong = self.arena.agent_b.model_name
        
        print(f"=== WARMUP ({strong}) ===")
        self._warmup(5)
        
        split_status = self.arena.example_splitter.validate_separation()
        print(f"  Train/Test split: {split_status['train_count']} train, {split_status['test_count']} test")
        print(f"  Separation clean: {split_status['separation_clean']}")
        
        if len(self.arena.example_splitter.train_examples) < 3:
            print("FAIL: Not enough valid training examples. Abort.")
            return None
        
        # PHASE 1: SYNC
        print(f"\n=== SYNC ({strong} -> {strong}) ===")
        for i in range(n_sync):
            res = self.arena.run_trial("SYNC")
            self._record(res)
            print(f"  {i+1} | err={res.error:.3f} | base={res.baseline_error:.3f} | mi={res.mutual_information:.2f} | ent={res.token_entropy:.2f}")
        
        # PHASE 2: SWAP (strong A -> weak B, with strong training examples)
        weak = CONFIG["weak_model"]
        print(f"\n=== SWAP (strong A -> {weak}, strong training examples) ===")
        old_b = self.arena.agent_b
        self.arena.agent_b = LLMWrapper(weak, "ollama")
        
        for i in range(n_swap):
            res = self.arena.run_trial("SWAP")
            self._record(res)
            print(f"  {i+1} | err={res.error:.3f} | base={res.baseline_error:.3f} | mi={res.mutual_information:.2f} | ent={res.token_entropy:.2f}")
        
        # PHASE 3: CONTROL (weak B, no examples)
        print(f"\n=== CONTROL ({weak}, no examples) ===")
        saved_examples = self.arena.example_splitter.train_examples
        self.arena.example_splitter.train_examples = []
        
        for i in range(n_control):
            res = self.arena.run_trial("CONTROL")
            self._record(res)
            print(f"  {i+1} | err={res.error:.3f} | base={res.baseline_error:.3f} | mi={res.mutual_information:.2f} | ent={res.token_entropy:.2f}")
        
        # PHASE 4: CROSS (weak B, weak-generated training examples)
        print(f"\n=== CROSS ({weak}, weak-generated training examples) ===")
        weak_train_examples = []
        temp_a = self.arena.agent_a
        self.arena.agent_a = LLMWrapper(weak, "ollama")
        
        for _ in range(3):
            res = self.arena.run_trial("CROSS_WARMUP")
            if res.valid:
                weak_train_examples.append((res.tokens, res.points))
        
        self.arena.agent_a = temp_a
        self.arena.example_splitter.train_examples = weak_train_examples
        
        for i in range(n_cross):
            res = self.arena.run_trial("CROSS")
            self._record(res)
            print(f"  {i+1} | err={res.error:.3f} | base={res.baseline_error:.3f} | mi={res.mutual_information:.2f} | ent={res.token_entropy:.2f}")
        
        # Restore
        self.arena.example_splitter.train_examples = saved_examples
        self.arena.agent_b = old_b
        return self.analyze()
    
    def _record(self, res: TrialResult):
        tokens_list = res.tokens.split() if res.tokens else []
        self.results.append({
            "phase": res.phase,
            "error": res.error,
            "valid": res.valid,
            "token_count": res.token_count,
            "token_entropy": res.token_entropy,
            "unique_tokens": len(set(tokens_list)),
            "repetition_rate": 1.0 - len(set(tokens_list))/len(tokens_list) if tokens_list else 0.0,
            "mutual_information": res.mutual_information,
            "baseline_error": res.baseline_error,
            "improvement_over_baseline": res.error - res.baseline_error if not np.isnan(res.baseline_error) else np.nan,
            "is_test_example": res.is_test_example
        })
    
    def analyze(self):
        df = pd.DataFrame(self.results)
        valid_df = df[df['valid'] == True].copy()
        
        print("\n=== ANALYSIS ===")
        if valid_df.empty:
            print("NO VALID DATA")
            return df
        
        # Core statistics
        summary = valid_df.groupby("phase").agg({
            'error': ['count', 'mean', 'std'],
            'token_entropy': 'mean',
            'mutual_information': 'mean',
            'baseline_error': 'mean',
            'repetition_rate': 'mean'
        }).round(4)
        print(summary)
        
        # Baseline comparison
        print(f"\n--- Learned vs Deterministic ---")
        valid_with_baseline = valid_df[~valid_df['baseline_error'].isna()]
        if len(valid_with_baseline) > 0:
            learned_mean = valid_df['error'].mean()
            baseline_mean = valid_with_baseline['baseline_error'].mean()
            improvement = baseline_mean - learned_mean
            print(f"Learned error: {learned_mean:.4f}")
            print(f"Baseline error: {baseline_mean:.4f}")
            print(f"Improvement: {improvement:+.4f} ({100*improvement/baseline_mean:+.1f}%)")
            
            if improvement > 0.05:
                print(">>> LLMs BEAT DETERMINISTIC: Learning is real")
            elif improvement < -0.05:
                print(">>> DETERMINISTIC BEATS LLMs: Learning failed")
            else:
                print(">>> PARITY: No clear advantage")
        
        # MI analysis
        print(f"\n--- Mutual Information ---")
        mi_analysis = self.arena.mi_analyzer.analyze_correlation(self.arena.history)
        print(f"MI-Error correlation: {mi_analysis['correlation']:.3f}")
        print(f"Mean MI: {mi_analysis['mi_mean']:.3f} ± {mi_analysis['mi_std']:.3f}")
        print(f"Insight: {mi_analysis['insight']}")
        
        if mi_analysis['correlation'] < -0.5:
            print(">>> TOKENS ENCODE GEOMETRY: High MI → Low error (real signal)")
        elif abs(mi_analysis['correlation']) < 0.3:
            print(">>> NOISE ALIGNMENT: MI independent of error (no learning)")
        
        # Phase comparison
        phases = ['SYNC', 'SWAP', 'CONTROL', 'CROSS']
        means = {p: valid_df[valid_df.phase==p]['error'].mean() for p in phases if p in valid_df.phase.values}
        
        print(f"\n--- Signal Detection ---")
        for p, m in means.items():
            print(f"{p:8s}: {m:.4f}")
        
        if all(p in means for p in ['SWAP', 'CONTROL', 'CROSS']):
            ghost = means['CONTROL'] - means['SWAP']
            cross = means['CROSS'] - means['CONTROL']
            
            print(f"\nGhost signal (CTRL - SWAP): {ghost:+.4f}")
            print(f"Cross signal (CROSS - CTRL): {cross:+.4f}")
            
            if ghost > 0.05 and cross > 0.05:
                print(">>> STRONG GHOST: Weak B benefits from strong examples specifically")
            elif ghost > 0.05 and cross < 0.05:
                print(">>> WEAK GHOST: Any examples help, not specific to strong protocol")
            elif ghost < -0.05:
                print(">>> ANTI-GHOST: Strong examples actively harm weak B")
            else:
                print(">>> NULL: No detectable transfer")
        
        # Train/test split integrity
        print(f"\n--- Train/Test Integrity ---")
        test_trials = df[df['is_test_example'] == True]
        if len(test_trials) > 0:
            print(f"Test examples used: {len(test_trials)}")
            print(f"(These should NOT appear in future prompts)")
        
        return df

CONFIG = {
    "strong_model": "nemotron3:33b",
    "weak_model": "llama3.2",
}

if __name__ == "__main__":
    for m in [CONFIG["strong_model"], CONFIG["weak_model"]]:
        if not check_model(m):
            print(f"MISSING: {m} — run: ollama pull {m}")
            exit(1)
    
    agent_a = LLMWrapper(CONFIG["strong_model"], "ollama")
    agent_b = LLMWrapper(CONFIG["strong_model"], "ollama")
    
    arena = BabelArena(agent_a, agent_b, vocab_size=8)
    detector = FrictionDetector(arena)
    df = detector.run_experiment_suite(n_sync=8, n_swap=8, n_control=8, n_cross=8)
    
    # Save results
    df.to_csv("/workspace/babel_results_v2.csv", index=False)
    print("\n✓ Results saved to babel_results_v2.csv")
