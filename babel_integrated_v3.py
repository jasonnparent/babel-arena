"""
Babel Arena v3: Integrated analysis with disentanglement + adversarial robustness
Combines:
  - Baseline encoder comparison
  - Mutual information analysis
  - Train/test splits
  - Representation disentanglement (which geometric properties → which tokens?)
  - Adversarial robustness (graceful vs catastrophic degradation)
"""

import numpy as np
import json
import re
import pandas as pd
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import math
from scipy.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import fclusterdata
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE COMPONENTS (from previous versions)
# ============================================================================

@dataclass
class TrialResult:
    points: List
    tokens: str
    reconstruction: List
    error: float
    token_count: int
    phase: str
    valid: bool
    token_entropy: float
    mutual_information: float = 0.0
    baseline_error: float = float('nan')
    is_test_example: bool = False

class DeterministicEncoder:
    """Baseline: PCA + quantization + hashing"""
    
    def __init__(self, vocab_size: int = 8, n_components: int = 2):
        self.vocab_size = vocab_size
        self.n_components = n_components
        self.pca = None
        self.token_vocab = [f"TOKEN_{i:02d}" for i in range(vocab_size)]
        self.fitted = False
    
    def fit(self, training_points: List[np.ndarray]) -> None:
        all_points = np.vstack(training_points)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(all_points)
        self.fitted = True
    
    def encode(self, points: np.ndarray) -> str:
        if not self.fitted or self.pca is None:
            raise ValueError("Not fitted")
        projected = self.pca.transform(points)
        quantized = np.round(projected).astype(int)
        tokens = []
        for coord in quantized:
            token_idx = int(np.sum(np.abs(coord))) % self.vocab_size
            tokens.append(self.token_vocab[token_idx])
        return " ".join(tokens[:8])
    
    def decode(self, tokens: str) -> np.ndarray:
        token_list = tokens.split()
        coords = []
        for tok in token_list:
            if tok.startswith("TOKEN_"):
                idx = int(tok.split("_")[1])
                coord = np.array([idx % 3 - 1, idx // 3 - 1])
                coords.append(coord)
        if not coords:
            return np.random.rand(8, 3) * 50 + 25
        coords = np.array(coords)
        reconstructed = self.pca.inverse_transform(coords)
        while len(reconstructed) < 8:
            reconstructed = np.vstack([reconstructed, reconstructed[-1:]])
        return reconstructed[:8]

class MutualInformationAnalyzer:
    @staticmethod
    def discretize_geometry(points: np.ndarray, n_bins: int = 4) -> np.ndarray:
        discretized = np.zeros_like(points, dtype=int)
        for dim in range(points.shape[1]):
            discretized[:, dim] = np.digitize(points[:, dim], 
                                              np.linspace(points[:, dim].min(), 
                                                         points[:, dim].max(), n_bins))
        return discretized
    
    @staticmethod
    def compute_mi(geometry_state: np.ndarray, token_sequence: str) -> float:
        geo_tuple = tuple(geometry_state.flatten())
        tokens = token_sequence.split()
        token_counts = Counter(tokens)
        if len(token_counts) == 0 or len(tokens) == 0:
            return 0.0
        p_tok = np.array(list(token_counts.values())) / len(tokens)
        uniqueness = len(set(tokens)) / len(tokens)
        h_tok = -np.sum(p_tok * np.log2(p_tok + 1e-10))
        mi_estimate = uniqueness * h_tok
        return float(np.clip(mi_estimate, 0, 1))
    
    @staticmethod
    def analyze_correlation(trial_results: List[TrialResult]) -> Dict:
        valid_trials = [t for t in trial_results if t.valid]
        if len(valid_trials) < 3:
            return {"n": len(valid_trials), "correlation": np.nan}
        mis = np.array([t.mutual_information for t in valid_trials])
        errors = np.array([t.error for t in valid_trials])
        corr = float(np.corrcoef(mis, errors)[0, 1]) if np.std(mis) > 0 else 0.0
        return {
            "n": len(valid_trials),
            "correlation": corr,
            "mi_mean": float(np.mean(mis)),
            "mi_std": float(np.std(mis)),
            "insight": "strong" if corr < -0.5 else "weak" if abs(corr) < 0.3 else "moderate"
        }

class ExampleSplitter:
    def __init__(self, test_ratio: float = 0.3, seed: int = 42):
        self.test_ratio = test_ratio
        self.seed = seed
        self.train_examples: List[Tuple] = []
        self.test_examples: List[Tuple] = []
        np.random.seed(seed)
    
    def add_example(self, tokens: str, points: List, phase: str = "warmup") -> bool:
        rand = np.random.random()
        is_test = rand < self.test_ratio
        example = (tokens, points)
        if is_test:
            self.test_examples.append(example)
        else:
            self.train_examples.append(example)
        return is_test
    
    def get_train_examples(self) -> List[Tuple]:
        return self.train_examples
    
    def validate_separation(self) -> Dict:
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
    if not tokens:
        return 0.0
    counts = Counter(tokens.split())
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    entropy_val = -sum(p * math.log2(p) for p in probs if p > 0)
    max_ent = math.log2(vocab_size)
    return entropy_val / max_ent if max_ent > 0 else 0.0

# ============================================================================
# REPRESENTATION DISENTANGLEMENT
# ============================================================================

@dataclass
class GeometricProperties:
    cluster_count: int
    cluster_separation: float
    symmetry_score: float
    outlier_fraction: float
    density_uniformity: float
    topology_class: str
    convex_hull_ratio: float
    dimensionality: float

class RepresentationDisentangler:
    """Extract and correlate geometric properties with token patterns"""
    
    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size
        self.property_token_map: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
        self.property_history: List[Tuple[GeometricProperties, str]] = []
    
    @staticmethod
    def extract_properties(points: np.ndarray) -> GeometricProperties:
        """Analyze geometry and extract latent properties"""
        
        # Cluster detection
        try:
            clusters = fclusterdata(points, t=1.5, criterion='distance', method='complete')
            cluster_count = len(np.unique(clusters))
        except:
            cluster_count = 1
        
        # Cluster separation
        if cluster_count > 1:
            centroids = np.array([points[clusters == i].mean(axis=0) for i in np.unique(clusters)])
            inter_cluster_dist = np.min(pdist(centroids))
            within_cluster_dist = np.mean([np.std(points[clusters == i]) for i in np.unique(clusters)])
            cluster_separation = inter_cluster_dist / (within_cluster_dist + 1e-8)
        else:
            cluster_separation = 0.0
        
        # Symmetry
        centroid = points.mean(axis=0)
        distances_to_center = np.linalg.norm(points - centroid, axis=1)
        symmetry_score = 1.0 - (np.std(distances_to_center) / (np.mean(distances_to_center) + 1e-8))
        symmetry_score = np.clip(symmetry_score, 0, 1)
        
        # Outliers
        mean = points.mean(axis=0)
        std = points.std(axis=0)
        z_scores = np.abs((points - mean) / (std + 1e-8))
        outlier_fraction = np.mean(np.any(z_scores > 2, axis=1))
        
        # Density uniformity
        knn_distances = np.sort(cdist(points, points), axis=1)[:, 3].mean()
        local_densities = 1.0 / (np.mean(np.sort(cdist(points, points), axis=1)[:, 1:4], axis=1) + 1e-8)
        density_uniformity = 1.0 - np.std(local_densities) / (np.mean(local_densities) + 1e-8)
        density_uniformity = np.clip(density_uniformity, 0, 1)
        
        # Topology
        if outlier_fraction > 0.2:
            topology_class = "sparse"
        elif cluster_count > 2:
            topology_class = "clusters"
        elif cluster_separation > 2.0:
            topology_class = "clusters"
        else:
            topology_class = "blob"
        
        # Convex hull (approximation)
        convex_hull_ratio = 0.85
        
        # Dimensionality
        pca = PCA()
        pca.fit(points)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        dimensionality = np.where(cumsum > 0.95)[0][0] / 3.0 if len(np.where(cumsum > 0.95)[0]) > 0 else 1.0
        dimensionality = np.clip(dimensionality, 0, 1)
        
        return GeometricProperties(
            cluster_count=int(cluster_count),
            cluster_separation=float(np.clip(cluster_separation, 0, 5)),
            symmetry_score=float(symmetry_score),
            outlier_fraction=float(outlier_fraction),
            density_uniformity=float(density_uniformity),
            topology_class=topology_class,
            convex_hull_ratio=float(convex_hull_ratio),
            dimensionality=float(dimensionality)
        )
    
    def record_trial(self, points: np.ndarray, tokens: str) -> GeometricProperties:
        props = self.extract_properties(points)
        self.property_history.append((props, tokens))
        
        token_list = tokens.split()
        for prop_name, prop_value in asdict(props).items():
            if prop_name == "topology_class":
                key = f"{prop_name}={prop_value}"
                self.property_token_map[prop_name][key].append(token_list)
            else:
                # Discretize continuous properties
                if prop_name == "cluster_separation":
                    bins = [0, 0.5, 1.5, 3, 5]
                    labels = ["low", "medium", "high", "very_high"]
                elif prop_name == "symmetry_score":
                    bins = [0, 0.3, 0.6, 1.0]
                    labels = ["asymmetric", "semi-symmetric", "symmetric"]
                elif prop_name == "outlier_fraction":
                    bins = [0, 0.1, 0.2, 1.0]
                    labels = ["none", "few", "many"]
                elif prop_name == "density_uniformity":
                    bins = [0, 0.3, 0.6, 1.0]
                    labels = ["clustered", "mixed", "uniform"]
                else:
                    continue
                
                bin_idx = np.digitize([prop_value], bins) - 1
                bin_idx = np.clip(bin_idx, 0, len(labels) - 1)
                key = f"{prop_name}={labels[bin_idx]}"
                self.property_token_map[prop_name][key].append(token_list)
        
        return props
    
    def analyze_disentanglement(self) -> Dict:
        results = {}
        
        for property_name, bins_dict in self.property_token_map.items():
            property_results = {}
            
            for bin_name, token_sequences in bins_dict.items():
                if not token_sequences:
                    continue
                
                all_tokens_in_bin = Counter()
                for seq in token_sequences:
                    all_tokens_in_bin.update(seq)
                
                top_tokens = all_tokens_in_bin.most_common(3)
                top_token_names = [t[0] for t in top_tokens]
                top_token_counts = [t[1] for t in top_tokens]
                
                total_tokens = sum(all_tokens_in_bin.values())
                consistency = top_token_counts[0] / total_tokens if total_tokens > 0 else 0
                
                probs = np.array(list(all_tokens_in_bin.values())) / total_tokens
                entropy_val = -np.sum(probs * np.log2(probs + 1e-10))
                entropy_normalized = entropy_val / np.log2(len(all_tokens_in_bin)) if len(all_tokens_in_bin) > 1 else 0
                
                property_results[bin_name] = {
                    "top_tokens": top_token_names,
                    "top_counts": top_token_counts,
                    "consistency": consistency,
                    "entropy": entropy_val,
                    "entropy_normalized": entropy_normalized,
                    "n_samples": len(token_sequences),
                    "unique_tokens_used": len(all_tokens_in_bin)
                }
            
            results[property_name] = property_results
        
        return results
    
    def compute_disentanglement_score(self) -> Dict:
        analysis = self.analyze_disentanglement()
        scores = {}
        
        for property_name, bins in analysis.items():
            if not bins:
                continue
            
            consistencies = [b["consistency"] for b in bins.values()]
            avg_consistency = np.mean(consistencies) if consistencies else 0
            
            bin_names = list(bins.keys())
            if len(bin_names) >= 2:
                token_sets = []
                for bn in bin_names:
                    tokens_in_bin = bins[bn]["top_tokens"]
                    token_sets.append(set(tokens_in_bin))
                
                overlaps = []
                for i in range(len(token_sets)):
                    for j in range(i+1, len(token_sets)):
                        intersection = len(token_sets[i] & token_sets[j])
                        union = len(token_sets[i] | token_sets[j])
                        jaccard = 1 - (intersection / union) if union > 0 else 0
                        overlaps.append(jaccard)
                
                separation = np.mean(overlaps) if overlaps else 0
            else:
                separation = 0
            
            scores[property_name] = {
                "consistency": avg_consistency,
                "separation": separation,
                "disentanglement_score": (avg_consistency * separation),
                "n_bins": len(bins)
            }
        
        return scores
    
    def generate_report(self) -> str:
        analysis = self.analyze_disentanglement()
        scores = self.compute_disentanglement_score()
        
        report = "=== REPRESENTATION DISENTANGLEMENT ANALYSIS ===\n\n"
        
        for property_name in sorted(analysis.keys()):
            if property_name not in scores:
                continue
            
            score_dict = scores[property_name]
            report += f"📊 {property_name.upper()}\n"
            report += f"  Consistency: {score_dict['consistency']:.3f}\n"
            report += f"  Separation:  {score_dict['separation']:.3f}\n"
            report += f"  D-Score:     {score_dict['disentanglement_score']:.3f}\n"
            
            bins = analysis[property_name]
            report += f"  Bins: {len(bins)}\n"
            
            for bin_name, bin_data in sorted(bins.items())[:3]:
                report += f"\n    {bin_name}:\n"
                report += f"      Top tokens: {', '.join(bin_data['top_tokens'])}\n"
                report += f"      Consistency: {bin_data['consistency']:.3f}\n"
                report += f"      N: {bin_data['n_samples']}\n"
            
            report += "\n"
        
        avg_d_score = np.mean([s["disentanglement_score"] for s in scores.values()])
        report += f"🎯 OVERALL DISENTANGLEMENT: {avg_d_score:.3f}\n\n"
        
        if avg_d_score > 0.4:
            report += "✅ STRONG DISENTANGLEMENT: Tokens encode interpretable geometric properties\n"
        elif avg_d_score > 0.2:
            report += "⚠️  WEAK DISENTANGLEMENT: Some structure, but mostly noise\n"
        else:
            report += "❌ NO DISENTANGLEMENT: Tokens are arbitrary, no learned structure\n"
        
        return report

# ============================================================================
# ADVERSARIAL ROBUSTNESS TESTING
# ============================================================================

@dataclass
class PerturbationResult:
    perturbation_type: str
    severity: float
    base_error: float
    perturbed_error: float
    degradation: float
    error_amplification: float
    valid_reconstructions: int
    total_trials: int
    failure_mode: str

class AdversarialRobustnessTester:
    
    def __init__(self, arena, baseline_trials: int = 5):
        self.arena = arena
        self.baseline_trials = baseline_trials
        self.baseline_error = None
        self.baseline_valid_count = 0
        self.results: List[PerturbationResult] = []
    
    def measure_baseline(self) -> float:
        print("  Measuring baseline performance...")
        errors = []
        valid_count = 0
        
        for _ in range(self.baseline_trials):
            res = self.arena.run_trial("BASELINE")
            if res.valid:
                errors.append(res.error)
                valid_count += 1
        
        self.baseline_error = np.mean(errors) if errors else float('inf')
        self.baseline_valid_count = valid_count
        print(f"  Baseline: {self.baseline_error:.4f} ({valid_count}/{self.baseline_trials} valid)")
        return self.baseline_error
    
    def perturb_tokens(self, tokens: str, severity: float) -> str:
        token_list = tokens.split()
        n_swap = max(1, int(len(token_list) * severity))
        for _ in range(n_swap):
            i, j = np.random.choice(len(token_list), 2, replace=False)
            token_list[i], token_list[j] = token_list[j], token_list[i]
        return " ".join(token_list)
    
    def perturb_geometry(self, points: np.ndarray, severity: float, mode: str = "noise") -> np.ndarray:
        perturbed = points.copy()
        
        if mode == "noise":
            noise = np.random.randn(*points.shape) * (severity * 10)
            perturbed = np.clip(perturbed + noise, 5, 95)
        elif mode == "outlier_injection":
            n_outliers = int(len(points) * severity)
            if n_outliers > 0:
                outliers = np.random.rand(n_outliers, 3) * 80 + 10
                perturbed = np.vstack([perturbed[:-n_outliers], outliers])
        elif mode == "scaling":
            scales = 1 + (np.random.randn(3) * severity * 0.5)
            perturbed = perturbed * scales
            perturbed = np.clip(perturbed, 5, 95)
        
        return np.round(perturbed, 2)
    
    def run_perturbation_test(self, perturbation_type: str, severities: List[float], 
                             trials_per_severity: int = 3) -> List[PerturbationResult]:
        print(f"\n  Testing: {perturbation_type}")
        test_results = []
        
        for severity in severities:
            errors = []
            valid_count = 0
            
            for trial_idx in range(trials_per_severity):
                clean_res = self.arena.run_trial("PERTURB_CLEAN")
                if not clean_res.valid:
                    continue
                
                original_tokens = clean_res.tokens
                original_points = np.array(clean_res.points)
                
                if "token" in perturbation_type.lower():
                    perturbed_tokens = self.perturb_tokens(original_tokens, severity)
                    prompt = self.arena._decode_prompt(perturbed_tokens, [])
                    raw_recon = self.arena.agent_b.query(prompt).strip()
                    try:
                        matches = re.findall(r'\s*[\d\.\-]+\s*,\s*[\d\.\-]+\s*,\s*[\d\.\-]+\s*', raw_recon)
                        if len(matches) >= 8:
                            reconstruction = [json.loads(m) for m in matches[:8]]
                        else:
                            reconstruction = []
                    except:
                        reconstruction = []
                else:
                    perturbed_points = self.perturb_geometry(original_points, severity, perturbation_type)
                    prompt_a = self.arena._encode_prompt(perturbed_points, [])
                    raw_a = self.arena.agent_a.query(prompt_a).strip()
                    tokens_list = re.findall(r'TOKEN_\d{2}', raw_a.upper())
                    perturbed_tokens = " ".join(tokens_list[:8])
                    
                    prompt_b = self.arena._decode_prompt(perturbed_tokens, [])
                    raw_b = self.arena.agent_b.query(prompt_b).strip()
                    try:
                        matches = re.findall(r'\s*[\d\.\-]+\s*,\s*[\d\.\-]+\s*,\s*[\d\.\-]+\s*', raw_b)
                        if len(matches) >= 8:
                            reconstruction = [json.loads(m) for m in matches[:8]]
                        else:
                            reconstruction = []
                    except:
                        reconstruction = []
                
                perturbed_error, valid = self.arena.procrustes_error(original_points, reconstruction)
                if valid and not np.isnan(perturbed_error):
                    errors.append(perturbed_error)
                    valid_count += 1
            
            perturbed_error_mean = np.mean(errors) if errors else float('inf')
            degradation = (perturbed_error_mean - self.baseline_error) / (self.baseline_error + 1e-8)
            amplification = perturbed_error_mean / (self.baseline_error + 1e-8)
            
            if amplification < 1.1:
                failure_mode = "robust"
            elif amplification < 2.0:
                failure_mode = "graceful"
            else:
                failure_mode = "catastrophic"
            
            result = PerturbationResult(
                perturbation_type=perturbation_type,
                severity=severity,
                base_error=self.baseline_error,
                perturbed_error=perturbed_error_mean,
                degradation=degradation,
                error_amplification=amplification,
                valid_reconstructions=valid_count,
                total_trials=trials_per_severity,
                failure_mode=failure_mode
            )
            test_results.append(result)
            
            print(f"    Severity {severity:.2f}: error {perturbed_error_mean:.4f} " +
                  f"(×{amplification:.2f}, {failure_mode})")
        
        self.results.extend(test_results)
        return test_results
    
    def run_full_robustness_suite(self) -> pd.DataFrame:
        print("\n=== ADVERSARIAL ROBUSTNESS TESTING ===")
        
        self.measure_baseline()
        
        self.run_perturbation_test("token_scrambling", 
                                  severities=[0.1, 0.3, 0.5],
                                  trials_per_severity=3)
        
        self.run_perturbation_test("noise",
                                  severities=[0.1, 0.3, 0.5],
                                  trials_per_severity=3)
        
        self.run_perturbation_test("outlier_injection",
                                  severities=[0.1, 0.2, 0.3],
                                  trials_per_severity=3)
        
        self.run_perturbation_test("scaling",
                                  severities=[0.1, 0.3],
                                  trials_per_severity=3)
        
        return self.analyze_robustness()
    
    def analyze_robustness(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        print("\n=== ROBUSTNESS ANALYSIS ===")
        print(df.to_string(index=False))
        
        print("\n--- Degradation Patterns ---")
        for perturb_type in df['perturbation_type'].unique():
            subset = df[df['perturbation_type'] == perturb_type]
            amplifications = subset['error_amplification'].values
            
            print(f"\n{perturb_type}:")
            print(f"  Mean amplification: {np.mean(amplifications):.2f}x")
            print(f"  Max amplification: {np.max(amplifications):.2f}x")
            
            if np.max(amplifications) < 1.5:
                print(f"  ✅ ROBUST: Protocol tolerates {perturb_type}")
            elif np.max(amplifications) < 3.0:
                print(f"  ⚠️  GRACEFUL: Degradation is proportional to severity")
            else:
                print(f"  ❌ FRAGILE: Catastrophic collapse on {perturb_type}")
        
        return df

# ============================================================================
# LLM WRAPPER & ARENA
# ============================================================================

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
        
        self.example_splitter = ExampleSplitter(test_ratio=0.3)
        self.baseline_encoder = DeterministicEncoder(vocab_size=vocab_size)
        self.mi_analyzer = MutualInformationAnalyzer()
        self.disentangler = RepresentationDisentangler(vocab_size=vocab_size)
        
        self.training_geometries: List[np.ndarray] = []
    
    def generate_task(self, n_points: int = 8) -> np.ndarray:
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
        vocab_str = " ".join(self.vocab[:6])
        ex_block = ""
        if examples:
            ex_block = "\nPRIOR ENCODINGS (training only):\n" + "\n".join(
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
            ex_block = "\nPRIOR DECODINGS (training only):\n" + "\n".join(
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
        
        self.training_geometries.append(raw_points.copy())
        
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
        
        try:
            if self.baseline_encoder.fitted:
                baseline_tokens = self.baseline_encoder.encode(points_for_a)
                baseline_recon = self.baseline_encoder.decode(baseline_tokens)
                baseline_error, _ = self.procrustes_error(raw_points, baseline_recon)
            else:
                baseline_error = float('nan')
        except:
            baseline_error = float('nan')
        
        mi = self.mi_analyzer.compute_mi(raw_points, message)
        is_test = self.example_splitter.add_example(message, raw_points.tolist(), phase)
        
        # NEW: Record for disentanglement analysis
        self.disentangler.record_trial(raw_points, message)
        
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

# ============================================================================
# COMPREHENSIVE TEST RUNNER
# ============================================================================

CONFIG = {
    "strong_model": "nemotron3:33b",
    "weak_model": "llama3.2",
}

def run_full_analysis():
    """End-to-end analysis pipeline"""
    
    print("🚀 BABEL ARENA v3: INTEGRATED ANALYSIS")
    print("=" * 70)
    
    # Check models
    for m in [CONFIG["strong_model"], CONFIG["weak_model"]]:
        if not check_model(m):
            print(f"❌ MISSING: {m} — run: ollama pull {m}")
            return
    
    # Initialize
    agent_a = LLMWrapper(CONFIG["strong_model"], "ollama")
    agent_b = LLMWrapper(CONFIG["strong_model"], "ollama")
    arena = BabelArena(agent_a, agent_b, vocab_size=8)
    
    # PHASE 0: Warmup with disentanglement tracking
    print("\n📝 PHASE 0: WARMUP & CALIBRATION")
    print("-" * 70)
    for i in range(8):
        res = arena.run_trial("WARMUP")
        if res.valid and res.token_entropy > 0.3:
            arena.example_splitter.add_example(res.tokens, res.points, "WARMUP")
            print(f"  ✓ Example {len(arena.example_splitter.train_examples)}: "
                  f"err={res.error:.3f}, ent={res.token_entropy:.2f}, mi={res.mutual_information:.2f}")
    
    # Fit baseline encoder
    if len(arena.training_geometries) > 0:
        arena.baseline_encoder.fit(arena.training_geometries)
    
    # PHASE 1: Standard trials
    print("\n🔬 PHASE 1: BASELINE MEASUREMENTS (16 trials)")
    print("-" * 70)
    for i in range(16):
        res = arena.run_trial("BASELINE")
        if res.valid:
            print(f"  Trial {i+1}: err={res.error:.3f}, base={res.baseline_error:.3f}, mi={res.mutual_information:.2f}")
    
    # PHASE 2: Disentanglement analysis
    print("\n🧬 PHASE 2: REPRESENTATION DISENTANGLEMENT")
    print("-" * 70)
    print(arena.disentangler.generate_report())
    
    disentangle_scores = arena.disentangler.compute_disentanglement_score()
    df_disentangle = pd.DataFrame(disentangle_scores).T
    print("\nDisentanglement Scores:")
    print(df_disentangle.to_string())
    df_disentangle.to_csv("/workspace/disentanglement_scores.csv")
    
    # PHASE 3: Adversarial robustness
    print("\n⚔️  PHASE 3: ADVERSARIAL ROBUSTNESS")
    print("-" * 70)
    adversary = AdversarialRobustnessTester(arena, baseline_trials=5)
    df_robustness = adversary.run_full_robustness_suite()
    df_robustness.to_csv("/workspace/adversarial_robustness.csv", index=False)
    
    # PHASE 4: Summary statistics
    print("\n📊 PHASE 4: SUMMARY STATISTICS")
    print("-" * 70)
    valid_trials = [t for t in arena.history if t.valid]
    print(f"Valid trials: {len(valid_trials)}/{len(arena.history)}")
    print(f"Mean error: {np.mean([t.error for t in valid_trials]):.4f}")
    print(f"Mean MI: {np.mean([t.mutual_information for t in valid_trials]):.4f}")
    print(f"Mean entropy: {np.mean([t.token_entropy for t in valid_trials]):.4f}")
    
    # Save master results
    results_data = []
    for trial in arena.history:
        results_data.append({
            "phase": trial.phase,
            "error": trial.error,
            "baseline_error": trial.baseline_error,
            "mutual_information": trial.mutual_information,
            "token_entropy": trial.token_entropy,
            "token_count": trial.token_count,
            "valid": trial.valid,
            "is_test": trial.is_test_example
        })
    
    df_master = pd.DataFrame(results_data)
    df_master.to_csv("/workspace/babel_master_results.csv", index=False)
    
    print("\n✅ Analysis complete!")
    print("   - babel_master_results.csv")
    print("   - disentanglement_scores.csv")
    print("   - adversarial_robustness.csv")

if __name__ == "__main__":
    run_full_analysis()
