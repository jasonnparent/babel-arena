import numpy as np
import json
import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import math

@dataclass
class GeometricProperties:
    """Extracted latent properties from point cloud geometry"""
    cluster_count: int
    cluster_separation: float  # min inter-cluster distance / within-cluster spread
    symmetry_score: float  # 0-1, how symmetric around centroid
    outlier_fraction: float  # proportion of points > 2 std from center
    density_uniformity: float  # 0-1, how uniform is point distribution
    topology_class: str  # "blob", "clusters", "line", "sparse"
    convex_hull_ratio: float  # volume of points / convex hull volume
    dimensionality: float  # intrinsic dimensionality (0-3)

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
        from scipy.cluster.hierarchy import fclusterdata
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
        
        # Symmetry: deviation from radial symmetry around centroid
        centroid = points.mean(axis=0)
        distances_to_center = np.linalg.norm(points - centroid, axis=1)
        symmetry_score = 1.0 - (np.std(distances_to_center) / (np.mean(distances_to_center) + 1e-8))
        symmetry_score = np.clip(symmetry_score, 0, 1)
        
        # Outlier detection
        mean = points.mean(axis=0)
        std = points.std(axis=0)
        z_scores = np.abs((points - mean) / (std + 1e-8))
        outlier_fraction = np.mean(np.any(z_scores > 2, axis=1))
        
        # Density uniformity: coefficient of variation in local density
        from scipy.spatial.distance import cdist
        knn_distances = np.sort(cdist(points, points), axis=1)[:, 3].mean()  # 3-NN
        local_densities = 1.0 / (np.mean(np.sort(cdist(points, points), axis=1)[:, 1:4], axis=1) + 1e-8)
        density_uniformity = 1.0 - np.std(local_densities) / (np.mean(local_densities) + 1e-8)
        density_uniformity = np.clip(density_uniformity, 0, 1)
        
        # Topology classification
        if outlier_fraction > 0.2:
            topology_class = "sparse"
        elif cluster_count > 2:
            topology_class = "clusters"
        elif cluster_separation > 2.0:
            topology_class = "clusters"
        elif np.std(distances_to_center) / np.mean(distances_to_center) < 0.3:
            topology_class = "blob"
        else:
            topology_class = "blob"
        
        # Convex hull ratio (approximation)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            convex_hull_ratio = 1.0  # Placeholder: exact calculation complex
        except:
            convex_hull_ratio = 0.8
        
        # Intrinsic dimensionality (PCA-based)
        from sklearn.decomposition import PCA
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
        """Extract properties and record mapping"""
        props = self.extract_properties(points)
        self.property_history.append((props, tokens))
        
        # Build token pattern map
        token_list = tokens.split()
        for prop_name, prop_value in asdict(props).items():
            if prop_name == "topology_class":
                key = f"{prop_name}={prop_value}"
                self.property_token_map[prop_name][key].append(token_list)
            else:
                # Discretize continuous properties into bins
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
                    bins = None
                
                if bins:
                    bin_idx = np.digitize([prop_value], bins) - 1
                    bin_idx = np.clip(bin_idx, 0, len(labels) - 1)
                    key = f"{prop_name}={labels[bin_idx]}"
                    self.property_token_map[prop_name][key].append(token_list)
        
        return props
    
    def analyze_disentanglement(self) -> Dict:
        """Compute property-token consistency"""
        results = {}
        
        for property_name, bins_dict in self.property_token_map.items():
            property_results = {}
            
            for bin_name, token_sequences in bins_dict.items():
                if not token_sequences:
                    continue
                
                # Extract unique tokens per property bin
                all_tokens_in_bin = Counter()
                for seq in token_sequences:
                    all_tokens_in_bin.update(seq)
                
                # Top tokens for this property value
                top_tokens = all_tokens_in_bin.most_common(3)
                top_token_names = [t[0] for t in top_tokens]
                top_token_counts = [t[1] for t in top_tokens]
                
                # Consistency: dominance of top token
                total_tokens = sum(all_tokens_in_bin.values())
                consistency = top_token_counts[0] / total_tokens if total_tokens > 0 else 0
                
                # Entropy: how focused is distribution on top tokens?
                probs = np.array(list(all_tokens_in_bin.values())) / total_tokens
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropy_normalized = entropy / np.log2(len(all_tokens_in_bin))
                
                property_results[bin_name] = {
                    "top_tokens": top_token_names,
                    "top_counts": top_token_counts,
                    "consistency": consistency,
                    "entropy": entropy,
                    "entropy_normalized": entropy_normalized,
                    "n_samples": len(token_sequences),
                    "unique_tokens_used": len(all_tokens_in_bin)
                }
            
            results[property_name] = property_results
        
        return results
    
    def compute_disentanglement_score(self) -> Dict:
        """Overall disentanglement quality metric"""
        analysis = self.analyze_disentanglement()
        
        scores = {}
        for property_name, bins in analysis.items():
            if not bins:
                continue
            
            # Average consistency across property bins
            consistencies = [b["consistency"] for b in bins.values()]
            avg_consistency = np.mean(consistencies) if consistencies else 0
            
            # Separation: how different are token distributions across bins?
            bin_names = list(bins.keys())
            if len(bin_names) >= 2:
                token_sets = []
                for bn in bin_names:
                    tokens_in_bin = bins[bn]["top_tokens"]
                    token_sets.append(set(tokens_in_bin))
                
                # Jaccard distance: 1 - intersection/union
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
                "disentanglement_score": (avg_consistency * separation),  # high consistency + low overlap
                "n_bins": len(bins)
            }
        
        return scores
    
    def generate_report(self) -> str:
        """Human-readable disentanglement report"""
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
            
            for bin_name, bin_data in sorted(bins.items())[:3]:  # Top 3 bins
                report += f"\n    {bin_name}:\n"
                report += f"      Top tokens: {', '.join(bin_data['top_tokens'])}\n"
                report += f"      Consistency: {bin_data['consistency']:.3f}\n"
                report += f"      N: {bin_data['n_samples']}\n"
            
            report += "\n"
        
        # Overall interpretation
        avg_d_score = np.mean([s["disentanglement_score"] for s in scores.values()])
        report += f"🎯 OVERALL DISENTANGLEMENT: {avg_d_score:.3f}\n\n"
        
        if avg_d_score > 0.4:
            report += "✅ STRONG DISENTANGLEMENT: Tokens encode interpretable geometric properties\n"
        elif avg_d_score > 0.2:
            report += "⚠️  WEAK DISENTANGLEMENT: Some structure, but mostly noise\n"
        else:
            report += "❌ NO DISENTANGLEMENT: Tokens are arbitrary, no learned structure\n"
        
        return report


class AdversarialRobustnessTester:
    """Systematically degrade protocol; measure graceful vs catastrophic collapse"""
    
    @dataclass
    class PerturbationResult:
        perturbation_type: str
        severity: float  # 0-1, how aggressive
        base_error: float
        perturbed_error: float
        degradation: float  # (perturbed - base) / base
        error_amplification: float  # perturbed / base
        valid_reconstructions: int
        total_trials: int
        failure_mode: str  # "graceful", "catastrophic", "None"
    
    def __init__(self, arena, baseline_trials: int = 5):
        self.arena = arena
        self.baseline_trials = baseline_trials
        self.baseline_error = None
        self.baseline_valid_count = 0
        self.results: List[self.PerturbationResult] = []
    
    def measure_baseline(self) -> float:
        """Establish clean protocol performance"""
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
        """Adversarial: scramble token sequence"""
        token_list = tokens.split()
        n_swap = max(1, int(len(token_list) * severity))
        
        # Random swaps
        for _ in range(n_swap):
            i, j = np.random.choice(len(token_list), 2, replace=False)
            token_list[i], token_list[j] = token_list[j], token_list[i]
        
        return " ".join(token_list)
    
    def perturb_geometry(self, points: np.ndarray, severity: float, mode: str = "noise") -> np.ndarray:
        """Adversarial: distort geometry"""
        perturbed = points.copy()
        
        if mode == "noise":
            # Gaussian noise
            noise = np.random.randn(*points.shape) * (severity * 10)
            perturbed = np.clip(perturbed + noise, 5, 95)
        
        elif mode == "rotation":
            # Random rotation by severity * 180°
            theta = np.radians(severity * 180)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            perturbed = perturbed @ R
        
        elif mode == "outlier_injection":
            # Add outliers
            n_outliers = int(len(points) * severity)
            if n_outliers > 0:
                outliers = np.random.rand(n_outliers, 3) * 80 + 10
                perturbed = np.vstack([perturbed[:-n_outliers], outliers])
        
        elif mode == "scaling":
            # Non-uniform scaling
            scales = 1 + (np.random.randn(3) * severity * 0.5)
            perturbed = perturbed * scales
            perturbed = np.clip(perturbed, 5, 95)
        
        return np.round(perturbed, 2)
    
    def run_perturbation_test(self, perturbation_type: str, severities: List[float], 
                             trials_per_severity: int = 5) -> List[PerturbationResult]:
        """Test robustness at multiple severity levels"""
        print(f"\n  Testing: {perturbation_type}")
        test_results = []
        
        for severity in severities:
            errors = []
            valid_count = 0
            
            for trial_idx in range(trials_per_severity):
                # Get clean trial
                clean_res = self.arena.run_trial("PERTURB_CLEAN")
                if not clean_res.valid:
                    continue
                
                # Store original for comparison
                original_tokens = clean_res.tokens
                original_points = np.array(clean_res.points)
                
                # Apply perturbation
                if "token" in perturbation_type.lower():
                    perturbed_tokens = self.perturb_tokens(original_tokens, severity)
                    # Decode perturbed tokens
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
                    # Geometry perturbation: apply to original points
                    perturbed_points = self.perturb_geometry(original_points, severity, perturbation_type)
                    # Re-encode perturbed geometry
                    prompt_a = self.arena._encode_prompt(perturbed_points, [])
                    raw_a = self.arena.agent_a.query(prompt_a).strip()
                    tokens_list = re.findall(r'TOKEN_\d{2}', raw_a.upper())
                    perturbed_tokens = " ".join(tokens_list[:8])
                    
                    # Decode
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
                
                # Measure error
                perturbed_error, valid = self.arena.procrustes_error(original_points, reconstruction)
                if valid and not np.isnan(perturbed_error):
                    errors.append(perturbed_error)
                    valid_count += 1
            
            perturbed_error_mean = np.mean(errors) if errors else float('inf')
            degradation = (perturbed_error_mean - self.baseline_error) / (self.baseline_error + 1e-8)
            amplification = perturbed_error_mean / (self.baseline_error + 1e-8)
            
            # Classify failure mode
            if amplification < 1.1:
                failure_mode = "robust"
            elif amplification < 2.0:
                failure_mode = "graceful"
            else:
                failure_mode = "catastrophic"
            
            result = self.PerturbationResult(
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
        """Comprehensive adversarial testing"""
        print("\n=== ADVERSARIAL ROBUSTNESS TESTING ===")
        
        self.measure_baseline()
        
        # Test 1: Token scrambling
        self.run_perturbation_test("token_scrambling", 
                                  severities=[0.1, 0.3, 0.5, 0.7],
                                  trials_per_severity=4)
        
        # Test 2: Geometry noise
        self.run_perturbation_test("noise",
                                  severities=[0.1, 0.3, 0.5, 0.7],
                                  trials_per_severity=4)
        
        # Test 3: Outlier injection
        self.run_perturbation_test("outlier_injection",
                                  severities=[0.1, 0.2, 0.3, 0.4],
                                  trials_per_severity=4)
        
        # Test 4: Non-uniform scaling
        self.run_perturbation_test("scaling",
                                  severities=[0.1, 0.3, 0.5],
                                  trials_per_severity=4)
        
        return self.analyze_robustness()
    
    def analyze_robustness(self) -> pd.DataFrame:
        """Summarize and interpret robustness"""
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        print("\n=== ROBUSTNESS ANALYSIS ===")
        print(df.to_string())
        
        print("\n--- Degradation Patterns ---")
        for perturb_type in df['perturbation_type'].unique():
            subset = df[df['perturbation_type'] == perturb_type]
            amplifications = subset['error_amplification'].values
            
            print(f"\n{perturb_type}:")
            print(f"  Mean amplification: {np.mean(amplifications):.2f}x")
            print(f"  Max amplification: {np.max(amplifications):.2f}x")
            print(f"  Failure modes: {subset['failure_mode'].value_counts().to_dict()}")
            
            # Interpret
            if np.max(amplifications) < 1.5:
                print(f"  ✅ ROBUST: Protocol tolerates {perturb_type}")
            elif np.max(amplifications) < 3.0:
                print(f"  ⚠️  GRACEFUL: Degradation is proportional to severity")
            else:
                print(f"  ❌ FRAGILE: Catastrophic collapse on {perturb_type}")
        
        return df
