import torch
import torch.nn as F
import numpy as np

class InterventionEngine:
    """
    v6.6-F: Causal Intervention Engine.
    Injects noise or clips rank in specific layers to test causal hypotheses.
    """
    def __init__(self, model):
        self.model = model
        self.active_interventions = {} # {layer_name: type}
        self.random_baseline_prob = 0.15

    def register_intervention(self, layer_name, type="noise", duration=5):
        self.active_interventions[layer_name] = {"type": type, "remaining": duration}
        print(f"🛠️ INTERVENTION: Registered {type} on {layer_name} for {duration} steps.")

    def apply(self, step):
        """
        Applies active interventions by attaching temporary hooks.
        """
        # Random Intervention Baseline (Scientific Control)
        if np.random.random() < self.random_baseline_prob:
            layer_names = [n for n, _ in self.model.named_modules() if "transformer.layers" in n]
            if layer_names:
                random_layer = np.random.choice(layer_names)
                self.register_intervention(random_layer, "noise", duration=1)

        # Decay durations and clean up
        to_remove = []
        for name, data in self.active_interventions.items():
            data["remaining"] -= 1
            if data["remaining"] <= 0:
                to_remove.append(name)
        for name in to_remove:
            self.active_interventions.pop(name, None)

    def hook_fn(self, module, input, output):
        # We find the name of the module that called this
        # In a real implementation, we'd use a more robust way to map module to name
        # For AlphaKnit v6.6-F, we assume this is called on a layer that is active
        # Simplified perturbation:
        if isinstance(output, tuple):
            h = output[0]
            # Add Gaussian noise scaled by the output's standard deviation
            noise = torch.randn_like(h) * (h.std() * 0.1)
            return (h + noise, *output[1:])
        else:
            noise = torch.randn_like(output) * (output.std() * 0.1)
            return output + noise


class HypothesisEngine:
    """
    v6.6-F: Causal Falsification Engine.
    Automates the "Discovery" process by verifying persistence and invariance.
    """
    def __init__(self, n_seeds_target=5):
        self.hypotheses = []
        self.n_seeds_target = n_seeds_target
        self.persistence_window = 3
        self.causal_confidence = {}

    def propose(self, name, description, condition_fn):
        self.hypotheses.append({
            "name": name,
            "desc": description,
            "condition": condition_fn,
            "streak": 0,
            "status": "PROPOSED",
            "history": []
        })

    def update(self, metrics, epoch):
        report = []
        for h in self.hypotheses:
            # Type-safe streak access
            streak_raw = h.get("streak", 0)
            streak = int(streak_raw) if isinstance(streak_raw, (int, float)) else 0
            
            condition_fn = h.get("condition")
            is_met = False
            if callable(condition_fn):
                try:
                    is_met = condition_fn(metrics)
                except Exception:
                    is_met = False

            if is_met:
                streak += 1
                if streak >= self.persistence_window:
                    h["status"] = "VERIFIED"
            else:
                if h["status"] == "VERIFIED":
                    print(f"🔴 FALSIFIED: Discovery '{h['name']}' failed persistence check at epoch {epoch}.")
                    h["status"] = "FALSIFIED"
                streak = 0
            
            h["streak"] = streak
            h["history"].append(h["status"])
            name = str(h.get("name", "Unknown"))
            report.append(f"{name}: {h['status']} ({streak}/{self.persistence_window})")
        
        return report

    def monitor_failure(self, real_metrics, null_metrics):
        """
        Automated Hypothesis Rejection: If Null Mode (Placebo) shows stronger structural
        emergence than the real run, then the current hypothesis is invalid.
        """
        for h in self.hypotheses:
            if h["status"] == "VERIFIED":
                if null_metrics.get("struct_acc", 0) > real_metrics.get("struct_acc", 0) * 1.5:
                    print(f"⚠️ REJECTED: Discovery '{h['name']}' rejected by Null Control at epoch {real_metrics['epoch']}.")
                    h["status"] = "REJECTED_BY_CONTROL"

    def get_survival_map(self):
        return {h["name"]: h["status"] for h in self.hypotheses}


class NullEmergenceSuite:
    """
    v6.6-F: Scientific Control Suite.
    Manages "Placebo" training seeds (Random Labels, Noise Inputs, Geometry Null).
    """
    def __init__(self, mode="real"):
        self.mode = mode # "real", "random_labels", "noise_inputs", "geometry_null"

    def transform_batch(self, batch):
        if self.mode == "real":
            return batch
        
        if self.mode == "random_labels":
            # Semantic Breakdown
            batch['type_labels'] = batch['type_labels'][torch.randperm(batch['type_labels'].size(0))]
        
        if self.mode == "noise_inputs":
            # Structural Breakdown
            batch['points'] = batch['points'] + torch.randn_like(batch['points']) * 0.5
            
        return batch

    def apply_geometry_null(self, model):
        if self.mode == "geometry_null":
            print("🧱 GEOMETRY NULL: Scrambling layer connectivity...")
            # Simple version: randomize weights to break representational paths
            for p in model.parameters():
                if p.dim() >= 2:
                    torch.nn.init.orthogonal_(p)
