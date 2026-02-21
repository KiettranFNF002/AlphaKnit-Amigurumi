import torch
import torch.nn.functional as F
import numpy as np
import os

@torch.no_grad()
def compute_phase_lag(model, optimizer, eps=1e-8):
    """
    v6.6-F: True Adam Update Direction Phase Lag.
    Cosine similarity between gradient direction and actual Adam update vector:
    update = m_hat / (sqrt(v_hat) + eps)
    """
    cosines = []
    update_energies = []
    
    # Target late layers for topological orientation
    target_keywords = ["transformer.layers.-1", "lm_head", "output_proj", "final_norm"]
    
    beta1, beta2 = 0.9, 0.999
    for group in optimizer.param_groups:
        beta1, beta2 = group.get('betas', (0.9, 0.999))
        break

    for name, p in model.named_parameters():
        if not any(k in name for k in target_keywords) or p.grad is None:
            continue
            
        state = optimizer.state.get(p, None)
        if not state or "exp_avg" not in state or "exp_avg_sq" not in state:
            continue
            
        step = state.get('step', 0)
        if isinstance(step, torch.Tensor):
            step = step.item()
            
        if step == 0: continue

        grad = p.grad.detach().flatten()
        m = state["exp_avg"].detach().flatten()
        v = state["exp_avg_sq"].detach().flatten()
        
        # Bias correction
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        
        # Actual Update Direction (Normalized by second moment)
        update_dir = m_hat / (torch.sqrt(v_hat) + eps)
        
        if grad.numel() == 0 or update_dir.numel() == 0:
            continue
            
        cos = F.cosine_similarity(grad.unsqueeze(0), update_dir.unsqueeze(0), dim=1)
        cosines.append(cos.item())
        update_energies.append(torch.norm(update_dir).item())

    if not cosines:
        return 1.0, 0.0
        
    avg_lag = sum(cosines) / len(cosines)
    avg_energy = sum(update_energies) / len(update_energies)
    return avg_lag, avg_energy


class HiddenProbePool:
    """
    v6.6-F: Observer Decoupling.
    Manages multiple orthogonal probe batches and rotates them to prevent instrument internalization.
    """
    def __init__(self, probe_loader, num_pools=3):
        self.pools = []
        self.active_idx = 0
        self.rotation_count = 0
        
        # Partition probe_loader into distinct pools
        all_batches = list(probe_loader)
        if len(all_batches) < num_pools:
            self.pools = [all_batches]
        else:
            size = len(all_batches) // num_pools
            for i in range(num_pools):
                self.pools.append(all_batches[i*size : (i+1)*size])
        
    def get_batch(self):
        pool = self.pools[self.active_idx]
        return pool[np.random.randint(len(pool))]

    def rotate(self):
        self.active_idx = (self.active_idx + 1) % len(self.pools)
        self.rotation_count += 1
        print(f"🔄 DECOUPLING: Probe Pool rotated to idx {self.active_idx}")

    def compute_pib(self, model, train_grads_dict, criterion, device):
        """
        Probe Influence Bound (PIB): cos(grad_train, grad_probe)
        Ensures measurement doesn't steer learning.
        """
        batch = self.get_batch()
        inputs = batch['points'].to(device)
        targets = batch['type_labels'].to(device)
        
        # Compute probe gradient without affecting optimizer state
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs['logits_type'].view(-1, outputs['logits_type'].size(-1)), targets.view(-1))
        loss.backward()
        
        cos_sims = []
        for name, p in model.named_parameters():
            if "transformer.layers.-1" in name and p.grad is not None and name in train_grads_dict:
                g_p = p.grad.detach().flatten()
                g_t = train_grads_dict[name].detach().flatten()
                sim = F.cosine_similarity(g_p.unsqueeze(0), g_t.unsqueeze(0))
                cos_sims.append(sim.item())
        
        model.zero_grad() # Clean up immediately
        
        if not cos_sims: return 0.0
        return sum(cos_sims) / len(cos_sims)


class LatentPhasePortrait:
    """
    VRAM-safe telemetry. Stores exactly ONE pooled structural embedding per epoch.
    Allows visualization of the learning trajectory (Phase Portrait).
    Basis Freezing: Locks pooling basis after stabilization.
    """
    def __init__(self):
        self.history = []
        self.locked_basis = None # TODO: Implement spectral locking in v6.6-F

    @torch.no_grad()
    def capture(self, hidden_states, structural_mask):
        """
        hidden_states: [B, T, D] (last layer hidden states)
        structural_mask: [B, T] (bool mask of topology-defining tokens)
        """
        if hidden_states is None:
            return

        h = hidden_states.detach()
        mask = structural_mask.unsqueeze(-1).float()
        
        # Mean pooling only over structural tokens across entire batch
        sum_hidden = (h * mask).sum(dim=(0, 1))
        denom = mask.sum() + 1e-6
        
        pooled = sum_hidden / denom # [D]
        self.history.append(pooled.cpu().float().numpy())

    def get_history(self):
        if not self.history:
            return None
        return np.stack(self.history)


class ModelRealityAnchors:
    """
    v6.6-F: Grounding telemetry in physical invariants.
    Tracks Normalized Weight Curvature and Representational Rank (SVD).
    """
    def __init__(self):
        self.prev_weights = {}
        self.history = {"curvature": [], "rank": [], "mi_leak": [], "stability": []}
        self.prev_latents = None

    @torch.no_grad()
    def update(self, model, current_latents):
        """
        current_latents: [D] (from LatentPhasePortrait)
        """
        # 1. Normalized Weight Curvature: ||W_t - W_{t-1}|| / ||W_{t-1}||
        curvatures = []
        for name, p in model.named_parameters():
            if "transformer.layers.-1" in name: 
                w = p.detach().cpu()
                if name in self.prev_weights:
                    diff = torch.norm(w - self.prev_weights[name])
                    norm = torch.norm(self.prev_weights[name]) + 1e-8
                    curvatures.append((diff / norm).item())
                self.prev_weights[name] = w
        
        if curvatures:
            self.history["curvature"].append(sum(curvatures) / len(curvatures))

        # 2. Representation Stability: cos(L_t, L_{t-1})
        if self.prev_latents is not None and current_latents is not None:
            l1 = torch.from_numpy(self.prev_latents)
            l2 = torch.from_numpy(current_latents)
            stab = F.cosine_similarity(l1.unsqueeze(0), l2.unsqueeze(0)).item()
            self.history["stability"].append(stab)
        
        self.prev_latents = current_latents

    @torch.no_grad()
    def compute_rank(self, latents_batch):
        """
        Spectral Rank: exp(Entropy(SingularValues))
        latents_batch: [B, D]
        """
        if latents_batch.size(0) < 2: return 0.0
        
        centered = latents_batch - latents_batch.mean(dim=0)
        _, S, _ = torch.svd(centered)
        
        probs = S / (S.sum() + 1e-8)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        rank = torch.exp(entropy).item()
        self.history["rank"].append(rank)
        return rank

    def track_mi_leak(self, latents, probe_labels):
        """
        Feature Mutual Information proxy: Correlation between probe identity and latents.
        High correlation = Instrument Internalization.
        """
        l_norm = torch.norm(latents, dim=-1).cpu().numpy()
        y = probe_labels.cpu().numpy().flatten()[:len(l_norm)]
        
        if len(np.unique(y)) < 2: return 0.0
        
        corr = np.abs(np.corrcoef(l_norm, y)[0, 1])
        self.history["mi_leak"].append(corr)
        return corr


class EmergenceTracker:
    """
    Detects the "Crystallization Window" for post-peak checkpoint saving.
    Monitors Velocity and Acceleration of competence metrics.
    """
    def __init__(self, window_size=5):
        self.history = []
        self.best_velocity = -1e9
        self.peak_epoch = None
        self.window_size = window_size

    def update(self, score, epoch):
        self.history.append(score)
        if len(self.history) < 2:
            return False

        vel = self.history[-1] - self.history[-2]
        
        if vel > self.best_velocity:
            self.best_velocity = vel
            self.peak_epoch = epoch
            print(f"📈 NEW EMERGENCE PEAK: Velocity {vel:.4f} at Epoch {epoch}")

        if self.peak_epoch is not None:
            age = epoch - self.peak_epoch
            if 3 <= age <= 5:
                return True
        return False
