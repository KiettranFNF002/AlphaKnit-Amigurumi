import torch
import torch.nn.functional as F

def topology_tension_field(node_degrees, edge_count, num_nodes, lambda_density=0.1, report_only=False):
    """
    Structural Regularization Loss.
    v6.6-F: Scheduled Observer Purity.
    """
    degree_var = node_degrees.var(dim=1).mean()
    edge_density = edge_count / (num_nodes + 1e-6)
    density_penalty = torch.relu(0.15 - edge_density)
    
    ttf_loss = degree_var + lambda_density * density_penalty
    
    stats = {
        "degree_var": degree_var.detach().item(),
        "edge_density": edge_density.detach().item(),
        "tension_loss": ttf_loss.detach().item()
    }
    
    if report_only:
        # Purity check: Zero the gradient influence
        return torch.tensor(0.0, device=ttf_loss.device, requires_grad=True), stats
        
    return ttf_loss, stats


def compute_structural_metrics(logits, targets, structural_mask, topk=(1, 3)):
    """
    Measures the "Logit Margin" and Top-K accuracy specifically for structural decisions.
    """
    logits_s = logits[structural_mask]
    targets_s = targets[structural_mask]
    
    if logits_s.numel() == 0:
        return {}
        
    true_logits = logits_s.gather(1, targets_s.unsqueeze(1)).squeeze(1)
    
    top2 = torch.topk(logits_s, k=2, dim=1).values
    best_wrong = torch.where(top2[:, 0] == true_logits, top2[:, 1], top2[:, 0])
    margin = (true_logits - best_wrong).mean()
    
    metrics = {"struct_margin": margin.item()}
    for k in topk:
        preds = torch.topk(logits_s, k=k, dim=1).indices
        correct = (preds == targets_s.unsqueeze(1)).any(dim=1)
        metrics[f"struct_top{k}_acc"] = correct.float().mean().item()
        
    probs = F.softmax(logits_s, dim=1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
    metrics["struct_entropy"] = entropy.item()
    
    return metrics


class FunctionalSharpness:
    """
    v6.6-F: Monitors grad_norm_t / grad_norm_{t-1} with EMA filtering.
    Detects local flattening of the landscape during emergence.
    """
    def __init__(self, ema_alpha=0.1):
        self.ema_alpha = ema_alpha
        self.prev_norm = None
        self.sharpness_ema = 1.0

    def update(self, current_norm):
        if self.prev_norm is None or self.prev_norm == 0:
            self.prev_norm = current_norm
            return 1.0
            
        ratio = current_norm / self.prev_norm
        self.sharpness_ema = (self.ema_alpha * ratio) + (1.0 - self.ema_alpha) * self.sharpness_ema
        self.prev_norm = current_norm
        return self.sharpness_ema
