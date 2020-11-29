import torch


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def entropy(q, dim=-1):
    p = q / torch.sum(q, dim=dim, keepdim=True)
    log_p = torch.log(clamp_probs(p))
    return -torch.sum(p * log_p, dim=dim)


def quantile(x, q):
    qt = torch.kthvalue(x, int(q * x.size(0)))[0]
    return qt
