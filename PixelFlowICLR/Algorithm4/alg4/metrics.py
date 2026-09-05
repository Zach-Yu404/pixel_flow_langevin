"""Image-quality metrics on [0,1] tensors (N,3,H,W). PSNR/SSIM: piq (data_range 1). LPIPS: official AlexNet
(richzhang/PerceptualSimilarity `lpips` package, inputs mapped to [-1,1]); piq-VGG (replace_pooling) kept as the
historical column. FID: InceptionV3 block-3 features."""
import numpy as np
import torch
import piq

_LPIPS_PIQ, _OFFICIAL, _INCEPTION = {}, {}, {}


def to01(x):
    """[-1,1] sampler output -> clamped [0,1]."""
    return (x.clamp(-1, 1) + 1) / 2


def psnr(recon01, gt01):
    return piq.psnr(recon01, gt01, data_range=1.0).item()


def ssim(recon01, gt01):
    return piq.ssim(recon01, gt01, data_range=1.0).item()


def lpips_piq(recon01, gt01, device):
    if device not in _LPIPS_PIQ:
        _LPIPS_PIQ[device] = piq.LPIPS(replace_pooling=True).to(device)
    return _LPIPS_PIQ[device](recon01.to(device), gt01.to(device)).item()


def _official(net, device):
    key = (net, str(device))
    if key not in _OFFICIAL:
        import lpips as _lp
        _OFFICIAL[key] = _lp.LPIPS(net=net, verbose=False).to(device).eval()
    return _OFFICIAL[key]


@torch.no_grad()
def lpips_official(recon01, gt01, device, net="alex", batch=32):
    model = _official(net, device)
    vals = []
    for i in range(0, recon01.shape[0], batch):
        r = recon01[i:i + batch].to(device) * 2.0 - 1.0
        g = gt01[i:i + batch].to(device) * 2.0 - 1.0
        vals.append(model(r, g).flatten().cpu())
    return torch.cat(vals).mean().item()


def lpips_alex(recon01, gt01, device):
    return lpips_official(recon01, gt01, device, "alex")


def _inception(device):
    if device not in _INCEPTION:
        from piq.feature_extractors import InceptionV3
        _INCEPTION[device] = InceptionV3(output_blocks=[3], normalize_input=True, requires_grad=False,
                                         use_fid_inception=True).to(device).eval()
    return _INCEPTION[device]


@torch.no_grad()
def inception_feats(images01, device, batch=32):
    net = _inception(device)
    feats = []
    for i in range(0, images01.shape[0], batch):
        f = net(images01[i:i + batch].to(device))[0]
        feats.append(f.reshape(f.shape[0], -1).cpu())
    return torch.cat(feats).numpy()


def fid(gt_feats, recon_feats):
    from scipy import linalg
    mu1, mu2 = gt_feats.mean(0), recon_feats.mean(0)
    s1, s2 = np.cov(gt_feats, rowvar=False), np.cov(recon_feats, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))
