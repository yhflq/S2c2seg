
import torch
import torch.nn.functional as F


DILATIONS = (1, 2, 4, 8, 12, 24)

MAX_SIDE = 1024

CHANNEL_CHUNK = 64


def _neighbors(maps, dilation):
    channels, height, width = maps.shape
    unfolded = F.unfold(maps[None], kernel_size=3,
                        dilation=dilation, padding=dilation)
    unfolded = unfolded.view(channels, 9, height, width)
    index = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=maps.device)
    return unfolded.index_select(1, index)


def _affinity(image, dilations, eps=1e-8):
    logits = []
    for dilation in dilations:
        diff = (_neighbors(image, dilation)
                - image[:, None]).abs().mean(dim=0)
        logits.append(diff)
    stacked = torch.cat(logits, dim=0)
    stacked = -stacked / (stacked.mean() + eps)
    return torch.softmax(stacked, dim=0)


@torch.no_grad()
def refine(image, probs, num_iter, dilations=DILATIONS, max_side=MAX_SIDE,
           channel_chunk=CHANNEL_CHUNK):
    if num_iter <= 0:
        return probs
    height, width = probs.shape[-2:]
    scale = max(height, width) / float(max_side)
    work_size = (height, width) if scale <= 1 else (
        int(round(height / scale)), int(round(width / scale)))

    img = image.float()[None]
    if img.shape[-2:] != work_size:
        img = F.interpolate(img, size=work_size, mode='bilinear',
                            align_corners=False)
    maps = probs.float()[None]
    if maps.shape[-2:] != work_size:
        maps = F.interpolate(maps, size=work_size, mode='bilinear',
                             align_corners=False)
    img, maps = img[0], maps[0]

    affinity = _affinity(img, dilations)
    chunk = max(1, int(channel_chunk))
    for _ in range(int(num_iter)):
        updated = torch.zeros_like(maps)
        for start in range(0, maps.shape[0], chunk):
            part = maps[start:start + chunk]
            offset = 0
            for dilation in dilations:
                weights = affinity[offset:offset + 8]
                updated[start:start + chunk] += (
                    _neighbors(part, dilation) * weights[None]).sum(dim=1)
                offset += 8
        maps = updated
    if maps.shape[-2:] != (height, width):
        maps = F.interpolate(maps[None], size=(height, width),
                             mode='bilinear', align_corners=False)[0]
    return maps
