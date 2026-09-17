import numpy as np

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def unnormalize(image, mean=CLIP_MEAN, std=CLIP_STD):
    output = image.clone().float()
    squeezed = output[0] if output.ndim == 4 else output
    for channel, m, s in zip(squeezed, mean, std):
        channel.mul_(s).add_(m)
    return output


def to_uint8_image(image):
    array = image.detach().permute(1, 2, 0).cpu().numpy()
    return (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
