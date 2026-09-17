import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from torchvision import transforms

from myutils import UnNormalize
from open_clip import create_model, tokenizer
from prompts.imagenet_template import openai_imagenet_template

def spatial_patch_grid(tensor, patch_size):
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = patch_size
    return (
        tensor.shape[-2] // patch_height,
        tensor.shape[-1] // patch_width,
    )

@MODELS.register_module()
class S2C2Segmentation(BaseSegmentor):
    SUPPORTED_X_OPTIONS = frozenset({
        'X_PAPER_PIPELINE',

        'X_PAPER_TAU',
        'X_PAPER_K_MIN',
        'X_PAPER_K_MAX',
        'X_PAPER_LAM',
        'X_PAPER_T_MAX',
        'X_PAPER_BETA',
        'X_PAPER_THETA_HIGH',
        'X_PAPER_DELTA',
        'X_PAPER_RESIDUAL_BETA',
        'X_PAPER_ADMIT_CAP',
        'X_PAPER_ADMIT_TRIAL',
        'X_PAPER_ADAPTIVE_ADMIT',
        'X_PAPER_GLOBAL_SCORE',
        'X_PAPER_UNSELECTED_FILL',
        'X_PAPER_RERANK_WEIGHTS',
        'X_PAPER_RERANK_CONF_WEIGHT',
        'X_PAPER_GUIDE_WEIGHT',
        'X_PAPER_LOCAL_TEXT',
        'X_PAPER_CLIPSEG_MODEL',
        'X_PAPER_CLIPSEG_QUERY_BATCH',
        'X_PAPER_MEMORY_CACHE_CROPS',
        'X_SPATIAL_REFINE',
        'X_SPATIAL_REFINE_DILATIONS',
        'X_SPATIAL_REFINE_TEMP',

        'X_ADAPTIVE_BG',

        'X_ADAPTIVE_BG_C',
    })

    PAPER_FIELD_TYPES = {
        'tau': float,
        'k_min': int,
        'k_max': 'optional_int',
        'lam': float,
        't_max': int,
        'beta': float,
        'theta_high': float,
        'delta': float,
        'residual_beta': float,
        'admit_cap': int,
        'admit_trial': bool,
        'adaptive_admit': bool,
        'global_score': str,
        'unselected_fill': float,
        'rerank_weights': 'optional_triple',
        'rerank_conf_weight': float,
        'guide_weight': float,
        'local_text': str,
        'clipseg_model': str,
        'clipseg_query_batch': int,
        'memory_cache_crops': int,
    }

    def __init__(self, clip_type, model_type, vfm_model, name_path, checkpoint=None,
                 device=torch.device('cuda'), prob_thd=0.0, logit_scale=40,
                 beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336,
                 x_options=None):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self._apply_x_options(x_options)
        self._paper_pipeline = None

        self._last_cls_scores = None

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = vfm_model
        if vfm_model == 'dino':
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        else:
            raise ValueError(
                "vfm_model must be 'dino', got %r" % (vfm_model,))

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.query_words = query_words
        self.query_to_class = list(self.query_idx)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

        if self._enabled('X_PAPER_PIPELINE'):
            self._build_paper_pipeline(device)
        self._setup_inference_options()

    def _build_paper_pipeline(self, device):
        if self.slide_crop <= 0:
            raise ValueError('S2C2 requires sliding-window inference (slide_crop > 0)')
        from models import S2C2Config, S2C2Pipeline

        config = S2C2Config(**self._paper_overrides())
        self._paper_pipeline = S2C2Pipeline(
            self.query_words, self.query_to_class, self.num_classes, config,
            device, logit_scale=self.logit_scale,
        )

    def _setup_inference_options(self):
        self._spatial_refine = int(self._opt('X_SPATIAL_REFINE', 0) or 0)
        if self._spatial_refine < 0:
            raise ValueError('X_SPATIAL_REFINE must be >= 0')
        self._refine_image = None
        temp = str(self._opt('X_SPATIAL_REFINE_TEMP', '') or '').strip()
        self._spatial_refine_temp = None
        if temp and temp.lower() not in {'none', 'null'}:
            self._spatial_refine_temp = float(temp)
            if self._spatial_refine_temp <= 0:
                raise ValueError('X_SPATIAL_REFINE_TEMP must be positive')
        raw = str(self._opt('X_SPATIAL_REFINE_DILATIONS', '') or '').strip()
        self._spatial_refine_dilations = None
        if raw:
            parts = [p for p in raw.strip('[]() ').split(',') if p.strip()]
            self._spatial_refine_dilations = tuple(int(p) for p in parts)
            if not self._spatial_refine_dilations \
                    or min(self._spatial_refine_dilations) < 1:
                raise ValueError('X_SPATIAL_REFINE_DILATIONS must be positive ints')
        self._adaptive_bg = str(
            self._opt('X_ADAPTIVE_BG', '') or '').strip().lower()
        if self._adaptive_bg in {'0', 'false', 'no', 'off', 'none'}:
            self._adaptive_bg = ''
        if self._adaptive_bg not in {'', 'otsu'}:
            raise ValueError(
                "X_ADAPTIVE_BG must be ''/otsu, got %r"
                % (self._adaptive_bg,))
        self._adaptive_bg_c = float(self._opt('X_ADAPTIVE_BG_C', 0.5))

    def _paper_overrides(self):
        overrides = {}
        for field, kind in self.PAPER_FIELD_TYPES.items():
            name = 'X_PAPER_' + field.upper()
            if name not in self.x_options and name not in os.environ:
                continue
            raw = self._opt(name)
            overrides[field] = self._coerce_paper_value(name, raw, kind)
        return overrides

    @staticmethod
    def _coerce_paper_value(name, raw, kind):
        text = str(raw).strip()
        lowered = text.lower()
        try:
            if kind is bool:
                if lowered in {'', '0', 'false', 'no', 'off', 'none'}:
                    return False
                if lowered in {'1', 'true', 'yes', 'on'}:
                    return True
                raise ValueError('expected a boolean')
            if kind is int:
                return int(text)
            if kind is float:
                return float(text)
            if kind is str:
                return text
            if kind == 'optional_float':
                return None if lowered in {'none', 'null', ''} else float(text)
            if kind == 'optional_int':
                return None if lowered in {'none', 'null', ''} else int(text)
            if kind == 'optional_str':
                return None if lowered in {'none', 'null', ''} else text
            if kind == 'optional_triple':
                if lowered in {'none', 'null', ''}:
                    return None
                parts = [p for p in text.strip('[]() ').split(',') if p.strip()]
                if len(parts) != 3:
                    raise ValueError('expected three comma-separated numbers')
                return tuple(float(p) for p in parts)
        except (TypeError, ValueError) as error:
            raise ValueError('%s=%r is invalid: %s' % (name, raw, error))
        raise ValueError('unknown coercion %r for %s' % (kind, name))

    def _apply_x_options(self, opts):
        self.x_options = {}
        for key, value in (opts or {}).items():
            if not key.startswith('X_'):
                raise ValueError('x_options keys must start with X_, got %r' % key)
            if key not in self.SUPPORTED_X_OPTIONS:
                raise ValueError(
                    'unknown switch %r; supported switches: %s'
                    % (key, sorted(self.SUPPORTED_X_OPTIONS)))
            self.x_options[key] = str(value)
        if self.x_options:
            print('Active switches: %s' % self.x_options)

    def _opt(self, name, default=None):
        if name not in self.SUPPORTED_X_OPTIONS:
            raise ValueError('unknown switch %r' % name)
        return self.x_options.get(name, os.environ.get(name, default))

    def _enabled(self, name, default=False):
        value = self._opt(name, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {'', '0', 'false', 'no', 'off', 'none'}

    @torch.no_grad()
    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]


        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)

        imgs_norm = imgs_norm.half()

        feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
        nb_im = feat.shape[0]
        patch_size = self.vfm.patch_embed.patch_size
        I, J = spatial_patch_grid(imgs_norm, patch_size)
        ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)

        if self._needs_cls_scores():
            image_features, cls_feature = self.clip.encode_image(
                img.half(), external_feats=ex_feats, beta=self.beta,
                gamma=self.gamma, return_cls=True)
            cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
            self._last_cls_scores = (cls_feature @ self.query_features.T).float()
        else:
            image_features = self.clip.encode_image(img.half(),
                                                   external_feats=ex_feats,
                                                   beta=self.beta,
                                                   gamma=self.gamma)
            self._last_cls_scores = None

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits

    def _forward_proxy_context(self, inputs, batch_img_metas):
        if self.slide_crop > 0:
            return self.forward_slide(
                inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        return self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        window_pipeline = None
        if self._paper_pipeline is not None:
            window_pipeline = self._paper_pipeline

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)
                crop_seg_logit = self.forward_feature(crop_img).detach()

                if window_pipeline is not None:

                    cls_scores = None
                    if self._last_cls_scores is not None:
                        cls_scores = self._last_cls_scores[0]
                    crop_seg_logit = window_pipeline.run_window(
                        crop_img, crop_seg_logit, cls_scores=cls_scores)

                torch.cuda.empty_cache()

                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat

        img_size = img_metas[0]['ori_shape'][:2]
        return nn.functional.interpolate(preds, size=img_size, mode='bilinear')

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        if self._spatial_refine > 0:
            self._refine_image = inputs

        seg_logits = self._forward_proxy_context(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _needs_cls_scores(self):
        return (self._paper_pipeline is not None
                and self._paper_pipeline.cfg.global_score == 'cls')

    def get_eval_metrics(self):
        if self._paper_pipeline is None:
            return {}
        return {
            key: round(float(value), 6)
            for key, value in self._paper_pipeline.evaluation_metrics().items()
        }

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            raw_logits = seg_logits[i]
            seg_logits = self._class_probabilities(raw_logits, self.logit_scale)

            sharp = None
            if self._spatial_refine > 0 and self._refine_image is not None:

                from models import spatial_refine
                image = F.interpolate(
                    self._refine_image[i:i + 1].float(),
                    size=seg_logits.shape[-2:], mode='bilinear',
                    align_corners=False)[0]
                refine_kwargs = {}
                if self._spatial_refine_dilations is not None:
                    refine_kwargs['dilations'] = self._spatial_refine_dilations
                temp = self._spatial_refine_temp
                if temp is not None and float(temp) != float(self.logit_scale):
                    # The label map is read from a sharper (temperature `temp`) refined map;
                    # the background threshold keeps the calibrated `logit_scale` probabilities.
                    # Only the label map is kept so that two full-resolution maps never coexist.
                    sharp = spatial_refine.refine(
                        image, self._class_probabilities(raw_logits, temp),
                        self._spatial_refine, **refine_kwargs).argmax(0, keepdim=True)
                if sharp is None or self.prob_thd > 0:
                    # The calibrated refined map is only consumed by the background
                    # threshold; without a background class it is not needed.
                    seg_logits = spatial_refine.refine(
                        image, seg_logits, self._spatial_refine, **refine_kwargs)

            seg_pred = sharp if sharp is not None else seg_logits.argmax(
                0, keepdim=True)
            max_prob = seg_logits.max(0, keepdim=True)[0]
            seg_pred[max_prob < self._bg_threshold(max_prob)] = 0

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def _class_probabilities(self, raw_logits, scale):
        probs = (raw_logits * float(scale)).softmax(0)
        num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
        if num_cls != num_queries:
            probs = torch.stack([
                probs[self.query_idx == class_index].max(0).values
                for class_index in range(num_cls)
            ], dim=0)
        return probs

    def _bg_threshold(self, max_prob):
        if not self._adaptive_bg or self.prob_thd <= 0:
            return self.prob_thd
        values = max_prob.reshape(-1).float()
        threshold = _otsu_threshold(values)
        return min(max(threshold, float(self.prob_thd)), self._adaptive_bg_c)

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass

def _otsu_threshold(values, bins=256):
    hist = torch.histc(values, bins=bins, min=0.0, max=1.0)
    total = hist.sum()
    if float(total) <= 0:
        return 0.0
    prob = hist / total
    centers = (torch.arange(bins, device=values.device, dtype=torch.float32)
               + 0.5) / bins
    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * centers, dim=0)
    denom = omega * (1.0 - omega)
    variance = (mu[-1] * omega - mu) ** 2 / denom.clamp_min(1e-12)
    variance = torch.where(denom > 0, variance, torch.zeros_like(variance))
    return float(centers[int(torch.argmax(variance))])

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]

    class_names = [item.rstrip('\r\n') for item in class_names]
    return class_names, class_indices
