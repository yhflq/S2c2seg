import torch
import torch.nn.functional as F

from .preprocess import to_uint8_image, unnormalize

MODEL_ALIASES = {
    "rd16": "CIDAS/clipseg-rd16",
    "rd64": "CIDAS/clipseg-rd64",
    "rd64-refined": "CIDAS/clipseg-rd64-refined",
}


class CLIPSegBackend:

    def __init__(self, query_words, cfg, device):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.cfg = cfg
        self.device = device
        self.query_words = list(query_words)
        model_id = MODEL_ALIASES.get(cfg.clipseg_model, cfg.clipseg_model)
        self.model_id = model_id
        self.processor = CLIPSegProcessor.from_pretrained(model_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_id)
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.conditional_embeddings = self._encode_conditions()

    @torch.no_grad()
    def _encode_conditions(self):
        if self.cfg.local_text == "plain":
            tokens = self.processor.tokenizer(
                self.query_words, return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            return self.model.clip.get_text_features(**tokens).detach()

        from prompts.imagenet_template import openai_imagenet_template

        embeddings = []
        for word in self.query_words:
            prompts = [template(word) for template in openai_imagenet_template]
            tokens = self.processor.tokenizer(
                prompts, return_tensors="pt", padding=True)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            features = self.model.clip.get_text_features(**tokens)
            embeddings.append(features.mean(dim=0, keepdim=True))
        return torch.cat(embeddings, dim=0).detach()

    def _pixel_values(self, normalized_crop):
        from PIL import Image

        batched = normalized_crop if normalized_crop.ndim == 4 \
            else normalized_crop.unsqueeze(0)
        unnormalized = unnormalize(batched)
        array = to_uint8_image(unnormalized[0])
        processed = self.processor.image_processor(
            images=[Image.fromarray(array)], return_tensors="pt")
        return processed["pixel_values"].to(self.device)

    @torch.no_grad()
    def dense_probabilities(self, normalized_crop, output_size):
        pixel_values = self._pixel_values(normalized_crop)
        vision_outputs = self.model.clip.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        activations = [
            vision_outputs.hidden_states[layer + 1]
            for layer in self.model.extract_layers
        ]
        count = len(self.query_words)
        batch = self.cfg.clipseg_query_batch or count
        chunks = []
        for start in range(0, count, batch):
            end = min(start + batch, count)
            expanded = [
                activation.expand(end - start, -1, -1).contiguous()
                for activation in activations
            ]
            logits = self.model.decoder(
                expanded,
                self.conditional_embeddings[start:end],
                return_dict=True,
            ).logits
            if logits.ndim == 2:
                logits = logits.unsqueeze(0)
            chunks.append(logits.float())
        native = torch.cat(chunks, dim=0)
        probabilities = torch.sigmoid(native)
        if tuple(probabilities.shape[-2:]) != tuple(output_size):
            probabilities = F.interpolate(
                probabilities.unsqueeze(0), size=tuple(output_size),
                mode="bilinear", align_corners=False).squeeze(0)
        return probabilities
