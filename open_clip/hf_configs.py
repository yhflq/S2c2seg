# HF architecture dict:
arch_dict = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    "roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    "xlm-roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    "mt5": {
        "config_names": {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # T5 uses the decoder vocabulary size for its token embeddings.
            "context_length": "",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "num_heads",
            "layers": "num_layers",
            "layer_attr": "block",
            "token_embeddings_attr": "embed_tokens"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    "bert": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
        },
        "pooler": "cls_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/m2m_100
    "m2m_100": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "encoder_attention_heads",
            "layers": "encoder_layers",
        },
        "pooler": "cls_pooler",
    },
}
