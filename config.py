"""
CONTRASTIVE LANGUAGE-IMAGE PRETRAINING(CLIP):
CLIP was developed by OpenAI and has been a revolution allowing open world object recognization by
"predicting with image and text pairing go together". CLIP will be able
to predict this by learning cosine similarity between image and text features.

"""

# ---------- Training Parameters ---------- #
parameters = {
    "emb_dim": 32,
    "vit_width": 9,
    "image_size": (28, 28),
    "patch_size": (14, 14),
    "n_channels": 1,
    "vit_layers": 3,
    "vit_heads" : 3,
    "vocab_size": 256,
    "text_width": 32,
    "max_sequence": 32,
    "text_heads": 8,
    "text_layers": 4,
    "lr": 1e-3,
    "epochs": 100,
    "batch_size": 128
}

