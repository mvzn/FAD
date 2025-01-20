# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from frechet_audio_distance import FrechetAudioDistance

background_embds_path = "C:/path/to/bg_emb.npy"
eval_embds_path = "C:/path/to/eval_emb.npy"

frechet = FrechetAudioDistance(
    model_name="encodec",
    sample_rate=48000,
    channels=2,
    verbose=False,
)

fad_score = frechet.score(
    "C:/path/to/bg_data",
    "C:/path/to/eval_data",
    background_embds_path=background_embds_path,
    eval_embds_path=eval_embds_path,
    dtype="float32"
)

print(fad_score)
