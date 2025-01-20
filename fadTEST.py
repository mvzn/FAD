# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from frechet_audio_distance import FrechetAudioDistance

background_embds_path = "C:/Users/markm/FAD/BGEmb/"
eval_embds_path = "C:/Users/markm/FAD/EVALEmb/"

frechet = FrechetAudioDistance(
    model_name="encodec",
    sample_rate=48000,
    channels=2,
    verbose=False,
)

fad_score = frechet.score(
    "C:/Users/markm/FAD/DataSet/",
    "C:/Users/markm/FAD/EvalSet/",
    dtype="float32"
)

print(fad_score)
