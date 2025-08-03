# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:26:01 2024

@author: dominika
"""

from transformers import AutoProcessor, BarkModel
import torch
from scipy.io.wavfile import write as write_wav
import numpy as np

#import os
#os.environ["SUNO_OFFLOAD_CPU"] = "True"
#os.environ["SUNO_USE_SMALL_MODELS"] = "True"

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)
model.enable_cpu_offload()
#model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float32).to(device)
inputs = processor("Incroyable! Je peux générer du son.", voice_preset="v2/fr_speaker_5").to(device)
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

#audio_array /=1.414
#audio_array *= 32767
#audio_array = audio_array.astype(np.int16)
#print(audio_array)
sample_rate = model.generation_config.sample_rate
write_wav("bark_generation.wav", rate=sample_rate, data=audio_array)