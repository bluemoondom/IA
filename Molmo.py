# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 19:10:10 2024

@author: dominika
"""

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import matplotlib.pyplot as plt

myimagepath = "roylichtenstein.jpg"
myimage = Image.open(myimagepath)
plt.figure()
plt.imshow(myimage) 
plt.show()

processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

inputs = processor.process(
    images=[myimage],
    text="Describe me this picture."
)
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer,
    temperature=0.7,
    do_sample=True
)

generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(generated_text)