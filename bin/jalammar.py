# from https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/Simple_Transformer_Language_Model.ipynb

# Errors on T4:
# "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenizer = AutoTokenizer.from_pretrained("distilgpt2") 
# model = AutoModelForCausalLM.from_pretrained("distilgpt2", output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") 
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", output_hidden_states=True)
# Move the model to the GPU
model.to(device)

text = """
You do not have to be good.
You do not have to walk on your knees
for a hundred miles through the desert repenting.
You only have to let the soft animal of your body
love what it loves.
Tell me about despair, yours, and I will tell you mine.
"""

# Tokenize the input string
input = tokenizer.encode(text, return_tensors="pt")
# Move input tensors to the GPU
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run the model
output = model.generate(input, max_length=128, do_sample=False)

# Print the output
print('\n',tokenizer.decode(output[0]))

