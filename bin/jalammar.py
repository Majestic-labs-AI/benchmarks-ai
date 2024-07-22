# from https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/Simple_Transformer_Language_Model.ipynb

# This is probably the ugliest code I've ever written, and would appreciate leniency
# and forgiveness when reading

# Errors on T4 & L4 with Meta-Llama-3-8B:
# "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU"

import ipdb, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print the memory allocated on GPU before loading the model
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB", file=sys.stderr)
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB", file=sys.stderr)
# print(f"Memory reserved: {torch.cuda.memory_summary(device=None, abbreviated=False)} %s", file=sys.stderr)

script_path = os.path.dirname(__file__)
llama_path = os.path.join(script_path, "../../Phi-3-mini-4k-instruct/")
tokenizer = AutoTokenizer.from_pretrained(llama_path)
model = AutoModelForCausalLM.from_pretrained(llama_path, output_hidden_states=True)
# Move the model to the GPU
model.to(device)

# Print the memory allocated on GPU after loading the model
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB", file=sys.stderr)
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB", file=sys.stderr)

# Thank you, Mary Oliver
text = """
You do not have to be good.
You do not have to walk on your knees
for a hundred miles through the desert repenting.
You only have to let the soft animal of your body
love what it loves.
Tell me about despair, yours, and I will tell you mine.

This is the beginning of a moving poem by Mary Oliver, which explores a
spectrum of human emotions. Contrast with Tennyson's poem, Ulysses.
"""

# Tokenize the input string, move to GPU
input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
# ipdb.set_trace()

# Run the model
start_time = time.time()
output = model.generate(input.input_ids.to(device), max_length=512, do_sample=False)
elapsed_time = time.time() - start_time
print('\n',tokenizer.decode(output[0]), file=sys.stderr)
print(f"elapsed time: {elapsed_time:.1f} seconds", file=sys.stderr)
print(f"tokens: {len(output[0])}, tokens / second {len(output[0])/elapsed_time:.1f}", file=sys.stderr)
