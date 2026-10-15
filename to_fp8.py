import torch
from safetensors.torch import load_file, save_file
import sys

if len(sys.argv) != 3:
    sys.exit("usage: python to_fp8.py <input.safetensors> <output.safetensors>")

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Loading {input_path}...")
state_dict = load_file(input_path)

print("Converting to FP8 e4m3fn...")
fp8_dict = {}
converted = 0
skipped = 0

for key, tensor in state_dict.items():
    # Only quantize float tensors, skip embeddings/norms/biases
    if tensor.dtype in (torch.float32, torch.bfloat16, torch.float16) and \
       tensor.ndim >= 2 and \
       not any(x in key for x in ['norm', 'bias', 'embed', 'pos_embed', 'time_embed']):
        fp8_dict[key] = tensor.to(torch.float8_e4m3fn)
        converted += 1
    else:
        fp8_dict[key] = tensor
        skipped += 1

print(f"Converted: {converted} tensors | Skipped: {skipped} tensors")
print(f"Saving to {output_path}...")
save_file(fp8_dict, output_path)
print("Done.")