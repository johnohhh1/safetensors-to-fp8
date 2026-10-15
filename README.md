# safetensors-to-fp8

A single-file script that converts a `.safetensors` model (for example a diffusion UNet/DiT or an LLM checkpoint) to **FP8 (`float8_e4m3fn`)**, roughly halving its size compared with FP16/BF16 so it fits in less VRAM.

It quantizes only the large weight matrices and leaves the precision-sensitive tensors alone:

- **Converted:** float32 / float16 / bfloat16 tensors with 2 or more dimensions
- **Kept as-is:** 1-D tensors and any tensor whose name contains `norm`, `bias`, `embed`, `pos_embed` or `time_embed`

## Install

Needs Python 3.9+ and PyTorch 2.1 or newer (the first version with `torch.float8_e4m3fn`).

```bash
git clone https://github.com/johnohhh1/safetensors-to-fp8
cd safetensors-to-fp8
pip install -r requirements.txt
```

## Usage

```bash
python to_fp8.py <input.safetensors> <output.safetensors>
```

Example:

```bash
python to_fp8.py flux1-dev.safetensors flux1-dev-fp8_e4m3fn.safetensors
# Loading flux1-dev.safetensors...
# Converting to FP8 e4m3fn...
# Converted: N tensors | Skipped: M tensors
# Saving to flux1-dev-fp8_e4m3fn.safetensors...
# Done.
```

The output loads anywhere that supports FP8 weights, such as ComfyUI's "Load Diffusion Model" node with `weight_dtype` set to `fp8_e4m3fn`.

**Notes**

- The whole model is loaded into RAM, so you need roughly the input file's size in free memory.
- This is plain casting with no scaling factors, which works well for most diffusion models. Check output quality before you delete the original.

## Contributing

Issues and pull requests are welcome, for example an `--e5m2` option, a custom skip list, or per-tensor scaling.

## License

MIT, see [LICENSE](LICENSE).
