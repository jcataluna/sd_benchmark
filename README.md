# SD benchmark

## Requirements

```
python3 -m venv sd
source sd/bin/activate
pip install -r requirements.txt
```

## Run benchmark

```
python sd_benchmark.py
```

## Example output

```
$ python sd_benchmark.py
Running with args:  Namespace(bs=4, compile=True, gpu=0, n=10, safety_checker=True, vae_slicing=True)
VAE slicing enabled
Using batch size 4
100%|███████████████████████████| 10/10 [03:06<00:00, 18.61s/it]
Mean 4.65 sec/img ± 0.01 (95%)
```

## Available options

Check `python sd_benchmark.py --help` for available options:

- `--vae-slicing/--no-vae-slicing`: enabling VAE slicing allows a bigger batch size
- `--safety-checker/--no-safety-checker`: enable NSFW filter
- `--compile/--no-compile`: compile the Unet (faster at the expense of much slower first batch)
- `-n`: number of test iterations
- `--gpu`: run inference on this GPU id
- `--bs`: batch size
