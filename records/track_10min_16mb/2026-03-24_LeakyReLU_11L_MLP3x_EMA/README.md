# LeakyReLU² + 11L + MLP3x + EMA

**val_bpb: TBD** | 1xH100 SXM (initial test run)

## Changes from Baseline

| Change | Impact |
|--------|--------|
| NUM_LAYERS 9→11 | +2 transformer blocks, more capacity within 16MB budget |
| MLP_MULT 2→3 | 3× MLP expansion, more expressive MLPs |
| LeakyReLU(0.5)² | Preserves negative gradient flow, eliminates dead neurons (-0.003 BPB per ablations in PR #493) |
| EMA (decay=0.997) | Exponential moving average weights used for eval/export |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| Weight avg | EMA(0.997) |
| Quantization | int8+zlib (baseline) |

## Key Innovation: LeakyReLU(0.5)²

```python
# Before (baseline relu²)
x = torch.relu(self.fc(x)).square()

# After (leaky relu²)
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

LeakyReLU with slope 0.5 preserves negative gradient flow through the MLP.
The squaring step still produces non-negative outputs, maintaining the relu²
inductive bias while eliminating dead neurons.

## Run Command

```bash
NUM_LAYERS=11 MLP_MULT=3 LEAKY_RELU_SLOPE=0.5 EMA_ENABLED=1 EMA_DECAY=0.997 \
RUN_ID=v1_leakyrelu_11l_mlp3x_ema \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Results

TBD — run in progress.
