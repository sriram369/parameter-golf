# v3: Full SOTA Stack with PyTorch SDPA

**val_bpb: TBD** | 1xH100 SXM (test) / 8xH100 (final)

## Changes from SOTA (2026-03-23_LeakyReLU_LegalTTT_ParallelMuon)

| Change | Reason |
|--------|--------|
| flash_attn_3 → PyTorch SDPA | FA3 not available on PyTorch 2.4.0 pods |

All other components identical to SOTA 1.1194.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| Attention | PyTorch SDPA (Flash Attention 2 backend) |
| TTT | Legal score-first TTT (for final eval) |

## SDPA Replacement

```python
# Before (Flash Attention 3):
y = flash_attn_3_func(q, k, v, causal=True)

# After (PyTorch SDPA with Flash Attention 2 backend):
q_sdpa = q.transpose(1, 2)   # [B,T,H,D] -> [B,H,T,D]
k_sdpa = k.transpose(1, 2)   # [B,T,Hkv,D] -> [B,Hkv,T,D]
v_sdpa = v.transpose(1, 2)   # [B,T,Hkv,D] -> [B,Hkv,T,D]
y = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True).transpose(1, 2)
```

Both FA3 and PyTorch SDPA use scale=1/√(head_dim) and GQA. Numerically equivalent.

## Run Command (8xH100)

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Run | Steps | val_bpb | Artifact |
|-----|-------|---------|---------|
| 1xH100 test (741 steps, 600s) | 741 | 1.4573 (pre-EMA) | 6.1MB |
| 8xH100 target (compute grant) | ~7200 | ~1.119 (predicted) | ~16MB |

Note: 1xH100 result is degraded due to LR warmdown tuned for 8xH100 timing.
8xH100 run pending OpenAI compute grant approval.
