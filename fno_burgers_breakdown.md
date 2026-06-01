# FNO (Burgers) parameter & pipeline breakdown

This document mirrors `fno_darcy_breakdown.md` for the **Burgers** setup, and explains how the **number of parameters (in millions)** changes with the number of Fourier layers.

## 1) Pipeline overview (input → output)

The forward path is identical to Darcy and is implemented in:
- `neuralop/models/fno.py` (for `arch="fno"`)
- `neuralop/models/fno_staged.py` (for `arch="fno_staged"`)

Pipeline steps:
1. **Positional embedding** (default `"grid"`)  
2. **Lifting ChannelMLP** (2× 1×1 convs)
3. **Fourier layers** (SpectralConv + fno_skip + ChannelMLP + channel_mlp_skip)
4. **Projection ChannelMLP**

## 2) Key default values (Burgers)

From `config/models.py` → `FNO_Small2d` (original):
- `C_in=1`, `C_out=1`, `H=24`, `n_modes=[16,16]`, `n_layers=4`

From `config/burgers_staged_config.py` (updated staged):
- `C_in=1`, `C_out=1`, `H=24`, `n_layers=6`
- `n_modes_per_layer = [[16,16],[16,16],[16,16],[8,8],[8,8],[8,8]]`

Assumptions:
- Real‑valued inputs → `m2_eff = m2//2 + 1`
- `lift_ratio = 2`, `proj_ratio = 2`, `channel_mlp_expansion = 0.5`
- `fno_skip = "linear"`, `channel_mlp_skip = "soft-gating"`
- `positional_embedding = "grid"` adds 2 input channels

## 3) Parameter formulas (per layer)

Let:
- `H = hidden_channels`
- `m1, m2` be the layer’s `n_modes`
- `m2_eff = m2//2 + 1` (real data)
- `h_mlp = round(H * channel_mlp_expansion)`

### Per‑layer parameters (one Fourier layer)

**SpectralConv (Fourier weights + bias)**  
```
weight params = 2 * H * H * m1 * m2_eff   # complex params → 2 real
bias params   = H
```

**fno_skip ("linear")**
```
fno_skip params = H * H
```

**channel_mlp (2 layers, Conv1d with bias)**
```
channel_mlp params = (H*h_mlp + h_mlp) + (h_mlp*H + H)
                   = 2 * H * h_mlp + h_mlp + H
```

**channel_mlp_skip ("soft-gating")**
```
channel_mlp_skip params = H
```

**Per‑layer total**
```
layer_params =
  (2*H*H*m1*m2_eff + H)   # SpectralConv
  + (H*H)                 # fno_skip
  + (2*H*h_mlp + h_mlp + H)  # channel_mlp
  + (H)                   # channel_mlp_skip
```

## 4) Lifting + Projection MLPs

**Lifting MLP**  
```
lift_in = C_in + d = 1 + 2 = 3
lift_hidden = round(lift_ratio * H) = 48
lifting_params = (3*48 + 48) + (48*24 + 24) = 1,368
```

**Projection MLP**  
```
proj_hidden = round(proj_ratio * H) = 48
projection_params = (24*48 + 48) + (48*1 + 1) = 1,249
```

## 5) Numeric breakdown for Burgers

### Shared constants
```
H = 24
h_mlp = round(24 * 0.5) = 12
m2_eff(16) = 16//2 + 1 = 9
m2_eff(8)  = 8//2 + 1 = 5
```

### Per‑layer totals

**Layer with n_modes=[16,16]**
```
SpectralConv = 2*24*24*16*9 + 24 = 165,912
fno_skip     = 24*24              =     576
channel_mlp  = 2*24*12 + 12 + 24  =     612
mlp_skip     = 24                 =      24
layer_total  = 167,124
```

**Layer with n_modes=[8,8]**
```
SpectralConv = 2*24*24*8*5 + 24 = 46,104
fno_skip     = 24*24            =    576
channel_mlp  = 2*24*12 + 12 + 24 =   612
mlp_skip     = 24               =    24
layer_total  = 47,316
```

### Total parameters

**Original Burgers FNO (4 layers, all [16,16])**
```
total_params = 1,368 + 1,249 + 4 * 167,124
             = 671,113  ≈ 0.671M
```

**Staged Burgers (6 layers: 3×[16,16], 3×[8,8])**
```
total_params = 1,368 + 1,249 + 3 * 167,124 + 3 * 47,316
             = 645,937  ≈ 0.646M
```

So the 6‑layer tapered model **has slightly fewer parameters** than the original 4‑layer model.

## 6) Where the values come from (code pointers)

| Component | Code path | Notes |
|---|---|---|
| Model config (original) | `config/models.py` → `FNO_Small2d` | `n_modes=[16,16]`, `hidden_channels=24`, `n_layers=4` |
| Model config (staged) | `config/burgers_staged_config.py` | `n_modes_per_layer` (6 layers) |
| Forward pipeline | `neuralop/models/fno.py`, `neuralop/models/fno_staged.py` | lifting → Fourier layers → projection |
| SpectralConv | `neuralop/layers/spectral_convolution.py` | FFT / contract / iFFT |
| fno_skip | `neuralop/layers/skip_connections.py` | `Flattened1dConv` for `"linear"` |
| channel_mlp | `neuralop/layers/channel_mlp.py` | 2× Conv1d(1×1) + GELU |
