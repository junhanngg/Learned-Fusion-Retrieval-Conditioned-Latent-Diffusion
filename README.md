# Learned Fusion Retrieval-Conditioned Latent Diffusion for Spectrogram Inpainting

## Overview

This project investigates **music inpainting**: reconstructing a missing segment of a musical recording from the surrounding context. Audio is represented as a **mel spectrogram**, a contiguous time region is masked, and the model is asked to reconstruct the missing content while preserving the observed context.

The experiments use classical piano recordings derived from the **MAESTRO dataset** and progress from deterministic convolutional baselines to latent diffusion and retrieval-conditioned generation.

The main modelling stages are:

1. **CNN Autoencoder** — convolutional baseline with MBConv, squeeze-and-excitation, and explicit temporal-context modelling.
2. **Attention U-Net** — skip-connected baseline with attention-gated skips, multi-scale mask injection, and a dilated bottleneck.
3. **Variational Autoencoder (VAE)** — learns the compressed latent representation used by the diffusion models.
4. **Baseline Latent Diffusion** — performs masked denoising in VAE latent space.
5. **Retrieval-Guided Sampling** — explores whether clean latent references retrieved from similar musical context can guide sampling.
6. **Retrieval-Conditioned Diffusion U-Net** — integrates retrieved references directly into the diffusion model using learned fusion.

The final notebook contains architecture inspection, training-history analysis, checkpoint loading, quantitative evaluation, qualitative spectrogram comparison, retrieval ablations, and cross-model comparison.

---

## Project Structure

```text
.
├── assets/
├── data/
│   ├── configs/
│   │   └── configs.yaml
│   └── data2parquet.ipynb
├── models/
│   ├── __init__.py
│   ├── base_diffusion.py
│   ├── cnn_ae.py
│   ├── enhance_diffusion.py
│   ├── unet.py
│   └── vae.py
├── notebooks/
│   ├── cnn_ae.ipynb
│   ├── di2_exploration.ipynb
│   ├── diffusion.ipynb
│   ├── enhanced_di2.ipynb
│   ├── reverse_pipeline.ipynb
│   └── unet.ipynb
├── runs/
│   ├── base_diffusion_history.csv
│   ├── base_diffusion_results.csv
│   ├── cnn_ae_history.csv
│   ├── cnn_ae_results.csv
│   ├── enhance_diffusion_history.csv
│   ├── enhance_diffusion_results.csv
│   ├── unet_history.csv
│   ├── unet_results.csv
│   └── vae_history.csv
├── test_audio/
│   ├── base_diffusion/
│   ├── cnn_ae/
│   ├── enhanced_diffusion/
│   └── unet/
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── data_load.py
│   ├── logging.py
│   ├── losses.py
│   └── mel2wav.py
├── project_main.ipynb
└── project-report.pdf
```

### Directory Roles

- `models/` contains the modular PyTorch implementations used by the main notebook.
- `utils/` contains shared data-loading, loss, checkpointing, logging, and mel-to-waveform utilities.
- `data/` contains the preprocessing notebook and YAML configuration.
- `notebooks/` contains exploratory and model-specific development notebooks.
- `runs/` stores exported training histories and evaluation results.
- `assets/` stores cached experimental outputs and plotting assets used by the notebook.
- `test_audio/` stores reconstructed waveform examples grouped by model and gap size.
- `project_main.ipynb` is the main reproducibility, evaluation, and comparison notebook.

---

## Data

The original audio source is the **MAESTRO** piano-performance dataset.

Data preprocessing is performed separately in:

```text
data/data2parquet.ipynb
```

Preprocessing settings are controlled through:

```text
data/configs/configs.yaml
```

The processed dataset used by the main notebook is hosted on Hugging Face:

**Dataset:**  
https://huggingface.co/datasets/han2o/grant-ortsaem-processedV3

The main notebook loads the `short_gaps` variant with:

- `masked_spectrogram` as the model input
- `spectrogram` as the clean target
- `mask` as the binary inpainting mask
- streaming enabled
- batch size `16`
- up to `10,000` training samples
- up to `1,000` validation samples
- up to `1,000` test samples

The principal evaluation gap buckets are:

```text
0.50 s, 0.75 s, 1.00 s, 1.50 s, 1.75 s, 2.00 s
```

Because the stored gap duration is continuous, evaluation assigns each example to its nearest target gap bucket.

---

## Model Weights and Cached Artifacts

Large checkpoints are not stored directly in the Git repository. They are hosted on Hugging Face and downloaded in the notebook with `hf_hub_download`.

**Model repository:**  
https://huggingface.co/han2o/inpaint_diffusion

Important files include:

```text
cnn_ae/best_model.pt
unet/best_model.pt
vae/best_model.pt
base_diffusion/best_model.pt
enhance_diffusion/best_model.pt

enhance_model_retrieval_bank/enhance_train_bank.pt
enhance_model_retrieval_bank/enhance_val_bank.pt
```

Additional cached inference outputs are also downloaded where appropriate so that expensive experiments do not need to be repeated simply to reproduce plots and comparisons.

---

## Installation

The project is implemented in Python using PyTorch.

A minimal environment for the imports used directly by `project_main.ipynb` is:

```bash
python -m venv .venv
source .venv/bin/activate

pip install \
  torch \
  numpy \
  pandas \
  matplotlib \
  tqdm \
  torchinfo \
  huggingface_hub \
  jupyter
```

The preprocessing and audio-conversion utilities may require additional packages depending on the implementations in `data/` and `utils/`.

A CUDA-capable GPU is strongly recommended for diffusion training and full reverse-process evaluation.

The project models were trained using **Google Colab**.

---

## Quick Start

### 1. Clone or Open the Repository

Run Jupyter from the repository root so that local modules and relative paths such as `./runs/` and `./assets/` resolve correctly.

```bash
jupyter notebook
```

Then open:

```text
project_main.ipynb
```

---

### 2. Load the Processed Data

The notebook constructs the data module using the Hugging Face dataset:

```python
dm = DataModule(
    repo_id="han2o/grant-ortsaem-processedV3",
    variant="short_gaps",
    input_key="masked_spectrogram",
    target_key="spectrogram",
    mask_key="mask",
    batch_size=16,
    num_workers=0,
    streaming=True,
    max_train_samples=10000,
    max_val_samples=1000,
    max_test_samples=1000,
)

train_loader, val_loader, test_loader = dm.setup()
```

---

### 3. Load Pretrained Checkpoints

The notebook automatically downloads model checkpoints from:

```python
repo_id = "han2o/inpaint_diffusion"
```

For example:

```python
best_model = hf_hub_download(
    repo_id=repo_id,
    filename="cnn_ae/best_model.pt"
)
```

---

### 4. Reproduce the Evaluation

For fast reproduction, use the saved CSV files in `runs/` together with the provided model checkpoints.

The computationally expensive training and full diffusion-evaluation calls are intentionally left commented in the main notebook.

They can be re-enabled when full retraining or regeneration is required.

---

# Model Architectures

## 1. CNN Autoencoder

The CNN autoencoder is the first convolutional inpainting baseline.

Its input consists of two channels:

```text
masked spectrogram + binary mask
```

The model combines:

- lightweight residual convolution blocks at high resolution
- **MBConv** blocks for efficient deeper feature extraction
- **squeeze-and-excitation (SE)** channel attention
- a dedicated **Temporal Context Block** with multiple temporal receptive fields
- an encoder-bottleneck-decoder structure
- bilinear upsampling in the decoder

The model predicts a residual only inside the masked region:

\[
\text{output} = x + m \odot \text{pred}
\]

where:

- \(x\) is the masked input
- \(m\) is the binary gap mask
- `pred` is the predicted reconstruction

This prevents the model from unnecessarily rewriting already-observed context.

---

## 2. Attention U-Net

The U-Net extends the deterministic baseline with direct encoder-to-decoder skip connections.

Inpainting-specific additions include:

- **attention-gated skip connections**
- **multi-scale mask injection**
- a **dilated bottleneck** with dilation rates `[2, 4, 8]`
- MBConv encoder blocks
- GroupNorm for small-batch stability
- learnable strided convolutions for downsampling
- bilinear interpolation for upsampling

The attention gates reduce the influence of unreliable encoder features originating from the masked region while preserving useful surrounding context.

The model uses four encoder stages with the channel progression:

```text
32 → 64 → 128 → 256 → 512
```

---

## 3. Variational Autoencoder

The VAE provides the latent representation used by the diffusion stages.

The configured latent space uses:

```text
latent_channels = 8
```

The VAE uses:

- residual blocks
- GroupNorm
- dilated processing
- self-attention
- latent mean and variance prediction

Its training objective combines:

- masked reconstruction
- multi-resolution supervision
- gradient consistency
- KL regularisation

The total objective can be written as:

$$
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\beta \mathcal{L}_{\text{KL}}
$$

with a KL warm-up schedule.

During training, the encoder is exposed to both clean and masked spectrograms so that the latent representation remains useful when conditioned on incomplete inputs.

After training, the VAE is frozen and reused by the diffusion models.

---

## 4. Baseline Latent Diffusion

The baseline diffusion model performs denoising in the VAE latent space rather than directly at full spectrogram resolution.

The conditional input concatenates:

```text
current noisy latent
+ masked-input latent
+ self-conditioning latent
+ latent-space mask
```

The diffusion U-Net includes:

- sinusoidal timestep embeddings
- FiLM-style timestep conditioning
- residual diffusion blocks
- SE channel attention
- dilated bottleneck blocks
- self-attention
- encoder-decoder skip connections

Training uses masked noise prediction so that optimisation focuses more strongly on the missing region.

The implementation uses a cosine diffusion schedule with:

```text
1000 diffusion steps
```

for the main experiments.

During reverse diffusion, the known region is explicitly reinserted at every timestep to prevent observed context from drifting.

---

# Retrieval Guidance

## Retrieval Bank

Retrieval uses the known context surrounding a gap to search for similar training examples.

For each bank item, the system stores:

1. a context-derived embedding used for nearest-neighbour search
2. the corresponding full clean latent used as the retrieval payload
3. the context latent
4. the latent mask
5. associated metadata

The provided enhanced-model banks contain:

```text
Training bank:   4,000 items
Validation bank: 1,000 items
Embedding size:  1,024
Latent shape:    (8, 16, 235)
```

The embeddings are L2-normalised and retrieval is based on contextual similarity.

---

## Retrieval-Guided Sampler Exploration

The notebook first studies retrieval as an external sampler-level guidance mechanism.

Rather than averaging all retrieved clean latents into a single target, the top candidates are kept separate.

This avoids forcing structurally different but contextually similar completions into a blurred average.

Experiments include:

- dynamic guidance
- selective mixing
- candidate weighting
- diversity filtering
- guidance scheduling
- self-conditioning ablations

These experiments motivate the final learned retrieval-conditioned model.

---

# Retrieval-Conditioned Diffusion U-Net

The final model integrates retrieval directly into the diffusion architecture.

The configuration used in the notebook is:

```text
Latent channels:              8
Base channels:                128
Timestep embedding:           256
Retrieval conditioning dim:   256
Retrieval encoder channels:   64
Retrieval cross-attn blocks:  2
Fusion mode:                  learned_fusion
Parameters:                   57,855,273
```

Training retrieves multiple candidate references and learns how strongly each candidate should influence the denoising process.

The main retrieval configuration uses:

```text
top_k = 5
max_keep = 3
diversity_threshold = 0.92
masked_weight = 5.0
Min-SNR gamma = 5.0
```

Retrieval strength is randomised during training across:

```text
No retrieval
Weak retrieval
Medium retrieval
Strong retrieval
```

This prevents the model from depending on retrieved examples unconditionally and encourages it to remain useful when retrieval quality varies.

---

# Training Configuration

| Model | Optimiser | Initial LR | Main Objective | Early-Stopping Monitor |
|---|---|---:|---|---|
| CNN Autoencoder | Adam | `1e-3` | Masked L1 | Validation gap RMSE |
| U-Net | AdamW | `5e-4` | Masked L1 + gradient term | Validation gap RMSE |
| VAE | AdamW | `1e-4` | Reconstruction + KL | Validation loss |
| Base Latent Diffusion | AdamW | `5e-5` | Masked noise MSE + latent L1 | Validation noise loss |
| Retrieval Diffusion | AdamW | `1.25e-5` | Retrieval-conditioned masked noise MSE + latent L1 | Validation noise loss |

Training calls are preserved in `project_main.ipynb` but are commented out where cached histories and pretrained checkpoints are available.

---

# Evaluation

Models are evaluated separately across six gap-duration buckets:

```text
0.50 s
0.75 s
1.00 s
1.50 s
1.75 s
2.00 s
```

The notebook reports the following metrics.

### Gap MAE

Mean absolute error measured only within the missing region.

### Gap RMSE

Root mean squared error measured within the missing region.

### Full MAE

Mean absolute error over the complete reconstructed spectrogram.

### Full RMSE

Root mean squared error over the complete reconstructed spectrogram.

### PSNR

Peak signal-to-noise ratio of the reconstructed spectrogram.

Both quantitative and qualitative comparisons are included.

---

## Final Enhanced-Model Ablation

Among the evaluated enhanced-diffusion inference modes, the notebook selects:

```text
enhanced_base_no_sc
```

as the best configuration according to weighted gap RMSE:

```text
Weighted Gap RMSE = 0.728963
```

The compared enhanced modes are:

```text
enhanced_base_no_sc
enhanced_base_with_sc
enhanced_sampler_no_sc
enhanced_sampler_with_sc
```

The final notebook then compares the selected enhanced model against:

```text
CNN Autoencoder
U-Net
Base Diffusion
Enhanced Diffusion
```

across:

- gap RMSE
- PSNR
- qualitative spectrogram reconstruction

---

# Reproducibility

The repository is structured so that the expensive parts of the project do not need to be rerun for basic verification.

## Provided for Reproducibility

- preprocessed dataset on Hugging Face
- pretrained model checkpoints on Hugging Face
- prebuilt retrieval banks
- saved training histories
- saved evaluation CSVs
- cached sampler outputs
- modular model implementations
- preprocessing notebook
- main evaluation notebook

---

## Recommended Reproduction Path

For a quick reproduction:

1. Install the project dependencies.
2. Open `project_main.ipynb`.
3. Stream the processed dataset from Hugging Face.
4. Download the pretrained checkpoints.
5. Load the precomputed CSV results.
6. Regenerate the training curves and model-comparison plots.
7. Run selected inference examples if GPU resources are available.

For a full reproduction, uncomment the corresponding training or evaluation cells and regenerate the checkpoints, retrieval banks, and result CSVs.

---

# Main Notebook Roadmap

`project_main.ipynb` is organised in the following order:

```text
1. Load Data
2. Convolutional Autoencoder
3. U-Net
4. Baseline Latent Diffusion
   ├── Variational Autoencoder
   └── Diffusion U-Net
5. Quantitative Baseline Comparison
6. Qualitative Baseline Comparison
7. Retrieval Guidance Sampler Exploration
8. Retrieval-Conditioned Diffusion U-Net
9. Final Comparison Across All Models
```

This progression is intentional: each stage introduces additional modelling capacity and motivates the next experiment.

---

# Outputs

Training and evaluation outputs are stored primarily in:

```text
runs/
```

Example files:

```text
cnn_ae_history.csv
cnn_ae_results.csv
unet_history.csv
unet_results.csv
vae_history.csv
base_diffusion_history.csv
base_diffusion_results.csv
enhance_diffusion_history.csv
enhance_diffusion_results.csv
```

These CSVs are loaded directly into pandas for analysis and plotting.

Reconstructed audio examples are stored under:

```text
test_audio/
```

and are grouped by model and gap size.

---

# Notes

- Full diffusion inference with a 1000-step reverse process is computationally expensive.
- Precomputed evaluation CSVs are included to make result reproduction substantially faster.
- Large model checkpoints and retrieval-bank tensors are stored on Hugging Face rather than committed directly to Git.
- The main notebook is intended primarily for reproducibility, evaluation, visualisation, and final model comparison.
- Exploratory work is separated into the notebooks under `notebooks/`.
