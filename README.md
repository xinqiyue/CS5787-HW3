# CS 5787 Deep Learning — Homework 3 (Questions 4–6)

## Project Overview

This project explores diffusion-based image generation through three experiments:

- **Question 4: Forward Diffusion** — Investigating how Gaussian noise gradually corrupts images over time.
- **Question 5: DDIM Sampling** — Evaluating how the number of reverse diffusion steps affects generated image quality.
- **Question 6: Prompt-to-Prompt Image Editing** — Exploring how cross-attention control enables text-guided image editing while preserving the original image composition.

These experiments demonstrate the relationship between the forward diffusion process, sampling efficiency, and controllable image generation.

---

## Repository Structure

| File | Description |
|---|---|
| `CS5787HW3 Q4.5.ipynb` | Question 4: Forward Diffusion and Question 5: DDIM Sampling |
| `HW3-6(1).ipynb` | Question 6: Prompt-to-Prompt Image Editing |
| `README.md` | Project documentation |

Additional dependencies referenced by the notebooks:

- Q4 requires five input images: `mona.jpg`, `cat.jpg`, `liberty.jpg`, `nyc.jpg`, and `cornell.jpg`.
- Q6 imports `ptp_utils.py` and `seq_aligner.py` from the Prompt-to-Prompt implementation.

These additional files must be available when reproducing the experiments.

## Environment and Dependencies

The experiments use Python and PyTorch in a Jupyter Notebook environment.

**Main libraries:**

- PyTorch
- NumPy
- Matplotlib
- Pillow
- Hugging Face Diffusers
- Accelerate
- Stable Diffusion

For Question 5, the notebook installs the primary dependencies using:

```bash
pip install diffusers torch accelerate
```

GPU acceleration is recommended for diffusion model inference
