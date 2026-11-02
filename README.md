# CS5787 HW3

## Overview

This project explores diffusion models for image generation and editing using PyTorch and pretrained diffusion models.

Three experiments are conducted:

- **Question 4 — Forward Diffusion:** Visualize how Gaussian noise gradually corrupts images at different diffusion time steps.
- **Question 5 — DDIM Sampling:** Compare image generation quality using different numbers of reverse diffusion steps.
- **Question 6 — Prompt-to-Prompt Image Editing:** Explore how cross-attention control enables image editing while preserving the original image composition.

The goal is to understand the diffusion process, evaluate sampling efficiency, and investigate controllable image generation.

## Input Images and Models

**Question 4 — Forward Diffusion**

Five input images are used:

- `mona.jpg`
- `cat.jpg`
- `liberty.jpg`
- `nyc.jpg`
- `cornell.jpg`

All images are resized to 128 × 128 pixels and normalized to [-1, 1].

**Question 5 — DDIM Sampling**

We use the pretrained `google/ddpm-cat-256` model from Hugging Face Diffusers.

**Question 6 — Prompt-to-Prompt**

We use `CompVis/stable-diffusion-v1-4` with the Prompt-to-Prompt implementation to explore cross-attention-based image editing.

## Model Architecture

### Forward Diffusion

The forward diffusion process gradually adds Gaussian noise to an image according to:

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
\]

where \(x_0\) is the original image, \(x_t\) is the noisy image, and \(\epsilon\) represents Gaussian noise.

We also calculate Mean Squared Error (MSE) to measure image corruption.

### DDIM Sampling

DDIM performs reverse diffusion to generate images from noise.

We compare different numbers of inference steps to evaluate the trade-off between image generation quality and computational efficiency.

### Prompt-to-Prompt

Prompt-to-Prompt modifies cross-attention maps during Stable Diffusion sampling.

The experiments include word replacement, prompt refinement, attention re-weighting, and LocalBlend to control image editing while preserving the original composition.

## Experimental Settings

| Parameter | Value |
|---|---|
| Q4 Image Size | 128 × 128 |
| Q4 Total Diffusion Steps | 1000 |
| Q4 Noise Schedule | Linear |
| Q4 Beta Range | 0.0001–0.02 |
| Q4 Evaluated Time Steps | 0, 50, 100, 300, 500 |
| Q5 Model | google/ddpm-cat-256 |
| Q5 Sampler | DDIM |
| Q5 Inference Steps | 500, 100, 10, 5 |
| Q5 Images per Configuration | 5 |
| Q5 Random Seed | 6 |
| Q5 Eta | 0.0 |
| Q6 Model | Stable Diffusion v1.4 |
| Q6 Diffusion Steps | 50 |
| Q6 Guidance Scale | 7.5 |
| Q6 Initial Example Seed | 8888 |

## Running the Notebooks

Install the required packages:

```bash
pip install torch torchvision numpy matplotlib pillow diffusers transformers accelerate jupyter
```

Open the notebooks in Jupyter Notebook, Google Colab, or VS Code.

**Questions 4 and 5:**

Open `CS5787HW3 Q4.5.ipynb`.

Make sure the five input images are available in the correct file directory.

Run the cells in order to generate the forward diffusion visualizations, calculate MSE, and compare DDIM sampling results.

**Question 6:**

Open `HW3-6(1).ipynb`.

The notebook requires the Prompt-to-Prompt helper files `ptp_utils.py` and `seq_aligner.py`, together with access to the Stable Diffusion v1.4 model.

Run the setup cells before performing the image editing experiments.

**Note:** GPU acceleration is recommended, particularly for Questions 5 and 6. Exact generated images may vary depending on the environment and software versions.

## Experimental Results

### Question 4 — Forward Diffusion

Five images were evaluated at different diffusion time steps.

| Time Step | Observation |
|---|---|
| 0 | Original image without added noise |
| 50 | Small amount of noise; image remains recognizable |
| 100 | More visible noise with most image structures preserved |
| 300 | Significant noise and reduced image clarity |
| 500 | Strong noise corruption; original structure becomes difficult to recognize |

As the diffusion time step increases, the images become progressively noisier and lose their original visual features.

The MSE calculations provide a quantitative measure of the difference between the original and noisy images.

The results demonstrate how the forward diffusion process gradually destroys image information.

### Question 5 — DDIM Sampling

We generated five images for each sampling configuration using 500, 100, 10, and 5 inference steps.

| DDIM Steps | Observation |
|---|---|
| 500 | High-quality cat images with recognizable structures and fine details |
| 100 | Images remain visually similar to the 500-step outputs |
| 10 | Main structures remain visible, but fine details become less clear |
| 5 | Significant degradation in image quality and object coherence |

The 100-step results were visually close to the 500-step results, suggesting that DDIM can generate good-quality images with substantially fewer inference steps.

However, reducing the number of steps to 10 or 5 resulted in noticeable quality degradation.

This demonstrates the trade-off between sampling efficiency and image quality.

### Question 6 — Prompt-to-Prompt Image Editing

We tested different cross-attention editing techniques using Stable Diffusion.

| Experiment | Prompt Modification | Observation |
|---|---|---|
| Word Replacement | Cat → Dog | Successfully changed the animal while preserving similar poses and backgrounds |
| Word Replacement | Cat → Person | Initial results were incomplete; adjusting attention settings and using LocalBlend improved the output |
| Word Replacement | Sofa → Bed | Modified the furniture while preserving the teddy bear |
| Prompt Refinement | Castle → Castle in Winter | Added snow while preserving the original castle structure |
| Prompt Refinement | Cake → Strawberry Cake | Results improved after adjusting attention injection |
| Attention Re-weighting | Smiling × 5 | Produced a wider smile |
| Attention Re-weighting | Fluffy | Produced limited visible changes |
| Attention Re-weighting | Strawberry × 5 | Increased pink coloration and strawberry-related details |

The experiments show that Prompt-to-Prompt can modify selected objects and attributes while preserving much of the original image composition.

Word replacement worked particularly well when the original and target objects had similar shapes, such as replacing a cat with a dog.

More significant structural changes, such as replacing a cat with a person, were more difficult and required additional attention parameter adjustments.

LocalBlend helped restrict modifications to selected image regions, while attention re-weighting allowed certain attributes to become more visually prominent.

However, attention adjustments did not always produce predictable results.

## Conclusion

The experiments demonstrate three important aspects of diffusion-based image generation.

In **Question 4**, increasing the diffusion time step progressively added noise and reduced image clarity, illustrating the forward diffusion process.

In **Question 5**, DDIM generated visually similar images using 100 and 500 inference steps, showing that fewer sampling steps can maintain strong image quality. However, very small step counts reduced image quality.

In **Question 6**, Prompt-to-Prompt demonstrated that cross-attention manipulation enables controlled image editing without retraining the diffusion model. Simple word replacements and attribute modifications worked well, while larger structural changes required additional tuning.

Overall, these experiments demonstrate how diffusion models support image generation, efficient sampling, and text-guided image editing.

## Repository Files

```text
CS5787-HW3/
├── CS5787HW3 Q4.5.ipynb
├── HW3-Q6.ipynb
└── README.md
```

Additional resources required to reproduce the notebooks include the five input images for Question 4 and the Prompt-to-Prompt helper files for Question 6.

## References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.*
- Song et al. (2020). *Denoising Diffusion Implicit Models.*
- Hertz et al. (2022). *Prompt-to-Prompt Image Editing with Cross Attention Control.*
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.*
- [Google Research Prompt-to-Prompt Repository](https://github.com/google/prompt-to-prompt)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)

## AI Usage Disclosure

ChatGPT was used to assist with understanding diffusion model concepts, reviewing parts of the Python implementation, interpreting experimental results, and organizing the README documentation.

The experiments and observations are documented in the accompanying notebooks. The final code and reported results should be reviewed and verified before submission.
