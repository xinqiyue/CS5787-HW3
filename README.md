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

GPU acceleration is recommended for diffusion model inference and image editing.

---

# Question 4 — Forward Diffusion Process

## Objective

The purpose of this experiment is to understand how the forward diffusion process gradually transforms an original image into a noisy representation.

We visualize images at different diffusion time steps and measure the amount of corruption using Mean Squared Error (MSE).

## Methodology

Five input images are loaded, resized to 128 × 128 pixels, and normalized to the range [-1, 1].

We use a linear noise schedule with 1,000 diffusion steps:

\[
T = 1000
\]

\[
\beta_t \in [0.0001, 0.02]
\]

The forward diffusion equation is:

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
\]

where:

- \(x_0\) is the original image.
- \(x_t\) is the noisy image at time step \(t\).
- \(\bar{\alpha}_t\) is the cumulative product of the noise schedule coefficients.
- \(\epsilon \sim \mathcal{N}(0,I)\) is Gaussian noise.

We visualize the images at the following time steps:

**0, 50, 100, 300, and 500.**

To measure image corruption, we calculate MSE between the original image and its noisy representation:

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(x_{t,i}-x_{0,i})^2
\]

## Experimental Results

The generated image grids show that increasing the diffusion time step progressively reduces the visibility of the original image.

| Time Step | Observation |
|---|---|
| 0 | Original image without added noise |
| 50 | Small amount of noise; image structure remains clear |
| 100 | More visible noise, but the image remains recognizable |
| 300 | Significant noise; image details become difficult to distinguish |
| 500 | Strong noise corruption; much of the original image structure is obscured |

The notebook also displays MSE values for each image and time step, providing a quantitative measure of corruption.

## Discussion

The results demonstrate that forward diffusion progressively destroys visual information by adding Gaussian noise.

At smaller time steps, much of the original image structure is preserved. As the time step increases, the noise contribution becomes stronger, making the underlying image increasingly difficult to recognize.

The experiment also demonstrates that the closed-form forward diffusion equation allows us to directly generate a noisy image at a selected time step without simulating every previous step.

## Key Takeaway

Forward diffusion gradually converts structured image data into noisy representations. This process forms the foundation of diffusion models, which learn to reverse the corruption process during image generation.

---

# Question 5 — DDIM Sampling Efficiency

## Objective

The objective of this experiment is to evaluate how the number of reverse diffusion sampling steps affects generated image quality.

We investigate whether DDIM can generate visually similar images with fewer inference steps.

## Model and Experimental Setup

We use the pretrained diffusion model:

**`google/ddpm-cat-256`**

The model is loaded using the Hugging Face `DDIMPipeline`.

We compare four different numbers of inference steps:

**500, 100, 10, and 5 steps.**

For each configuration:

- Five images are generated.
- The random seed is fixed at 6.
- The DDIM parameter `eta` is set to 0.0.
- The same pretrained model is used.

Fixing the random seed helps make comparisons more consistent across different sampling configurations.

## Experimental Results

| DDIM Steps | Observations |
|---|---|
| 500 | High-quality cat images with recognizable structures and fine details |
| 100 | Images appear almost identical to the 500-step results, maintaining similar poses, colors, and backgrounds |
| 10 | Overall structures remain similar, but fine details such as fur and whiskers become less distinct |
| 5 | Significant degradation in image quality; some outputs become difficult to recognize as cats |

## Discussion

The experiment demonstrates that reducing the number of DDIM sampling steps does not necessarily cause a proportional reduction in image quality.

In our results, images generated with 100 sampling steps are visually very similar to those generated with 500 steps.

This suggests that many reverse sampling steps can be removed while maintaining much of the generated image quality.

However, when the number of sampling steps is reduced to 10 or 5, the model has fewer opportunities to refine the generated image.

As a result, fine visual details deteriorate, and the generated objects become less coherent.

This illustrates the trade-off between sampling efficiency and generation quality.

Although fewer sampling steps generally reduce the amount of denoising computation, this experiment focuses on qualitative image comparisons rather than directly measuring runtime.

## Key Takeaway

DDIM can maintain strong visual quality with substantially fewer inference steps. However, reducing the number of steps too aggressively can cause significant losses in image detail and structure.

In this experiment, 100 steps provided results visually close to the 500-step baseline.

---

# Question 6 — Prompt-to-Prompt Image Editing

## Objective

The purpose of this experiment is to investigate how Prompt-to-Prompt image editing can modify specific objects or attributes in a generated image while preserving the original composition.

Instead of retraining a diffusion model, Prompt-to-Prompt manipulates cross-attention maps during the image generation process.

We explore four main techniques:

1. Cross-attention visualization
2. Word replacement
3. Prompt refinement
4. Attention re-weighting

## Model Configuration

The experiment uses Stable Diffusion with the Prompt-to-Prompt implementation.

| Parameter | Value |
|---|---|
| Stable Diffusion model | `CompVis/stable-diffusion-v1-4` |
| Diffusion steps | 50 |
| Guidance scale | 7.5 |
| Low-resource mode | False |
| Maximum token slots | 77 |
| Initial example seed | 8888 |

We generate an initial image and reuse its latent representation for editing experiments to help preserve the original composition.

## Experiment 1 — Cross-Attention Visualization

**Original prompt:**

`a photo of a cat sitting on a bench`

We visualize cross-attention maps associated with individual prompt tokens.

These maps illustrate how different words influence spatial regions during image generation.

### Observation

Different prompt words produce different spatial attention patterns.

This helps explain how cross-attention can be used to control specific regions and attributes within an image.

### Interpretation

Cross-attention provides a connection between textual descriptions and spatial image features.

By modifying these attention patterns, Prompt-to-Prompt can guide changes to selected visual elements without necessarily regenerating the entire scene.

---

## Experiment 2 — Word Replacement

Word replacement changes a specific word in the original prompt while attempting to preserve the overall image structure.

### Experiment A: Cat to Dog

**Original prompt:**

`a photo of a cat sitting on a bench`

**Modified prompt:**

`a photo of a dog sitting on a bench`

### Observation

The cat-to-dog replacement worked well.

The generated dogs maintained similar poses and positions to the original cats, while the background remained largely unchanged.

Compared with ordinary prompt-based generation, Prompt-to-Prompt provided stronger preservation of the original scene.

### Experiment B: Cat to Person

**Original prompt:**

`a photo of a cat sitting on a bench`

**Modified prompt:**

`a photo of a person sitting on a bench`

### Observation

The initial replacement did not successfully generate a convincing person.

The output contained incomplete human-like structures, including a white box and a pair of legs.

This suggests that replacing an animal with a human is more difficult than replacing one animal with another because their shapes and spatial structures differ significantly.

### Adjustment: Cross-Attention Injection

We experimented with reducing the word-specific attention injection for `person`.

This allowed the model more flexibility to generate the new object rather than strongly preserving the original cat structure.

We also applied `LocalBlend` to restrict the editing region.

After adjusting the attention replacement settings and using LocalBlend, a person appeared in the edited image.

### Additional Experiment: Sofa to Bed

We also tested replacing `sofa` with `bed` in a teddy bear image.

The experiment demonstrated that local editing could change the furniture while preserving the teddy bear.

### Interpretation

Word replacement works particularly well when the original and target objects have similar spatial structures.

For larger changes in object geometry, reducing attention injection and applying localized editing can improve results.

---

## Experiment 3 — Prompt Refinement

Prompt refinement modifies an existing prompt by adding new descriptive information while attempting to preserve the original image.

### Experiment A: Castle in Winter

**Original prompt:**

`a photo of a castle on a hill`

**Modified prompt:**

`a photo of a castle on a hill in winter`

### Observation

The refined image preserved the original castle structure while introducing winter characteristics, including snow.

This demonstrated that Prompt-to-Prompt can modify environmental attributes without substantially changing the scene composition.

### Experiment B: Strawberry Cake

**Original prompt:**

`cake`

**Modified prompt:**

`strawberry cake`

We experimented with different attention injection settings and LocalBlend.

We also tested:

`cream cake` → `strawberry cream cake`

### Observation

Stronger attention injection sometimes preserved too many characteristics of the original cake, limiting the introduction of strawberry-related details.

Reducing the cross-attention injection parameter to 0.2 produced better results for the cream cake refinement.

### Interpretation

Prompt refinement is effective when adding attributes that do not require major structural changes.

However, strong preservation constraints can limit the model's ability to incorporate new visual features.

Adjusting attention injection allows greater flexibility while retaining important elements of the original image.

---

## Experiment 4 — Attention Re-Weighting

Attention re-weighting adjusts the influence of selected words in the prompt.

Instead of replacing or adding words, we modify how strongly the model attends to particular descriptions.

### Experiment A: Smiling Old Man

**Prompt:**

`a smiling old man`

We increased the attention weight of the word `smiling` by a factor of 5.

### Observation

The generated man displayed a wider smile after increasing the attention weight.

This demonstrates that attention re-weighting can emphasize selected visual attributes.

### Experiment B: Fluffy White Dog

**Prompt:**

`a fluffy white dog`

We experimented with changing the attention weight associated with `fluffy`.

### Observation

The adjustment produced little visible difference in the generated image.

This indicates that attention re-weighting does not always produce predictable visual changes.

### Experiment C: Strawberry Cream Cake

We combined prompt refinement with attention re-weighting.

The attention weight of `strawberry` was increased by a factor of 5.

### Observation

The generated cake became pinker and displayed more strawberry-related details.

This demonstrated that combining refinement and attention re-weighting can strengthen selected visual characteristics.

---

## Question 6 — Overall Findings

The experiments demonstrate several strengths and limitations of Prompt-to-Prompt editing.

### What Worked Well

**1. Preserving image composition**

Replacing a cat with a dog maintained much of the original pose, position, and background.

**2. Adding environmental attributes**

Adding `in winter` changed the appearance of the castle scene while preserving its underlying structure.

**3. Localized image modifications**

LocalBlend helped restrict changes to relevant image regions, reducing unwanted changes to the surrounding scene.

**4. Controlling visual attributes**

Increasing attention weights for words such as `smiling` and `strawberry` produced noticeable visual differences.

### Challenges and Limitations

**1. Large geometry changes are difficult**

Replacing a cat with a person was less successful because the target object required a significantly different spatial structure.

**2. Attention injection requires tuning**

Different edits required different cross-attention replacement settings.

**3. Attention re-weighting is not always effective**

Changing the importance of `fluffy` produced little visible improvement.

**4. Excessive preservation can restrict editing**

Strong attention injection sometimes prevented the model from introducing new strawberry features.

### What Surprised Me

One of the most interesting findings was that changing a single word in the prompt could significantly alter the target object while preserving much of the surrounding image.

By reusing the initial latent representation and manipulating attention, Prompt-to-Prompt allowed targeted modifications without requiring additional model training.

However, the experiments also showed that successful editing depends on the relationship between the original and target objects and the choice of attention parameters.

## Key Takeaway

Prompt-to-Prompt enables controlled image editing by manipulating cross-attention during diffusion sampling.

The technique is particularly effective for small semantic changes, attribute refinement, and localized editing.

However, edits requiring major geometric transformations or stronger visual modifications may require additional parameter tuning and may still produce imperfect results.

---

# Overall Conclusion

Through Questions 4–6, we explored three important components of diffusion-based image generation.

**Question 4 — Forward Diffusion**

Adding Gaussian noise progressively destroys image structure, demonstrating how diffusion models transform images into increasingly noisy representations.

**Question 5 — DDIM Sampling**

Reducing the number of reverse diffusion steps can improve computational efficiency while maintaining strong visual quality. However, using too few steps can significantly reduce output quality.

**Question 6 — Prompt-to-Prompt Editing**

Manipulating cross-attention enables targeted image editing while preserving much of the original composition. The effectiveness of these edits depends on object similarity, attention injection settings, and the requested modification.

Together, these experiments demonstrate how diffusion models can be used not only for image generation but also for efficient sampling and controllable image editing.

---

# Reproducibility

To reproduce these experiments:

1. Install the required Python libraries.
2. Open `CS5787HW3 Q4.5.ipynb` and make the five input images available in the notebook's working directory.
3. Run Question 4 to visualize forward diffusion and MSE.
4. Run Question 5 to compare DDIM sampling at 500, 100, 10, and 5 steps.
5. Open `HW3-6(1).ipynb`.
6. Ensure `ptp_utils.py` and `seq_aligner.py` are available, and configure access to the Stable Diffusion checkpoint.
7. Run the setup and initial image-generation cells before executing the Prompt-to-Prompt editing experiments.

The experiments use fixed seeds where specified in the notebooks, but exact outputs may vary with software versions and environment configurations.

The image comparisons presented in this project are primarily qualitative.

---

# References

- Ho et al. (2020). *Denoising Diffusion Probabilistic Models.*
- Song et al. (2020). *Denoising Diffusion Implicit Models.*
- Hertz et al. (2022). *Prompt-to-Prompt Image Editing with Cross Attention Control.*
- Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.*
- [Google Research — Prompt-to-Prompt GitHub Repository](https://github.com/google/prompt-to-prompt)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers)

**Attribution:** The Question 6 notebook is adapted from the Google Research Prompt-to-Prompt implementation, which includes an Apache License 2.0 notice. The original copyright and license notices should be retained in redistributed source code.
