# CS5787-HW3
Overview
This project explores three parts of diffusion-based image generation:
- Question 4 — Forward diffusion: add Gaussian noise to five images at selected diffusion time steps and measure how far the noisy images depart from the originals.
- Question 5 — DDIM sampling: compare images generated from a pretrained cat diffusion model using different numbers of reverse-sampling steps.
- Question 6 — Prompt-to-Prompt editing: investigate how manipulating cross-attention during Stable Diffusion sampling can change a target object or attribute while preserving other image content.
The experiments and observations below are based on the submitted notebooks. Reported results are qualitative unless explicitly described as a numerical measurement.
Files
File	Purpose
CS5787HW3 Q4.5.ipynb	Forward diffusion (Q4) and DDIM sampling (Q5), including saved image grids.
HW3-6(1).ipynb	Prompt-to-Prompt experiments (Q6), code, visual outputs, and written observations.
mona.jpg, cat.jpg, liberty.jpg, nyc.jpg, cornell.jpg	Local input images referenced in Q4. These must be provided in the working directory to rerun that section.
ptp_utils.py, seq_aligner.py	External helper modules imported by the Q6 notebook; required to rerun it.


Note: The five JPEG files and the two Python helper modules are dependencies referenced by the notebooks, not files included in the two notebook uploads reviewed here.
Environment and dependencies
- Python in a Jupyter/Colab environment
- PyTorch, NumPy, Matplotlib, Pillow
- Hugging Face diffusers and accelerate
- Stable Diffusion model access and dependencies appropriate for the installed diffusers version
- GPU recommended for image generation and Prompt-to-Prompt editing
Q5 installs packages with:
pip install diffusers torch accelerate
Q6 imports ptp_utils and seq_aligner, which need to be available in the notebook's Python path. Its model-loading cell refers to CompVis/stable-diffusion-v1-4. The notebook contains a placeholder token and a warning that use_auth_token is ignored by the installed pipeline version; access/authentication may therefore need adjustment for a fresh run. Do not commit an actual access token to a public repository.
Question 4 — Visualizing the forward diffusion process
Objective
Investigate how increasing the diffusion time step changes the appearance of an image, and quantify the corruption using mean squared error (MSE).
Method
Five RGB images are loaded from disk, resized to 128 × 128, and converted to PyTorch tensors in [-1, 1]. The implementation defines T = 1,000 forward-diffusion steps and a linear beta schedule:
\[
\beta_t \in [10^{-4}, 0.02], \qquad
\alpha_t = 1-\beta_t, \qquad
\bar\alpha_t = \prod_{s=1}^{t}\alpha_s.
\]
A noisy image is sampled directly from the original image using the closed-form forward diffusion equation:
\[
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]
The notebook visualizes each image at displayed steps 0, 50, 100, 300, and 500. Step 0 is shown as the unchanged original; for other displayed steps the implementation passes the zero-based schedule index t-1 into q_sample. A new noise tensor is drawn for each image/time-step combination.

The notebook also computes
\[
\operatorname{MSE}=\frac{1}{N}\sum_{i=1}^N(x_{t,i}-x_{0,i})^2
\]
on the normalized tensor values before display clipping. Its figure contains five rows (one per input image) and five columns (one per displayed time step), with MSE annotated above each output.
Observations and interpretation
At t = 0, the displayed image is unchanged and its MSE is set to zero. As diffusion time increases, more random noise is mixed into the image and recognizable structure becomes harder to see. The progression illustrates the forward diffusion mechanism: over time, image information is progressively obscured by noise.

The MSE labels provide a numerical measure of corruption for each example, but the notebook does not provide a separate table of MSE values or an aggregated MSE curve. Therefore, no particular MSE values are claimed in this README.
Takeaway
Forward diffusion gradually transforms structured images toward noisy representations. The closed-form sampling equation lets us visualize arbitrary time steps without simulating every earlier step one by one.
Question 5 — Effect of the number of DDIM sampling steps
Objective
Examine the trade-off between the number of DDIM reverse-sampling steps and the visual quality of generated images.
Method
The notebook loads the pretrained google/ddpm-cat-256 checkpoint with DDIMPipeline. It compares 500, 100, 10, and 5 inference steps, generating five images per setting. For each setting, a PyTorch generator is reset to seed 6, and DDIM uses eta=0.0 (deterministic DDIM sampling given the initial noise and model setup).
steps_list = [500, 100, 10, 5]
n = 5
for s in steps_list:
    g = torch.Generator(device).manual_seed(6)
    results[s] = ddim(
        batch_size=n,
        num_inference_steps=s,
        eta=0.0,
        generator=g
    ).images
Keeping the starting random seed fixed supports a more meaningful visual comparison across different step counts, although the denoising trajectories and time-step schedules are not identical.
Results
Inference steps	Observations documented in the notebook
500	Cat images with clear structure and fine visual details; reference for the comparison.
100	Almost identical to the 500-step outputs in pose, color, background, and fine detail.
10	Overall structure remains similar, but fine details such as fur and whiskers become blurred or disappear.
5	Strong quality degradation; the generated cats become difficult to recognize.


Interpretation
DDIM can produce similar-looking images using many fewer steps than a long sampling trajectory: in these saved examples, the 100-step results are close to the 500-step results. However, reducing the number of steps too aggressively loses visual detail and eventually object coherence.

The experiment evaluates visual quality, not runtime, perceptual scores, or FID. Its outcome should therefore be presented as a qualitative trade-off, not as a measured speedup or a universal optimal step count. The notebook's discussion suggests that 50–100 steps can be effective, but this particular experiment actually tests 100 steps, not 50.
Takeaway
There is a practical balance between sampling efficiency and output fidelity. DDIM can reduce the number of inference steps while preserving recognizable images, but very small step counts can cause substantial quality loss.
Question 6 — Prompt-to-Prompt image editing with Stable Diffusion
Objective
Explore how modifying attention maps during image generation makes it possible to edit individual words or attributes in a text prompt while retaining much of the original composition.
Model and shared settings
The notebook adapts the Prompt-to-Prompt implementation using CompVis/stable-diffusion-v1-4. Its main configuration is:
Setting	Value
Diffusion model	CompVis/stable-diffusion-v1-4
Sampling steps	NUM_DIFFUSION_STEPS = 50
Guidance scale	GUIDANCE_SCALE = 7.5
Low-resource flag	LOW_RESOURCE = False
Maximum prompt token slots	MAX_NUM_WORDS = 77
Initial example seed	8888 (for generating the first latent/image)


The initial image is generated from “a photo of a cat sitting on a bench” using a seeded generator; the returned x_t latent is then reused for subsequent editing experiments. This controls a major source of variation across image edits.

Prompt-to-Prompt changes the attention behavior within the U-Net rather than retraining the diffusion model. The notebook implements controller classes such as AttentionReplace, AttentionRefine, AttentionReweight, and LocalBlend.
6.1 Cross-attention visualization
The notebook first records cross-attention from the image-generation process and visualizes 16 × 16 attention maps for the text tokens in “a photo of a cat sitting on a bench.” The displayed token maps help illustrate how words become associated with different spatial regions of a generated scene.

Interpretation: cross-attention offers a mechanism for steering where in the image a prompt word has influence. Attention maps are informative but should not be treated as perfect pixel-level object segmentation.
6.2 Word replacement
Word replacement modifies the target word and uses attention injection to help preserve the original layout.
Edit	Parameters	Observed result
cat → person	cross_replace_steps=0.8, self_replace_steps=0.4	Strong injection failed to form a convincing person: the notebook describes a white box and a pair of legs.
cat → dog	cross_replace_steps=0.8, self_replace_steps=0.4	Worked well: the changed animal kept a similar pose and location while the background remained largely unchanged.
cat → person	cross_replace_steps={"default_": 1., "person": .4}, self_replace_steps=0.4	The notebook tests weaker word-specific replacement to allow more freedom for the new object.
cat → person with LocalBlend	Same word-specific parameters plus LocalBlend(("cat", "person"))	A person appeared after reducing injection and using a local mask, according to the notebook's notes.
sofa → bed with LocalBlend	cross_replace_steps={"default_": 1., "bed": .2}, self_replace_steps=0.4	The furniture changed while the teddy bear was preserved, as reported in the notebook.


For some replacement examples, run_baseline=True also shows unedited-constraint baseline generations, making the composition-preservation effect of Prompt-to-Prompt visible.

Interpretation: attention injection is well suited to local semantic changes when the target objects share a roughly compatible shape and position. Large shape changes may require weaker injection and localized control, and can still fail.
6.3 Prompt refinement
Refinement keeps the original prompt and adds more descriptive words.
Original → refined prompt	Key settings	Observed result
a photo of a castle on a hill → a photo of a castle on a hill in winter	AttentionRefine, cross replacement 0.5, self replacement 0.4	The castle structure stayed consistent while snow/winter appearance was added.
cake → strawberry cake	With LocalBlend; cross/self values 0.8/0.4 and then 0.2/0.2	Tested stronger versus weaker refinement. The notebook reports that the generic starting prompt tended to preserve original cake features too strongly.
cream cake → strawberry cream cake	With LocalBlend; cross/self values 0.2/0.2	Low injection was reported to work better for introducing strawberry features.


Interpretation: an added descriptor can preserve the original image identity while changing conditions such as season or appearance. For more substantial texture and object-detail changes, overly strong attention preservation can restrict the requested edit.
6.4 Attention re-weighting
Attention re-weighting scales the influence of selected prompt tokens.
Prompt / edit	Re-weighting setting	Observed result
a smiling old man	Multiply smiling attention by 5	The man smiled more widely.
a fluffy white dog	Equalizer value -2 for fluffy, with a dog LocalBlend mask	The notebook reports little visible change.
cream cake → strawberry cream cake	Multiply strawberry attention by 5, combined with refinement and LocalBlend	The cake became pinker and displayed more strawberry features.


Interpretation: stronger token attention can emphasize some desired attributes, but not all words respond equally. Re-weighting is a controllable edit parameter rather than a guarantee that the visual property will change by a predictable amount. The negative value for fluffy is what the code specifies; it should not be described simply as a standard 0–1 reduction factor.
What worked well
1. Word replacement could preserve the original scene, especially when replacing similar objects such as a cat with a dog.
2. Refinement could make a seasonal change (adding winter) without reconstructing the entire composition.
3. LocalBlend helped constrain modifications to relevant spatial regions.
4. Increasing the attention weight of selected descriptors such as smiling or strawberry produced visible changes.
5. The experiment required no new model training for these edits.
Limitations and unsuccessful cases
- Geometry mismatch: an entirely different object, such as a person replacing a cat, may not fit the original spatial arrangement.
- Hyperparameter sensitivity: replacement fractions, attention strengths, and local masks must often be adjusted for different prompts.
- Uneven attribute control: weakening fluffy showed little change in the recorded example.
- Preservation can become a constraint: strong attention injection may retain original cake shape/color when the desired edit needs greater freedom.
- Real photographs: this workflow edits generated images. Adapting it to an existing photograph generally requires an inversion procedure to obtain a suitable latent representation.
Key takeaway
Prompt-to-Prompt provides a useful way to edit generated images through cross-attention control. It is particularly effective for localized or modest changes that should preserve the original composition, but it is less reliable when an edit demands major geometry changes or when its hyperparameters are poorly matched to the prompt.
Reproducing the experiments
1. Open CS5787HW3 Q4.5.ipynb in Jupyter or Google Colab and ensure the five local JPEG files are accessible from the notebook's working directory.
2. Run Q4 cells in order to create the forward-diffusion image/MSE grid.
3. Install the required packages, then run Q5 with an appropriate environment for downloading and executing google/ddpm-cat-256. Compare the generated grids for 500, 100, 10, and 5 steps.
4. Open HW3-6(1).ipynb and make sure ptp_utils.py and seq_aligner.py are importable. Set up working model access/authentication if necessary; do not leave private tokens in shared code.
5. Run the Q6 setup and initial image generation first. Subsequent edits reuse the initial latent x_t; execute the notebook in order so that variable exists and refers to the intended base image.
6. Review the images alongside each markdown observation, since the reported comparisons are qualitative.
Reproducibility notes
- Q4 sets PyTorch and NumPy seeds to 42, but its noisy samples are generated independently for each image and time step.
- Q5 resets a PyTorch random generator to seed 6 for each sampling-step setting and uses eta=0.0.
- Q6 seeds the initial generated image with 8888, then reuses its latent across editing experiments. Some notebook cells also use global state (prompts, x_t), so running cells out of order can give unexpected results.
- Model versions, hardware, library versions, and attention-processor compatibility can affect the ability to reproduce images exactly.
Conclusion
Questions 4–6 show three complementary aspects of diffusion models. Forward diffusion gradually destroys image structure by adding noise. DDIM sampling can reconstruct recognizable images with substantially fewer reverse steps, although an excessively short schedule degrades output quality. Prompt-to-Prompt editing changes the effect of specific words during generation through attention control, often preserving the original composition without model retraining. Together, these experiments illustrate the relationships among the diffusion process, sampling efficiency, and controllable image generation.
Attribution
The Question 6 notebook retains the original Copyright 2022 Google LLC header and Apache License 2.0 notice. Its Prompt-to-Prompt architecture and helper-code dependencies originate from that project. Keep the relevant attribution and license information when redistributing adapted source code.
