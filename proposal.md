## Project proposal form

Please provide the information requested in the following form. Try provide concise and informative answers.

**1. What is your project title?**
Context-Enhances Latent Diffusion for Music Spectrogram Inpainting

**2. What is the problem that you want to solve?**
Can stronger context conditioning improve latent diffusion models for music spectrogram inpainting, particularly as the duration of the missing segment increases?

The project focuses on the task of music inpainting, where a model is given a music clip with a missing segment and is asked to reconstruct the missing portion using the surrounding musical context. We plan to represent the audio as mel spectrogram and study the problem in the context of classical piano music.

The main motivation for this project is that while latent diffusion models have proven to be strong in generative audio tasks, reconstruction quality can still degrade as the missing gap becomes longer or more structurally ambiguous. In particular, a standard latent diffusion model may not make sufficiently effective use of the left and right surrounding musical context when restoring the missing region. Our project would therefore investigate whether stronger context conditioning can improve music inpainting performance, especially for longer missing gaps.

We expect to evaluate the models under multiple missing-gap durations (for e.g., 0.5s, 1.0s, and 2.0s) using reconstruction-based metrics such as masked-region MSE / MAE and log-spectral distance, together with qualitative spectrogram and audio comparisons. We also hope to discuss the trade-off between reconstruction quality and model complexity.


**3. What deep learning methodologies do you plan to use in your project?**

The proposed system would consist of:

- ⁠a VAE to encode spectrograms into a compact latent space,
- a conditional latent diffusion model operating in that latent space,
- ⁠and an improved conditioning mechanism that explicitly incorporates:
-     left/right context information,
-     mask-aware conditioning,
-     and gap-length conditioning.

To evaluate the effectiveness of the proposed approach, we plan to compare:

1. ⁠A simple CNN autoencoder baseline,
2.⁠ ⁠A U-Net baseline,
3.⁠ ⁠A plain conditional latent diffusion baseline,
4.⁠ ⁠and the context-enhanced latent diffusion model.


**4. What dataset will you use? Provide information about the dataset, and a URL for the dataset if available. Briefly discuss suitability of the dataset for your problem.**
We plan to represent the audio as mel spectrogram and study the problem in the context of classical piano music, likely using the MAESTRO dataset due to its high quality and relative tractability and extend it to MusicNet if the model performs better than expected.

https://www.kaggle.com/datasets/alonhaviv/the-maestro-dataset-v3-0-0

**5. List key references (e.g. research papers) that your project will be based on?**
- Chen, Y., Zhang, Q., Li, Z., Liu, Y., & Yang, F. (2025). Audio editing with diffusion models: A unified framework. arXiv. https://doi.org/10.48550/arXiv.2506.08457
- Dror, T., Shoham, I., Buchris, M., Gal, O., Permuter, H., Katz, G., & Nachmani, E. (2025). Token-based audio inpainting via discrete diffusion. arXiv. https://doi.org/10.48550/arXiv.2507.08333
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. arXiv. https://doi.org/10.48550/arXiv.2006.11239
- Lin, L., Xia, G., Zhang, Y., & Jiang, J. (2024). Arrange, inpaint, and refine: Steerable long-term music audio generation and editing via content-based controls. arXiv. https://doi.org/10.48550/arXiv.2402.09508
- Moliner, E., & Välimäki, V. (2023). Diffusion-based audio inpainting. arXiv. https://doi.org/10.48550/arXiv.2305.15266


**Please indicate whether your project proposal is ready for review (Yes/No):**
Yes

## Feedback (to be provided by the course lecturer)
