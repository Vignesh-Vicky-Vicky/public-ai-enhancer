# Detail Lab

## Version 7: generated microtexture rather than sharpening

Restart the launcher and use **AI micro-detail at output resolution**. New defaults: creativity **0.28**, AI microtexture strength **1.25**, and sharpening **0 (off)**. Microtexture strength now extends to 2.0. Reset restores these defaults. In the non-AI GPU texture mode you must raise the sharpening slider to get an effect.

The texture blend now isolates fine-scale generated luminance texture, removes the locally correlated source-edge component, and transfers the remaining texture onto the original-resolution output canvas. It amplifies weak model-supplied texture with a bounded gain, without injecting an independent grain layer. A final high-pass filter limits broad tonal changes. Strong-edge suppression, aligned section noise, and overlap blending remain. This is intended to distinguish added AI texture from increased edge contrast; it cannot guarantee that generated texture is biologically or materially accurate, or that facial geometry is perfectly unchanged.

The existing SD 1.5 model is unchanged. A blend cannot recover detailed pores or fibers if the model does not generate them. Convincing detail at 200–400% is **not established**. Version 7 has not been run or tested, per your instruction. Earlier test assertions and timing records describe older blend versions and are not validation of this revision.

## Version 6: consistency and studio controls

Restart the launcher to load the refreshed UI: framed preview panels, numbered settings groups, an accent-colored Enhance button, a fixed action bar, and dedicated **100% / 200% / 400% / Fit** buttons. Mouse-wheel zoom and linked drag-to-pan remain available. The zoom percentage is relative to source pixels so both views show the same scene area.

**Cancel** requests a stop at the next GPU checkpoint. The current operation must finish first. **Reset** clears both previews, returns all settings and zoom to defaults, and leaves saved files untouched. During a job it first requests cancellation and resets after the worker exits; late results are discarded. Reset keeps the already-loaded model available on the GPU for the next job.

Native output mode still generates at the final enlarged resolution. Overlapping sections now share a coordinate-aligned noise field, use deterministic VAE encoding, and have broad color/illumination matched to the source before feathered blending. These changes aim to reduce inconsistent texture and visible boundaries. They do not establish perfect seams, exact identity, or new factual detail. The full model prompt explicitly prioritizes identity and structure over aggressive generation.

**No execution, benchmarks, or tests were performed for version 6, as requested.** Quality, speed, cancellation behavior, and the visual layout remain unverified in this version.

## Version 5: AI at the output resolution

Restart **Start Detail Lab.bat** and select **AI micro-detail at output resolution**, then **2× output** or **4× output**. This mode enlarges the source canvas first and runs diffusion across every enlarged section, with overlapping GPU blending. It does not reduce the image to 896/512 pixels before generation or simply resize the final result. The quality preset controls steps per section; native mode always uses the full output dimensions. Clarity defaults to 0.35 to reduce reliance on sharpening.

The default **Stop unfinished jobs near 180 seconds** remains enabled. Native AI may need substantially longer, especially at 4×. Uncheck this option only if you want to allow a longer job; Cancel remains available. The limit is checked between GPU operations and is not an exact wall-clock guarantee. No incomplete output is reported as finished. GPU memory may also limit large outputs; there is no CPU inference fallback.

The existing **AI detail generation** mode remains a reduced-working-resolution alternative, with bicubic resizing afterwards. All AI modes use the full enhancement instructions. Native processing creates an opportunity for higher-frequency generated texture; it does **not** guarantee believable pores/fibers, exact identity preservation, or convincing 200–400% results with the existing SD 1.5 model. Overlap reduces boundaries but cannot guarantee uniform texture.

**Version 5 has not been run or tested, at your request.** Previous timings and tests below do not validate this mode.

A local Windows detail booster for your GTX 1660 Ti. No accounts, API keys, credits, or cloud inference. AI enhancement and texture processing run on CUDA; there is no CPU inference or model offloading. The CPU handles the desktop window, model/file loading, and image decoding/export.

## Start the app

1. Open this folder: `C:\Users\Vicky\Desktop\AI upscaler`.
2. On a fresh installation, double-click **setup.bat**. Python 3.12 is required (already present on this PC). Keep internet connected while dependencies and the public model download. Allow roughly 12 GB of free space, plus installer caches. No login is required. If a download fails, rerun setup to resume.
3. Double-click **Start Detail Lab.bat**.
4. Click **Open image**, choose **AI detail generation** and **Detail · 896**. Choose Fast if you need a shorter run.
5. Describe the subject and texture, such as “portrait, natural skin texture, fine hair” or “leather bag, fine leather grain, stitching”. Start at creativity **0.20** and texture intensity **0.8**.
6. Set **Clarity** to **1.0**, click **Boost details**, compare the previews, then **Save PNG**. Settings and elapsed time are embedded in the saved PNG.

## Zoom and compare

## Model instructions and output scale (version 4)

Every **AI detail generation** run now encodes the complete built-in enhancement brief and passes it to the diffusion model as `prompt_embeds`. The optional guidance box adds to that brief; clearing it does not remove the instructions. The entire positive and negative prompts are encoded in CLIP-sized chunks, with matching embedding lengths, so the long brief is not silently cut off at the model's normal 77-token input length. Text encoding and inference run on CUDA. This is conditioning for Stable Diffusion, not a guarantee that the model follows every instruction or recovers factual missing detail. The non-generative GPU texture mode does not use a model prompt.

Choose **1×**, **2×**, or **4×** output. Scaling is GPU bicubic resizing after AI texture enhancement; it does not run diffusion at the final enlarged resolution or provide native 4× AI reconstruction. Output is limited to 64 megapixels and can still exceed available VRAM. Both previews stay aligned to the same scene area even when their pixel dimensions differ. The viewer percentage is relative to the original image's pixels.

Version 4 changes have **not been tested**, at the user's request. Earlier timings below apply to older versions and do not predict the new long-prompt path.

### Viewer controls

Use **+ / −**, the mouse wheel over either image, **100%** for actual pixels, or **Fit** for the complete picture. Drag either image to pan both together. Double-click switches to 100%. Both views always use the same zoom and image coordinates. Zoom is limited to 5–800%; only the visible crop is rendered. At 100% and above the viewer uses nearest-neighbor display so interpolation does not hide pixel-level differences. Viewing does not change the exported image.

After setup the app is offline, including model loading. Images are never uploaded. Keep the terminal window open while using the app; closing the app releases its GPU memory.

## What the controls do

- **AI detail generation (version 3):** Stable Diffusion 1.5 proposes texture. Detail mode works at up to a 896-pixel long edge, generating overlapping 512-pixel sections with 128-pixel minimum overlap and feathered blending. It retains finer spatial detail than the old 512-pixel full-frame pass. The blend adds bounded luminance detail, rejects opposing texture and major structure changes, and protects strong edges. Same model, higher working resolution; no new download required.
- **Clarity:** independently enhances existing texture and local contrast at the original resolution on the GPU. This is sharpening, not invented AI detail. Start at 1.0; zero disables it. High values can emphasize noise or halos.
- **GPU texture boost:** fast sharpening of existing texture; it does not invent new details and does not need the model.
- **Detail / Fast / Balanced / Fine:** maximum AI working edge of 896 / 512 / 640 / 768 pixels, with 6 steps per section in Detail, and 8 / 10 / 12 steps per full frame for the other settings. Detail can require up to 4 sections. All settings may hit the memory or time limit depending on GPU load.
- **Creativity:** more allows the diffusion model to alter the texture more. It can introduce unwanted details.
- **Texture intensity:** scales the texture contribution; too much can create halos or grain.

Output retains the input dimensions. Detail mode uses feathered overlap to reduce patch boundaries, but texture consistency and artifact-free results are not guaranteed. Larger images are reduced for AI processing; this does **not** provide native 4K microdetail. Input limit: 16 megapixels. Photos become RGB PNGs; transparency and original camera metadata are not retained. Photo quality still needs evaluation on the actual source image; screenshots alone are insufficient for a pixel-level comparison.

## Speed and GPU memory

Measured on this PC: **20.84 seconds** for a 512 × 512 synthetic brick image in AI Fast mode, including model loading, with **4,876 MiB peak PyTorch GPU allocation**. The CUDA enhancement/cancellation tests and desktop-window smoke test also passed. This confirms the pipeline runs; it does not establish photo quality or timings for larger settings.

Under 180 seconds is a target, not a guarantee. The visible timer includes model loading and processing. A cooperative 170-second budget stops unfinished AI jobs between steps; a long GPU operation can exceed that budget. No partial result is presented as successful. The first job loads the model; later jobs reuse it on the GPU.

All model components stay on CUDA, in FP32 for GTX 16-series compatibility. No CPU fallback, offload, or disk inference. If memory fills, close games and other GPU-heavy applications, restart Detail Lab, and use Fast. Windows may itself manage shared GPU memory; this app cannot override driver behavior.

## Verification and troubleshooting

Run from PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m unittest test_engine.py
.\.venv\Scripts\python.exe benchmark.py
# Test your own image:
.\.venv\Scripts\python.exe benchmark.py --image "C:\path\photo.jpg"
```

The benchmark writes images and measured GPU/time metadata to `outputs`. The default brick illustration is a functional test, not a quality benchmark for photographs.

If the model is missing, run `setup.bat`. If CUDA is unavailable, rerun setup and check the NVIDIA driver. This app intentionally refuses CPU inference. Cancel takes effect between GPU steps, not in the middle of an operation.

Model: [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), CreativeML OpenRAIL-M. Its model card is downloaded with the weights and links to the model license. Libraries: [Diffusers](https://huggingface.co/docs/diffusers/index), [PyTorch](https://pytorch.org/).
