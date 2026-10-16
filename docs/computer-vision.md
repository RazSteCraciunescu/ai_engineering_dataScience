# Computer vision

The OpenCV workflow measures dimensions, mean brightness, contrast, Laplacian variance and a simple noise estimate. Transparent thresholds flag underexposure, overexposure, low contrast and blur.

Stable Diffusion is kept as an optional local example because image generation and image-quality measurement solve different problems. A real quality gate should calibrate thresholds against labelled examples and camera/scene variation.
