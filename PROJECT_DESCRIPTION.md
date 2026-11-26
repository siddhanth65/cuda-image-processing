# Project Description: CUDA Batch Image Processing

## Purpose

This project demonstrates GPU-accelerated processing of a collection of images using custom CUDA kernels. Rather than processing a single image, the program discovers and processes every binary PPM image in an input directory during one execution. The included reproducible dataset contains 100 synthetic RGB images at 128 x 96 pixels.

## Algorithms and CUDA kernels

1. **Grayscale conversion:** Each CUDA thread handles one pixel and converts RGB values to luminance using `0.299R + 0.587G + 0.114B`.
2. **Gaussian blur:** Each thread applies a 3 x 3 Gaussian approximation with a normalized weight matrix. Border coordinates are clamped to the nearest valid pixel.
3. **Sobel edge detection:** Each thread computes horizontal and vertical image gradients using Sobel masks and stores the gradient magnitude as an edge intensity.

A two-dimensional grid of 16 x 16 thread blocks covers each image. Bounds checks handle image dimensions that are not exact multiples of the block dimensions.

## CPU comparison and measurements

Equivalent CPU routines apply the same formulas and border policy. For each image, the program records CPU elapsed time and GPU elapsed time in a CSV file. GPU elapsed time includes allocation, host-to-device transfer, kernel launches, synchronization, device-to-host transfer and deallocation. The per-image speedup is calculated as CPU time divided by GPU time. Small images may not show a speedup because transfer and allocation overhead can exceed the kernel work.

## Engineering choices and challenges

- Binary PPM was selected to keep image loading self-contained and avoid an external image dependency in the CUDA lab.
- The program validates command-line options, input files, CUDA availability and CUDA API calls.
- Boundary handling is explicit in both CPU and GPU implementations to avoid out-of-range reads at image edges.
- Device buffers are currently allocated per image. This makes the first implementation straightforward to inspect, but adds overhead; persistent allocations, pinned memory and CUDA streams are natural future optimizations.
- The batch-size option groups files for progress reporting. Images within each group are currently processed sequentially, not concurrently.

## Execution evidence

The repository includes a deterministic generator for 100 sample images. A real run on the CUDA lab GPU should produce processed images, a per-image benchmark CSV and a terminal log. Those outputs must be generated on the target GPU and committed as execution evidence; performance values should not be fabricated.
