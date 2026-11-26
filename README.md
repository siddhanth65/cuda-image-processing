# CUDA Batch Image Processing

A CUDA C++ project that processes a directory of images in one execution. It applies three image-processing operations using custom GPU kernels and compares their results against CPU reference implementations.

## Features

- **RGB to grayscale:** weighted luminance conversion (`0.299 R + 0.587 G + 0.114 B`).
- **Gaussian blur:** 3 x 3 Gaussian approximation with weights `[[1,2,1],[2,4,2],[1,2,1]] / 16`.
- **Sobel edge detection:** horizontal and vertical 3 x 3 Sobel gradients, combined into an edge magnitude image.
- **Batch execution:** discovers all `.ppm` files in an input directory and processes them in groups controlled by `--batch-size`.
- **CPU reference:** runs equivalent CPU implementations and records CPU/GPU timings for every image.
- **Benchmark output:** writes per-image timings and speedup to CSV.
- **No third-party image library:** uses binary PPM (P6), keeping the build self-contained apart from the CUDA Toolkit.

The GPU timing includes device-memory allocation, host-to-device transfer, kernel execution, synchronization, device-to-host transfer and deallocation. It is therefore an end-to-end per-image GPU path measurement, not just kernel-only timing. The three kernels are launched for every image; `--operation` controls which processed outputs are written.

## Requirements

- NVIDIA GPU with a working CUDA driver.
- CUDA Toolkit with `nvcc`.
- GNU Make.
- C++17-capable CUDA compiler.
- Python 3 only if you want to regenerate the synthetic sample dataset.

This project requires actual CUDA execution. It deliberately exits with an error if no CUDA-capable GPU is detected; the CPU implementation is a reference for comparison, not a CPU-only fallback submission.

## Repository layout

```text
cuda-image-processing/
├── src/
│   └── image_processing.cu   # CPU reference and CUDA kernels/pipeline
├── data/
│   └── sample_images/        # 100 generated P6 PPM images (128 x 96)
├── output/                   # Processed PPM files are written here
├── results/                  # Per-image benchmark CSV is written here
├── screenshots/              # Add screenshots/logs from your CUDA lab run
├── Makefile
├── run.sh
├── generate_dataset.py
└── README.md
```

## Build

From the repository root:

```bash
make clean
make build
```

If your lab requires a specific GPU architecture, pass it through `CUDA_ARCH`, for example:

```bash
make clean
make build CUDA_ARCH=-arch=sm_75
```

Use the architecture supported by the GPU in your lab. If unsure, omit `CUDA_ARCH` and let `nvcc` use its default.

## Run

Process the included 100-image dataset with all three operations:

```bash
./cuda_image_processing \
  --input data/sample_images \
  --output output \
  --results results/benchmark.csv \
  --operation all \
  --batch-size 16
```

Or use the wrapper:

```bash
./run.sh
```

The wrapper accepts positional overrides in this order: input directory, output directory, benchmark CSV path, operation, batch size.

### CLI options

| Argument | Description | Default |
|---|---|---|
| `--input DIR` | Directory containing binary PPM (`.ppm`) images | `data/sample_images` |
| `--output DIR` | Destination for processed images | `output` |
| `--results FILE` | Destination for benchmark CSV | `results/benchmark.csv` |
| `--operation MODE` | `all`, `grayscale`, `blur`, or `sobel` | `all` |
| `--batch-size N` | Number of images in each reported processing group | `16` |
| `--help` | Print usage information | — |

Examples:

```bash
# Grayscale outputs only
./cuda_image_processing --operation grayscale

# Sobel outputs, with 8 images per group
./cuda_image_processing --operation sobel --batch-size 8

# Use a different dataset and output folder
./cuda_image_processing --input /path/to/ppm_images --output /tmp/processed
```

## Input format

The loader accepts binary PPM files (P6), with 8-bit RGB channels and maximum value 255. The included dataset generator creates 100 deterministic synthetic images of size 128 x 96:

```bash
python3 generate_dataset.py --count 100 --width 128 --height 96
```

To use another image dataset, convert its images to P6 PPM first. Keep the input images in one directory; nested directories are not recursively scanned.

## Outputs

For each input image, the program writes the selected output(s):

- `image_001_grayscale.ppm`
- `image_001_blur.ppm`
- `image_001_sobel.ppm`

The benchmark CSV contains:

```text
image,width,height,cpu_ms,gpu_ms,gpu_speedup
```

`gpu_speedup` is `cpu_ms / gpu_ms`. A value above 1 means the measured GPU path was faster for that image; small images may be slower on the GPU because allocation and transfer overhead can dominate kernel execution.

## Implementation notes

Each CUDA thread computes one output pixel. A two-dimensional grid of 16 x 16 thread blocks covers each image, with boundary checks for dimensions that are not multiples of the block size. The blur and Sobel kernels clamp sample coordinates at image boundaries. Device buffers are allocated per image, input pixels are copied to the GPU, and the resulting arrays are copied back before output files are written.

The CPU routines use the same formulas and boundary policy as the GPU kernels, making the benchmark comparison interpretable. The project prioritizes a transparent baseline over advanced optimization. Potential next steps include pinned host memory, persistent device buffers, streams, asynchronous transfers, and fusing compatible operations.

## Reproducible execution evidence for peer review

Run the full dataset in the CUDA lab and save the terminal output:

```bash
make clean && make build
./cuda_image_processing --input data/sample_images --output output \
  --results results/benchmark.csv --operation all --batch-size 16 \
  | tee results/execution.log
```

Then verify the number of inputs and outputs:

```bash
find data/sample_images -maxdepth 1 -name '*.ppm' | wc -l
find output -maxdepth 1 -name '*_grayscale.ppm' | wc -l
find output -maxdepth 1 -name '*_blur.ppm' | wc -l
find output -maxdepth 1 -name '*_sobel.ppm' | wc -l
head results/benchmark.csv
```

For the included dataset, each count should be 100. Add `results/execution.log`, `results/benchmark.csv`, and clear screenshots from the actual CUDA lab run to the repository. Do not fabricate execution logs or benchmark numbers; they must come from running the program on a CUDA GPU.

## Limitations

- P6 PPM only; JPEG/PNG conversion must be performed before running this program.
- Input images are processed one at a time within each reported group. The batch size organizes the run and progress reporting; it does not imply concurrent multi-image GPU execution.
- The current implementation allocates and frees device buffers for every image, so timings include this overhead.
- The CPU and GPU paths are compared by elapsed wall-clock time. Results depend on GPU model, driver, image dimensions and system load.
