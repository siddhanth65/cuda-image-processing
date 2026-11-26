#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t error = (call);                                                   \
    if (error != cudaSuccess) {                                                   \
        std::ostringstream message;                                               \
        message << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "       \
                << cudaGetErrorString(error);                                    \
        throw std::runtime_error(message.str());                                  \
    }                                                                             \
} while (0)

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

struct Options {
    fs::path inputDir = "data/sample_images";
    fs::path outputDir = "output";
    fs::path resultsFile = "results/benchmark.csv";
    std::string operation = "all";
    std::size_t batchSize = 16;
    bool showHelp = false;
};

std::string readPpmToken(std::istream& input) {
    std::string token;
    char ch = 0;
    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }
    while (input.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (ch == '\r' && input.peek() == '\n') input.get();
            break;
        }
        token.push_back(ch);
    }
    return token;
}

Image loadPpm(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open image: " + path.string());
    const std::string magic = readPpmToken(input);
    if (magic != "P6") throw std::runtime_error("Only binary PPM (P6) is supported: " + path.string());
    Image image;
    image.width = std::stoi(readPpmToken(input));
    image.height = std::stoi(readPpmToken(input));
    const int maxValue = std::stoi(readPpmToken(input));
    if (image.width <= 0 || image.height <= 0 || maxValue != 255)
        throw std::runtime_error("Invalid PPM dimensions or max value: " + path.string());
    const std::size_t byteCount = static_cast<std::size_t>(image.width) * image.height * 3;
    image.pixels.resize(byteCount);
    input.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(byteCount));
    if (input.gcount() != static_cast<std::streamsize>(byteCount))
        throw std::runtime_error("Incomplete PPM pixel data: " + path.string());
    return image;
}

void savePpm(const fs::path& path, int width, int height, const std::vector<unsigned char>& pixels) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot write image: " + path.string());
    output << "P6\n" << width << ' ' << height << "\n255\n";
    output.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

Options parseArguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto nextValue = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + arg);
            return argv[++i];
        };
        if (arg == "--input") options.inputDir = nextValue();
        else if (arg == "--output") options.outputDir = nextValue();
        else if (arg == "--results") options.resultsFile = nextValue();
        else if (arg == "--operation") options.operation = nextValue();
        else if (arg == "--batch-size") options.batchSize = std::stoul(nextValue());
        else if (arg == "--help" || arg == "-h") options.showHelp = true;
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    if (options.operation != "all" && options.operation != "grayscale" &&
        options.operation != "blur" && options.operation != "sobel")
        throw std::runtime_error("--operation must be all, grayscale, blur, or sobel");
    if (options.batchSize == 0) throw std::runtime_error("--batch-size must be greater than zero");
    return options;
}

void printHelp() {
    std::cout << "CUDA Batch Image Processing\n"
              << "Usage: ./cuda_image_processing [options]\n\n"
              << "  --input DIR         Input directory containing P6 PPM images\n"
              << "  --output DIR        Output directory (default: output)\n"
              << "  --results FILE      Benchmark CSV (default: results/benchmark.csv)\n"
              << "  --operation MODE    all | grayscale | blur | sobel (default: all)\n"
              << "  --batch-size N      Images per reported batch (default: 16)\n"
              << "  --help              Show this message\n";
}

__global__ void grayscaleKernel(const unsigned char* input, unsigned char* gray, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int pixel = y * width + x;
    const int offset = pixel * 3;
    const float value = 0.299f * input[offset] + 0.587f * input[offset + 1] + 0.114f * input[offset + 2];
    gray[pixel] = static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, value)));
}

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    for (int channel = 0; channel < 3; ++channel) {
        int sum = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int sx = min(max(x + dx, 0), width - 1);
                const int sy = min(max(y + dy, 0), height - 1);
                sum += weights[dy + 1][dx + 1] * input[(sy * width + sx) * 3 + channel];
            }
        }
        output[(y * width + x) * 3 + channel] = static_cast<unsigned char>((sum + 8) / 16);
    }
}

__global__ void sobelKernel(const unsigned char* gray, unsigned char* edges, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int gxWeights[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gyWeights[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int gx = 0;
    int gy = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int sx = min(max(x + dx, 0), width - 1);
            const int sy = min(max(y + dy, 0), height - 1);
            const int value = gray[sy * width + sx];
            gx += gxWeights[dy + 1][dx + 1] * value;
            gy += gyWeights[dy + 1][dx + 1] * value;
        }
    }
    const float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
    edges[y * width + x] = static_cast<unsigned char>(fminf(255.0f, magnitude));
}

std::vector<unsigned char> grayscaleCpu(const Image& image) {
    const std::size_t count = static_cast<std::size_t>(image.width) * image.height;
    std::vector<unsigned char> output(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float value = 0.299f * image.pixels[i * 3] + 0.587f * image.pixels[i * 3 + 1] + 0.114f * image.pixels[i * 3 + 2];
        output[i] = static_cast<unsigned char>(std::clamp(value, 0.0f, 255.0f));
    }
    return output;
}

std::vector<unsigned char> blurCpu(const Image& image) {
    const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    std::vector<unsigned char> output(image.pixels.size());
    for (int y = 0; y < image.height; ++y) for (int x = 0; x < image.width; ++x) {
        for (int c = 0; c < 3; ++c) {
            int sum = 0;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const int sx = std::clamp(x + dx, 0, image.width - 1);
                const int sy = std::clamp(y + dy, 0, image.height - 1);
                sum += weights[dy + 1][dx + 1] * image.pixels[(sy * image.width + sx) * 3 + c];
            }
            output[(y * image.width + x) * 3 + c] = static_cast<unsigned char>((sum + 8) / 16);
        }
    }
    return output;
}

std::vector<unsigned char> sobelCpu(const std::vector<unsigned char>& gray, int width, int height) {
    const int gxWeights[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gyWeights[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    std::vector<unsigned char> output(gray.size());
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        int gx = 0, gy = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            const int sx = std::clamp(x + dx, 0, width - 1);
            const int sy = std::clamp(y + dy, 0, height - 1);
            const int value = gray[sy * width + sx];
            gx += gxWeights[dy + 1][dx + 1] * value;
            gy += gyWeights[dy + 1][dx + 1] * value;
        }
        output[y * width + x] = static_cast<unsigned char>(std::min(255.0f, std::sqrt(static_cast<float>(gx * gx + gy * gy))));
    }
    return output;
}

std::vector<unsigned char> grayToRgb(const std::vector<unsigned char>& gray) {
    std::vector<unsigned char> rgb(gray.size() * 3);
    for (std::size_t i = 0; i < gray.size(); ++i)
        rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = gray[i];
    return rgb;
}

struct GpuResult {
    std::vector<unsigned char> gray;
    std::vector<unsigned char> blur;
    std::vector<unsigned char> edges;
    double milliseconds = 0.0;
};

GpuResult processGpu(const Image& image) {
    const std::size_t pixelCount = static_cast<std::size_t>(image.width) * image.height;
    const std::size_t rgbBytes = pixelCount * 3;
    const std::size_t grayBytes = pixelCount;
    unsigned char *dInput = nullptr, *dGray = nullptr, *dBlur = nullptr, *dEdges = nullptr;
    GpuResult result;
    result.gray.resize(grayBytes);
    result.blur.resize(rgbBytes);
    result.edges.resize(grayBytes);
    const dim3 block(16, 16);
    const dim3 grid((image.width + block.x - 1) / block.x, (image.height + block.y - 1) / block.y);
    const auto start = Clock::now();
    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dInput), rgbBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGray), grayBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dBlur), rgbBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dEdges), grayBytes));
        CUDA_CHECK(cudaMemcpy(dInput, image.pixels.data(), rgbBytes, cudaMemcpyHostToDevice));
        grayscaleKernel<<<grid, block>>>(dInput, dGray, image.width, image.height);
        gaussianBlurKernel<<<grid, block>>>(dInput, dBlur, image.width, image.height);
        sobelKernel<<<grid, block>>>(dGray, dEdges, image.width, image.height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(result.gray.data(), dGray, grayBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.blur.data(), dBlur, rgbBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(result.edges.data(), dEdges, grayBytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        if (dInput) cudaFree(dInput);
        if (dGray) cudaFree(dGray);
        if (dBlur) cudaFree(dBlur);
        if (dEdges) cudaFree(dEdges);
        throw;
    }
    if (dInput) CUDA_CHECK(cudaFree(dInput));
    if (dGray) CUDA_CHECK(cudaFree(dGray));
    if (dBlur) CUDA_CHECK(cudaFree(dBlur));
    if (dEdges) CUDA_CHECK(cudaFree(dEdges));
    result.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);
        if (options.showHelp) { printHelp(); return 0; }
        if (!fs::exists(options.inputDir)) throw std::runtime_error("Input directory does not exist: " + options.inputDir.string());
        fs::create_directories(options.outputDir);
        fs::create_directories(options.resultsFile.parent_path().empty() ? fs::path(".") : options.resultsFile.parent_path());

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) throw std::runtime_error("No CUDA-capable GPU was detected. This project requires GPU execution.");
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << "\n";

        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(options.inputDir))
            if (entry.is_regular_file() && entry.path().extension() == ".ppm") files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        if (files.empty()) throw std::runtime_error("No .ppm images found in " + options.inputDir.string());

        std::ofstream csv(options.resultsFile);
        if (!csv) throw std::runtime_error("Cannot create benchmark CSV: " + options.resultsFile.string());
        csv << "image,width,height,cpu_ms,gpu_ms,gpu_speedup\n";
        double totalCpuMs = 0.0, totalGpuMs = 0.0;
        std::size_t processed = 0;
        std::cout << "Found " << files.size() << " images; batch size = " << options.batchSize << "\n";

        for (std::size_t batchStart = 0; batchStart < files.size(); batchStart += options.batchSize) {
            const std::size_t batchEnd = std::min(files.size(), batchStart + options.batchSize);
            std::cout << "Batch " << (batchStart / options.batchSize + 1) << ": images "
                      << (batchStart + 1) << "-" << batchEnd << "\n";
            for (std::size_t index = batchStart; index < batchEnd; ++index) {
                const Image image = loadPpm(files[index]);
                const auto cpuStart = Clock::now();
                const auto cpuGray = grayscaleCpu(image);
                const auto cpuBlur = blurCpu(image);
                const auto cpuEdges = sobelCpu(cpuGray, image.width, image.height);
                const double cpuMs = std::chrono::duration<double, std::milli>(Clock::now() - cpuStart).count();
                const GpuResult gpu = processGpu(image);
                totalCpuMs += cpuMs;
                totalGpuMs += gpu.milliseconds;
                const double speedup = gpu.milliseconds > 0.0 ? cpuMs / gpu.milliseconds : 0.0;
                csv << files[index].filename().string() << ',' << image.width << ',' << image.height << ','
                    << std::fixed << std::setprecision(4) << cpuMs << ',' << gpu.milliseconds << ',' << speedup << '\n';

                const std::string stem = files[index].stem().string();
                if (options.operation == "all" || options.operation == "grayscale")
                    savePpm(options.outputDir / (stem + "_grayscale.ppm"), image.width, image.height, grayToRgb(gpu.gray));
                if (options.operation == "all" || options.operation == "blur")
                    savePpm(options.outputDir / (stem + "_blur.ppm"), image.width, image.height, gpu.blur);
                if (options.operation == "all" || options.operation == "sobel")
                    savePpm(options.outputDir / (stem + "_sobel.ppm"), image.width, image.height, grayToRgb(gpu.edges));
                ++processed;
                std::cout << "  [" << processed << '/' << files.size() << "] " << files[index].filename().string()
                          << " | CPU " << std::fixed << std::setprecision(2) << cpuMs << " ms"
                          << " | GPU " << gpu.milliseconds << " ms | speedup " << speedup << "x\n";
            }
        }
        csv.close();
        std::cout << "\nCompleted " << processed << " images.\n"
                  << "Total CPU time: " << totalCpuMs << " ms\n"
                  << "Total GPU time (including transfers and allocation): " << totalGpuMs << " ms\n"
                  << "Overall speedup: " << (totalGpuMs > 0.0 ? totalCpuMs / totalGpuMs : 0.0) << "x\n"
                  << "Outputs: " << options.outputDir << "\nBenchmark: " << options.resultsFile << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
