NVCC ?= nvcc
CUDA_ARCH ?=
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
TARGET := cuda_image_processing
SOURCE := src/image_processing.cu

.PHONY: all build clean run help

all: build

build: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) $< -o $@

run: build
	./$(TARGET) --input data/sample_images --output output --results results/benchmark.csv --operation all --batch-size 16

clean:
	rm -f $(TARGET)

help:
	@echo "Targets: build, run, clean"
