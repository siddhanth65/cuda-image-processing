#!/usr/bin/env python3
"""Generate a deterministic collection of binary PPM test images."""
from pathlib import Path
import argparse
import math


def make_image(index: int, width: int, height: int) -> bytes:
    pixels = bytearray()
    phase = index * 0.17
    for y in range(height):
        for x in range(width):
            wave = 0.5 + 0.5 * math.sin(x * 0.075 + y * 0.043 + phase)
            ring = 0.5 + 0.5 * math.cos(math.hypot(x - width / 2, y - height / 2) * 0.11 + phase)
            checker = ((x // 12 + y // 12 + index) % 2) * 35
            red = int(max(0, min(255, (x / max(1, width - 1)) * 170 + wave * 65 + checker)))
            green = int(max(0, min(255, (y / max(1, height - 1)) * 170 + ring * 70)))
            blue = int(max(0, min(255, (1.0 - x / max(1, width - 1)) * 150 + (1.0 - wave) * 100)))
            pixels.extend((red, green, blue))
    return bytes(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/sample_images')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=96)
    args = parser.parse_args()
    if min(args.count, args.width, args.height) <= 0:
        parser.error('count, width and height must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        path = output / f'image_{index + 1:03d}.ppm'
        with path.open('wb') as image_file:
            image_file.write(f'P6\n{args.width} {args.height}\n255\n'.encode('ascii'))
            image_file.write(make_image(index, args.width, args.height))
    print(f'Generated {args.count} PPM images ({args.width}x{args.height}) in {output}')


if __name__ == '__main__':
    main()
