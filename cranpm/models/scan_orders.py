import math


def wind_band_hilbert(height, width, wind_angle):
    dx = math.cos(wind_angle)
    dy = math.sin(wind_angle)

    positions = []
    for i in range(height):
        for j in range(width):
            proj = i * dy + j * dx
            positions.append((proj, i, j))

    positions.sort(key=lambda x: x[0])

    num_bands = max(height, width)
    band_size = len(positions) // num_bands

    order = []
    for band_idx in range(num_bands):
        start = band_idx * band_size
        end = start + band_size if band_idx < num_bands - 1 else len(positions)
        band = positions[start:end]
        band_indices = [(p[1], p[2]) for p in band]
        band_indices.sort(key=lambda x: (x[0] + x[1]) % 2 * 1000 + x[0] * width + x[1])
        for i, j in band_indices:
            order.append(i * width + j)

    return order
