import numpy as np
import pygame


def generate_naca4(number, chord_length, num_points=100):
    """
    Generates the coordinates for a NACA 4-digit airfoil.
    """
    if len(number) < 4:
        number = "0012"

    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0
    t = int(number[2:]) / 100.0

    beta = np.linspace(0, np.pi, num_points)
    x = (1 - np.cos(beta)) / 2.0

    # Thickness Distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) -
                  0.1260 * x -
                  0.3516 * x**2 +
                  0.2843 * x**3 -
                  0.1015 * x**4)

    # Camber Line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < p:
            yc[i] = (m / p**2) * (2*p*x[i] - x[i]**2)
            dyc_dx[i] = (2*m / p**2) * (p - x[i])
        else:
            if (1-p) ** 2 > 0:
                yc[i] = (m / (1-p)**2) * ((1-2*p) + 2*p*x[i] - x[i]**2)
                dyc_dx[i] = (2*m / (1-p)**2) * (p - x[i])

    # Upper/Lower Surface
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    xu *= chord_length
    xl *= chord_length
    yu *= chord_length
    yl *= chord_length

    points_top = list(zip(xu, yu))
    points_bot = list(zip(xl, yl))

    return points_top + points_bot[::-1]


def stamp_airfoil(obstacle_grid, number, cx, cy, chord, angle_deg=0):
    """
    Stamps the airfoil directly onto the boolean fluid grid.
    """
    points = generate_naca4(number, chord)

    rad = np.radians(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    transformed_points = []
    x_offset = 0.25 * chord

    for px, py in points:
        tx = px - x_offset
        ty = py
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        transformed_points.append((rx + cx, ry + cy))

    w, h = obstacle_grid.shape
    surf = pygame.Surface((w, h))

    pygame.draw.polygon(surf, (255, 255, 255), transformed_points)

    pixel_array = pygame.surfarray.array2d(surf)
    mask = pixel_array > 0
    obstacle_grid[mask] = True

    count = np.sum(mask)
    print(f"Stamped Airfoil with {count} pixels.")
