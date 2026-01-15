# 2D Computational Fluid Dynamics (CFD) Wind Tunnel

![Language](https://img.shields.io/badge/Language-Python_3.12-blue.svg)
![Engine](https://img.shields.io/badge/Engine-Taichi_(CUDA)-green.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)

> **A real-time, GPU-accelerated fluid simulation that solves the Navier-Stokes equations to visualize aerodynamics and generate lift/drag polars for any NACA airfoil.**

---

## Project Overview

This project is a laptop-runnable 2D wind tunnel designed to bridge the gap between textbook aerodynamics and real-time visualization. Unlike standard game physics, this engine runs a high-fidelity **Lattice Boltzmann Method (LBM)** solver on the GPU, allowing it to simulate **200,000 interactive particles** at 60 FPS.

It allows users to generate **NACA 4-digit airfoils** on the fly, visualize pressure/velocity fields, and perform automated angle-of-attack sweeps to generate professional Lift/Drag polar graphs.

## Tech Stack & Physics

* **Method:** Lattice Boltzmann Method (LBM) using the **D2Q9** configuration (2 Dimensions, 9 Discrete Velocities).
* **Engine:** **Taichi Lang** (Python) for compiling high-performance CUDA kernels directly to the GPU.
* **Visualization:** **Pygame** for the GUI and surface rendering.
* **Math:** **NumPy** for data aggregation and airfoil geometry generation.

**Why Taichi?**
Solving fluid dynamics requires iterating over hundreds of thousands of cells every frame. Standard Python is too slow for this. Taichi allows us to write Python-like syntax that compiles down to highly optimized GPU machine code, giving us C++ level performance with Python's flexibility.

## Performance & Optimizations

This simulation was optimized to run on consumer hardware (laptops with dedicated GPUs). Key engineering challenges included:

* **Massive Parallelism:** Rendering **200,000 particles** individually using a custom GPU kernel rather than CPU loops.
* **Zero-Allocation Rendering:** To prevent "Garbage Collection Stutter," all memory buffers (NumPy arrays and Pygame surfaces) are pre-allocated at startup. Frames are drawn by injecting data into existing memory slots rather than creating new objects.
* **Numerical Stability:** High-velocity fluid simulations are prone to "exploding" (values hitting infinity). We implemented strict **CFL (Courant–Friedrichs–Lewy) conditions**, limiting the lattice speed to maintain stability while using an **Exponential Moving Average (EMA)** to filter out high-frequency acoustic noise.
* **Stair-Step Smoothing:** Because the simulation runs on a pixel grid, curved airfoils suffer from "voxelization artifacts" that trap fluid. We calibrated the viscosity and smoothing algorithms to mitigate these spikes in drag/lift data.

## Key Features

* **Automated Data Sweeps:** Press 'D' to initiate a full autonomous sweep from -5° to +20° Angle of Attack. The system waits for convergence, records $C_l$ and $C_d$, and rotates the wing automatically.
* **Professional Polar Plots:** Generates a real-time Lift vs. Drag polar graph. Click the graph to expand it into a detailed scientific plot with axes, ticks, and calculated **Max L/D Ratio**.
* **Multi-Modal Visualization:**
    * **Speed:** Heatmap of velocity magnitude.
    * **Curl:** Visualizes vorticity and turbulence (red/blue).
    * **Pressure:** Visualizes high (red) and low (blue) pressure zones (Bernoulli's Principle).
    * **Particles:** 200k Lagrangian particles for flow visualization.
* **Live Geometry:** Type any 4-digit code (e.g., `2412`, `0010`) to generate and test custom airfoils instantly.

## Controls

| Key | Action |
| :--- | :--- |
| **SPACE** | Pause / Resume Simulation |
| **R** | Soft Reset (Clear Airflow) |
| **C** | Hard Reset (Clear Airflow & Obstacles) |
| **A** | Open Airfoil Menu (Type NACA Code) |
| **D** | Start Data Sweep |
| **X** | Cancel Active Sweep |
| **1** | View Mode: **Curl** (Vorticity) |
| **2** | View Mode: **Speed** (Velocity Magnitude) |
| **3** | View Mode: **Particles** (Flow Lines) |
| **4** | View Mode: **Pressure** (Density) |
| **H** | Toggle HUD |
| **Click Graph** | Expand/Collapse Scientific Plot |

## Dependencies & Credits

This project relies on the open-source Python ecosystem:
* **[Taichi Lang](https://github.com/taichi-dev/taichi):** For the GPU physics backend.
* **[Pygame](https://www.pygame.org/):** For window management and rendering.
* **[NumPy](https://numpy.org/):** For vector math and array manipulation.
* **[Numba](https://numba.pydata.org/):** For auxiliary CPU optimization.

---
Jace Hawkins Jan 2026
