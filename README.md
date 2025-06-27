# Lenia Web Simulation

This repository presents an interactive implementation of Lenia, a continuous artificial life model, fully executable in the browser. Unlike discrete cellular automata such as Conway’s Game of Life, Lenia operates in a continuous space, generating complex organic behaviors.

**Online Demo:** [https://wartets.github.io/Lenia-Web/](https://wartets.github.io/Lenia-Web/)

## Mathematical Concepts of Lenia

### Theoretical Foundations

Lenia extends the concept of cellular automata by using:

* **Continuous states**: Each cell has a value in \[0,1]
* **Continuous space**: Real positions and distances
* **Continuous time**: Evolves with a tunable time step Δt

Its evolution is governed by the equation:

```math
A_{t+1} = A_t + \Delta t \cdot G(K * A_t)
```

where:

* `A`: State grid
* `K`: Convolution kernel
* `G`: Growth function
* `*`: Convolution operation

### Convolution Kernel

The radial kernel is defined by:

```math
K(\mathbf{r}) = \sum_{n=0}^{N-1} \frac{BS_n}{\|BS\|} \exp\left(-\frac{( \|\mathbf{r}\|/R \cdot N - n - 0.5)^2}{2\sigma^2}\right)
```

* `R`: Influence radius (default: 10)
* `BS`: Ring coefficients (default: \[1, 5/12, 2/3])
* `σ = 0.15`: Gaussian smoothing

### Growth Function

```math
G(u) = 2 \exp\left(-\frac{(u - \mu)^2}{2\sigma^2}\right) - 1
```

* `μ ∈ [0,1]`: Center of growth zone
* `σ > 0`: Width of the growth zone

### Multichannel Convolution

This implementation supports multiple channels:

```javascript
MU = [0.156, 0.193, 0.342]   // Growth centers
SIGMA = [0.0118, 0.049, 0.0891] // Widths
BS = [[1, 5/12, 2/3], [1/12, 1], [1]] // Rings
```

The final growth is the average of the channels' contributions.

### Visual Mapping

Value-to-color transformation (Viridis colormap):

```math
\text{remap}(t,k,a) = a \cdot \text{remap}_1(t,k) + (1-a) \cdot \text{remap}_2(t,k)
```

where:

```math
\text{remap}_1(t,k) = 0.5 + \frac{\arctan(k(t-0.5))}{2\arctan(k/2)}
```

```math
\text{remap}_2(t,k) = 0.5 + 4^k |t-0.5|^{2k+1} \cdot \text{sign}(t-0.5)
```

## Implemented Features

1. **Simulation Engine:**

   * Multichannel convolution
   * Periodic boundary conditions
   * Adjustable time step (DT)
   * Step-by-step mode

2. **Interactive Controls:**

   * Adjustment of μ, σ, BS parameters
   * Add/remove channels
   * Dynamic visual mapping (k, a)
   * Automatic contrast adjustment mode

3. **Visualization:**

   * Viridis colormap with remapping
   * Bicubic interpolation
   * FPS display
   * Frame counter

4. **Initial Configurations:**

   * Predefined "Poisson" pattern
   * Orbium (typical organism)
   * Chromium (radial symmetry)
   * Cyanor (concentric rings)
   * Random multi-fish pattern
   * Random grid

## User Guide

Adjustable parameters:

* `DT`: Time step (default: 0.1)
* `R`: Influence radius (10)
* `MU`, `SIGMA`, `BS`: Per channel
* `k`, `a`: Visual mapping parameters

## References

1. Chan, B. W\.-C. (2019). [Lenia: Biology of Artificial Life](https://arxiv.org/abs/1812.05433). Complex Systems, 28(3), 251–286.
2. Lenia homepage by Chan: [https://chakazul.github.io/lenia.html](https://chakazul.github.io/lenia.html)
3. Python version: [https://github.com/Wartets/Lenia-Simulation](https://github.com/Wartets/Lenia-Simulation)
4. Colormap reference: [Matplotlib Viridis](https://bids.github.io/colormap/)

## Dependencies

No external dependencies – Pure JavaScript/HTML5 Canvas

## Author

Colin Bossu
Project developed in July 2024

---
