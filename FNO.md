Link: https://arxiv.org/pdf/2010.08895

Github: https://github.com/neuraloperator/neuraloperator

# Understanding of the Paper

## Motivation

Machine learning methods may hold the key to revolutionizing scientific disciplines by providing
fast solvers that approximate or enhance traditional ones. However, classical neural networks map between finite-dimensional spaces and can therefore only learn solutions tied to a specific discretization.

On the other hand, data-driven methods can directly learn the trajectory of the family of equations from the data. As a result, the learning-based method can be orders of magnitude faster than the conventional solvers.

Few DL based approaches:

1. **Finite Dimensional Operator:** These approaches parameterize the solution operator as a deep convolutional neural network between finite-dimensional Euclidean spaces, hence mesh-dependent and needs tuning for different resolutions.
These approaches are limited to the discretization size and geometry of the training data and hence, it is not possible to query solutions at new points in the domain.
2. **Neural-FEM:** The second approach directly parameterizes the solution function as a neural network. This approach is designed to model one specific instance of the PDE, not the solution operator. It is mesh-independent and accurate, but for any given new instance of the functional parameter/coefficient, it requires training a new neural network. The optimization problem needs to be solved for every new instance. Furthermore, the approach is limited to a setting in which the underlying PDE is known.
3. **Neural Operator:** The neural operator remedies the mesh-dependent nature of the finite-dimensional operator methods discussed above by producing a single set of network parameters that may be used with different discretizations. It has the ability to transfer solu-
tions between meshes(REUSE here?). Furthermore, the neural operator needs to be trained only once(REUSE of weights in batched inference). Obtaining a solution for a new instance of the parameter requires only a forward pass of the network, alleviating the major computational issues incurred in Neural-FEM methods. Lastly, the neural operator requires no knowledge of the underlying PDE, only data. But accuracy seems bad, so Fast Fourier Transform is used.

***AI Support:***

Traditional PDE solvers (like Finite Element or Finite Difference methods) solve equations by breaking space into a grid and calculating the solution step-by-step. If you change the grid resolution or the physical parameters (like the shape of an airfoil), you have to run the expensive solver all over again. Standard neural networks and Physics-Informed Neural Networks (PINNs) suffer from a similar issue: they are often tied to a specific grid resolution, or they optimize a single instance of a PDE rather than learning the general physical rules.

The **Fourier Neural Operator (FNO)** is different because it learns the *mapping between infinite-dimensional function spaces*. In plain English: it learns the underlying physics operators themselves. Once trained, you can feed it a new set of parameters (like a new material distribution), and it instantly predicts the solution field in milliseconds, regardless of the grid resolution

## Fourier Neural Operator Architecture

![image.png](attachment:b43d1b17-f355-4c5d-9c7a-07273c65641f:image.png)

The FNO architecture is a 3-step pipeline. It is like expanding the data, processing it globally and locally, and then shrinking it back down to a final answer.

> Consider this example, to understand dimensionality at each step. Let's define the numerical dimensions for our example:
> 
> - **Batch size (**$*B*$**)**: 10 (processing 10 physical scenarios at once).
> - **Grid Resolution (**$*H*$,$*W*$**)**: 64×64 points (representing our 2D physical space).
> - **Input Features (**$*C_{in}*$ **or** $*d_{a}*$**)**: 3 (e.g., x-coordinate, y-coordinate, and a spatial physical parameter like diffusion coefficient).
> - **Latent Channels (**$*C_{latent}*$ **or** $*d_{v}*$**)**: 32 (the expanded feature space inside the network).
> - **Truncated Fourier Modes (**$*K_{max}*$**)**: 12 (the number of low-frequency waves we keep per spatial dimension).
> - **Output Features (**$*C_{out}*$ **or** $*d_{u}*$**)**: 1 (the final predicted value at each point, such as scalar pressure)
> 
> The input is evaluated at the 64×64 grid points. The data structure is a 4D tensor formatted as **`(Batch, Channels, Height, Width)`**.
> 
> **Initial Dimension:** **`(10, 3, 64, 64)`**
> 

**1: Lifting (Encoding)** The raw input features $a(x)$ (e.g., spatial coordinates, boundary conditions) are passed through a shallow, point-wise neural network (an MLP) $P$. This expands the data from a simple scalar/vector into a much richer, high-dimensional feature space (or "channel" space) at every single point in your grid.

> Dimension: `(10, 32, 64, 64)` **(Real)**
> 

**2: The Fourier Layers (The core workload)** This is where the actual PDE solving happens. The network passes the data through a stack of ***T*** Fourier layers. Each layer asks two questions simultaneously - *What is happening globally?* and *What is happening locally?* 

- **The Global Spectral Path:**
    - **FFT:** The layer applies a Fast Fourier Transform (FFT) to move the spatial data into the frequency domain. So the new 64 x 64 is not spatial, but Frequencies now.
    
    > Dimension: `(10, 32, 64, 64)` **(Complex)**
    > 
    - **Truncation:** It throws away all the high-frequency modes and keeps only a fixed number of low-frequency modes (called $*K_{cut}*$). Basically a grid point, whose value is above the threshold, is removed. This is not random points filtered and hence sparse points, BUT a corner with lowest integer values which are at the corner is taken, and rest is discarded. Hence $R$ dimension is also deterministic.
    
    > Dimension: `(10, 32, 12, 12)` **(Complex)**
    > 
    > 
    > Basically each grid point behaves as a frequency mode, and series of 10 x 32 points for the same grid point are seen.
    > 
    - **Channel Mixing:** The network multiplies these remaining low-frequency modes by a learned weight tensor $*R$*  and learning based on Frequencies happen here. This mixes information across the channels.
    
    > Dimension of $R$ = `(12, 12, 32, 32)`
    Dimension: `(10, 32, 12, 12)` **(Complex)**
    > 
    - **Zero-Padding + Inverse FFT:** First the 12 x 12 is converted to 64 x 64 with Zero Padding and then it transforms the data back into the physical, spatial domain.
    
    > **Dimension:** `(10, 32, 64, 64)`
    > 
- **The Local Bypass Path:** Simultaneously, the original spatial data is multiplied by a local, point-wise weight matrix $*W*$ (a skip connection).

> Dimension:  `(10, 32, 64, 64)`
> 
- **Activation:** The results of the global path and the local path are added together and passed through a non-linear activation function (like GELU or ReLU).

**3: Projection (Decoding)** Finally, the deep feature channels are collapsed back down by another point-wise neural network with $Q$ to give you the final physical values you care about (e.g., velocity, pressure)

> Dimension: `(10, 1, 64, 64)`  - 1 is for the output.
> 

## **Microarchitectural and Optimization Places**

These are suggested by AI, of places to optimize and see.

- **The FFT Bottleneck:** The $*O(NlogN)*$ FFT and inverse-FFT operations will dominate your profiling charts. Furthermore, FFTs demand uniform Cartesian grids. Highly tuned FFT libraries (like cuFFT on NVIDIA GPUs) are essential, but allocating memory for intermediate complex-number arrays between the FFT, the frequency-mixing, and the iFFT is a massive memory-bandwidth bottleneck. **Kernel fusion**—fusing the FFT output directly into the channel-mixing matrix multiplication without writing to global memory is a prime optimization target.
- **Parameter Explosion (Curse of Dimensionality):** The weight tensor $*R*$ has a size that scales as *O*($*K^d_{cut}$* × channels × channels) where $*d*$ is the number of spatial dimensions. For 3D or 4D (space + time) problems, this parameter count explodes, destroying cache locality and making the model heavily memory-bound. A massive optimization area in the literature is **Tensorization** (e.g., using Tucker Tensor factorization on the weights) to compress the Fourier layers by up to 90%, massively reducing FLOPs and memory traffic.

FNO is essentially bouncing data between the physical domain (for boundaries and non-linearities) and the frequency domain (for cheap global mixing).

# Fast Fourier Transformation(FFT)

Here is a decomposition of FFT and analysis of how reuse can be done inside it. More details on Tucker Decomposition.

## FFT on a Matrix, and optimizations

$F[u,v] = \sum_{i=0}^{M-1}\sum_{j=0}^{N-1} A[i,j] e^{-2\pi i\left(\frac{ui}{M}+\frac{vj}{N}\right)}$

Basically FFT is Matrix Multiplication of $A$  and $C_0$, where $C_0[i,j] = w^{ij}$ . But instead of Matrix multiplication, it uses Butterfly Decomposition, which exploits structure of this $C_0$ and creates small subpermutation matrices which makes it $O(NlogN)$ computation.

## Tucker Decomposition

Tucker decomposition decomposes a tensor into a set of matrices and a smaller core tensor. In a three-mode case, given the original tensor $X∈R^{I×J×K}$ , Tucker decomposition outputs a tensor $Z ∈ R^{P ×Q×R}$ and three matrices $A ∈R^{I×P}$, $B ∈R^{J×Q}$, $C ∈R^{K×R}$ : 
$X ≈ Z ×_1 A ×_2 B ×_3 C$

with $×_n$ indicating the tensor product along the n-th mode. Elements of the core tensor $Z$ show the level of interaction between the different components. Typically, $P$, $Q$, $R$ are smaller than $I$, $J$, $K$ respectively, so $Z$ can be thought of as a compressed version of $X$.

## Meeting 1

**Q1.** What is the order of number of Fourier Layers? Ans: 1- 20 ish I found. Order of $10^1$.

**Q2.** What is the number of parameters of the model?

**Answer:** In the paper, nothing is mentioned. Based on the code, I have calculated it to be-

- for Darcy, Layers = 4, Net  = 0.167 * 4 + other = 0.668 + other = 0.671 Million
- for navier stokes = 4, Net  = 8,768 + 16,897 + L * 17,309,920 = 69.27 Million

So most parameters go into each Fourier layer, and pre/post MLP’s are very small.
Hence, number of parameters is almost linear to the number of layers.

**Q3.** Are the parameters changing in each layer?

**Answer:** No, in the paper it isn’t mentioned. Neither in the code.

**Q4.** What happens when you change parameters in each layer?

I experimented with Darcy. 

**Q5.** Why is $k_{max,j} = 12$ used in many places, rather than a power of 2?

## Important Files in the Repo

| Component | Code path | Navier–Stokes defaults |
| --- | --- | --- |
| Model config | `config/navier_stokes_config.py` → `FNO_Medium2d` | `n_modes=[64,64]`, `hidden_channels=64`, `n_layers=4` |
| Forward pipeline | `neuralop/models/fno.py` | `self.lifting → self.fno_blocks → self.projection` |
| Fourier layer | `neuralop/layers/spectral_convolution.py` | FFT/contract/iFFT |
| FNO block | `neuralop/layers/fno_block.py` | convs, skips, channel_mlp |
| fno_skip | `neuralop/layers/skip_connections.py` | `Flattened1dConv` for `"linear"` |
| channel_mlp | `neuralop/layers/channel_mlp.py` | 2× Conv1d(1×1) + GELU |
|  |  |  |