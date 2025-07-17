Okay so a reminder:

$$
dg=g_\mathbf{p}[d\mathbf{p}]-({\color{red}g_\mathbf{x}\mathbf{A}^{-1}})\cdot(\mathbf{A}'(\mathbf{p})[d\mathbf{p}]\cdot\mathbf{x}-{\mathbf{b}'(\mathbf{p})[d\mathbf{p}]})
$$

where the adjoint equation is: $\mathbf{v}^\dagger=\color{red}g_\mathbf{x}\mathbf{A}^{-1}$ (i.e. solving $\mathbf{A}^\dagger\mathbf{v}=g_\mathbf{x}^\dagger$). The symbols for the EM wave equation are:

- $\mathbf{A}=\nabla^2-(j\omega\mu\sigma-\omega^2\mu\varepsilon)$
- $\mathbf{b}=j\omega\mu\mathbf{d}\cdot\delta(\mathbf{r}-\mathbf{r}_0)$ to excite the qubit via a point current-dipole source with moment $\mathbf{d}[\textnormal{A}\cdot\textnormal{m}]$
- $\mathbf{x}=\mathbf{E}$ as the field-vectors being the solved unknown
- $g$ is the custom objective function that samples $\mathbf{E}$ at a finite set of points

### Simple example

So now let's take a simple example where the resonant frequency is to be optimised. In this case, one may sample the electric field strength $\mathbf{E}^\dagger\mathbf{E}$ at one point. To isolate this to a single point (or points), one can filter the points via $\mathbf{S}\mathbf{E}$ where $\mathbf{S}$ is a diagonal matrix of zeros with 1 only at the $x,y,z$ of the point to be sampled. Thus, one may write the objective function as:

$$
g=\mathbf{E}^\dagger\mathbf{S}\mathbf{E}
$$

where one notes that $\mathbf{S}^\dagger\mathbf{S}=\mathbf{S}$. Now to find the required derivative:

$$
g_\mathbf{E}=d\mathbf{E}^\dagger\mathbf{S}\mathbf{E}+\mathbf{E}^\dagger\mathbf{S}d\mathbf{E}=2\Re(\mathbf{E}^\dagger\mathbf{S}d\mathbf{E}).
$$

Now this quantity is a real-valued vector quantity. Thus, $g_\mathbf{E}^\dagger=g_\mathbf{E}^T$. That is, the adjoint equation to solve is:

$$
\mathbf{A}^\dagger\mathbf{v}=2\Re(\mathbf{E}^\dagger\mathbf{S}d\mathbf{E})^T
$$

This $\mathbf{A}^\dagger$ effectively implies that the EM wave equation needs to be solved for $-\omega$. This may be difficult; thus, a hack is to take the conjugate of the adjoint equation (noting that the RHS stays the same) and noting that $\mathbf{A}$ is symmetric to realise:

$$
\mathbf{A}\mathbf{v}^*=2\Re(\mathbf{E}^\dagger\mathbf{S}d\mathbf{E})^T.
$$

Now given that the sourcing term is given as $j\omega\mu\mathbf{d}\cdot\delta(\mathbf{r}-\mathbf{r}_0)$, this implies that:

$$
\mathbf{d}_\textnormal{adjoint}=\frac{2\Re(\mathbf{E}^\dagger\mathbf{S})^T}{j\omega\mu}.
$$

To summarise:

- The adjoint simulation is run by placing the electric current dipole source $\mathbf{d}_\textnormal{adjoint}$ at the point being sampled by the objective function
- The $j$ could probably be ignored on the notion that it's effectively a $e^{j\pi/2}$ phase term on the source... After all, the full product is preferably real...
- When solving the resulting fields, take the transpose of this solution to obtain $\mathbf{v}^\dagger$ (as the conjugation is already done by construction)
- Evaluate $\mathbf{v}^\dagger$ at the mesh points of the original simulation

To calculate $dg$, just note:

- $\mathbf{b}'(\mathbf{p})[d\mathbf{p}]=\mathbf{0}$ as the sourcing term should not be changing with the planar geometry
- $\mathbf{A}'(\mathbf{p})[d\mathbf{p}]=-j\omega\mu\sigma'(\mathbf{p})$ since $\nabla^2$ does not change with $\mathbf{p}$. Similarly, $\omega^2\mu\varepsilon$ does not change as the substrate block is not manipulated by changing the geometry.
- Evaluate at mesh points of the original simulation so that the points all match


