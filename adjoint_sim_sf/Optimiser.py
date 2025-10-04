class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.params = np.asarray(initial_params, float)
        self.lr = float(lr)
        self.evaluator = evaluator
        self.history = []  # list of (params, loss)

    def step(self, verbose: bool = False):
        grad, loss = self.evaluator.evaluate(self.params, verbose=verbose)
        self.params -= self.lr * grad
        self.history.append((self.params.copy(), loss))
        return grad, loss
    
    def _make_param_range(self, center, width, num):
        return np.linspace(center - width, center + width, num)

    def _open_file(self, filename, tag=None):
        fn = filename if str(filename).endswith(".dat") else f"{filename}.dat"
        if tag:
            fn = os.path.join(filename, f"{tag}.dat")
        d = os.path.dirname(fn)
        if d:
            os.makedirs(d, exist_ok=True)
        exists = os.path.exists(fn)
        f = open(fn, "a" if exists else "w")
        if not exists:
            f.write("param\tloss\treal_grad\timag_grad\tabs_grad\n")
        return f

    def _write_row(self, f, x, loss, grad):
        L = float(np.asarray(loss).ravel()[0])
        G = complex(np.asarray(grad).ravel()[0])
        f.write(f"{x:.10e}\t{L:.10e}\t{G.real:.10e}\t{G.imag:.10e}\t{abs(G):.10e}\n")

    def sweep(self, center: float = 0.199, width: float = 0.04, num: int = 21,
              adj_rotation=None,
              perturbation=None, verbose: bool = False, filename=None):
        if perturbation is None:
            perturbation = self.evaluator.param_perturbation

        param_range = self._make_param_range(center, width, num)
        losses, grads = [], []

        for x in param_range:
            p = [x]
            if adj_rotation: self.evaluator.adjoint_rotation = adj_rotation

            grad, loss = self.evaluator.evaluate(p, perturbation, verbose=verbose)
            grads.append(grad)
            losses.append(loss)

        if filename:
            with self._open_file(filename) as f:
                for x, L, G in zip(param_range, losses, grads):
                    self._write_row(f, x, L, G)
        else: 
            print("no filename to save data")

        return param_range, losses, grads

    def sweep_reusing_fields(self, center=0.199, width=0.04, num=21,
                             angles=(0.0,),     
                             perturbation=None, verbose=False, filename_base=None):
        if perturbation is None:
            perturbation = self.evaluator.param_perturbation

        param_range = self._make_param_range(center, width, num)

        for x in param_range:
            p = [x]
            fwd, adj, loss = self.evaluator.sims(p, perturbation)
            boundary_velocity_field, reference_coord, _ = \
                self.evaluator.parametric_designer.compute_boundary_velocity(p, perturbation)

            
            for ang in angles:
                grad = -self.evaluator._calc_adjoint_forward_product(
                    boundary_velocity_field, reference_coord,
                    fwd, adj, ang)
                if filename_base:
                    tag = f"ang={float(ang):.4f}rad"
                    with self._open_file(filename_base, tag=tag) as f:
                        self._write_row(f, x, loss, grad)

        if verbose:
            print(f"x={x:.6f} done")

        return param_range

    def gradient_descent(self, initial_param, lr=0.01, pertubation=None, num_steps=50, verbose=False):
        # Accept the existing (misspelled) argument name; map to the internal variable used elsewhere
        perturbation = self.evaluator.param_perturbation if pertubation is None else pertubation

        # Initialise params and (optionally) learning rate for this run
        self.params = np.asarray(initial_param, dtype=float)
        local_lr = float(lr)

        grads = []
        losses = []

        for k in range(int(num_steps)):
            grad, loss = self.evaluator.evaluate(self.params, perturbation, verbose=False)
           
            self.params -= local_lr * np.array([grad.imag])
            # Log minimal state
            self.history.append((self.params.copy(), loss))
            print(params)
            grads.append(grad)
            losses.append(loss)
            if verbose:
                G = complex(np.asarray(grad).ravel()[0])
                L = float(np.asarray(loss).ravel()[0])
                print(f"step={k:03d}  param={self.params.ravel()[0]:.10e}  loss={L:.10e}  grad={G.real:.10e}+{G.imag:.10e}j")

        return np.asarray(self.params, dtype=self.params.dtype), np.array(losses), np.array(grads)
