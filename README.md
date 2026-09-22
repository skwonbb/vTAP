# vTAP — Verifiable Task-Aligned Projection

Code and data for *vTAP: Verifiable Task-Aligned Projection for
Privacy-Preserving Outsourced Inference*.

vTAP is a closed-form linear transform applied to a released embedding,

$$\hat U = (UW)M + b$$

where $W$ solves a shrinkage-regularized discriminant problem
($S_B w = \lambda \hat S_W w$) and $M, b$ come from a least-squares fit.
There is no gradient descent, no epochs, no seeds: the same fitting sample
always yields the same $W, M, b$, so the deployed map can be checked against
its specification on the published matrices alone.

## Layout

```
src/           vTAP and the comparison methods; figures, tables, exports
zk/            circom + snarkjs (Groth16 over BN254)
zk2/           halo2 (KZG over BN254), with a custom-gate Poseidon chip
results/       measurement records the paper is computed from
paper_data/    the values shown in each figure and table
```

Figure files are not committed; the scripts below regenerate them from
`results/`.

## Reproducing the figures and tables

```bash
pip install numpy pandas scipy matplotlib

python src/fig2.py              # what the raw embedding gives up
python src/fig4.py              # method retention, dense sweep
python src/fig5.py              # ablation
python src/fig6.py              # receiver models given Uhat
python src/fig7.py              # role assignment, encoder swap
python src/fig8.py              # arithmetization error
python src/fig9.py              # vTAP vs commitment share
python src/fig10.py             # deployment cost

python src/tables.py            # per-family retention, and Table 5
python src/tab_quant.py         # Table 7 values
python src/tab_compat_quant.py  # Table 8 values
python src/tab_roles_enc.py     # values behind Figure 7

python src/paper_data.py        # every figure and table value -> paper_data/
```

Figures are written to `results/figures/`, numbered as in the paper.

On Windows, set `PYTHONIOENCODING=utf-8` if the console rejects the output.

## paper_data/

| file | contents |
|---|---|
| `figure2.csv` | what each attribute reads from the raw embedding, per family |
| `figure4.csv` | (a)–(f) retention, method × probe family × target |
| `figure4g.csv` | (g) the three points read off the sweep |
| `figure5.csv` | four ablation variants × family × target |
| `figure6.csv` | raw-$U$ and $\hat U$ accuracy, retention, difference |
| `figure7.csv` | (a) 4×4 role grid, (b) two encoders |
| `figure8.csv` | total error and per-stage share against $f$ |
| `figure10.csv` | 12 panels × 4 points, both proof systems |
| `figure9.csv` | vTAP/commitment split at eight points |
| `table5.csv` | ablation averaged over the five probe families |
| `table6.csv` | factored against direct arithmetization |
| `table7.csv` | accuracy under fixed-point arithmetic at $f=16$ |
| `table8.csv` | a receiver trained on raw $U$, given $\hat U_f$ |


## Re-running the experiments

Building `results/` needs the dataset and the embeddings.

1. Obtain **RAF-DB** and place it under `data/` (not redistributable here).
2. `src/03_build_rafdb_labels.py`, then `src/05_embed_rafdb.py`
   (CLIP ViT-L/14) and `src/11_embed_rafdb_dino_large.py` (DINOv2 ViT-L/14).
3. `src/07_run.py [run]` — trains the methods and runs the probes, writing
   `probe_runs_<run>.csv`. Run it twice: no argument for the main CLIP run,
   then `rafdb_swap` for the permitted-task swap.
4. `src/sweep_dense.py` — the dense grid behind Figure 4(g).
5. Side experiments: `run_roles.py` (role assignment), `run_dinol.py`
   (encoder swap), `run_between.py` (ablation stage 1), `recon_byfamily.py`
   (the deployed transform, per family), `compat_byfamily.py` (receiver
   compatibility), `compat_quant.py` (the same receivers under fixed-point
   arithmetic), `recon_appendix.py`, `quant_ablation.py`,
   `quant_sweep.py`, `quant_form.py` (fixed-point arithmetic).
6. `run_swap_leace2.py`, then `run_appendix.py` — both append to the swap run
   from step 3.

`src/lib.py` holds vTAP and every comparison method (VIB, adversarial
removal, LEACE, PCA, random projection).

## Verifying the deployed map

Checkable on the published matrices:

```python
import numpy as np
W = np.load("results/tap_params/W.npy")   # 768 x 4
M = np.load("results/tap_params/M.npy")   #   4 x 768
P = W @ M
np.linalg.matrix_rank(P)          # 4 = r
np.abs(P @ P - P).max()           # 1.2e-16, idempotent to machine precision
np.abs(M @ W - np.eye(4)).max()   # 1.4e-15
```

$P = WM$ is the rank-$r$ idempotent the paper specifies; reproducing $W, M, b$
from the fitting sample reproduces it exactly.

## Zero-knowledge proofs

The relation proved is

$$\mathcal{R}_{\mathrm{vTAP}} = \{((c,\hat U),(\tilde U,\rho)) :
c = \mathrm{Com}(\tilde U;\rho) \;\wedge\; \hat U = \mathsf{vTAP}(\tilde U)\}$$

lifted into two arithmetizations. `zk/check_equal.py` confirms that all 768
coordinates of $\hat U$ agree between them.

Both sweeps read the same parameters: `bench_scaling.py` and `sweep_halo2.py`
both call `gen_circuits.py --random --nbits 48`, whose generator is seeded at
0, so at a given $(d,r)$ the two systems use bit-identical
$\tilde W, \tilde M, \tilde b$.

Value dependence differs between them. R1CS constraint counts are exactly
independent of the coefficients — a constant multiplication is absorbed into
a linear combination — and re-measuring with the deployed parameters
reproduces 37,056. halo2 advice cells depend weakly: `inner_product` skips
terms with coefficient 0, and three coefficients round to zero in the real
parameters but none in the random ones, so vTAP cells differ by 60,101 against
60,098 at $d=768, r=4$ (0.005%).

### Prerequisites

`cache/embeddings/rafdb_clip.npy`, from step 2 above — `gen_circuits.py` draws
the sample input from it.

Powers of tau at $2^{17}$, the smallest power that holds the largest circuit
measured (151 MB, so not committed):

```bash
mkdir -p zk/ptau && cd zk/ptau
snarkjs powersoftau new bn128 17 pot_0.ptau -v
snarkjs powersoftau contribute pot_0.ptau pot_1.ptau --name="reproduction" -v
snarkjs powersoftau prepare phase2 pot_1.ptau pot17_final.ptau -v
```

### Running

```bash
python zk/gen_circuits.py --d 768 --r 4 --f 16    # circuits and constants

cd zk && npm install
python bench_scaling.py     # d, r sweep       -> zk_scaling.csv
python sweep_commit.py      # commitment only  -> zk_commit_groth.csv

cd zk2 && cargo run --release --bin tap_halo2 -- \
    --params ../results/tap_params --d 768 --r 4 --f 16 \
    --lookup_bits 12 --max_cols 4
python ../zk/sweep_halo2.py                     # -> zk_halo2_gated.csv
```

The sweep removes its build directory, so build the deployed point once more
before comparing the two arithmetizations:

```bash
python zk/build_full.py     # -> zk/build/tap_full.*
python zk/check_equal.py    # d=768  match 768/768
```

Both sweeps skip points already present in their CSV; `--dry` lists what they
would measure. The direct form fails witness generation at $d \ge 512$ (WASM
function size limit); that failure is itself a result, so the row is kept up to
the stage reached.

`sweep_commit.py` measures the commitment circuit at the four values of $d$
Figure 9 needs; `bench_scaling.py` measures it only at the deployed point.

## results/

| file | contents |
|---|---|
| `probe_runs_rafdb.csv` | main experiment — method × $k$ × target × probe family |
| `probe_runs_rafdb_roles.csv` | role assignment varied |
| `probe_runs_rafdb_swap.csv` | permitted task swapped |
| `probe_runs_rafdb_dinol.csv` | DINOv2 ViT-L/14 encoder |
| `sweep_dense.csv` | dense grid over the learned baselines' handle |
| `compat_5fam.csv` | receiver-model compatibility |
| `compat_quant.csv` | the same receivers, given $\hat U_f$ |
| `recon_appendix_*.csv` | role and encoder runs measured on $\hat U$ |
| `quant_error_ablation.csv` | stage-wise arithmetization error over $f$ |
| `quant_sweep.csv` | probe accuracy under fixed-point arithmetic |
| `quant_form.csv` | factored against direct form |
| `zk_scaling.csv` | Groth16 over $d$ and $r$ |
| `zk_commit_groth.csv` | Groth16, commitment circuit over $d$ |
| `zk_halo2_gated.csv` | halo2-KZG over $d$ and $r$ |

`results/tap_params/{W,M,b}.npy` (54 KB) are the deployed parameters, and the
input to both the check above and the circuits.

## Environment

- Python 3.12, PyTorch (CUDA), scikit-learn
- circom 2.1.8, snarkjs 0.7.6
- Rust, halo2-axiom 0.5.3 with halo2-base 0.5.5
- Intel Core i7-11700, 8 cores / 16 threads, 32 GB
- NVIDIA GeForce RTX 3060, for the embeddings and the learned baselines

`zk2/rust-toolchain.toml` selects nightly because halo2-base 0.5.5 requires
rustc 1.91.1 or newer; no nightly feature is used, so stable 1.91+ also works.

## License

MIT, see `LICENSE`. RAF-DB itself is not redistributed here; `results/`
holds measurements derived from it, not the images or their labels.
