"""Generate the circom circuits and the quantized constants."""
import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CIRC = HERE / "circuits"


CHUNK = 128        


def _chunks(coeffs, sig):
    parts = [(int(c), i) for i, c in enumerate(coeffs) if int(c) != 0]
    if not parts:
        return ["0"]
    out = []
    for s in range(0, len(parts), CHUNK):
        expr = ""
        for n, (c, i) in enumerate(parts[s:s + CHUNK]):
            tok = f"{abs(c)}*{sig}[{i}]"
            expr += (("-" if c < 0 else "") + tok if n == 0
                     else f" {'-' if c < 0 else '+'} {tok}")
        out.append(expr)
    return out


def lincomb(coeffs, sig, acc, tag):
    ch = _chunks(coeffs, sig)
    if len(ch) == 1:
        return [], ch[0]
    L = [f"    signal {acc}[{len(ch)}];"]
    for n, e in enumerate(ch):
        prev = f"{acc}[{n - 1}] + " if n else ""
        L.append(f"    {acc}[{n}] <== {prev}{e};")
    return L, f"{acc}[{len(ch) - 1}]"


RESCALE = """
// Rescale a 2f-bit accumulator back to f bits. Division is not a field
// operation, so it is expressed as a quotient-remainder witness with a
// range check; the sign is folded into a bias constant so that negative
// values use the same decomposition.
template Rescale() {{
    signal input  in;
    signal output out;
    component n2b = Num2Bits({NB});
    n2b.in <== in + {BIAS};
    component b2n = Bits2Num({NB} - {F});
    for (var k = 0; k < {NB} - {F}; k++) {{ b2n.in[k] <== n2b.out[k + {F}]; }}
    out <== b2n.out - {OFF};
}}
"""


def header(f, nb):
    return ('pragma circom 2.1.8;\n\ninclude "bitify.circom";\n'
            + RESCALE.format(NB=nb, F=f, BIAS=(1 << (f - 1)) + (1 << (nb - 1)),
                             OFF=1 << (nb - 1 - f)))


def emit_factored(path, Wq, Mq, bq, f, nb):
    d, r = Wq.shape
    L = [header(f, nb),
         f"// factored form, through Z=UW: {r}+{d} rescalings, {2*d*r} products\n"
         f"template Tap() {{\n    signal input  U[{d}];\n"
         f"    signal output Uhat[{d}];\n    component rz[{r}];\n    signal Z[{r}];"]
    for j in range(r):
        code, sig = lincomb(Wq[:, j], "U", f"za{j}", "z")
        L += code
        L.append(f"    rz[{j}] = Rescale();\n    rz[{j}].in <== {sig};\n"
                 f"    Z[{j}] <== rz[{j}].out;")
    L.append(f"    component ru[{d}];")
    for k in range(d):
        code, sig = lincomb(Mq[:, k], "Z", f"ua{k}", "u")
        L += code
        L.append(f"    ru[{k}] = Rescale();\n    ru[{k}].in <== {sig};\n"
                 f"    Uhat[{k}] <== ru[{k}].out + {int(bq[k])};")
    L.append("}\n\ncomponent main = Tap();")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def emit_direct(path, Pq, bq, f, nb):
    d = Pq.shape[0]
    L = [header(f, nb),
         f"// direct form: {d} rescalings, {d*d} products\n"
         f"template Tap() {{\n    signal input  U[{d}];\n"
         f"    signal output Uhat[{d}];\n    component ru[{d}];"]
    for k in range(d):
        code, sig = lincomb(Pq[:, k], "U", f"ua{k}", "u")
        L += code
        L.append(f"    ru[{k}] = Rescale();\n    ru[{k}].in <== {sig};\n"
                 f"    Uhat[{k}] <== ru[{k}].out + {int(bq[k])};")
    L.append("}\n\ncomponent main = Tap();")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def emit_commit(path, d, rate=15):
    nblk = -(-d // rate)
    L = ['pragma circom 2.1.8;\n\ninclude "poseidon.circom";\n',
         f"// absorb {d} elements plus the opening randomness, rate={rate} ({nblk} permutations).",
         f"template Commit() {{\n    signal input  U[{d}];\n"
         f"    signal input  rho;\n    signal output c;\n"
         f"    component h[{nblk}];\n    signal st[{nblk + 1}];\n    st[0] <== 0;"]
    for t in range(nblk):
        L.append(f"    h[{t}] = Poseidon({rate + 1});\n    h[{t}].inputs[0] <== st[{t}];")
        for s in range(rate):
            i = t * rate + s
            src = f"U[{i}]" if i < d else "rho"
            L.append(f"    h[{t}].inputs[{s + 1}] <== {src};")
        L.append(f"    st[{t + 1}] <== h[{t}].out;")
    L.append(f"    c <== st[{nblk}];\n}}\n\ncomponent main = Commit();")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def emit_full(path, Wq, Mq, bq, f, nb, rate=15):
    d, r = Wq.shape
    nblk = -(-d // rate)
    L = [header(f, nb).replace('include "bitify.circom";',
                               'include "bitify.circom";\ninclude "poseidon.circom";'),
         f"template TapFull() {{\n    signal input  U[{d}];\n    signal input  rho;\n"
         f"    signal output c;\n    signal output Uhat[{d}];",
         f"    component h[{nblk}];\n    signal st[{nblk + 1}];\n    st[0] <== 0;"]
    for t in range(nblk):
        L.append(f"    h[{t}] = Poseidon({rate + 1});\n    h[{t}].inputs[0] <== st[{t}];")
        for s in range(rate):
            i = t * rate + s
            L.append(f"    h[{t}].inputs[{s + 1}] <== "
                     + (f"U[{i}];" if i < d else "rho;"))
        L.append(f"    st[{t + 1}] <== h[{t}].out;")
    L.append(f"    c <== st[{nblk}];")
    L.append(f"    component rz[{r}];\n    signal Z[{r}];")
    for j in range(r):
        code, sig = lincomb(Wq[:, j], "U", f"za{j}", "z")
        L += code
        L.append(f"    rz[{j}] = Rescale();\n    rz[{j}].in <== {sig};\n"
                 f"    Z[{j}] <== rz[{j}].out;")
    L.append(f"    component ru[{d}];")
    for k in range(d):
        code, sig = lincomb(Mq[:, k], "Z", f"ua{k}", "u")
        L += code
        L.append(f"    ru[{k}] = Rescale();\n    ru[{k}].in <== {sig};\n"
                 f"    Uhat[{k}] <== ru[{k}].out + {int(bq[k])};")
    L.append("}\n\ncomponent main = TapFull();")
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def dump_json(W, M, b, Pq, f, nb, umax, d, r, out_dir):
    import json
    S = 1 << f
    Wq = np.rint(W * S).astype(np.int64)
    Mq = np.rint(M * S).astype(np.int64)
    bq = np.rint(np.asarray(b).reshape(-1) * S).astype(np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quantized.json").write_text(json.dumps({
        "d": d, "r": r, "f": f, "nbits": nb, "umax": umax,
        "W_cols": [Wq[:, j].tolist() for j in range(r)],   
        "M_cols": [Mq[:, k].tolist() for k in range(d)],   


        "P_cols": [Pq[:, k].tolist() for k in range(d)],   
        "b": bq.tolist(),
    }), encoding="utf-8")


    U = np.resize(np.load(ROOT / "cache" / "embeddings" / "rafdb_clip.npy")[0], d)
    (out_dir / "u_sample.json").write_text(json.dumps(
        np.rint(np.asarray(U, np.float64) * S).astype(np.int64).tolist()), "utf-8")
    return out_dir / "quantized.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--f", type=int, default=16)
    ap.add_argument("--rate", type=int, default=15)
    ap.add_argument("--nbits", type=int, default=0,
                    help="rescaling bit width; fix it so that a sweep varies only d and r "
                         "(0 = derive it from the parameter magnitudes)")
    ap.add_argument("--umax", type=float, default=14.0,
                    help="max |U|, used to bound the rescaling width")
    ap.add_argument("--random", action="store_true",
                    help="use random parameters instead of the real ones")
    a = ap.parse_args()
    CIRC.mkdir(parents=True, exist_ok=True)
    S = 1 << a.f

    if a.random:
        rng = np.random.default_rng(0)
        W = rng.normal(0, .5, (a.d, a.r)); M = rng.normal(0, .2, (a.r, a.d))
        b = rng.normal(0, 1, a.d)
    else:
        P = ROOT / "results" / "tap_params"
        W, M, b = (np.load(P / f"{n}.npy") for n in ("W", "M", "b"))
        b = b.reshape(-1)
        W, M, b = W[:a.d, :a.r], M[:a.r, :a.d], b[:a.d]

    Wq = np.rint(W * S).astype(np.int64)
    Mq = np.rint(M * S).astype(np.int64)
    Pq = np.rint((W @ M) * S).astype(np.int64)     
    bq = np.rint(b * S).astype(np.int64)


    umax, zmax = a.umax, a.umax * float(np.abs(W).sum(0).max())
    bound = max(umax * S * float(np.abs(Wq).sum(0).max()),
                zmax * S * float(np.abs(Mq).sum(0).max()),
                umax * S * float(np.abs(Pq).sum(0).max()))
    need = int(np.ceil(np.log2(bound + 1)))
    nb = need + 2                                  
    if a.nbits:                                    
        if a.nbits < nb:
            raise SystemExit(f"--nbits {a.nbits} is smaller than the required {nb}")
        nb = a.nbits
    if nb >= 250:
        raise SystemExit(f"NBITS={nb} exceeds the BN254 field")
    print(f"d={a.d} r={a.r} f={a.f}  accumulator bound ~2^{need}  -> NBITS={nb}"
          f"  ({nb} constraints per rescaling)")

    made = [emit_factored(CIRC / "tap_factored.circom", Wq, Mq, bq, a.f, nb),
            emit_direct(CIRC / "tap_direct.circom", Pq, bq, a.f, nb),
            emit_commit(CIRC / "commit.circom", a.d, a.rate),
            emit_full(CIRC / "tap_full.circom", Wq, Mq, bq, a.f, nb, a.rate)]
    dump_json(W, M, b, Pq, a.f, nb, a.umax, a.d, a.r, ROOT / "results" / "tap_params")
    (CIRC / "params.json").write_text(json.dumps(
        dict(d=a.d, r=a.r, f=a.f, nbits=nb, rate=a.rate, random=a.random)), "utf-8")
    for p in made:
        print(f"  {p.name:22s} {p.stat().st_size/1e6:8.2f} MB")


if __name__ == "__main__":
    main()

