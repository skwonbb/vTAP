use halo2_base::{
    halo2_proofs::{
        circuit::{Cell, Region, Value},
        halo2curves::bn256::Fr,
        plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector},
        poly::Rotation,
    },
    poseidon::hasher::spec::OptimizedPoseidonSpec,
};

pub const T: usize = 3;
pub const RATE: usize = 2;
pub const R_F: usize = 8;
pub const R_P: usize = 57;

pub const ROUNDS: usize = R_F + R_P;
pub const PERM_ROWS: usize = 1 + ROUNDS;

const ZERO: Fr = Fr::zero();
const ONE: Fr = Fr::one();

pub type Spec = OptimizedPoseidonSpec<Fr, T, RATE>;

pub fn spec() -> Spec {
    OptimizedPoseidonSpec::new::<R_F, R_P, 0>()
}

pub fn blocks(n: usize) -> usize {
    n.div_ceil(RATE) + usize::from(n % RATE == 0)
}

pub fn rows(n: usize) -> usize {
    blocks(n) * PERM_ROWS + 1
}

pub fn advice_cells(n: usize) -> usize {
    rows(n) * T + blocks(n) * (RATE + R_P)
}

fn sbox(x: Fr, c: Fr) -> Fr {
    let x2 = x * x;
    x2 * x2 * x + c
}

#[derive(Clone, Debug)]
pub struct Config {
    pub state: [Column<Advice>; T],
    pub inp: [Column<Advice>; RATE],
    x5: Column<Advice>,
    rc: [Column<Fixed>; T],
    srow: [Column<Fixed>; T],
    scol: [Column<Fixed>; RATE],
    pad: [Column<Fixed>; RATE],
    consts: Column<Fixed>,

    s_abs: Selector,
    s_full: Selector, // MDS
    s_pre: Selector,  // pre_sparse_mds
    s_part: Selector, // sparse
}

impl Config {
    pub fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let sp = spec();
        let mds = *sp.mds_matrices().mds().as_ref();
        let pre = *sp.mds_matrices().pre_sparse_mds().as_ref();

        let state = [(); T].map(|_| meta.advice_column());
        let inp = [(); RATE].map(|_| meta.advice_column());
        let x5c = meta.advice_column();
        let rc = [(); T].map(|_| meta.fixed_column());
        let srow = [(); T].map(|_| meta.fixed_column());
        let scol = [(); RATE].map(|_| meta.fixed_column());
        let pad = [(); RATE].map(|_| meta.fixed_column());
        let consts = meta.fixed_column();

        for c in state {
            meta.enable_equality(c);
        }
        for c in inp {
            meta.enable_equality(c);
        }
        meta.enable_constant(consts);

        let (s_abs, s_full, s_pre, s_part) =
            (meta.selector(), meta.selector(), meta.selector(), meta.selector());

        meta.create_gate("absorb", |m| {
            let s = m.query_selector(s_abs);
            let mut out = Vec::new();
            for i in 0..T {
                let mut e = m.query_advice(state[i], Rotation::cur())
                    + m.query_fixed(rc[i], Rotation::cur());
                if i > 0 {
                    e = e + m.query_advice(inp[i - 1], Rotation::cur());
                }
                out.push(s.clone() * (m.query_advice(state[i], Rotation::next()) - e));
            }
            for j in 0..RATE {
                out.push(
                    s.clone()
                        * m.query_fixed(pad[j], Rotation::cur())
                        * m.query_advice(inp[j], Rotation::cur()),
                );
            }
            out
        });

        let mut full_gate = |sel: Selector, mc: [[Fr; T]; T], name: &'static str| {
            meta.create_gate(name, move |m| {
                let s = m.query_selector(sel);
                let p: Vec<_> = (0..T)
                    .map(|j| {
                        let x = m.query_advice(state[j], Rotation::cur());
                        let x2 = x.clone() * x.clone();
                        x2.clone() * x2 * x + m.query_fixed(rc[j], Rotation::cur())
                    })
                    .collect();
                (0..T)
                    .map(|i| {
                        let acc = (0..T)
                            .map(|j| Expression::Constant(mc[i][j]) * p[j].clone())
                            .reduce(|a, b| a + b)
                            .unwrap();
                        s.clone() * (m.query_advice(state[i], Rotation::next()) - acc)
                    })
                    .collect::<Vec<_>>()
            });
        };
        full_gate(s_full, mds, "full round (mds)");
        full_gate(s_pre, pre, "full round (pre-sparse mds)");

        //   x5      = cur[0]^5 + rc[0]
        //   next[0] = row[0]*x5 + Σ_{j>0} row[j]*cur[j]
        //   next[i] = col_hat[i-1]*x5 + cur[i]
        meta.create_gate("partial round", |m| {
            let s = m.query_selector(s_part);
            let x = m.query_advice(state[0], Rotation::cur());
            let x2 = x.clone() * x.clone();
            let x5 = m.query_advice(x5c, Rotation::cur());
            let mut out = vec![
                s.clone()
                    * (x5.clone() - (x2.clone() * x2 * x + m.query_fixed(rc[0], Rotation::cur()))),
            ];
            let acc = (0..T)
                .map(|j| {
                    let w = if j == 0 {
                        x5.clone()
                    } else {
                        m.query_advice(state[j], Rotation::cur())
                    };
                    m.query_fixed(srow[j], Rotation::cur()) * w
                })
                .reduce(|a, b| a + b)
                .unwrap();
            out.push(s.clone() * (m.query_advice(state[0], Rotation::next()) - acc));
            for i in 1..T {
                let e = m.query_fixed(scol[i - 1], Rotation::cur()) * x5.clone()
                    + m.query_advice(state[i], Rotation::cur());
                out.push(s.clone() * (m.query_advice(state[i], Rotation::next()) - e));
            }
            out
        });

        Config {
            state, inp, x5: x5c, rc, srow, scol, pad, consts,
            s_abs, s_full, s_pre, s_part,
        }
    }
}

fn perm_trace(sp: &Spec, pre: [Fr; T], blk: &[Fr]) -> Vec<[Fr; T]> {
    let cst = sp.constants();
    let mds = sp.mds_matrices().mds().as_ref();
    let pres = sp.mds_matrices().pre_sparse_mds().as_ref();
    let sm = sp.mds_matrices().sparse_matrices();
    let apply = |s: &[Fr; T], m: &[[Fr; T]; T]| -> [Fr; T] {
        core::array::from_fn(|i| (0..T).map(|j| m[i][j] * s[j]).sum())
    };

    let mut out = vec![pre];
    let pc = &cst.start()[0];
    let mut s = pre;
    s[0] += pc[0];
    for i in 1..T {
        s[i] += pc[i]
            + blk.get(i - 1).copied().unwrap_or(ZERO)
            + if i - 1 == blk.len() { ONE } else { ZERO };
    }
    out.push(s);

    for c in cst.start().iter().skip(1).take(R_F / 2 - 1) {
        s = apply(&core::array::from_fn(|j| sbox(s[j], c[j])), mds);
        out.push(s);
    }
    let c = *cst.start().last().unwrap();
    s = apply(&core::array::from_fn(|j| sbox(s[j], c[j])), pres);
    out.push(s);

    for (c, m) in cst.partial().iter().zip(sm.iter()) {
        let x5 = sbox(s[0], *c);
        let (row, col) = (m.row(), m.col_hat());
        let mut ns = [ZERO; T];
        ns[0] = (0..T).map(|j| row[j] * if j == 0 { x5 } else { s[j] }).sum();
        for i in 1..T {
            ns[i] = col[i - 1] * x5 + s[i];
        }
        s = ns;
        out.push(s);
    }
    for c in cst.end().iter() {
        s = apply(&core::array::from_fn(|j| sbox(s[j], c[j])), mds);
        out.push(s);
    }
    s = apply(&core::array::from_fn(|j| sbox(s[j], ZERO)), mds);
    out.push(s);

    debug_assert_eq!(out.len(), PERM_ROWS + 1);
    out
}

pub fn native(sp: &Spec, inputs: &[Fr]) -> Fr {
    let mut s = [ZERO; T];
    s[0] = Fr::from_raw([0, 1, 0, 0]); // capacity 2^64
    for blk in inputs.chunks(RATE) {
        s = *perm_trace(sp, s, blk).last().unwrap();
    }
    if inputs.len() % RATE == 0 {
        s = *perm_trace(sp, s, &[]).last().unwrap();
    }
    s[1]
}

pub struct Assigned {
    pub digest: Cell,
    pub digest_value: Fr,
    pub inputs: Vec<Cell>,
}

pub fn assign(
    cfg: &Config,
    region: &mut Region<Fr>,
    sp: &Spec,
    inputs: &[Fr],
    row0: usize,
) -> Result<Assigned, Error> {
    let cst = sp.constants();
    let sm = sp.mds_matrices().sparse_matrices();

    let mut s = [ZERO; T];
    s[0] = Fr::from_raw([0, 1, 0, 0]);
    let init = s;

    let mut blks: Vec<&[Fr]> = inputs.chunks(RATE).collect();
    if inputs.len() % RATE == 0 {
        blks.push(&[]);
    }

    let mut inp_cells = Vec::with_capacity(inputs.len());
    let mut last = [ZERO; T];

    for (p, blk) in blks.iter().enumerate() {
        let base = row0 + p * PERM_ROWS;
        let tr = perm_trace(sp, s, blk);

        for (k, st) in tr.iter().take(PERM_ROWS).enumerate() {
            for j in 0..T {
                let c = region.assign_advice(cfg.state[j], base + k, Value::known(st[j]));
                if p == 0 && k == 0 {
                    let cc = region.assign_fixed(cfg.consts, row0 + j, init[j]);
                    region.constrain_equal(c.cell(), cc);
                }
            }
        }

        cfg.s_abs.enable(region, base)?;
        let pc = &cst.start()[0];
        region.assign_fixed(cfg.rc[0], base, pc[0]);
        for i in 1..T {
            let extra = if i - 1 == blk.len() { ONE } else { ZERO };
            region.assign_fixed(cfg.rc[i], base, pc[i] + extra);
        }
        for j in 0..RATE {
            let v = blk.get(j).copied().unwrap_or(ZERO);
            let c = region.assign_advice(cfg.inp[j], base, Value::known(v));
            region.assign_fixed(cfg.pad[j], base, if j < blk.len() { ZERO } else { ONE });
            if j < blk.len() {
                inp_cells.push(c.cell());
            }
        }

        for k in 1..=(R_F / 2) {
            let c = cst.start()[k];
            for j in 0..T {
                region.assign_fixed(cfg.rc[j], base + k, c[j]);
            }
            if k < R_F / 2 {
                cfg.s_full.enable(region, base + k)?;
            } else {
                cfg.s_pre.enable(region, base + k)?;
            }
        }
        for i in 0..R_P {
            let k = R_F / 2 + 1 + i;
            cfg.s_part.enable(region, base + k)?;
            region.assign_advice(cfg.x5, base + k, Value::known(sbox(tr[k][0], cst.partial()[i])));
            region.assign_fixed(cfg.rc[0], base + k, cst.partial()[i]);
            for j in 1..T {
                region.assign_fixed(cfg.rc[j], base + k, ZERO);
            }
            for j in 0..T {
                region.assign_fixed(cfg.srow[j], base + k, sm[i].row()[j]);
            }
            for j in 0..RATE {
                region.assign_fixed(cfg.scol[j], base + k, sm[i].col_hat()[j]);
            }
        }
        for i in 0..(R_F / 2) {
            let k = R_F / 2 + 1 + R_P + i;
            cfg.s_full.enable(region, base + k)?;
            let c = if i < cst.end().len() { cst.end()[i] } else { [ZERO; T] };
            for j in 0..T {
                region.assign_fixed(cfg.rc[j], base + k, c[j]);
            }
        }

        s = *tr.last().unwrap();
        last = s;
    }

    let tail = row0 + blks.len() * PERM_ROWS;
    let mut digest = None;
    for j in 0..T {
        let c = region.assign_advice(cfg.state[j], tail, Value::known(last[j]));
        if j == 1 {
            digest = Some(c.cell());
        }
    }
    Ok(Assigned { digest: digest.unwrap(), digest_value: last[1], inputs: inp_cells })
}
