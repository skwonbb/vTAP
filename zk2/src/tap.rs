use std::{fs, path::Path};

use halo2_base::{
    gates::{GateInstructions, RangeChip, RangeInstructions},
    halo2_proofs::halo2curves::bn256::Fr,
    utils::BigPrimeField,
    AssignedValue, Context,
    QuantumCell::{Constant, Existing},
};

#[derive(Clone)]
pub struct Params {
    pub d: usize,
    pub r: usize,
    pub f: u32,
    pub b_bits: usize,
    pub w: Vec<Vec<i64>>,
    pub m: Vec<Vec<i64>>,
    pub pmat: Vec<Vec<i64>>,
    pub bias: Vec<i64>,
}

pub fn fe(x: i64) -> Fr {
    if x >= 0 {
        Fr::from(x as u64)
    } else {
        -Fr::from((-x) as u64)
    }
}

pub fn rescale<F: BigPrimeField>(
    ctx: &mut Context<F>,
    range: &RangeChip<F>,
    acc: AssignedValue<F>,
    f: u32,
    b_bits: usize,
) -> AssignedValue<F> {
    let half = F::from(1u64 << (f - 1));
    let mid = F::from(2u64).pow_vartime([(b_bits - 1) as u64]);
    let biased = range.gate().add(ctx, acc, Constant(half + mid));
    let (q, _rem) = range.div_mod(ctx, biased, 1u128 << f, b_bits);
    let off = F::from(2u64).pow_vartime([(b_bits as u64) - 1 - f as u64]);
    range.gate().sub(ctx, q, Constant(off))
}

pub fn load_u(ctx: &mut Context<Fr>, u: &[i64]) -> Vec<AssignedValue<Fr>> {
    u.iter().map(|&x| ctx.load_witness(fe(x))).collect()
}

pub fn arith(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    p: &Params,
    u_cells: &[AssignedValue<Fr>],
    form: &str,
) -> Vec<AssignedValue<Fr>> {
    let mut out = Vec::with_capacity(p.d);
    if form == "direct" {
        for k in 0..p.d {
            let coeff: Vec<_> = p.pmat[k].iter().map(|&c| Constant(fe(c))).collect();
            let cells: Vec<_> = u_cells.iter().map(|c| Existing(*c)).collect();
            let acc = range.gate().inner_product(ctx, cells, coeff);
            let q = rescale(ctx, range, acc, p.f, p.b_bits);
            out.push(range.gate().add(ctx, q, Constant(fe(p.bias[k]))));
        }
    } else if form != "commit" {
        let mut z = Vec::with_capacity(p.r);
        for j in 0..p.r {
            let coeff: Vec<_> = p.w[j].iter().map(|&c| Constant(fe(c))).collect();
            let cells: Vec<_> = u_cells.iter().map(|c| Existing(*c)).collect();
            let acc = range.gate().inner_product(ctx, cells, coeff);
            z.push(rescale(ctx, range, acc, p.f, p.b_bits));
        }
        for k in 0..p.d {
            let coeff: Vec<_> = p.m[k].iter().map(|&c| Constant(fe(c))).collect();
            let cells: Vec<_> = z.iter().map(|c| Existing(*c)).collect();
            let acc = range.gate().inner_product(ctx, cells, coeff);
            let q = rescale(ctx, range, acc, p.f, p.b_bits);
            out.push(range.gate().add(ctx, q, Constant(fe(p.bias[k]))));
        }
    }
    out
}

pub fn load_params(dir: &Path, d: usize, r: usize, f: u32) -> Params {
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(dir.join("quantized.json")).unwrap()).unwrap();
    let take = |k: &str| -> Vec<Vec<i64>> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect())
            .collect()
    };
    Params {
        d,
        r,
        f,
        b_bits: v["nbits"].as_u64().unwrap() as usize,
        w: take("W_cols"),
        m: take("M_cols"),
        pmat: take("P_cols"),
        bias: v["b"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect(),
    }
}
