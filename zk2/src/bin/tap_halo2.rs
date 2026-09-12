use std::{env, fs, path::PathBuf, time::Instant};

use halo2_base::{
    gates::circuit::{
        builder::BaseCircuitBuilder, BaseCircuitParams, BaseConfig, CircuitBuilderStage,
    },
    halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        halo2curves::bn256::{Bn256, Fr, G1Affine},
        plonk::{
            create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, Column, ConstraintSystem,
            Error, Instance,
        },
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
        SerdeFormat,
    },
    ContextCell,
};
use rand::{rngs::StdRng, SeedableRng};

const MIN_ROWS: usize = 20;

const LOOKUP_COLS: usize = 1;

use tap_halo2::{
    poseidon,
    tap::{arith, load_params, load_u, Params},
};

struct Built {
    builder: BaseCircuitBuilder<Fr>,
    hash_cells: Vec<ContextCell>,
    hash_values: Vec<Fr>,
    commit: Fr,
    out: Vec<Fr>,
}

fn build(stage: CircuitBuilderStage, cfg: Option<BaseCircuitParams>, k: u32, lookup_bits: usize,
         p: &Params, u: &[i64], rho: u64, form: &str) -> Built {
    let mut builder = BaseCircuitBuilder::<Fr>::from_stage(stage).use_instance_columns(1);
    builder = match cfg {
        Some(c) => builder.use_params(c),
        None => builder.use_k(k as usize).use_lookup_bits(lookup_bits),
    };

    let range = builder.range_chip();
    let ctx = builder.main(0);

    let mut cells = load_u(ctx, u);
    cells.push(ctx.load_witness(Fr::from(rho)));
    let hash_cells: Vec<ContextCell> = cells.iter().filter_map(|c| c.cell).collect();
    let hash_values: Vec<Fr> = cells.iter().map(|c| *c.value()).collect();

    let out_cells = arith(ctx, &range, p, &cells[..u.len()], form);
    let out: Vec<Fr> = out_cells.iter().map(|c| *c.value()).collect();
    for c in out_cells {
        builder.assigned_instances[0].push(c);
    }

    let commit = poseidon::native(&poseidon::spec(), &hash_values);
    Built { builder, hash_cells, hash_values, commit, out }
}

struct TapCircuit {
    b: Built,
    params: BaseCircuitParams,
}

impl Circuit<Fr> for TapCircuit {
    type Config = (BaseConfig<Fr>, poseidon::Config, Column<Instance>);
    type FloorPlanner = SimpleFloorPlanner;
    type Params = BaseCircuitParams;

    fn params(&self) -> BaseCircuitParams {
        self.params.clone()
    }
    fn without_witnesses(&self) -> Self {
        unimplemented!()
    }
    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        params: BaseCircuitParams,
    ) -> Self::Config {
        let base = BaseConfig::configure(meta, params);
        let pos = poseidon::Config::configure(meta);
        let inst = meta.instance_column();
        meta.enable_equality(inst);
        (base, pos, inst)
    }
    fn configure(_: &mut ConstraintSystem<Fr>) -> Self::Config {
        unreachable!("use configure_with_params")
    }
    fn synthesize(
        &self,
        (base_cfg, pos_cfg, inst): Self::Config,
        mut ly: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        Circuit::synthesize(&self.b.builder, base_cfg, ly.namespace(|| "tap"))?;

        let sp = poseidon::spec();
        let wit_only = self.b.builder.core().witness_gen_only();
        let a = ly.assign_region(
            || "commitment",
            |mut r| {
                let a = poseidon::assign(&pos_cfg, &mut r, &sp, &self.b.hash_values, 0)?;
                if !wit_only {
                    let cm = self.b.builder.core().copy_manager.lock().unwrap();
                    for (mine, theirs) in a.inputs.iter().zip(self.b.hash_cells.iter()) {
                        let phys = cm.assigned_advices[theirs];
                        r.constrain_equal(*mine, phys);
                    }
                }
                Ok(a)
            },
        )?;
        ly.constrain_instance(a.digest, inst, 0);
        Ok(())
    }
}

fn main() {
    std::env::set_var("MAX_DEGREE", "6");

    let args: Vec<String> = env::args().collect();
    let get = |k: &str, dflt: &str| -> String {
        args.windows(2)
            .find(|w| w[0] == format!("--{k}"))
            .map(|w| w[1].clone())
            .unwrap_or_else(|| dflt.to_string())
    };
    let dir = PathBuf::from(get("params", "../results/tap_params"));
    let d: usize = get("d", "768").parse().unwrap();
    let r: usize = get("r", "4").parse().unwrap();
    let f: u32 = get("f", "16").parse().unwrap();
    let form = get("form", "factored");
    let kmin: u32 = get("kmin", "12").parse().unwrap();
    let kmax: u32 = get("kmax", "22").parse().unwrap();
    let max_cols: usize = get("max_cols", "4").parse().unwrap();
    let mock = args.iter().any(|a| a == "--mock");

    let p = load_params(&dir, d, r, f);
    let raw: Vec<i64> =
        serde_json::from_str(&fs::read_to_string(dir.join("u_sample.json")).unwrap()).unwrap();
    let u: Vec<i64> = (0..d).map(|i| raw[i % raw.len()]).collect();
    let u = &u[..];
    let rho = 12345678901234567890u64;

    let lookup_bits: usize = get("lookup_bits", &f.to_string()).parse().unwrap();
    let pos_rows = poseidon::rows(d + 1);

    let mut chosen = None;
    for kk in kmin.max(lookup_bits as u32 + 1)..=kmax {
        let mut b = build(CircuitBuilderStage::Keygen, None, kk, lookup_bits, &p, u, rho, &form);
        let st = b.builder.statistics();
        let adv: usize = st.gate.total_advice_per_phase.iter().sum();
        let lk: usize = st.total_lookup_advice_per_phase.iter().sum();
        let max_rows = (1usize << kk) - MIN_ROWS;
        if adv.div_ceil(max_cols) <= max_rows
            && lk.div_ceil(LOOKUP_COLS) <= max_rows
            && (1usize << lookup_bits) < (1usize << kk)
            && pos_rows < max_rows
        {
            let mut cfg = b.builder.calculate_params(Some(MIN_ROWS));
            cfg.num_advice_per_phase = vec![max_cols];
            cfg.num_lookup_advice_per_phase = vec![LOOKUP_COLS];
            chosen = Some((kk, b, cfg, adv, lk, st.gate.total_fixed));
            break;
        }
    }
    let (k, built, cfg, advice, lookup, fixed) = chosen.expect("no k within kmax fits the circuit");

    let commit = built.commit;
    let out = built.out.clone();
    let inst0 = out.clone();
    let inst1 = vec![commit];

    let cfg2 = cfg.clone();
    let circuit = TapCircuit { b: built, params: cfg.clone() };

    if mock {
        let mb = build(CircuitBuilderStage::Mock, Some(cfg.clone()), k, lookup_bits, &p, u, rho, &form);
        let mc = TapCircuit { b: mb, params: cfg.clone() };
        let t = Instant::now();
        let res = MockProver::run(k, &mc, vec![inst0.clone(), inst1.clone()])
            .unwrap()
            .verify();
        match res {
            Ok(()) => println!("{{\"mock\":\"ok\",\"secs\":{:.2}}}", t.elapsed().as_secs_f64()),
            Err(e) => {
                println!("{{\"mock\":\"FAIL\"}}");
                for x in e.iter().take(6) {
                    println!("  {x:?}");
                }
                return;
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(0);
    let t = Instant::now();
    let kzg = ParamsKZG::<Bn256>::setup(k, &mut rng);
    let srs_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let vk = keygen_vk(&kzg, &circuit).unwrap();
    let pk = keygen_pk(&kzg, vk.clone(), &circuit).unwrap();
    let keygen_s = t.elapsed().as_secs_f64();

    let mut buf = Vec::new();
    kzg.write_custom(&mut buf, SerdeFormat::RawBytes).unwrap();
    let srs_bytes = buf.len();
    let pk_bytes = pk.to_bytes(SerdeFormat::RawBytes).len();
    let vk_bytes = vk.to_bytes(SerdeFormat::RawBytes).len();

    let bp = circuit.b.builder.break_points();
    drop(circuit);

    let t = Instant::now();
    let mut pb = build(CircuitBuilderStage::Prover, Some(cfg.clone()), k, lookup_bits, &p, u, rho, &form);
    pb.builder.set_break_points(bp);
    let pc = TapCircuit { b: pb, params: cfg };
    let mut tr = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<Bn256>, _, _, _, _>(
        &kzg, &pk, &[pc], &[&[&inst0, &inst1]], &mut rng, &mut tr,
    )
    .unwrap();
    let proof = tr.finalize();
    let prove_s = t.elapsed().as_secs_f64();

    let mut best = f64::INFINITY;
    for _ in 0..20 {
        let t = Instant::now();
        let mut tr = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
        verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
            &kzg, &vk, SingleStrategy::new(&kzg), &[&[&inst0, &inst1]], &mut tr,
        )
        .unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "{{\"system\":\"halo2-kzg-custom\",\"form\":\"{form}\",\"d\":{d},\"r\":{r},\"f\":{f},\
\"k\":{k},\"lookup_bits\":{lookup_bits},\"advice_cols\":{},\"lookup_cols\":{},\"pos_cols\":6,\"tap_advice\":{advice},\"lookup_cells\":{lookup},\
\"pos_rows\":{pos_rows},\"pos_advice\":{},\"fixed\":{fixed},\
\"srs_s\":{srs_s:.2},\"keygen_s\":{keygen_s:.2},\"prove_s\":{prove_s:.3},\
\"verify_ms\":{best:.3},\"proof_bytes\":{},\"srs_bytes\":{srs_bytes},\"pk_bytes\":{pk_bytes},\
\"vk_bytes\":{vk_bytes},\"instances\":{}}}",
        cfg2.num_advice_per_phase.iter().sum::<usize>(),
        cfg2.num_lookup_advice_per_phase.iter().sum::<usize>(),
        poseidon::advice_cells(d + 1),
        proof.len(),
        inst0.len() + inst1.len()
    );

    let dump: Vec<String> = out.iter().map(|x| format!("{x:?}")).collect();
    fs::write(
        format!("halo2_output_{form}.json"),
        serde_json::to_string(&dump).unwrap(),
    )
    .unwrap();
    fs::write("halo2_commit.json", format!("\"{commit:?}\"")).unwrap();
}
