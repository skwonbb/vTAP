//!     cargo run --release --bin poseidon_chip -- --n 769 --k 15

use std::{env, time::Instant};

use halo2_base::{
    gates::{circuit::builder::BaseCircuitBuilder, GateChip},
    halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        halo2curves::bn256::{Bn256, Fr},
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
    },
    poseidon::hasher::PoseidonHasher,
};
use rand::rngs::OsRng;

use tap_halo2::poseidon::{self, advice_cells, blocks, rows, spec, RATE, T};

#[derive(Clone)]
struct PoseidonCircuit {
    inputs: Vec<Fr>,
}

impl Circuit<Fr> for PoseidonCircuit {
    type Config = (poseidon::Config, Column<Instance>);
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        Self { inputs: vec![Fr::zero(); self.inputs.len()] }
    }
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let cfg = poseidon::Config::configure(meta);
        let inst = meta.instance_column();
        meta.enable_equality(inst);
        (cfg, inst)
    }
    fn synthesize(&self, (cfg, inst): Self::Config, mut ly: impl Layouter<Fr>) -> Result<(), Error> {
        let sp = spec();
        let a = ly.assign_region(
            || "poseidon",
            |mut r| poseidon::assign(&cfg, &mut r, &sp, &self.inputs, 0),
        )?;
        ly.constrain_instance(a.digest, inst, 0);
        Ok(())
    }
}

fn generic_reference(inputs: &[Fr]) -> (Fr, usize) {
    let mut b = BaseCircuitBuilder::<Fr>::new(false).use_k(20).use_lookup_bits(16);
    let gate = GateChip::<Fr>::default();
    let ctx = b.main(0);
    let cells: Vec<_> = inputs.iter().map(|x| ctx.load_witness(*x)).collect();
    let mut h = PoseidonHasher::<Fr, T, RATE>::new(spec());
    h.initialize_consts(ctx, &gate);
    let d = h.hash_fix_len_array(ctx, &gate, &cells);
    let v = *d.value();
    let n: usize = b.statistics().gate.total_advice_per_phase.iter().sum();
    (v, n)
}

fn main() {
    std::env::set_var("MAX_DEGREE", "6");

    let a: Vec<String> = env::args().collect();
    let get = |k: &str, d: &str| {
        a.windows(2)
            .find(|w| w[0] == format!("--{k}"))
            .map(|w| w[1].clone())
            .unwrap_or_else(|| d.into())
    };
    let n: usize = get("n", "769").parse().unwrap();
    let k: u32 = get("k", "15").parse().unwrap();

    let sp = spec();
    let inputs: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64 + 1)).collect();

    let want = poseidon::native(&sp, &inputs);
    let (gen_digest, gen_cells) = generic_reference(&inputs);
    println!(
        "{{\"digest_custom\":\"{want:?}\",\"digest_generic\":\"{gen_digest:?}\",\"match\":{}}}",
        want == gen_digest
    );
    assert_eq!(want, gen_digest, "custom-gate trace differs from the generic implementation");

    println!(
        "{{\"n\":{n},\"perms\":{},\"rows\":{},\"advice_cols\":{},\"advice_cells_custom\":{},\
\"advice_cells_generic\":{gen_cells},\"ratio\":{:.1}}}",
        blocks(n),
        rows(n),
        T + RATE + 1,
        advice_cells(n),
        gen_cells as f64 / advice_cells(n) as f64
    );

    let mut circuit = PoseidonCircuit { inputs };
    let t0 = Instant::now();
    match MockProver::run(k, &circuit, vec![vec![want]]).unwrap().verify() {
        Ok(()) => println!("{{\"mock\":\"ok\",\"secs\":{:.2}}}", t0.elapsed().as_secs_f64()),
        Err(e) => {
            println!("{{\"mock\":\"FAIL\"}}");
            for x in e.iter().take(5) {
                println!("  {x:?}");
            }
            return;
        }
    }

    let t0 = Instant::now();
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    let srs_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk, &circuit).unwrap();
    let keygen_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut tr = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<Bn256>, _, _, _, _>(
        &params,
        &pk,
        std::slice::from_mut(&mut circuit),
        &[&[&[want]]],
        OsRng,
        &mut tr,
    )
    .unwrap();
    let proof = tr.finalize();
    let prove_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut tr = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let ok = verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<Bn256>, _, _, _>(
        &params,
        pk.get_vk(),
        SingleStrategy::new(&params),
        &[&[&[want]]],
        &mut tr,
    )
    .is_ok();
    let verify_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!(
        "{{\"k\":{k},\"srs_s\":{srs_s:.2},\"keygen_s\":{keygen_s:.2},\"prove_s\":{prove_s:.2},\
\"verify_ms\":{verify_ms:.2},\"proof_bytes\":{},\"verified\":{ok}}}",
        proof.len()
    );
}
