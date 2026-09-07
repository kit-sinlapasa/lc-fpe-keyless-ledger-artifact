extern crate bulletproofs;
extern crate curve25519_dalek;
extern crate merlin;
extern crate rand;
extern crate sha2;

use std::env;
use std::process;
use std::time::Instant;

use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek::ristretto::CompressedRistretto;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use rand::thread_rng;
use sha2::{Digest, Sha256};

const REAL_ROWS: usize = 20_000;
const OMEGA: u64 = 1_u64 << 40;

struct BatchResult {
    commitments: Vec<CompressedRistretto>,
    blindings: Vec<Scalar>,
}


fn expected_size(bits: usize, values: usize) -> usize {
    let dimension = bits * values;
    assert!(dimension.is_power_of_two());
    (9 + 2 * dimension.trailing_zeros() as usize) * 32
}


fn signed_amount(index: usize) -> i64 {
    let pair = index / 2;
    let magnitude = ((pair as u64 * 7_919 + 17) % 1_000_000 + 1) as i64;
    if index % 2 == 0 { magnitude } else { -magnitude }
}

fn deterministic_values(kind: &str, values_count: usize) -> Vec<u64> {
    if kind == "amount" {
        let real_rows = if values_count < REAL_ROWS { values_count } else { REAL_ROWS };
        return (0..values_count).map(|i| {
            if i < real_rows { (OMEGA as i64 + signed_amount(i)) as u64 } else { 0 }
        }).collect();
    }
    let row_capacity = values_count / 2;
    let real_rows = if row_capacity < REAL_ROWS { row_capacity } else { REAL_ROWS };
    (0..values_count).map(|i| {
        let row = i % row_capacity;
        if row >= real_rows {
            0
        } else {
            let value = signed_amount(row);
            if i < row_capacity {
                if value > 0 { value as u64 } else { 0 }
            } else if value < 0 {
                (-value) as u64
            } else {
                0
            }
        }
    }).collect()
}


fn checked_from_bytes(bytes: &[u8], bits: usize, values: usize) -> Result<RangeProof, String> {
    let want = expected_size(bits, values);
    if bytes.len() != want {
        return Err(format!("length guard: got {}, expected {}", bytes.len(), want));
    }
    RangeProof::from_bytes(bytes).map_err(|_| "non-canonical proof encoding".to_string())
}


fn run_batch(bits: usize, values_count: usize, kind: &str) -> Result<BatchResult, String> {
    if !values_count.is_power_of_two() || ![8, 16, 32, 64].contains(&bits) {
        return Err("bits and padded value count must be supported powers of two".to_string());
    }
    let setup_start = Instant::now();
    let bp_gens = BulletproofGens::new(bits, values_count);
    let pc_gens = PedersenGens::default();
    let setup = setup_start.elapsed();

    let values = deterministic_values(kind, values_count);
    let mut rng = thread_rng();
    let blindings: Vec<Scalar> = (0..values_count)
        .map(|_| Scalar::random(&mut rng))
        .collect();

    let prove_start = Instant::now();
    let mut prover_transcript = Transcript::new(b"PaperB/audited-range/v1");
    let (proof, commitments) = RangeProof::prove_multiple(
        &bp_gens,
        &pc_gens,
        &mut prover_transcript,
        &values,
        &blindings,
        bits,
    ).map_err(|e| format!("prove failed: {:?}", e))?;
    let prove = prove_start.elapsed();

    let bytes = proof.to_bytes();
    let parsed = checked_from_bytes(&bytes, bits, values_count)?;
    let verify_start = Instant::now();
    let mut verifier_transcript = Transcript::new(b"PaperB/audited-range/v1");
    parsed.verify_multiple(
        &bp_gens,
        &pc_gens,
        &mut verifier_transcript,
        &commitments,
        bits,
    ).map_err(|e| format!("verify failed: {:?}", e))?;
    let verify = verify_start.elapsed();

    let mut tampered = bytes.clone();
    let tamper_index = tampered.len() / 2;
    tampered[tamper_index] ^= 1;
    let tamper_rejected = match checked_from_bytes(&tampered, bits, values_count) {
        Err(_) => true,
        Ok(bad) => {
            let mut transcript = Transcript::new(b"PaperB/audited-range/v1");
            bad.verify_multiple(&bp_gens, &pc_gens, &mut transcript, &commitments, bits).is_err()
        }
    };
    let short_rejected = checked_from_bytes(&bytes[..bytes.len() - 1], bits, values_count).is_err();

    println!("batch={} bits={} padded_values={} bit_values={}",
             kind, bits, values_count, bits * values_count);
    println!("setup_s={:.3}", setup.as_secs() as f64 + setup.subsec_nanos() as f64 / 1e9);
    println!("prove_s={:.3}", prove.as_secs() as f64 + prove.subsec_nanos() as f64 / 1e9);
    println!("verify_s={:.3}", verify.as_secs() as f64 + verify.subsec_nanos() as f64 / 1e9);
    println!("proof_bytes={}", bytes.len());
    println!("expected_proof_bytes={}", expected_size(bits, values_count));
    println!("honest_verify=true");
    println!("tampered_proof_rejected={}", tamper_rejected);
    println!("wrong_length_rejected={}", short_rejected);
    if !tamper_rejected || !short_rejected {
        return Err("negative control accepted".to_string());
    }
    Ok(BatchResult { commitments, blindings })
}


fn record_digest(amount: &BatchResult, side: &BatchResult,
                 row_capacity: usize, real_rows: usize, tamper_first_side: bool) -> Vec<u8> {
    let pc_gens = PedersenGens::default();
    let mut digest = Sha256::new();
    digest.input(b"PaperB/full-ledger-anchor/v1");
    digest.input(&(real_rows as u64).to_be_bytes());
    for i in 0..real_rows {
        let c = amount.commitments[i];
        let mut d = side.commitments[i].decompress().unwrap();
        if tamper_first_side && i == 0 {
            d = d - pc_gens.B;
        }
        let d_bytes = d.compress();
        let k = side.commitments[row_capacity + i];
        let lambda = side.blindings[i] - side.blindings[row_capacity + i]
                     - amount.blindings[i];
        digest.input(&(i as u64).to_be_bytes());
        digest.input(c.as_bytes());
        digest.input(d_bytes.as_bytes());
        digest.input(k.as_bytes());
        digest.input(lambda.as_bytes());
    }
    digest.result().to_vec()
}


fn verify_same_ledger(amount: &BatchResult, side: &BatchResult,
                      row_capacity: usize, real_rows: usize) -> Result<(), String> {
    let pc_gens = PedersenGens::default();
    let mut sum_c = None;
    let mut sum_d = None;
    let mut sum_k = None;
    let mut sum_rho = Scalar::zero();
    let mut sum_eta = Scalar::zero();
    let mut sum_zeta = Scalar::zero();
    let mut turnover = 0_u64;
    let mut all_links = true;
    let mut understated_rejected = false;

    for i in 0..real_rows {
        let c = amount.commitments[i].decompress()
            .ok_or_else(|| "amount commitment did not decompress".to_string())?;
        let d = side.commitments[i].decompress()
            .ok_or_else(|| "debit commitment did not decompress".to_string())?;
        let k = side.commitments[row_capacity + i].decompress()
            .ok_or_else(|| "credit commitment did not decompress".to_string())?;
        let lambda = side.blindings[i] - side.blindings[row_capacity + i]
                     - amount.blindings[i];
        let link_lhs = d - k - c + Scalar::from(OMEGA) * pc_gens.B;
        let link_rhs = lambda * pc_gens.B_blinding;
        all_links &= link_lhs == link_rhs;
        if i == 0 {
            understated_rejected = d - pc_gens.B - k - c
                + Scalar::from(OMEGA) * pc_gens.B != link_rhs;
        }
        sum_c = Some(match sum_c { None => c, Some(v) => v + c });
        sum_d = Some(match sum_d { None => d, Some(v) => v + d });
        sum_k = Some(match sum_k { None => k, Some(v) => v + k });
        sum_rho += amount.blindings[i];
        sum_eta += side.blindings[i];
        sum_zeta += side.blindings[row_capacity + i];
        let value = signed_amount(i);
        if value > 0 { turnover += value as u64; }
    }
    let balance_c = sum_c.unwrap()
        == Scalar::from((real_rows as u64) * OMEGA) * pc_gens.B
           + sum_rho * pc_gens.B_blinding;
    let balance_d = sum_d.unwrap()
        == Scalar::from(turnover) * pc_gens.B + sum_eta * pc_gens.B_blinding;
    let balance_k = sum_k.unwrap()
        == Scalar::from(turnover) * pc_gens.B + sum_zeta * pc_gens.B_blinding;
    let anchor = record_digest(amount, side, row_capacity, real_rows, false);
    let anchor_recomputed = anchor == record_digest(amount, side, row_capacity, real_rows, false);
    let anchor_tamper_rejected = anchor != record_digest(amount, side, row_capacity, real_rows, true);

    println!("ledger_rows={}", real_rows);
    println!("same_record_set_for_both_batches=true");
    println!("side_links_verified={}", if all_links { real_rows } else { 0 });
    println!("amount_balance_verified={}", balance_c);
    println!("debit_credit_totals_verified={}", balance_d && balance_k);
    println!("understated_side_rejected={}", understated_rejected);
    println!("anchor_recomputed={}", anchor_recomputed);
    println!("anchored_record_tamper_rejected={}", anchor_tamper_rejected);
    if !(all_links && balance_c && balance_d && balance_k && understated_rejected
         && anchor_recomputed && anchor_tamper_rejected) {
        return Err("full-ledger consistency check failed".to_string());
    }
    Ok(())
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let result = if args.iter().any(|x| x == "--full") {
        run_batch(64, 32_768, "amount").and_then(|amount| {
            run_batch(32, 65_536, "side").and_then(|side| {
                verify_same_ledger(&amount, &side, 32_768, REAL_ROWS)
            })
        })
    } else if args.iter().any(|x| x == "--medium") {
        run_batch(64, 4_096, "amount").and_then(|amount| {
            run_batch(32, 8_192, "side").and_then(|side| {
                verify_same_ledger(&amount, &side, 4_096, 4_096)
            })
        })
    } else if args.iter().any(|x| x == "--quick") {
        run_batch(32, 16, "side").map(|_| ())
    } else {
        Err("use --quick, --medium, or --full".to_string())
    };
    if let Err(error) = result {
        eprintln!("error: {}", error);
        process::exit(1);
    }
}
