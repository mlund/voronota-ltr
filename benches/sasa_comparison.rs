//! Benchmark comparing SASA computation between voronota-ltr and rust-sasa.
//!
//! - voronota-ltr: Radical Voronoi tessellation (exact geometric solution)
//! - rust-sasa: Shrake-Rupley algorithm (numerical sphere point sampling)

#![allow(clippy::cast_possible_truncation)] // f64 to f32 casts are intentional

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rust_sasa::{Atom, calculate_sasa_internal};
use std::fs;
use std::hint::black_box;
use voronota_ltr::{Ball, Results, compute_tessellation};

fn parse_xyzr(content: &str) -> Vec<Ball> {
    content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return None;
            }
            let n = parts.len();
            let x: f64 = parts[n - 4].parse().ok()?;
            let y: f64 = parts[n - 3].parse().ok()?;
            let z: f64 = parts[n - 2].parse().ok()?;
            let r: f64 = parts[n - 1].parse().ok()?;
            Some(Ball::new(x, y, z, r))
        })
        .collect()
}

fn load_balls(name: &str) -> Vec<Ball> {
    let path = format!("benches/data/{name}.xyzr");
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    parse_xyzr(&content)
}

fn balls_to_atoms(balls: &[Ball]) -> Vec<Atom> {
    balls
        .iter()
        .enumerate()
        .map(|(i, b)| Atom {
            position: [b.x as f32, b.y as f32, b.z as f32],
            radius: b.r as f32,
            id: i,
            parent_id: None,
        })
        .collect()
}

/// Macro to benchmark both libraries with given thread configuration
macro_rules! bench_both {
    ($group:expr, $balls:expr, $atoms:expr, $probe:expr, $threads:expr) => {
        if $threads == 1 {
            // Single-threaded voronota via custom rayon pool
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            $group.bench_function("voronota", |b| {
                b.iter(|| {
                    pool.install(|| {
                        let result =
                            compute_tessellation(black_box($balls), black_box($probe), None, None);
                        black_box(result.total_sas_area())
                    })
                });
            });
        } else {
            $group.bench_function("voronota", |b| {
                b.iter(|| {
                    let result =
                        compute_tessellation(black_box($balls), black_box($probe), None, None);
                    black_box(result.total_sas_area())
                });
            });
        }

        $group.bench_function("rust_sasa", |b| {
            b.iter(|| {
                let sasa = calculate_sasa_internal(
                    black_box($atoms),
                    black_box($probe as f32),
                    100,
                    $threads,
                );
                black_box(sasa.iter().sum::<f32>())
            });
        });
    };
}

fn bench_sasa_single_thread(c: &mut Criterion) {
    let datasets = [
        ("balls_cs_1x1", 2.0), // 100 balls
        ("balls_2zsk", 1.4),   // 3545 balls
        ("balls_3dlb", 1.4),   // 9745 balls
    ];

    for (name, probe) in datasets {
        let balls = load_balls(name);
        let atoms = balls_to_atoms(&balls);

        let mut group = c.benchmark_group(format!("sasa_1_thread/{name}"));
        group.throughput(Throughput::Elements(balls.len() as u64));

        bench_both!(group, &balls, &atoms, probe, 1);

        group.finish();
    }
}

fn bench_sasa_multi_thread(c: &mut Criterion) {
    let datasets = [
        ("balls_cs_1x1", 2.0), // 100 balls
        ("balls_2zsk", 1.4),   // 3545 balls
        ("balls_3dlb", 1.4),   // 9745 balls
    ];

    for (name, probe) in datasets {
        let balls = load_balls(name);
        let atoms = balls_to_atoms(&balls);

        let mut group = c.benchmark_group(format!("sasa_multi_thread/{name}"));
        group.throughput(Throughput::Elements(balls.len() as u64));

        bench_both!(group, &balls, &atoms, probe, -1);

        group.finish();
    }
}

criterion_group!(benches, bench_sasa_single_thread, bench_sasa_multi_thread);
criterion_main!(benches);
