use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rust_voronota::{Ball, PeriodicBox, compute_tessellation};
use std::fs;

/// Parse xyzr file - last 4 numeric columns are x, y, z, r
fn parse_xyzr(content: &str) -> Vec<Ball> {
    content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return None;
            }
            // Take last 4 columns as x y z r
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
    let path = format!("benches/data/{}.xyzr", name);
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {path}: {e}"));
    parse_xyzr(&content)
}

fn bench_tessellation(c: &mut Criterion) {
    let datasets = [
        ("balls_cs_1x1", 2.0), // 100 balls, probe 2.0
        ("balls_2zsk", 1.4),   // 3545 balls, probe 1.4
        ("balls_3dlb", 1.4),   // 9745 balls, probe 1.4
    ];

    let mut group = c.benchmark_group("tessellation");

    for (name, probe) in datasets {
        let balls = load_balls(name);
        let n = balls.len();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("compute", name), &balls, |b, balls| {
            b.iter(|| compute_tessellation(black_box(balls), black_box(probe), None, None));
        });
    }

    group.finish();
}

fn bench_tessellation_periodic(c: &mut Criterion) {
    // balls_cs_1x1 with periodic box (0,0,0) to (200,250,300)
    let balls = load_balls("balls_cs_1x1");
    let pbox = PeriodicBox::from_corners((0.0, 0.0, 0.0), (200.0, 250.0, 300.0));
    let probe = 2.0;

    let mut group = c.benchmark_group("tessellation_periodic");
    group.throughput(Throughput::Elements(balls.len() as u64));

    group.bench_function("compute/balls_cs_1x1", |b| {
        b.iter(|| {
            compute_tessellation(
                black_box(&balls),
                black_box(probe),
                Some(black_box(&pbox)),
                None,
            )
        });
    });

    group.finish();
}

fn bench_tessellation_with_groups(c: &mut Criterion) {
    // Use balls_2zsk (3545 balls) - assign groups in blocks of 10
    let balls = load_balls("balls_2zsk");
    let probe = 1.4;
    let n = balls.len();

    // Create groups: every 10 balls belong to the same group
    let groups: Vec<i32> = (0..n).map(|i| (i / 10) as i32).collect();

    let mut group = c.benchmark_group("tessellation_groups");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("no_groups/balls_2zsk", |b| {
        b.iter(|| compute_tessellation(black_box(&balls), black_box(probe), None, None));
    });

    group.bench_function("with_groups/balls_2zsk", |b| {
        b.iter(|| {
            compute_tessellation(black_box(&balls), black_box(probe), None, Some(black_box(&groups)))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tessellation,
    bench_tessellation_periodic,
    bench_tessellation_with_groups
);
criterion_main!(benches);
