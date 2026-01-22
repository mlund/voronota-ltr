use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;
use voronota_ltr::{Ball, PeriodicBox, Results, TessellationResult, compute_tessellation};

/// Extended JSON output including per-ball SASA/volumes and totals
#[derive(Serialize)]
struct JsonOutput {
    #[serde(flatten)]
    result: TessellationResult,
    sas_areas: Vec<f64>,
    volumes: Vec<f64>,
    total_sas_area: f64,
    total_volume: f64,
    total_contact_area: f64,
}

#[derive(Parser)]
#[command(name = "voronota_ltr")]
#[command(about = "Compute radical Voronoi tessellation of atomic balls")]
#[command(
    long_about = "Constructs a radical Voronoi tessellation of atomic balls \
    constrained inside a solvent-accessible surface defined by a rolling probe. \
    Computes inter-atom contact areas, solvent accessible surface areas, and volumes.\n\n\
    Input format: .xyzr file with whitespace-separated values, last 4 columns are x y z radius."
)]
struct Cli {
    /// Rolling probe radius
    #[arg(long, default_value_t = 1.4)]
    probe: f64,

    /// Input .xyzr file (x y z radius per line). Reads from stdin if not specified
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output JSON file. Writes to stdout if not specified
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Periodic box corners: x1 y1 z1 x2 y2 z2
    #[arg(long, num_args = 6, value_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"])]
    periodic_box_corners: Option<Vec<f64>>,

    /// Suppress log messages
    #[arg(short, long)]
    quiet: bool,
}

fn parse_last_four_columns(line: &str) -> Option<Ball> {
    let mut parts = line.split_whitespace().rev();
    let r: f64 = parts.next()?.parse().ok()?;
    let z: f64 = parts.next()?.parse().ok()?;
    let y: f64 = parts.next()?.parse().ok()?;
    let x: f64 = parts.next()?.parse().ok()?;
    Some(Ball::new(x, y, z, r))
}

#[allow(clippy::many_single_char_names)]
fn parse_balls(reader: impl BufRead) -> Vec<Ball> {
    reader
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| parse_last_four_columns(&line))
        .collect()
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    // Read input balls
    let balls: Vec<Ball> = match &cli.input {
        Some(path) => {
            let file = File::open(path)?;
            parse_balls(BufReader::new(file))
        }
        None => parse_balls(io::stdin().lock()),
    };

    if !cli.quiet {
        eprintln!("Read {} balls", balls.len());
    }

    // Parse periodic box if specified
    let periodic_box = cli.periodic_box_corners.as_ref().map(|coords| {
        PeriodicBox::from_corners(
            (coords[0], coords[1], coords[2]),
            (coords[3], coords[4], coords[5]),
        )
    });

    // Compute tessellation
    let result = compute_tessellation(&balls, cli.probe, periodic_box.as_ref(), None);

    if !cli.quiet {
        eprintln!(
            "Computed {} contacts, {} cells",
            result.contacts.len(),
            result.cells.len()
        );
    }

    // Build extended JSON output with per-ball SASA/volumes and totals
    let output = JsonOutput {
        sas_areas: result.sas_areas(),
        volumes: result.volumes(),
        total_sas_area: result.total_sas_area(),
        total_volume: result.total_volume(),
        total_contact_area: result.total_contact_area(),
        result,
    };

    // Write JSON output
    if let Some(path) = &cli.output {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &output)?;
    } else {
        let stdout = io::stdout().lock();
        serde_json::to_writer_pretty(stdout, &output)?;
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_balls_reads_last_four_columns() {
        let input = "a b c 1.0 2.0 3.0 4.0\n0 0 0 1\n";
        let balls = parse_balls(input.as_bytes());
        assert_eq!(balls.len(), 2);
        assert_eq!(balls[0], Ball::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(balls[1], Ball::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn parse_balls_skips_short_lines() {
        let input = "1 2 3\n1 2 3 4\n";
        let balls = parse_balls(input.as_bytes());
        assert_eq!(balls, vec![Ball::new(1.0, 2.0, 3.0, 4.0)]);
    }
}
