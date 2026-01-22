//! CLI for computing radical Voronoi tessellation of molecular structures.

use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::time::Instant;

use clap::{ArgAction, Parser};
use log::{debug, info};
use serde::Serialize;
use voronota_ltr::input::{
    InputFormat, ParseOptions, RadiiLookup, build_chain_grouping, build_residue_grouping,
    parse_file_with_records, parse_reader,
};
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
    Supports PDB, mmCIF, and XYZR input formats (auto-detected from extension or content)."
)]
#[allow(clippy::struct_excessive_bools)]
struct Cli {
    /// Rolling probe radius
    #[arg(long, default_value_t = 1.4)]
    probe: f64,

    /// Input file (PDB, mmCIF, or XYZR format). Reads from stdin if not specified
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output JSON file. Writes to stdout if not specified
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Include hydrogen atoms
    #[arg(long)]
    include_hydrogens: bool,

    /// Exclude heteroatoms (HETATM records)
    #[arg(long)]
    exclude_heteroatoms: bool,

    /// Custom radii file (format: residue atom radius per line)
    #[arg(long)]
    radii_file: Option<PathBuf>,

    /// Periodic box corners: x1 y1 z1 x2 y2 z2
    #[arg(long, num_args = 6, value_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"])]
    periodic_box_corners: Option<Vec<f64>>,

    /// Only compute inter-chain contacts (exclude intra-chain)
    #[arg(long)]
    inter_chain_only: bool,

    /// Only compute inter-residue contacts (exclude intra-residue)
    #[arg(long)]
    inter_residue_only: bool,

    /// Increase verbosity (-v: debug, -vv: trace)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    /// Reduce verbosity to warnings only
    #[arg(short, long)]
    quiet: bool,

    /// Maximum number of threads to use (default: all available)
    #[arg(long)]
    processors: Option<usize>,

    /// Measure and output running time to stderr
    #[arg(long)]
    measure_running_time: bool,
}

#[allow(clippy::too_many_lines)]
fn main() -> io::Result<()> {
    let cli = Cli::parse();

    // Initialize logging based on verbosity
    let log_level = if cli.quiet {
        "warn"
    } else {
        match cli.verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Configure thread pool if --processors specified
    if let Some(num_threads) = cli.processors {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .map_err(io::Error::other)?;
        info!("Using {num_threads} threads");
    }

    // Setup radii lookup
    let mut radii = RadiiLookup::new();
    if let Some(ref radii_path) = cli.radii_file {
        let content = std::fs::read_to_string(radii_path)?;
        radii
            .load_from_text(&content)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        info!("Loaded custom radii from {}", radii_path.display());
    }

    let options = ParseOptions {
        exclude_heteroatoms: cli.exclude_heteroatoms,
        include_hydrogens: cli.include_hydrogens,
        as_assembly: false,
    };

    // Read input balls and records
    let (balls, grouping): (Vec<Ball>, Option<Vec<i32>>) = if let Some(path) = &cli.input {
        let parsed = parse_file_with_records(path, &options, &radii)?;

        // Build grouping if inter-chain or inter-residue filtering requested
        let grouping = if cli.inter_chain_only {
            if parsed.records.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "--inter-chain-only requires PDB or mmCIF input (not XYZR)",
                ));
            }
            Some(build_chain_grouping(&parsed.records))
        } else if cli.inter_residue_only {
            if parsed.records.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "--inter-residue-only requires PDB or mmCIF input (not XYZR)",
                ));
            }
            Some(build_residue_grouping(&parsed.records))
        } else {
            None
        };

        (parsed.balls, grouping)
    } else {
        // For stdin, try to detect format
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;

        let trimmed = first_line.trim();

        // Detect format from first line
        let format = if trimmed.starts_with("data_")
            || trimmed.starts_with('_')
            || trimmed.starts_with("loop_")
        {
            InputFormat::Mmcif
        } else if trimmed.starts_with("ATOM")
            || trimmed.starts_with("HETATM")
            || trimmed.starts_with("HEADER")
            || trimmed.starts_with("REMARK")
        {
            InputFormat::Pdb
        } else {
            InputFormat::Xyzr
        };

        debug!("Detected stdin format: {format:?}");

        // Note: stdin doesn't support grouping yet (would need parse_reader_with_records)
        if cli.inter_chain_only || cli.inter_residue_only {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--inter-chain-only and --inter-residue-only require file input (-i)",
            ));
        }

        (
            parse_reader(reader, format, Some(&first_line), &options, &radii)?,
            None,
        )
    };

    info!("Read {} balls", balls.len());

    // Parse periodic box if specified
    let periodic_box = cli.periodic_box_corners.as_ref().map(|coords| {
        PeriodicBox::from_corners(
            (coords[0], coords[1], coords[2]),
            (coords[3], coords[4], coords[5]),
        )
    });

    // Compute tessellation
    let start = Instant::now();
    let result = compute_tessellation(
        &balls,
        cli.probe,
        periodic_box.as_ref(),
        grouping.as_deref(),
    );
    let elapsed = start.elapsed();

    info!(
        "Computed {} contacts, {} cells",
        result.contacts.len(),
        result.cells.len()
    );

    if cli.measure_running_time {
        info!("Tessellation time: {} ms", elapsed.as_millis());
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
        let balls: Vec<Ball> = input
            .lines()
            .filter_map(|line| {
                let mut parts = line.split_whitespace().rev();
                let r: f64 = parts.next()?.parse().ok()?;
                let z: f64 = parts.next()?.parse().ok()?;
                let y: f64 = parts.next()?.parse().ok()?;
                let x: f64 = parts.next()?.parse().ok()?;
                Some(Ball::new(x, y, z, r))
            })
            .collect();
        assert_eq!(balls.len(), 2);
        assert_eq!(balls[0], Ball::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(balls[1], Ball::new(0.0, 0.0, 0.0, 1.0));
    }
}
