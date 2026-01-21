use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use voronotalt::{Ball, PeriodicBox, compute_tessellation};

#[derive(Parser)]
#[command(name = "voronotalt")]
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

    /// Periodic box corners: x1 y1 z1 x2 y2 z2
    #[arg(long, num_args = 6, value_names = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"])]
    periodic_box_corners: Option<Vec<f64>>,

    /// Print contacts table to stdout
    #[arg(long)]
    print_contacts: bool,

    /// Print cells table to stdout
    #[arg(long)]
    print_cells: bool,

    /// Suppress log messages
    #[arg(short, long)]
    quiet: bool,
}

fn parse_balls(reader: impl BufRead) -> Vec<Ball> {
    reader
        .lines()
        .filter_map(|line| {
            let line = line.ok()?;
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

    let mut stdout = io::stdout().lock();

    if cli.print_contacts {
        for c in &result.contacts {
            writeln!(
                stdout,
                "{} {} {:.6} {:.6}",
                c.id_a, c.id_b, c.area, c.arc_length
            )?;
        }
    }

    if cli.print_cells {
        for c in &result.cells {
            writeln!(stdout, "{} {:.6} {:.6}", c.index, c.sas_area, c.volume)?;
        }
    }

    // Default: print summary if nothing else requested
    if !cli.print_contacts && !cli.print_cells {
        let total_contact_area: f64 = result.contacts.iter().map(|c| c.area).sum();
        let total_sas_area: f64 = result.cells.iter().map(|c| c.sas_area).sum();
        let total_volume: f64 = result.cells.iter().map(|c| c.volume).sum();

        println!("contacts: {}", result.contacts.len());
        println!("cells: {}", result.cells.len());
        println!("total_contact_area: {:.4}", total_contact_area);
        println!("total_sas_area: {:.4}", total_sas_area);
        println!("total_volume: {:.4}", total_volume);
    }

    Ok(())
}
