use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use voronota_ltr::{Ball, PeriodicBox, compute_tessellation};

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

fn write_contacts(out: &mut impl Write, contacts: &[voronota_ltr::Contact]) -> io::Result<()> {
    for c in contacts {
        writeln!(
            out,
            "{} {} {:.6} {:.6}",
            c.id_a, c.id_b, c.area, c.arc_length
        )?;
    }
    Ok(())
}

fn write_cells(out: &mut impl Write, cells: &[voronota_ltr::Cell]) -> io::Result<()> {
    for c in cells {
        writeln!(out, "{} {:.6} {:.6}", c.index, c.sas_area, c.volume)?;
    }
    Ok(())
}

fn write_summary(
    out: &mut impl Write,
    result: &voronota_ltr::TessellationResult,
) -> io::Result<()> {
    let total_contact_area: f64 = result.contacts.iter().map(|c| c.area).sum();
    let total_sas_area: f64 = result.cells.iter().map(|c| c.sas_area).sum();
    let total_volume: f64 = result.cells.iter().map(|c| c.volume).sum();

    writeln!(out, "contacts: {}", result.contacts.len())?;
    writeln!(out, "cells: {}", result.cells.len())?;
    writeln!(out, "total_contact_area: {total_contact_area:.4}")?;
    writeln!(out, "total_sas_area: {total_sas_area:.4}")?;
    writeln!(out, "total_volume: {total_volume:.4}")?;
    Ok(())
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
        write_contacts(&mut stdout, &result.contacts)?;
    }

    if cli.print_cells {
        write_cells(&mut stdout, &result.cells)?;
    }

    if !(cli.print_contacts || cli.print_cells) {
        write_summary(&mut stdout, &result)?;
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
