// Copyright (c) 2026 Kliment Olechnovic and Mikael Lund
// Part of the voronota-ltr project, licensed under the MIT License.
// SPDX-License-Identifier: MIT

//! Atomic radii lookup with pattern matching support.
//!
//! Radii are assigned based on residue name + atom name patterns using wildcard matching.
//! Lookup priority: exact match → prefix match → wildcard residue → element default → 1.8

use std::collections::HashMap;

const DEFAULT_RADIUS: f64 = 1.8;

/// Radii lookup table with pattern matching.
pub struct RadiiLookup {
    rules: HashMap<(String, String), f64>,
}

impl Default for RadiiLookup {
    fn default() -> Self {
        Self::new()
    }
}

impl RadiiLookup {
    /// Create a new radii lookup with default rules from C++ voronota-lt.
    #[must_use]
    pub fn new() -> Self {
        let mut rules = HashMap::new();

        // Element-based defaults (wildcard residue)
        for (res, atom, radius) in DEFAULT_RULES {
            rules.insert(((*res).to_string(), (*atom).to_string()), *radius);
        }

        Self { rules }
    }

    /// Create an empty radii lookup for custom rules.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Load custom radii rules from text (one rule per line: resName atomName radius).
    /// Lines starting with '#' are comments.
    ///
    /// # Errors
    /// Returns error if a line has invalid format or a negative radius.
    pub fn load_from_text(&mut self, text: &str) -> Result<(), String> {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(format!("Invalid radii rule: {line}"));
            }

            let radius: f64 = parts[2]
                .parse()
                .map_err(|_| format!("Invalid radius value: {}", parts[2]))?;

            if radius < 0.0 {
                return Err(format!("Negative radius: {radius}"));
            }

            self.rules
                .insert((parts[0].to_string(), parts[1].to_string()), radius);
        }
        Ok(())
    }

    /// Look up radius for a given residue name and atom name.
    /// Uses pattern matching with wildcard support.
    #[must_use]
    pub fn get_radius(&self, res_name: &str, atom_name: &str) -> f64 {
        if atom_name.is_empty() || atom_name == "*" {
            return DEFAULT_RADIUS;
        }

        // Try exact match then prefix matches, first with specific residue then wildcard
        for res in [res_name, "*"] {
            // Exact match
            if let Some(&r) = self.rules.get(&(res.to_string(), atom_name.to_string())) {
                return r;
            }

            // Prefix matches (e.g., "CA" matches "C*")
            for prefix_len in (1..=atom_name.len()).rev() {
                let pattern = format!("{}*", &atom_name[..prefix_len]);
                if let Some(&r) = self.rules.get(&(res.to_string(), pattern)) {
                    return r;
                }
            }
        }

        // Wildcard atom match for specific residue then wildcard
        for res in [res_name, "*"] {
            if let Some(&r) = self.rules.get(&(res.to_string(), "*".to_string())) {
                return r;
            }
        }

        DEFAULT_RADIUS
    }
}

// All default radii rules ported from C++ voronota-lt
const DEFAULT_RULES: &[(&str, &str, f64)] = &[
    // Element-based defaults
    ("*", "C*", 1.80),
    ("*", "N*", 1.60),
    ("*", "O*", 1.50),
    ("*", "P*", 1.90),
    ("*", "S*", 1.90),
    ("*", "H*", 1.30),
    // Backbone atoms
    ("*", "C", 1.75),
    ("*", "CA", 1.90),
    ("*", "N", 1.70),
    ("*", "O", 1.49),
    // ALA
    ("ALA", "CB", 1.92),
    // ARG
    ("ARG", "CB", 1.91),
    ("ARG", "CD*", 1.88),
    ("ARG", "CG*", 1.92),
    ("ARG", "CZ*", 1.80),
    ("ARG", "NE*", 1.62),
    ("ARG", "NH1", 1.62),
    ("ARG", "NH2", 1.67),
    // ASN
    ("ASN", "CB", 1.91),
    ("ASN", "CG*", 1.81),
    ("ASN", "ND2", 1.62),
    ("ASN", "OD1", 1.52),
    // ASP
    ("ASP", "CB", 1.91),
    ("ASP", "CG*", 1.76),
    ("ASP", "OD1", 1.49),
    ("ASP", "OD2", 1.49),
    // CYS
    ("CYS", "CB", 1.91),
    ("CYS", "S*", 1.88),
    // GLN
    ("GLN", "CB", 1.91),
    ("GLN", "CD*", 1.81),
    ("GLN", "CG*", 1.80),
    ("GLN", "NE2", 1.62),
    ("GLN", "OE1", 1.52),
    // GLU
    ("GLU", "CB", 1.91),
    ("GLU", "CD*", 1.76),
    ("GLU", "CG*", 1.88),
    ("GLU", "OE1", 1.49),
    ("GLU", "OE2", 1.49),
    // HIS
    ("HIS", "CB", 1.91),
    ("HIS", "CD*", 1.74),
    ("HIS", "CE*", 1.74),
    ("HIS", "CG*", 1.80),
    ("HIS", "ND1", 1.60),
    ("HIS", "NE2", 1.60),
    // ILE
    ("ILE", "CB", 2.01),
    ("ILE", "CD1", 1.92),
    ("ILE", "CG1", 1.92),
    ("ILE", "CG2", 1.92),
    // LEU
    ("LEU", "CB", 1.91),
    ("LEU", "CD1", 1.92),
    ("LEU", "CD2", 1.92),
    ("LEU", "CG*", 2.01),
    // LYS
    ("LYS", "CB", 1.91),
    ("LYS", "CD*", 1.92),
    ("LYS", "CE*", 1.88),
    ("LYS", "CG*", 1.92),
    ("LYS", "NZ*", 1.67),
    // MET
    ("MET", "CB", 1.91),
    ("MET", "CE*", 1.80),
    ("MET", "CG*", 1.92),
    ("MET", "S*", 1.94),
    // PHE
    ("PHE", "CB", 1.91),
    ("PHE", "CD*", 1.82),
    ("PHE", "CE*", 1.82),
    ("PHE", "CG*", 1.74),
    ("PHE", "CZ*", 1.82),
    // PRO
    ("PRO", "CB", 1.91),
    ("PRO", "CD*", 1.92),
    ("PRO", "CG*", 1.92),
    // SER
    ("SER", "CB", 1.91),
    ("SER", "OG*", 1.54),
    // THR
    ("THR", "CB", 2.01),
    ("THR", "CG2", 1.92),
    ("THR", "OG*", 1.54),
    // TRP
    ("TRP", "CB", 1.91),
    ("TRP", "CD*", 1.82),
    ("TRP", "CE*", 1.82),
    ("TRP", "CE2", 1.74),
    ("TRP", "CG*", 1.74),
    ("TRP", "CH*", 1.82),
    ("TRP", "CZ*", 1.82),
    ("TRP", "NE1", 1.66),
    // TYR
    ("TYR", "CB", 1.91),
    ("TYR", "CD*", 1.82),
    ("TYR", "CE*", 1.82),
    ("TYR", "CG*", 1.74),
    ("TYR", "CZ*", 1.80),
    ("TYR", "OH*", 1.54),
    // VAL
    ("VAL", "CB", 2.01),
    ("VAL", "CG1", 1.92),
    ("VAL", "CG2", 1.92),
    // Halogens
    ("*", "F*", 1.33),
    ("*", "CL*", 1.81),
    ("*", "BR*", 1.96),
    ("*", "I*", 2.20),
    // Metal ions (residue name = atom name for ions)
    ("AL", "AL", 0.60),
    ("AS", "AS", 0.58),
    ("AU", "AU", 1.37),
    ("BA", "BA", 1.35),
    ("BE", "BE", 0.45),
    ("BI", "BI", 1.03),
    ("CA", "CA", 1.00),
    ("CD", "CD", 0.95),
    ("CO", "CO", 0.65),
    ("CR", "CR", 0.73),
    ("CS", "CS", 1.67),
    ("CU", "CU", 0.73),
    ("FE", "FE", 0.61),
    ("HE*", "FE", 0.61), // HEM residue iron
    ("GA", "GA", 0.62),
    ("GE", "GE", 0.73),
    ("HG", "HG", 1.02),
    ("K", "K", 1.38),
    ("LI", "LI", 0.76),
    ("MG", "MG", 0.72),
    ("MN", "MN", 0.83),
    ("MO", "MO", 0.69),
    ("NA", "NA", 1.02),
    ("NI", "NI", 0.69),
    ("PB", "PB", 1.19),
    ("PD", "PD", 0.86),
    ("PT", "PT", 0.80),
    ("RB", "RB", 1.52),
    ("SB", "SB", 0.76),
    ("SC", "SC", 0.75),
    ("SN", "SN", 0.69),
    ("SR", "SR", 1.18),
    ("TC", "TC", 0.65),
    ("TI", "TI", 0.86),
    ("V", "V", 0.79),
    ("ZN", "ZN", 0.74),
    ("ZR", "ZR", 0.72),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_atoms() {
        let lookup = RadiiLookup::new();
        assert!((lookup.get_radius("ALA", "CA") - 1.90).abs() < 0.001);
        assert!((lookup.get_radius("GLY", "C") - 1.75).abs() < 0.001);
        assert!((lookup.get_radius("ALA", "N") - 1.70).abs() < 0.001);
        assert!((lookup.get_radius("ALA", "O") - 1.49).abs() < 0.001);
    }

    #[test]
    fn sidechain_atoms() {
        let lookup = RadiiLookup::new();
        assert!((lookup.get_radius("ALA", "CB") - 1.92).abs() < 0.001);
        assert!((lookup.get_radius("ARG", "NH1") - 1.62).abs() < 0.001);
        assert!((lookup.get_radius("ARG", "NH2") - 1.67).abs() < 0.001);
    }

    #[test]
    fn wildcard_matching() {
        let lookup = RadiiLookup::new();
        // CD* pattern for ARG
        assert!((lookup.get_radius("ARG", "CD") - 1.88).abs() < 0.001);
        // C* fallback
        assert!((lookup.get_radius("UNK", "CX") - 1.80).abs() < 0.001);
    }

    #[test]
    fn metal_ions() {
        let lookup = RadiiLookup::new();
        assert!((lookup.get_radius("ZN", "ZN") - 0.74).abs() < 0.001);
        assert!((lookup.get_radius("CA", "CA") - 1.00).abs() < 0.001);
        assert!((lookup.get_radius("FE", "FE") - 0.61).abs() < 0.001);
    }

    #[test]
    fn default_radius() {
        let lookup = RadiiLookup::new();
        assert!((lookup.get_radius("UNK", "XYZ") - 1.8).abs() < 0.001);
    }

    #[test]
    fn custom_rules() {
        let mut lookup = RadiiLookup::empty();
        lookup.load_from_text("* C* 2.0\nALA CB 1.5").unwrap();
        assert!((lookup.get_radius("ALA", "CB") - 1.5).abs() < 0.001);
        assert!((lookup.get_radius("GLY", "CA") - 2.0).abs() < 0.001);
    }
}
