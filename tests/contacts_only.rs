use voronota_ltr::{Ball, PeriodicBox, compute_contacts_only, compute_tessellation};

#[test]
fn contacts_only_matches_full_tessellation() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.5),
        Ball::new(2.5, 0.0, 0.0, 1.5),
        Ball::new(1.25, 2.0, 0.0, 1.5),
    ];
    let full = compute_tessellation(&balls, 1.4, None, None, false);
    let mut contacts_only = compute_contacts_only(&balls, 1.4, None, None);

    // Sort both by (id_a, id_b) for comparison
    let mut full_contacts = full.contacts.clone();
    full_contacts.sort_by_key(|c| (c.id_a, c.id_b));
    contacts_only.sort_by_key(|c| (c.id_a, c.id_b));

    assert_eq!(
        full_contacts.len(),
        contacts_only.len(),
        "contact count mismatch"
    );
    for (f, c) in full_contacts.iter().zip(contacts_only.iter()) {
        assert_eq!(f.id_a, c.id_a);
        assert_eq!(f.id_b, c.id_b);
        assert!(
            (f.area - c.area).abs() < 1e-10,
            "area mismatch for ({}, {}): {} vs {}",
            f.id_a,
            f.id_b,
            f.area,
            c.area
        );
        assert!(
            (f.arc_length - c.arc_length).abs() < 1e-10,
            "arc_length mismatch for ({}, {}): {} vs {}",
            f.id_a,
            f.id_b,
            f.arc_length,
            c.arc_length
        );
    }
}

#[test]
fn contacts_only_with_groups_filters_intra_group() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.5),
        Ball::new(1.0, 0.0, 0.0, 1.5), // same group as first
        Ball::new(3.0, 0.0, 0.0, 1.5), // different group
    ];
    let groups = vec![0, 0, 1];
    let contacts = compute_contacts_only(&balls, 1.4, None, Some(&groups));

    for c in &contacts {
        assert_ne!(
            groups[c.id_a], groups[c.id_b],
            "intra-group contact found: ({}, {})",
            c.id_a, c.id_b
        );
    }
    assert!(
        !contacts.is_empty(),
        "expected at least one inter-group contact"
    );
}

#[test]
fn contacts_only_periodic_matches_full() {
    let balls = vec![Ball::new(0.0, 0.0, 0.0, 2.0), Ball::new(9.0, 0.0, 0.0, 2.0)];
    let pbox = PeriodicBox::from_corners((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0));
    let full = compute_tessellation(&balls, 1.4, Some(&pbox), None, false);
    let contacts_only = compute_contacts_only(&balls, 1.4, Some(&pbox), None);

    assert_eq!(
        full.contacts.len(),
        contacts_only.len(),
        "periodic contact count mismatch"
    );
    for c in &contacts_only {
        assert!(c.id_a < balls.len(), "non-canonical id_a: {}", c.id_a);
        assert!(c.id_b < balls.len(), "non-canonical id_b: {}", c.id_b);
    }
    // Verify areas match
    let mut full_contacts = full.contacts.clone();
    full_contacts.sort_by_key(|c| (c.id_a, c.id_b));
    let mut co = contacts_only;
    co.sort_by_key(|c| (c.id_a, c.id_b));
    for (f, c) in full_contacts.iter().zip(co.iter()) {
        assert!(
            (f.area - c.area).abs() < 1e-10,
            "periodic area mismatch: {} vs {}",
            f.area,
            c.area
        );
    }
}

#[test]
fn contacts_only_empty_input() {
    let contacts = compute_contacts_only(&[], 1.4, None, None);
    assert!(contacts.is_empty());
}

#[test]
fn contacts_only_no_overlap() {
    let balls = vec![
        Ball::new(0.0, 0.0, 0.0, 1.0),
        Ball::new(100.0, 0.0, 0.0, 1.0),
    ];
    let contacts = compute_contacts_only(&balls, 1.4, None, None);
    assert!(contacts.is_empty());
}
