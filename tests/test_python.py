"""Tests for voronota-ltr Python bindings."""

import unittest


class TestVoronotaLtr(unittest.TestCase):
    """Test cases for voronota-ltr Python bindings."""

    def test_import(self):
        """Module can be imported."""
        import voronota_ltr

        self.assertTrue(hasattr(voronota_ltr, "compute_tessellation"))

    def test_tuples(self):
        """Balls as list of tuples."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5)],
            probe=1.4,
        )
        self.assertEqual(result["num_balls"], 2)
        self.assertEqual(len(result["contacts"]), 1)
        self.assertGreater(result["total_sas_area"], 0)
        self.assertGreater(result["total_volume"], 0)

    def test_lists(self):
        """Balls as list of lists."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[[0, 0, 0, 1.5], [3, 0, 0, 1.5]],
            probe=1.4,
        )
        self.assertEqual(result["num_balls"], 2)
        self.assertEqual(len(result["contacts"]), 1)

    def test_dicts(self):
        """Balls as list of dicts."""
        import voronota_ltr

        balls = [
            {"x": 0, "y": 0, "z": 0, "r": 1.5},
            {"x": 3, "y": 0, "z": 0, "r": 1.5},
        ]
        result = voronota_ltr.compute_tessellation(balls=balls, probe=1.4)
        self.assertEqual(result["num_balls"], 2)
        self.assertEqual(len(result["contacts"]), 1)

    def test_numpy(self):
        """Balls as numpy array."""
        import voronota_ltr

        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")

        balls = np.array([[0, 0, 0, 1.5], [3, 0, 0, 1.5]])
        result = voronota_ltr.compute_tessellation(balls=balls, probe=1.4)
        self.assertEqual(result["num_balls"], 2)
        self.assertEqual(len(result["contacts"]), 1)

    def test_periodic_box_corners(self):
        """Periodic boundary with corners."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5)],
            probe=1.4,
            periodic_box={"corners": [(0, 0, 0), (10, 10, 10)]},
        )
        self.assertEqual(result["num_balls"], 2)

    def test_periodic_box_vectors(self):
        """Periodic boundary with vectors (triclinic cell)."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5)],
            probe=1.4,
            periodic_box={"vectors": [(10, 0, 0), (0, 10, 0), (0, 0, 10)]},
        )
        self.assertEqual(result["num_balls"], 2)

    def test_cell_vertices(self):
        """Request cell vertices and edges output."""
        import voronota_ltr

        # Need 3+ balls to generate vertices (2 balls only create a single contact plane)
        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5), (1.5, 2.5, 0, 1.5)],
            probe=1.4,
            with_cell_vertices=True,
        )
        self.assertIn("cell_vertices", result)
        self.assertIn("cell_edges", result)
        self.assertGreater(len(result["cell_vertices"]), 0)
        self.assertGreater(len(result["cell_edges"]), 0)

        # Check vertex structure
        v = result["cell_vertices"][0]
        self.assertIn("ball_indices", v)
        self.assertIn("x", v)
        self.assertIn("y", v)
        self.assertIn("z", v)
        self.assertIn("is_on_sas", v)

        # Check edge structure
        e = result["cell_edges"][0]
        self.assertIn("ball_indices", e)
        self.assertIn("length", e)
        self.assertIn("is_on_sas", e)

    def test_groups(self):
        """Groups parameter for inter-group contacts."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5), (6, 0, 0, 1.5)],
            probe=1.4,
            groups=[0, 0, 1],  # First two in group 0, third in group 1
        )
        self.assertEqual(result["num_balls"], 3)

    def test_contact_structure(self):
        """Check contact dict structure."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5)],
            probe=1.4,
        )
        contact = result["contacts"][0]
        self.assertIn("id_a", contact)
        self.assertIn("id_b", contact)
        self.assertIn("area", contact)
        self.assertIn("arc_length", contact)
        self.assertLess(contact["id_a"], contact["id_b"])  # Ordered

    def test_cell_structure(self):
        """Check cell dict structure."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5), (3, 0, 0, 1.5)],
            probe=1.4,
        )
        cell = result["cells"][0]
        self.assertIn("index", cell)
        self.assertIn("sas_area", cell)
        self.assertIn("volume", cell)

    def test_empty_balls(self):
        """Empty ball list."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(balls=[], probe=1.4)
        self.assertEqual(result["num_balls"], 0)
        self.assertEqual(len(result["contacts"]), 0)

    def test_single_ball(self):
        """Single ball (no contacts)."""
        import voronota_ltr

        result = voronota_ltr.compute_tessellation(
            balls=[(0, 0, 0, 1.5)],
            probe=1.4,
        )
        self.assertEqual(result["num_balls"], 1)
        self.assertEqual(len(result["contacts"]), 0)

    def test_invalid_ball_tuple(self):
        """Invalid ball tuple raises error."""
        import voronota_ltr

        with self.assertRaises(Exception):
            voronota_ltr.compute_tessellation(balls=[(0, 0, 0)], probe=1.4)  # Missing r

    def test_invalid_periodic_box(self):
        """Invalid periodic box raises error."""
        import voronota_ltr

        with self.assertRaises(Exception):
            voronota_ltr.compute_tessellation(
                balls=[(0, 0, 0, 1.5)],
                probe=1.4,
                periodic_box={"invalid": "key"},
            )


class TestComputeFromFile(unittest.TestCase):
    """Test cases for compute_tessellation_from_file."""

    def test_import(self):
        """Function is exported."""
        import voronota_ltr

        self.assertTrue(hasattr(voronota_ltr, "compute_tessellation_from_file"))

    def test_pdb_file(self):
        """Read PDB file."""
        import os
        import voronota_ltr

        path = os.path.join(os.path.dirname(__file__), "data", "assembly_1ctf.pdb1")
        if not os.path.exists(path):
            self.skipTest("Test file not found")

        result = voronota_ltr.compute_tessellation_from_file(path, probe=1.4)
        self.assertEqual(result["num_balls"], 492)
        self.assertEqual(len(result["contacts"]), 3078)
        self.assertGreater(result["total_sas_area"], 4000)

    def test_mmcif_file(self):
        """Read mmCIF file."""
        import os
        import voronota_ltr

        path = os.path.join(os.path.dirname(__file__), "data", "assembly_1ctf.cif")
        if not os.path.exists(path):
            self.skipTest("Test file not found")

        result = voronota_ltr.compute_tessellation_from_file(path, probe=1.4)
        self.assertEqual(result["num_balls"], 984)

    def test_with_selections(self):
        """Read file with VMD-like selections."""
        import os
        import voronota_ltr

        path = os.path.join(os.path.dirname(__file__), "data", "assembly_1ctf.cif")
        if not os.path.exists(path):
            self.skipTest("Test file not found")

        result = voronota_ltr.compute_tessellation_from_file(
            path, probe=1.4, group_selections=["chain A", "chain A-2"]
        )
        # Inter-chain contacts only (fewer than all contacts)
        self.assertLess(len(result["contacts"]), 6354)

    def test_with_cell_vertices(self):
        """Request cell vertices from file."""
        import os
        import voronota_ltr

        path = os.path.join(os.path.dirname(__file__), "data", "assembly_1ctf.pdb1")
        if not os.path.exists(path):
            self.skipTest("Test file not found")

        result = voronota_ltr.compute_tessellation_from_file(
            path, probe=1.4, with_cell_vertices=True
        )
        self.assertIn("cell_vertices", result)
        self.assertIn("cell_edges", result)

    def test_too_few_selections(self):
        """Error with fewer than two selections."""
        import os
        import voronota_ltr

        path = os.path.join(os.path.dirname(__file__), "data", "assembly_1ctf.pdb1")
        if not os.path.exists(path):
            self.skipTest("Test file not found")

        with self.assertRaises(IOError):
            voronota_ltr.compute_tessellation_from_file(
                path, probe=1.4, group_selections=["protein"]
            )

    def test_file_not_found(self):
        """Error when file doesn't exist."""
        import voronota_ltr

        with self.assertRaises(IOError):
            voronota_ltr.compute_tessellation_from_file("nonexistent.pdb", probe=1.4)


if __name__ == "__main__":
    unittest.main()
