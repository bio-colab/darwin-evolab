"""test_map_elites_dispatch.py — Unit Tests for MAP-Elites Archive and Specialized Dispatch."""

import json
from pathlib import Path
import tempfile
import pytest

from experimental.pdf2rtf.corpus import create_golden_corpus
from experimental.pdf2rtf.genome import ProfileGenome, ProfilePolicy
from experimental.pdf2rtf.map_elites_archive import (
    ArchiveCell,
    MAPElitesArchive,
    MAPElitesPolicyDispatcher,
    compute_document_descriptors,
)
from experimental.pdf2rtf.pdf_extractor import PDFExtractor


def test_compute_document_descriptors():
    corpus = create_golden_corpus()
    # Test article (narrative text)
    art = next(c for c in corpus if c.name == "article_standard")
    d1, d2 = compute_document_descriptors(art.pdf_bytes)
    assert 0.0 <= d1 <= 1.0
    assert 0.0 <= d2 <= 1.0
    # Financial table should have significantly higher table sensitivity than narrative article
    fin = next(c for c in corpus if c.name == "financial_table")
    d1_fin, d2_fin = compute_document_descriptors(fin.pdf_bytes)
    assert d2_fin > d2


def test_archive_cell_creation_and_update():
    archive = MAPElitesArchive(grid_x=10, grid_y=10)
    coord = (3, 5)
    policy1 = ProfilePolicy(space_gap_ratio=0.20, para_split_delta_ratio=1.0)
    policy2 = ProfilePolicy(space_gap_ratio=0.35, para_split_delta_ratio=1.8)

    # Initial insertion
    updated = archive.update_cell(
        coord=coord,
        fitness=0.85,
        policy=policy1,
        descriptors={"cluster_density": 0.35, "table_sensitivity": 0.55},
        fingerprint="fp1",
    )
    assert updated is True
    assert archive.occupied_count == 1
    assert archive.coverage == 0.01

    # Inferior fitness should NOT update
    updated_inferior = archive.update_cell(
        coord=coord,
        fitness=0.80,
        policy=policy2,
        descriptors={"cluster_density": 0.35, "table_sensitivity": 0.55},
        fingerprint="fp2",
    )
    assert updated_inferior is False
    assert archive.get_cell(coord).fitness == 0.85

    # Superior fitness SHOULD update
    updated_superior = archive.update_cell(
        coord=coord,
        fitness=0.95,
        policy=policy2,
        descriptors={"cluster_density": 0.35, "table_sensitivity": 0.55},
        fingerprint="fp2",
    )
    assert updated_superior is True
    assert archive.get_cell(coord).fitness == 0.95
    assert archive.get_cell(coord).policy.space_gap_ratio == 0.35


def test_archive_nearest_neighbor_and_champion():
    archive = MAPElitesArchive(grid_x=10, grid_y=10)
    p_text = ProfilePolicy(space_gap_ratio=0.20)
    p_table = ProfilePolicy(table_col_align_tol_pt=12.0)

    archive.update_cell((2, 2), 0.90, p_text, {"cluster_density": 0.25, "table_sensitivity": 0.25}, "fp_text")
    archive.update_cell((8, 8), 0.98, p_table, {"cluster_density": 0.85, "table_sensitivity": 0.85}, "fp_table")

    # Champion must be cell (8, 8)
    champ = archive.get_champion()
    assert champ is not None
    assert champ.coord == (8, 8)
    assert champ.fitness == 0.98

    # Query near (2, 2) should return cell (2, 2)
    near_text = archive.get_nearest_elite(0.20, 0.30)
    assert near_text is not None
    assert near_text.coord == (2, 2)

    # Query near (8, 8) should return cell (8, 8)
    near_table = archive.get_nearest_elite(0.90, 0.80)
    assert near_table is not None
    assert near_table.coord == (8, 8)


def test_archive_save_load_json(tmp_path):
    archive = MAPElitesArchive(grid_x=10, grid_y=10)
    p = ProfilePolicy(space_gap_ratio=0.30, align_tolerance_pt=6.5)
    archive.update_cell((4, 6), 0.965, p, {"cluster_density": 0.45, "table_sensitivity": 0.65}, "fp_test")

    out_file = tmp_path / "test_archive.json"
    archive.save_json(out_file)

    loaded = MAPElitesArchive.load_json(out_file)
    assert loaded.grid_x == 10
    assert loaded.grid_y == 10
    assert loaded.occupied_count == 1
    c = loaded.get_cell((4, 6))
    assert c is not None
    assert c.fitness == 0.965
    assert c.policy.align_tolerance_pt == 6.5


def test_extractor_with_archive_dispatch():
    corpus = create_golden_corpus()
    art = next(c for c in corpus if c.name == "article_standard")

    archive = MAPElitesArchive(grid_x=10, grid_y=10)
    p_specialized = ProfilePolicy(space_gap_ratio=0.33, para_split_delta_ratio=1.45)
    # Insert elite near article coordinates
    archive.update_cell((2, 0), 0.99, p_specialized, {"cluster_density": 0.25, "table_sensitivity": 0.05}, "fp_art")

    extractor = PDFExtractor(archive=archive)
    ir = extractor.extract(art.pdf_bytes)

    assert ir is not None
    assert len(ir.pages) > 0
    assert extractor.last_dispatched_cell is not None
    assert extractor.policy.space_gap_ratio == 0.33
