"""Tests for utility classes."""
import pytest
import numpy as np
import tempfile
from utils.output_manager import OutputManager
from utils.data_logger import DataLogger


class TestOutputManager:
    def test_creates_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            for subdir in OutputManager.SUBDIRS:
                assert (om._base / subdir).is_dir()

    def test_plot_path_has_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            path = om.plot_path("test")
            assert "test" in path.name
            assert path.suffix == ".png"

    def test_animation_path_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            path = om.animation_path("anim")
            assert path.suffix == ".gif"
            assert "anim" in path.name

    def test_data_path_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            path = om.data_path("run")
            assert path.suffix == ".npz"

    def test_log_path_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            path = om.log_path("session")
            assert path.suffix == ".log"

    def test_save_trajectory_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            N = 100
            result = {
                't': np.linspace(0, 1, N),
                'q': np.zeros((N, 4)),
                'dq': np.zeros((N, 4)),
                'u': np.zeros(N),
            }
            path = om.save_trajectory(result, name="traj")
            assert path.exists()

    def test_save_trajectory_skips_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            om = OutputManager(tmpdir)
            result = {'t': np.array([0.0, 0.1]), 'q': None}
            path = om.save_trajectory(result, name="partial")
            with np.load(str(path)) as data:
                assert 't' in data
                assert 'q' not in data


class TestDataLogger:
    def test_initial_empty(self):
        dl = DataLogger(100)
        assert dl.count == 0
        assert len(dl.time) == 0

    def test_record_and_retrieve(self):
        dl = DataLogger(100)
        dl.on_step(0.0, np.array([1, 2, 3, 4.0]),
                   np.array([5, 6, 7, 8.0]), 10.0)
        assert dl.count == 1
        assert dl.time[0] == 0.0
        np.testing.assert_array_equal(dl.states[0], [1, 2, 3, 4])
        assert dl.inputs[0] == 10.0

    def test_multiple_records(self):
        dl = DataLogger(100)
        for i in range(10):
            dl.on_step(float(i), np.ones(4) * i, np.zeros(4), float(i))
        assert dl.count == 10
        np.testing.assert_array_equal(dl.time, np.arange(10, dtype=float))
        np.testing.assert_array_equal(dl.inputs, np.arange(10, dtype=float))

    def test_expansion(self):
        dl = DataLogger(2)
        for i in range(5):
            dl.on_step(float(i), np.zeros(4), np.zeros(4), 0.0)
        assert dl.count == 5
        assert dl._capacity >= 5

    def test_expansion_preserves_data(self):
        dl = DataLogger(3)
        for i in range(7):
            dl.on_step(float(i), np.full(4, i), np.zeros(4), float(i * 2))
        assert dl.count == 7
        np.testing.assert_array_equal(dl.time, np.arange(7, dtype=float))
        np.testing.assert_array_equal(dl.inputs, np.arange(0, 14, 2, dtype=float))
        for i in range(7):
            np.testing.assert_array_equal(dl.states[i], np.full(4, i))

    def test_velocities_recorded(self):
        dl = DataLogger(10)
        dq = np.array([0.1, 0.2, 0.3, 0.4])
        dl.on_step(0.0, np.zeros(4), dq, 0.0)
        np.testing.assert_array_almost_equal(dl.velocities[0], dq)
