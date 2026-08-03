import numpy as np

from graydataviz.io import read_hdf5_mat, read_legacy_mat, save_mat


def test_legacy_mat_roundtrip(tmp_path):
    path = tmp_path / "roundtrip.mat"
    save_mat(path, {"x": np.array([1.0, 2.0, 3.0])})

    loaded = read_legacy_mat(path)
    np.testing.assert_allclose(loaded["x"].squeeze(), [1.0, 2.0, 3.0])


def test_read_hdf5_mat(tmp_path):
    import h5py

    path = tmp_path / "file.mat"
    with h5py.File(path, "w") as f:
        f.create_dataset("y", data=np.array([4.0, 5.0]))

    with read_hdf5_mat(path) as f:
        np.testing.assert_allclose(f["y"][()], [4.0, 5.0])
