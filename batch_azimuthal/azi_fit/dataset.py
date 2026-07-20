"""
Thin wrapper around one integrated HDF5 scan.

The notebooks reopened the HDF5 file on *every* redraw (`load_intensity` did
`with h5py.File(...) as f:` each time). That is the main interactive-lag source.
Here the file is opened once and kept open, axes are read once, and the most
recent pattern is cached, so slider scrubbing only reads one 2D slice.
"""

from __future__ import annotations

import numpy as np
import h5py


class Dataset:
    def __init__(self, scan_path, intensity_path, twotheta_path, chi_path):
        self.scan_path = scan_path
        self.intensity_path = intensity_path
        self.twotheta_path = twotheta_path
        self.chi_path = chi_path

        self._f = h5py.File(scan_path, "r")
        try:
            self._dset = self._f[intensity_path]
        except KeyError:
            self._f.close()
            raise KeyError(f"intensity path not found: {intensity_path}")

        self.n_patterns = int(self._dset.shape[0])
        self.n_chi = int(self._dset.shape[1])
        self.n_twotheta = int(self._dset.shape[2])

        self.two_theta = np.asarray(self._f[twotheta_path][:], dtype=float)
        self.chi = np.asarray(self._f[chi_path][:], dtype=float)

        self.x_min, self.x_max = float(np.nanmin(self.two_theta)), float(np.nanmax(self.two_theta))
        self.y_min, self.y_max = float(np.nanmin(self.chi)), float(np.nanmax(self.chi))

        self._cache_index = None
        self._cache_intensity = None

    # -- pattern access -------------------------------------------------
    def intensity(self, index):
        """Return the 2D I(chi, 2theta) for a pattern, cached."""
        if index == self._cache_index and self._cache_intensity is not None:
            return self._cache_intensity
        arr = np.asarray(self._dset[index, :, :], dtype=float)
        self._cache_index = index
        self._cache_intensity = arr
        return arr

    def i_vs_chi(self, index, roi_min, roi_max):
        """Integrate a 2theta ROI of one pattern down to I(chi)."""
        mask = (self.two_theta >= roi_min) & (self.two_theta <= roi_max)
        if not np.any(mask):
            return None
        intensity = self.intensity(index)
        return np.nansum(intensity[:, mask], axis=1)

    def roi_slice(self, roi_min, roi_max):
        """Column index slice (j0, j1) for a 2theta ROI, or None."""
        ri = np.where((self.two_theta >= roi_min) & (self.two_theta <= roi_max))[0]
        if ri.size == 0:
            return None
        return int(ri.min()), int(ri.max()) + 1

    def tree(self):
        """List of all HDF5 object paths (for the path pickers)."""
        names = []
        self._f.visititems(lambda name, obj: names.append(name))
        return names

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


def peek_tree(scan_path):
    """Return (all_paths, dataset_paths) for a file without fully loading it."""
    all_paths, dset_paths = [], []
    with h5py.File(scan_path, "r") as f:
        def _v(name, obj):
            all_paths.append(name)
            if isinstance(obj, h5py.Dataset):
                dset_paths.append(name)
        f.visititems(_v)
    return all_paths, dset_paths
