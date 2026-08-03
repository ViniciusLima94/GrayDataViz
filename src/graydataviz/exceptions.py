class GrayDataVizError(Exception):
    """Base class for all graydataviz errors."""


class InvalidMonkeyError(GrayDataVizError):
    """Raised when a monkey name outside of the known set is used."""


class RawDataNotFoundError(GrayDataVizError):
    """Raised when raw session files (recording_info.mat, trial_info.mat, LFP) are missing."""


class DerivedDataNotFoundError(GrayDataVizError):
    """Raised when a derived NetCDF product (power, coherence, ...) cannot be found on disk."""
