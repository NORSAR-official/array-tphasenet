import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from beamforming import compute_beam_time_delays, create_beam, load_array_geometries


def _trace(station: str, data: np.ndarray) -> Trace:
    tr = Trace(data=np.asarray(data, dtype=np.float32))
    tr.stats.station = station
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = 20.0
    tr.stats.starttime = UTCDateTime("2023-01-01T00:00:00")
    return tr


@pytest.mark.smoke
def test_array_geometry_loads():
    geometries = load_array_geometries()
    assert "ARCES" in geometries
    assert len(geometries["ARCES"]) > 0


@pytest.mark.smoke
def test_create_beam_from_toy_stream():
    geometry = {
        "STA1": {"dx": 0.0, "dy": 0.0},
        "STA2": {"dx": 0.5, "dy": 0.0},
        "STA3": {"dx": 0.0, "dy": 0.5},
    }
    delays = compute_beam_time_delays(geometry, azimuth_deg=45.0, velocity_km_sec=6.0)

    n = 200
    t = np.linspace(0, 2 * np.pi, n)
    stream = Stream(
        [
            _trace("STA1", np.sin(t)),
            _trace("STA2", np.sin(t + 0.1)),
            _trace("STA3", np.sin(t - 0.1)),
        ]
    )
    beam = create_beam(stream, delays, station_name="TESTBEAM")

    assert beam.stats.station == "TESTBEAM"
    assert beam.stats.channel == "BHZ"
    assert len(beam.data) >= n
