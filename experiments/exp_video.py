from datasets.navier_stokes import NavierStokesDataset
from datasets.sevir import SEVIRDataset
from datasets.era5 import ERA5Dataset
from algorithms.diffusion_forcing import (
    DiffusionForcingNS,
    DiffusionForcingSEVIR,
    DiffusionForcingERA5,
)
from .exp_base import BaseLightningExperiment


class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A spatiotemporal sequence experiment shared by the Navier-Stokes, SEVIR,
    and ERA5 data-assimilation domains.
    """

    compatible_algorithms = dict(
        df_ns=DiffusionForcingNS,
        df_ns_dit=DiffusionForcingNS,
        df_sevir=DiffusionForcingSEVIR,
        df_sevir_dit=DiffusionForcingSEVIR,
        df_era5=DiffusionForcingERA5,
        df_era5_dit=DiffusionForcingERA5,
    )

    compatible_datasets = dict(
        # Navier-Stokes vorticity
        ns_vorticity=NavierStokesDataset,
        # SEVIR VIL precipitation
        sevir_vil=SEVIRDataset,
        # ERA5 multi-variable weather
        era5=ERA5Dataset,
    )
