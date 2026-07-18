"""
Transformation statistics for EuroSAT
"""

from __future__ import annotations
from dataclasses import dataclass
from satx.data.band_info import RGB_CHANNEL_INDICES

"""
Dataclass hold transformation statistics for each channel
"""
@dataclass(frozen=True)
class ChannelStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    # TODO: Add transformation statistics here
    # resize...?

"""
Normalization & Std Statistics for EuroSAT
"""
# TorchGeo EuroSATDataModule statistics
# https://docs.torchgeo.org/en/stable/_modules/torchgeo/datamodules/eurosat.html
STANDARD_MS_STATS = ChannelStats(
    mean=(
        1354.40546513,
        1118.24399958,
        1042.92983953,
        947.62620298,
        1199.47283961,
        1999.79090914,
        2369.22292565,
        2296.82608323,
        732.08340178,
        12.11327804,
        1819.01027855,
        1118.92391149,
        2594.14080798,
    ),
    std=(
        245.71762908,
        333.00778264,
        395.09249139,
        593.75055589,
        566.41700170,
        861.18399006,
        1086.63139075,
        1117.98170791,
        404.91978886,
        4.77584468,
        1002.58768311,
        761.30323499,
        1231.58581042,
    ),
)

STANDARD_RGB_STATS = ChannelStats(
    mean=tuple(
        STANDARD_MS_STATS.mean[index]
        for index in RGB_CHANNEL_INDICES
    ),
    std=tuple(
        STANDARD_MS_STATS.std[index]
        for index in RGB_CHANNEL_INDICES
    ),
)

SPATIAL_MS_STATS = ChannelStats(
    mean=(
        1375.9932,
        1142.6339,
        1077.5502,
        1003.8445,
        1280.7300,
        2130.3491,
        2524.0549,
        2454.1938,
        785.4963,
        12.4639,
        1969.9224,
        1206.2421,
        2779.4104,
    ),
    std=(
        249.8516,
        337.9465,
        392.5661,
        612.4237,
        562.2878,
        806.8271,
        1022.6378,
        1065.4312,
        410.5831,
        4.8878,
        958.4751,
        740.6196,
        1157.2896,
    ),
)

SPATIAL_RGB_STATS = ChannelStats(
    mean=tuple(
        SPATIAL_MS_STATS.mean[index]
        for index in RGB_CHANNEL_INDICES
    ),
    std=tuple(
        SPATIAL_MS_STATS.std[index]
        for index in RGB_CHANNEL_INDICES
    ),
)

NORMALIZATION_STATS = {
    "standard": {
        "ms": STANDARD_MS_STATS,
        "rgb": STANDARD_RGB_STATS,
    },
    "spatial": {
        "ms": SPATIAL_MS_STATS,
        "rgb": SPATIAL_RGB_STATS,
    },
}

def get_normalization_stats(split_type: str, modality: str) -> ChannelStats:
    try:
        split_statistics = NORMALIZATION_STATS[split_type]
        return split_statistics[modality]
    except KeyError as exc:
        raise ValueError(
            "No normalization statistics are registered for "
            f"split_type={split_type!r}, modality={modality!r}."
        ) from exc