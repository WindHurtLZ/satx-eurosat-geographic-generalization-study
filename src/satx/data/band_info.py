MS_BAND_ORDER = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
)

RGB_BAND_ORDER = ("B04", "B03", "B02")

RGB_CHANNEL_INDICES = (3, 2, 1)

SPECTRAL_BAND_GROUPS: dict[str, tuple[str, ...]] = {
    "VISIBLE": ("B02", "B03", "B04"),
    "RED_EDGE_NIR": ("B05", "B06", "B07", "B08", "B8A"),
    "ATMOSPHERIC": ("B01", "B09", "B10"),
    "SWIR": ("B11", "B12"),
}

SPECTRAL_GROUP_NAMES = tuple(SPECTRAL_BAND_GROUPS)

SPECTRAL_GROUP_INDICES: dict[str, tuple[int, ...]] = {
    group_name: tuple(MS_BAND_ORDER.index(band_name) for band_name in band_names)
    for group_name, band_names in SPECTRAL_BAND_GROUPS.items()
}

if tuple(MS_BAND_ORDER[index] for index in RGB_CHANNEL_INDICES) != RGB_BAND_ORDER:
    raise RuntimeError(
        "RGB_CHANNEL_INDICES must select B04, B03, and B02 from MS_BAND_ORDER."
    )


def validate_band_groups() -> None:
    grouped_bands = tuple(
        band_name
        for band_names in SPECTRAL_BAND_GROUPS.values()
        for band_name in band_names
    )

    if len(grouped_bands) != len(set(grouped_bands)):
        raise RuntimeError("Spectral band groups must be disjoint.")

    if set(grouped_bands) != set(MS_BAND_ORDER):
        missing = set(MS_BAND_ORDER) - set(grouped_bands)
        extra = set(grouped_bands) - set(MS_BAND_ORDER)
        raise RuntimeError(
            f"Spectral groups must cover all MS bands exactly once. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}."
        )

validate_band_groups()