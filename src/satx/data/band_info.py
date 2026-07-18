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

if tuple(MS_BAND_ORDER[index] for index in RGB_CHANNEL_INDICES) != RGB_BAND_ORDER:
    raise RuntimeError(
        "RGB_CHANNEL_INDICES must select B04, B03, and B02 from MS_BAND_ORDER."
    )