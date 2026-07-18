from satx.data.band_info import (
    MS_BAND_ORDER,
    RGB_BAND_ORDER,
    RGB_CHANNEL_INDICES,
)

def test_ms_band_order_matching():
    assert MS_BAND_ORDER == (
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

def test_rgb_indices_select():
    selected_bands = tuple(
        MS_BAND_ORDER[index]
        for index in RGB_CHANNEL_INDICES
    )

    assert RGB_CHANNEL_INDICES == (3, 2, 1)
    assert selected_bands == RGB_BAND_ORDER
    assert selected_bands == ("B04", "B03", "B02")