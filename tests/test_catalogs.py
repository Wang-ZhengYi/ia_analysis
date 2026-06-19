from ia_analysis.catalogs.catalog_loader import _hdf5_chunk_sort_key


def test_hdf5_chunk_sort_is_numeric():
    files = [
        "groups_021.10.hdf5",
        "groups_021.2.hdf5",
        "groups_021.1.hdf5",
        "groups_021.0.hdf5",
    ]
    ordered = sorted(files, key=_hdf5_chunk_sort_key)
    assert ordered == [
        "groups_021.0.hdf5",
        "groups_021.1.hdf5",
        "groups_021.2.hdf5",
        "groups_021.10.hdf5",
    ]

