from .branch_frontier import branch_location, parse_branches


def test_branch_location_maps_ids_to_native_deepcache_coordinates():
    assert branch_location(0) == {"block_id": 0, "layer_id": 0}
    assert branch_location(2) == {"block_id": 0, "layer_id": 2}
    assert branch_location(3) == {"block_id": 1, "layer_id": 0}
    assert branch_location(11) == {"block_id": 3, "layer_id": 2}


def test_branch_parser_preserves_requested_order():
    assert parse_branches("0,2,5,8,11") == (0, 2, 5, 8, 11)
