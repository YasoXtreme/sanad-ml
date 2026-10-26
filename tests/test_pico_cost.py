from sanad.eval.pico_cost import activation_bytes


def test_activation_memory_matches_hand_count():
    # 3-layer Sanad-Tiny (16, 32, 64): pool1 in (200x16) + out (100x16) = 4,800 bytes beats
    # conv1 in (200x7) + out (200x16) = 4,600
    assert activation_bytes((16, 32, 64)) == 200 * 16 + 100 * 16
    # wider first layer: pool1 in (200x48) + out (100x48) = 14,400 > conv1 (200x7 + 200x48 = 11,000)
    assert activation_bytes((48, 96, 176)) == 200 * 48 + 100 * 48
