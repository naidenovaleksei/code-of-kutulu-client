import pytest
from src.envs.distance import find_path

@pytest.fixture
def sample_map():
    # (7, 3)
    return [
        #0123456
        '#######', # 0
        '#.....#', # 1
        '#######', # 2
    ]

@pytest.fixture
def normal_map():
    # (15, 14)
    return [
        #012345678901234
        '###############', # 0
        '#.............#', # 1
        '#.#.#.#.#.#.#.#', # 2
        '#..w.......w..#', # 3
        '#.#.#.#.#.#.#.#', # 4
        '#.............#', # 5
        '#.#.#.#.#.#.#.#', # 6
        '#.............#', # 7
        '#.#.#.#.#.#.#.#', # 8
        '#..w.......w..#', # 9
        '#.#.#.#.#.#.#.#', # 0
        '#.............#', # 1
        '###############', # 2
    ]
 
@pytest.fixture
def big_map():
    # (19, 19)
    return [
        #0123456789012345678
        '#########.#########',
        '#...#...#.#...#...#',
        '#.#...#.#.#.#...#.#',
        '#.##.##.###.##.##.#',
        '#.................#',
        '#.##.##.#.#.##.##.#',
        '#.##.#..#.#..#.##.#',
        '#......##.##......#',
        '####.###w.w###.####',
        '...#.....#.....#...',
        '####.###w.w###.####',
        '#......##.##......#',
        '#.##.#..#.#..#.##.#',
        '#.##.##.#.#.##.##.#',
        '#.................#',
        '#.##.##.###.##.##.#',
        '#.#...#.#.#.#...#.#',
        '#...#...#.#...#...#',
        '#########.#########']

@pytest.mark.parametrize("start_point, finish_point, expected", [
    [(1, 1), (1, 1), []],
    [(1, 1), (2, 1), [(2, 1)]],
    [(1, 1), (4, 1), [(4, 1), (3, 1), (2, 1)]],
])
def test_find_path_sample(start_point, finish_point, expected, sample_map):
    result = find_path(start_point, finish_point, sample_map)
    # assert len(result) == len(expected)
    assert result == expected

@pytest.mark.parametrize("start_point, finish_point, expected", [
    [(1, 1), (13, 11), [
        (13, 11),
        (13, 10),
        (13, 9),
        (13, 8),
        (13, 7),
        (13, 6),
        (13, 5),
        (13, 4),
        (13, 3),
        (13, 2),
        (13, 1),(12, 1),(11, 1),(10, 1),(9, 1),(8, 1),(7, 1),(6, 1),(5, 1),(4, 1),(3, 1),(2, 1),
    ]]
])
def test_find_path_normal(start_point, finish_point, expected, normal_map):
    result = find_path(start_point, finish_point, normal_map)
    assert len(result) == len(expected)
    assert result == expected

@pytest.mark.parametrize("start_point, finish_point, expected", [
    [(17, 2), (7, 2), [(7, 2), (7, 3), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (12, 4), (13, 4), (14, 4), (15, 4), (16, 4), (17, 4), (17, 3)]],
])
def test_find_path_big(start_point, finish_point, expected, big_map):
    result = find_path((17, 2), (7, 2), [
        #0123456789012345678
        '#########.#########',
        '#...#...#.#...#...#',
        '#.#...#.#.#.#...#.#',
        '#.##.##.###.##.##.#',
        '#.................#',
        '#.##.##.#.#.##.##.#',
        '#.##.#..#.#..#.##.#',
        '#......##.##......#',
        '####.###w.w###.####',
        '...#.....#.....#...',
        '####.###w.w###.####',
        '#......##.##......#',
        '#.##.#..#.#..#.##.#',
        '#.##.##.#.#.##.##.#',
        '#.................#',
        '#.##.##.###.##.##.#',
        '#.#...#.#.#.#...#.#',
        '#...#...#.#...#...#',
        '#########.#########'])
    # assert len(result) == len(expected)
    assert result == expected

@pytest.mark.parametrize("start_point, finish_point", [
    [(1, 1), (6, 1)],
])
def test_wrong_positions(start_point, finish_point, sample_map):
    with pytest.raises(AssertionError):
        find_path(start_point, finish_point, sample_map)
