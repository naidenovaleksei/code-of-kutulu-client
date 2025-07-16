from src.game.template import (
    find_path as find_path,
    distance as distance,
    UnreachedPositionError as UnreachedPositionError,
)

if __name__ == '__main__':
    lines, start_point, finish_point = ['#####', '#...#', '#####'], (1, 1), (1, 2)
    find_path(start_point, finish_point, lines)