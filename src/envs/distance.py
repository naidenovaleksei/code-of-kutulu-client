import heapq

CELL_EMPTY = '.'
CELL_WALL = '#'
CELL_SPAWN = 'w'


def distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def add(point1, point2):
    return (point1[0] + point2[0], point1[1] + point2[1])

def find_path(start_point, finish_point, lines):
    choices = ((0,1), (1,0), (0,-1), (-1,0))
    g_dict = {start_point: 0}
    f_dict = {start_point: distance(start_point, finish_point)}
    common_candidates = [(f_dict[start_point], start_point)]
    close_set = set()
    pred_point = {}

    while len(common_candidates):
        point = heapq.heappop(common_candidates)[1]
        if point == finish_point:
            path = []
            while point != start_point:
                path.append(point)
                point = pred_point[point]
            return path

        close_set.add(point)
        for choise in choices:
            candidate = add(point, choise)
            x,y = candidate
            # if lines[y][x] not in (CELL_EMPTY, CELL_SPAWN):
            if lines[y][x] == CELL_WALL:
                continue
            g = g_dict[point] + 1
            h = distance(candidate, finish_point)
            f = g + h
            if candidate in close_set and g >= g_dict.get(candidate, 0):
                continue
            pred_point[candidate] = point
            if g < g_dict.get(candidate, 0):
                g_dict[candidate] = g
            if candidate in [x[1] for x in common_candidates]:
                continue
            if candidate in f_dict:
                continue
            g_dict[candidate] = g
            f_dict[candidate] = f
            heapq.heappush(common_candidates, (f, candidate))
    assert False

if __name__ == '__main__':
    lines, start_point, finish_point = ['#####', '#...#', '#####'], (1, 1), (1, 2)
    find_path(start_point, finish_point, lines)