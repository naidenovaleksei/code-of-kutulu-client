def calculate_entities(player_pos, explorers=[], wanderers=[]):
    return [
        None,
        f'EXPLORER 0 {player_pos[0]} {player_pos[1]} 0 0 0'
    ] + [
        f'EXPLORER {i + 1} {x} {y} 0 0 0'
        for i, (x, y) in enumerate(explorers)
    ] + [
        f'WANDERER {i + 1} {x} {y} 10 {w} 0'
        for i, (x, y, w) in enumerate(wanderers)
    ]
