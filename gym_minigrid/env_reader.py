from gym_minigrid.envs.deceptive import DeceptiveEnv
from gym_minigrid.wrappers import SimpleObsWrapper
from gym_extensions.continuous.gym_navigation_2d import StateBasedMDPNavigation2DEnv
import numpy as np

FILE_PATH = 'gym_minigrid/maps/drl/drl.GR'
MAPS_ROOT = 'gym_minigrid/maps/drl'


def read_map(number, random_start=True, terminate_at_any_goal=True, goal_name='rg', discrete=True,
             max_episode_steps=49 ** 2, dilate=False, max_speed=1, destination_tolerance_range=0.5,
             reward_type='value_table'):
    with open(FILE_PATH, 'r') as f:
        maps = f.readlines()
        map = maps[number]
        map = map.split(",")
        map_name = map[0]
        optcost = (map[1])
        num_goals = int(map[2])
        start_pos = (int(map[3]), int(map[4]))
        real_goal = (int(map[5]), int(map[6]), 'rg')
        fake_goals = [(int(map[7 + 2 * i]), int(map[8 + 2 * i]), f'fg{i + 1}') for i in range(num_goals)]
    all_names = get_all_model_names(map_num=number)
    value_tables = {name: read_q_table(map_number=number, map_name=map_name, goal_name=name, show=False) for name in
                    all_names} \
        if reward_type == 'value_table' \
        else None

    if discrete:
        return SimpleObsWrapper(DeceptiveEnv.load_from_file(fp=f'{MAPS_ROOT}/{map_name}.map',
                                                            optcost=optcost,
                                                            start_pos=start_pos,
                                                            real_goal=real_goal,
                                                            fake_goals=fake_goals,
                                                            random_start=random_start,
                                                            max_episode_steps=max_episode_steps,
                                                            terminate_at_any_goal=terminate_at_any_goal,
                                                            goal_name=goal_name,
                                                            dilate=dilate)), map_name
    else:
        return StateBasedMDPNavigation2DEnv.load_from_file(fp=f'{MAPS_ROOT}/{map_name}.map',
                                                           optcost=optcost,
                                                           start_pos=start_pos,
                                                           goal_bonus=100,
                                                           real_goal=real_goal,
                                                           fake_goals=fake_goals,
                                                           max_episode_steps=max_episode_steps,
                                                           destination_tolerance_range=destination_tolerance_range,
                                                           random_start=random_start,
                                                           terminate_at_any_goal=terminate_at_any_goal,
                                                           goal_name=goal_name,
                                                           visibility_graph=None,
                                                           value_tables=value_tables,
                                                           max_speed=max_speed,
                                                           reward_type=reward_type), '2DContinuousNav'


def read_q_table(map_number, map_name, goal_name='rg', show=False):
    q_table = np.load(f'/Users/alanlewis/PycharmProjects/DeceptiveReinforcementLearning/model_storage/value_iteration/{map_name}{map_number}-{goal_name}.npy')
    # q_table = np.load(f'/data/projects/punim1607/spinningup/value_iteration/{map_name}{map_number}-{goal_name}.npy')
    grid_size = read_grid_size(number=map_number)[0]
    value_table = [[0 for _ in range(grid_size + 1)] for _ in range(grid_size + 1)]
    for y in range(grid_size):
        for x in range(grid_size):
            state = (x, y, 0)
            value_table[y][x] = q_table[state].max()
    return value_table


def read_goals(number, include_start=False, discrete=True):
    if number == -1:
        if include_start:
            return [(1, 1, 'rg'), (47, 1, 'fg1')], 'simple', (47, 47)
        else:
            return [(1, 1, 'rg'), (47, 1, 'fg1')], 'simple'
    with open(FILE_PATH, 'r') as f:
        maps = f.readlines()
        map = maps[number]
        map = map.split(",")
        map_name = map[0]
        # optcost = float(map[1])
        num_goals = int(map[2])
        start_pos = (int(map[3]), int(map[4]))
        real_goal = (int(map[5]), int(map[6]), 'rg')
        fake_goals = [(int(map[7 + 2 * i]), int(map[8 + 2 * i]), f'fg{i + 1}') for i in range(num_goals)]
    map_name = map_name if discrete else "2DContinuousNav"
    if include_start:
        return [real_goal] + fake_goals, map_name, start_pos
    else:
        return [real_goal] + fake_goals, map_name


def read_start(number):
    if number == -1:
        return (47, 47)
    with open(FILE_PATH, 'r') as f:
        maps = f.readlines()
        map = maps[number]
        map = map.split(",")
        start_pos = (int(map[3]), int(map[4]))
        return start_pos


def read_name(number, discrete=True):
    if number == -1:
        return 'simple'
    with open(FILE_PATH, 'r') as f:
        maps = f.readlines()
        map = maps[number]
        map = map.split(",")
        map_name = map[0]
    map_name = map_name if discrete else "2DContinuousNav"
    return map_name


def read_grid_size(number):
    if number == -1:
        return 49, 49
    with open(FILE_PATH, 'r') as f:
        maps = f.readlines()
        map = maps[number]
        map = map.split(",")
        map_name = map[0]
    if map_name == "empty":
        return 49, 49
    size = int(map_name.split("_")[0])
    return size, size


def read_blocked_states(number):
    # figure out the file path of the map we need to read
    if number == -1:
        number = 1
    with open(FILE_PATH, 'r') as f:
        maps = f.readlines()
        map = maps[number]
        map = map.split(",")
        map_name = map[0]
    fp = f'{MAPS_ROOT}/{map_name}.map'

    # read the important details of the map along with all the lines
    with open(fp, 'r') as f:
        type = f.readline().split(" ")[1]
        height = int(f.readline().split(" ")[1])
        width = int(f.readline().split(" ")[1])
        f.readline()  # read the map heading
        map_array = []
        for y in range(height):
            map_array.append(f.readline())

    # determine which of the cells are blocked cells
    blocked_cells = []
    for y in range(height):
        for x in range(width):
            cell = map_array[y][x]
            if cell == 'T':
                blocked_cells.append((x, y))
    return blocked_cells


def get_all_model_names(map_num):
    if map_num in {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 25, 26, 27, 28, 33, 34, 35, 36}:
        return ['rg', 'fg1', 'fg2']
    else:
        return ['rg', 'fg1', 'fg2', 'fg3', 'fg4']
