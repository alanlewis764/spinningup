from spinningup.gym_minigrid.minigrid import *

ACTION_TO_PATH_COST = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: np.sqrt(2),
    5: np.sqrt(2),
    6: np.sqrt(2),
    7: np.sqrt(2),
    8: 0,
}


class DeceptiveEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
            self,
            size=16,
            agent_start_pos=(14, 14),
            agent_start_dir=0,
            candidate_goals=None,
            walls=None,
            random_start=False,
            terminate_at_any_goal=True,
            goal_name='rg',
            max_episode_steps=49 ** 2,
            dilate=False
    ):
        if not random_start:
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
        else:
            self.agent_start_pos = None
            self.agent_start_dir = None

        self.candidate_goals = candidate_goals
        # self.goal_pos = (candidate_goals[0].x, candidate_goals[0].y)
        self.goal_name = goal_name
        self.goal_pos = None
        for goal in self.candidate_goals:
            if goal.name == goal_name:
                self.goal_pos = (goal.x, goal.y)
        assert self.goal_pos is not None, "There must be a true goal position"
        self.walls = walls
        super().__init__(
            grid_size=size,
            max_steps=max_episode_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            terminate_at_any_goal=terminate_at_any_goal,
            dilate=dilate
        )

    @classmethod
    def load_from_file(cls, fp, optcost, start_pos, real_goal, fake_goals, random_start=False,
                       terminate_at_any_goal=True, goal_name='rg', max_episode_steps=49 ** 2, dilate=False):
        with open(fp, 'r') as f:
            type = f.readline().split(" ")[1]
            height = int(f.readline().split(" ")[1])
            width = int(f.readline().split(" ")[1])
            f.readline()  # read the map heading
            map_array = []
            for y in range(height):
                map_array.append(f.readline())
        fake_goal_list = list(map(lambda fake_goal: FakeGoal(fake_goal[0], fake_goal[1], fake_goal[2]), fake_goals))
        candidate_goals = [Goal(real_goal[0], real_goal[1], real_goal[2])] + fake_goal_list

        walls = []
        for y in range(height):
            for x in range(width):
                cell = map_array[y][x]
                if cell == 'T':
                    walls.append((x, y))

        return cls(size=height,
                   agent_start_pos=start_pos,
                   candidate_goals=candidate_goals,
                   walls=walls,
                   random_start=random_start,
                   max_episode_steps=max_episode_steps,
                   terminate_at_any_goal=terminate_at_any_goal,
                   goal_name=goal_name,
                   dilate=dilate)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # put the all the candidate goals onto the grid
        for goal in self.candidate_goals:
            self.put_obj(goal, goal.x, goal.y)
            if self.dilate:
                self.put_obj(goal, goal.x + 1, goal.y)
                self.put_obj(goal, goal.x, goal.y + 1)
                self.put_obj(goal, goal.x + 1, goal.y + 1)

        if self.walls is not None:
            for wall in self.walls:
                x, y = wall
                self.put_obj(Wall(), x, y)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square as deceptively as possible"

    def step(self, action):
        self.step_count += 1

        reward = 100
        candidate_rewards = {goal.name: 0 for goal in self.candidate_goals}

        done = False
        self.action = action
        # Get the position in front of the agent
        next_pos = self.next_pos

        # Get the contents of the cell in front of the agent
        next_cell = self.grid.get(*next_pos)

        if self.dilate:
            inbetween_pos = next_pos - self.transition_vec
            inbetween_cell = self.grid.get(*inbetween_pos)
            if (inbetween_cell is None or inbetween_cell.can_overlap()) and (
                    next_cell is None or next_cell.can_overlap()):
                self.agent_pos = next_pos
            elif inbetween_cell is None or inbetween_cell.can_overlap():
                self.agent_pos = inbetween_pos
        else:
            if next_cell is None or next_cell.can_overlap():
                self.agent_pos = next_pos

        # Done action (not used by default)
        if self.action == self.actions.done:
            candidate_rewards = {goal.name: -np.sqrt(2) for goal in self.candidate_goals}
            # done = True
            pass
        elif self.action == self.actions.up or self.action == self.actions.down or self.action == self.actions.left or self.action == self.actions.right:
            candidate_rewards = {goal.name: -1 for goal in self.candidate_goals}
            self.normal_moves += 1
        elif self.action == self.actions.up_left or self.action == self.actions.up_right or self.action == self.actions.down_left or self.action == self.actions.down_right:
            candidate_rewards = {goal.name: -np.sqrt(2) for goal in self.candidate_goals}
            self.diag_moves += 1
        else:
            raise ValueError("Invalid move")

        if next_cell != None and next_cell.type == 'goal':
            candidate_rewards[next_cell.name] = next_cell.reward
            if self.terminate_at_any_goal or next_cell.name == self.goal_name:
                reward = self._reward()
                # we should just be able to hash into the candidate_reward set and change the reward value according to the
                # reward function
                # candidate_rewards[next_cell.name] = self._reward()
                done = True
            next_cell.reward = 0

        if self.step_count >= self.max_steps:
            reward = self._reward()
            done = True
        #
        # for candidate_goal in self.candidate_goals:
        #     candidate_rewards[candidate_goal.name] += self.candidate_reward(candidate_goal=candidate_goal,
        #                                                                     agent_pos=self.agent_pos)

        # if done:
        #     for candidate_goal in self.candidate_goals:
        #         candidate_rewards[candidate_goal.name] += (100 - self.distance(candidate_goal, self.agent_pos))

        obs = self.gen_obs()

        return obs, candidate_rewards, done, {}

    @staticmethod
    def distance(candidate_goal, agent_pos):
        return np.sqrt((candidate_goal.x - agent_pos[0]) ** 2 + (candidate_goal.y - agent_pos[1]) ** 2)

    @staticmethod
    def manhattan_distance(candidate_goal, agent_pos):
        return (candidate_goal.x - agent_pos[0]) + (candidate_goal.y - agent_pos[1])

    def candidate_reward(self, candidate_goal, agent_pos):
        distance_from_goal = self.distance(candidate_goal=candidate_goal, agent_pos=agent_pos)
        distance_from_start = self.distance(candidate_goal=candidate_goal, agent_pos=self.agent_start_pos)
        return - distance_from_goal / distance_from_start

    def reset(self):
        # reset reward
        for goal in self.candidate_goals:
            goal.reward = 100
        return super().reset()

# register(
#     id='MiniGrid-Deceptive',
#     entry_point='gym_minigrid.envs:DeceptiveEnv'
# )
