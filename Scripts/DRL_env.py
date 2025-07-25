import gymnasium as gym
from gymnasium import spaces
import numpy as np
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from commands import Command, CommandType

class AirTrafficControlEnv(gym.Env):
    """Custom Gymnasium environment for tower control using simulator."""

    def __init__(self, max_planes=10, max_ticks=5000):
        super().__init__()
        self.max_planes = max_planes
        self.max_ticks = max_ticks

        # Sim setup
        self.manager = PlaneManager()
        self.sim = FlightSimulator(plane_manager=self.manager)
        self.current_tick = 0

        # === Spaces ===
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim(),), dtype=np.float32)

        # Action = (command_id, plane_id, argument)
        self.action_space = spaces.Dict({
            "command": spaces.Discrete(len(CommandType)),
            "plane_id": spaces.Discrete(max_planes),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = 0

        # Create a new flight simulator class
        self.manager = PlaneManager()
        self.sim = FlightSimulator(plane_manager=self.manager)

        # Create starting state
        self.sim.initialize_scenario_from_real_data(...)  # placeholder for OpenSky snapshot
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # === Decode action ===
        command = Command(
            command_type=CommandType(action['command']),
            target_id=action['plane_id'],
            last_update=self.current_tick
        )
        self.sim.add_command(command)

        # Advance simulation by one tick
        self.sim.next_tick()
        self.current_tick += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        return obs, reward, done, False, info

    def _get_obs(self):
        """
        Extract current observation as a flat vector from all planes.
        Assumes padded or fixed-size format.
        """
        plane_states = self.manager.get_plane_states()

        obs = []
        for state in plane_states:
            # Encode each plane's state into a vector
            obs.append(self._encode_plane_state(state))

        while len(obs) < self.max_planes:
            obs.append(np.zeros_like(obs[0]))  # padding

        return np.concatenate(obs)

    def _encode_plane_state(self, state):
        """Convert a plane's state into a flat array (e.g., lat, lon, alt, gspd, hdg, ...)."""
        return np.array([
            state.lat, state.lon, state.alt, state.gspd, state.hdg,
            state.v_z, state.command.command_type.value
        ], dtype=np.float32)

    def _state_dim(self):
        return self.max_planes * 12  # 12 features per plane (adjust as needed)

    def _compute_reward(self):
        """Define your reward logic: +1 for successful landing, -0.01 delay, -100 for crash, etc."""
        reward = 0.0
        for plane in self.manager.planes:
            if plane.has_landed:
                reward += 1.0
            if plane.crashed:
                reward -= 100.0
            reward -= 0.001  # time penalty
        return reward

    def _check_done(self):
        if self.current_tick >= self.max_ticks:
            return True
        if self.manager.all_planes_landed_or_crashed():
            return True
        return False

