import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Scripts import plane
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from commands import Command, CommandType
from airport import Runway, Airport
from commands import *
from planestates import PlaneState

class AirTrafficControlEnv(gym.Env):
    """Custom Gymnasium environment for tower control using simulator."""

    def __init__(self, max_planes=10, max_ticks=5000):
        super().__init__()
        self.max_planes = max_planes
        self.max_ticks = max_ticks

        self.test_runways = {
	    9: Runway((32.73713, -117.20433), (32.73, -117.175), 9),  # NW-SE runway
	    27: Runway((32.73, -117.175), (32.73713, -117.20433), 27),  # SE-NW runway
        }

        self.test_airport = Airport(self.test_runways)

        # Create a new flight simulator class
        self.rolling_initial_state = [] # TODO: implement a way to load initial states
        self.fs = FlightSimulator(airport=self.test_airport, plane_manager=PlaneManager(), rolling_initial_state=self.rolling_initial_state)
        self.current_tick = 0

        # === Spaces ===
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim(),), dtype=np.float32)

        # Action = (command_id, plane_id, argument)
        self.action_space = spaces.Dict({
            "command": spaces.Discrete(6),  # NONE to TURN (0 to 5)
            "plane_id": spaces.Discrete(max_planes),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = 0

        # Create a new flight simulator class
        self.rolling_initial_state = []
        self.fs = FlightSimulator(airport=self.test_airport, plane_manager=PlaneManager(), rolling_initial_state=self.rolling_initial_state)

        # Create starting state
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # === Decode action ===
        command = Command(
            command_type=CommandType(action['command']),
            target_id=action['plane_id'],
            last_update=self.current_tick + 1
        )
        self.fs.add_command(command)

        # Advance simulation by one tick
        self.fs.tick()
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
        planes = self.fs.plane_manager.planes

        obs = []
        for plane in planes:
            # Encode each plane's state into a vector
            obs.append(self._encode_plane_state(plane))

        while len(obs) < self.max_planes:
            obs.append(np.zeros_like(obs[0]))  # padding

        return np.concatenate(obs)

    def _encode_plane_state(self, plane):
        """Convert a plane's state into a flat array (e.g., lat, lon, alt, gspd, hdg, ...)."""
        return np.array([
            plane.id, plane.lat, plane.lon, plane.alt, plane.gspd, plane.hdg,
            plane.v_z, plane.command.command_type.value
        ], dtype=np.float32)

    def _state_dim(self):
        return self.max_planes * 8  # 8 features per plane (adjust as needed)

    def _compute_reward(self):
        reward = 0.0

        for plane in self.fs.plane_manager.planes:
			# Reward for successful landings
            if plane.landed_this_tick == True:
                reward += 50.0

			# Reward for successful takeoff
            if plane.tookoff_this_tick == True:
                reward += 50.0

			# Penalty for crashing
            if plane.crashed_this_tick == True:
                reward -= 1000.0

    	# Penalty for invalid or illegal commands
        if self.fs.invalid_command_executed:
            reward -= 10.0

		# Reward for valid command execution
		# This is to encourage the DRL to issue valid commands
        if self.fs.valid_command_executed:
            reward += 5.0

    	# Small time pressure penalty per plane still in air
        reward -= 0.01 * len(self.fs.plane_manager.planes)

        return reward

    def _check_done(self):
        if self.current_tick >= self.max_ticks:
            return True
        if self.fs.plane_manager.all_planes_landed_or_crashed():
            # TODO: Implement a way to check if all planes are done
            return True
        return False

