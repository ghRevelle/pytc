import gymnasium as gym
from gymnasium import spaces
import numpy as np
from plane import Plane
from flightsim import FlightSimulator
from plane_manager import PlaneManager
from commands import Command, CommandType
from airport import Runway, Airport
from commands import *
from planestates import PlaneState
from rolling_initial_state_20250301 import *

class AirTrafficControlEnv(gym.Env):
    """Custom Gymnasium environment for tower control using simulator."""

    def __init__(self, max_planes=10, max_ticks=2000, test=False):
        super().__init__()
        self.max_planes = max_planes
        self.max_ticks = max_ticks
        self.test = test

        self.test_runways = {
        9: Runway((32.73713, -117.20433), (32.73, -117.175), 9),  # NW-SE runway
        27: Runway((32.73, -117.175), (32.73713, -117.20433), 27),  # SE-NW runway
        }

        self.test_airport = Airport(self.test_runways)

        self.all_initial_states = []
        # Load all initial states from the rolling initial state file
        for i in range(541):
            state = getattr(__import__('rolling_initial_state_20250301'), f'rolling_initial_state_{i:02d}')
            self.all_initial_states.append(state)
            for j in range(len(state)):
                if self.all_initial_states[-1][j]['state'] == 'takeoff':
                    self.all_initial_states[-1][j]['state'] = PlaneState.QUEUED
                elif self.all_initial_states[-1][j]['state'] == 'landing':
                    self.all_initial_states[-1][j]['state'] = PlaneState.AIR

        # 90/10 train/test split
        split_idx = int(0.9 * len(self.all_initial_states))
        self.train_initial_states = self.all_initial_states[:split_idx]
        self.test_initial_states = self.all_initial_states[split_idx:]
        self.rolling_initial_state = []
        self.fs = FlightSimulator(airport=self.test_airport, plane_manager=PlaneManager(), rolling_initial_state=self.rolling_initial_state, no_display=not self.test)
        self.current_tick = 0

        # === Spaces ===
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._state_dim(),), dtype=np.float32)

        # Action = (command_id, plane_id, argument)
        self.action_space = spaces.Dict({
            "command": spaces.Discrete(4),  # NONE to GO_AROUND (0 to 4)
            "plane_id": spaces.Discrete(max_planes),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = 0

        # Select initial state from train or test set
        initial_states = self.test_initial_states if self.test else self.train_initial_states
        if initial_states:
            import random
            self.rolling_initial_state = random.choice(initial_states)
        else:
            self.rolling_initial_state = []
        self.fs = FlightSimulator(airport=self.test_airport, plane_manager=PlaneManager(), rolling_initial_state=self.rolling_initial_state, no_display=not self.test)

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
        if command.command_type == CommandType.NONE:
            self.fs.no_command_executed = True
        else:
            self.fs.add_command(command)
        # Advance simulation by one tick
        self.fs.tick()
        self.current_tick += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        self.fs.no_command_executed = False  # Reset flag for next step
        return obs, reward, done, False, info

    def _get_obs(self):
        planes = self.fs.plane_manager.planes
        # Pre-allocate the full observation array
        obs = np.zeros((self.max_planes, 7), dtype=np.float32)
        
        # Fill only the slots for existing planes
        for i, plane in enumerate(planes[:self.max_planes]):
            obs[i] = self._encode_plane_state(plane)
        
        return obs.flatten()

    def _encode_plane_state(self, plane):
        """Convert a plane's state into a flat array (e.g., lat, lon, alt, gspd, hdg, ...)."""
        return np.array([
            plane.id, plane.lat, plane.lon, plane.alt, plane.gspd, plane.hdg,
            plane.command.command_type.value
        ], dtype=np.float32)

    def _state_dim(self):
        return self.max_planes * 7  # 7 features per plane (adjust as needed)

    def _compute_reward(self):
        reward = 0.0

        for plane in self.fs.plane_manager.planes:
            # Reward for successful landings
            if plane.landed_this_tick == True:
                reward += 100.0

            # Reward for successful takeoff
            if plane.tookoff_this_tick == True:
                reward += 100.0

            # Penalty for crashing
            if plane.crashed_this_tick == True and plane.close_call != True:
                plane.close_call = True
                reward -= 1000.0
                #print("Close call punishment")

        # Penalty for invalid or illegal commands
        if self.fs.invalid_command_executed:
            reward -= 2.0

        # Reward for valid command execution
        # This is to encourage the DRL to issue valid commands
        if self.fs.valid_command_executed:
            reward += 20.0

        # Penalty for abusing go-around command
        # This is to discourage the DRL from issuing go-around commands unnecessarily
        if self.fs.go_around_issued:
            reward -= 1.0

        if self.fs.no_command_executed:
            reward += 0.5  # Reward for deliberately not issuing a command

        # Small time pressure penalty per plane still in air
        for plane in self.fs.plane_manager.planes:
            if plane.state == PlaneState.QUEUED:
                reward -= 0.05

        if self._check_done():
            reward += 1.0 * (self.max_ticks - self.current_tick)  # Reward for finishing early

        return reward

    def _check_done(self):
        if self.current_tick >= self.max_ticks:
            return True
        if self.fs.check_end_state():
            return True
        return False

