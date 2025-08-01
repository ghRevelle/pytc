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

    def __init__(self, max_planes=10, max_ticks=2000, test=False, record_data=False, no_display=None):
        super().__init__()
        self.max_planes = max_planes
        self.max_ticks = max_ticks
        self.test = test
        self.record_data = record_data
        # If no_display is not specified, use the opposite of test (test=True shows display by default)
        self.no_display = no_display if no_display is not None else not test

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

                # Episode tracking for debugging and data recording
        self.episode_stats = {
            'total_reward': 0.0,
            'max_reward': 0,  # Set to your target maximum
            'ending_time': 0,
            'planes_taken_off': 0,
            'planes_landed': 0,
            'planes_encountered': 10,
            'go_arounds': 0,
            'crashes': 0,
            'processed_planes': 0,
            'reward_efficiency': 0
        }

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
        self.fs = FlightSimulator(airport=self.test_airport, plane_manager=PlaneManager(), rolling_initial_state=self.rolling_initial_state, no_display=self.no_display)
        
        # Reset episode stats
        self.episode_stats = {
            'total_reward': 0.0,
            'max_reward': self.compute_max_reward(self.fs),
            'ending_time': 0,
            'planes_taken_off': 0,
            'planes_landed': 0,
            'planes_encountered': 0,
            'go_arounds': 0,
            'crashes': 0,
            'processed_planes': 0,
            'reward_efficiency': 0
        }

        # Set total planes encountered in this episode
        self.episode_stats['planes_encountered'] = 10

        # Create starting state
        observation = self._get_obs()
        info = {'action_mask': self._get_action_mask()}
        return observation, info

    def step(self, action):
        # === Decode action ===
            
        command = Command(
            command_type=CommandType(action['command']),
            target_id=action['plane_id'],
            last_update=self.current_tick + 1,
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
        
        # Add action mask to info for next step
        info['action_mask'] = self._get_action_mask()
        
        # Add episode stats to info when episode ends
        if done:
            self.episode_stats['ending_time'] = self.current_tick
            self.episode_stats['total_reward'] = self.compute_external_reward(self.episode_stats)
            self.episode_stats['reward_efficiency'] = self.episode_stats['total_reward'] / self.episode_stats['max_reward']
            info['episode_stats'] = self.episode_stats.copy()

        self.fs.no_command_executed = False  # Reset flag for next step
        return obs, reward, done, False, info

    def _get_obs(self):
        planes = self.fs.plane_manager.planes
        # Pre-allocate the full observation array
        obs = np.zeros((self.max_planes, 3), dtype=np.float32)  # 3 features per plane
        
        # Fill only the slots for existing planes
        for i, plane in enumerate(planes[:self.max_planes]):
            obs[i] = self._encode_plane_state(plane)
        
        # Flatten plane data and append current tick
        plane_data = obs.flatten()
        
        # Add current tick as a normalized feature (normalize by max_ticks)
        current_tick_normalized = self.current_tick / self.max_ticks
        
        # Combine plane data with current tick
        full_obs = np.append(plane_data, current_tick_normalized)
        
        return full_obs

    def _encode_plane_state(self, plane):
        """Convert a plane's state into a flat array with PlaneState and has_gone_around."""
        return np.array([
            plane.id,
            plane.state.value,  # PlaneState enum value (0-5)
            float(plane.has_gone_around)  # Boolean converted to float (0.0 or 1.0)
        ], dtype=np.float32)

    def _state_dim(self):
        return self.max_planes * 3 + 1  # 3 features per plane + 1 for current tick

    def _compute_reward(self):
        reward = 0.0

        # Reward for successful landing orders
        if self.fs.landing_issued:
            self.fs.landing_issued = False
            reward += 50.0  # RESTORED to original training value

        # Reward for successful takeoff orders
        if self.fs.takeoff_issued:
            self.fs.takeoff_issued = False
            reward += 25.0  # RESTORED to original training value

        # Reward for successful go-around orders
        if self.fs.go_around_issued:
            self.fs.go_around_issued = False
            reward += 1.0
            self.episode_stats['go_arounds'] += 1

        for plane in self.fs.plane_manager.planes:
            # Recording if the plane landed
            if plane.landed_this_tick and not plane.close_call:
                plane.landed_this_tick = False
                self.episode_stats['planes_landed'] += 1

            if plane.tookoff_this_tick and not plane.close_call:
                plane.tookoff_this_tick = False
                self.episode_stats['planes_taken_off'] += 1

            # Penalty for crashing
            if plane.crashed_this_tick and not plane.close_call:
                plane.close_call = True
                plane.crashed_this_tick = False
                reward -= 30.0
                self.episode_stats['crashes'] += 1
                #print("Close call punishment")

        # Penalty for invalid or illegal commands
        if self.fs.invalid_command_executed:
            self.fs.invalid_command_executed = False
            reward -= 0.1

        if self.fs.no_command_executed:
            reward += 0.05  # Reward for deliberately not issuing a command
            self.fs.no_command_executed = False

        # Small time pressure penalty per plane still in the queue
        queued_planes = 0
        for plane in self.fs.plane_manager.planes:
            if plane.state == PlaneState.QUEUED:
                reward -= 0.01
                queued_planes += 1

        self.episode_stats['max_reward'] = self.compute_max_reward(self.fs)
        self.episode_stats['processed_planes'] = (self.episode_stats['planes_landed'] + self.episode_stats['planes_taken_off']) / self.episode_stats['planes_encountered']

        return reward

    def _check_done(self):
        if self.current_tick >= self.max_ticks:
            return True
        if self.fs.check_end_state():
            return True
        return False
    
    def _get_action_mask(self):
        """Return a mask indicating which actions are valid.
        
        Returns:
            dict: Action mask with same structure as action_space
                 - 'command': always all True (all commands valid)  
                 - 'plane_id': True only for planes that can execute ANY command
                 - 'command_plane_combinations': 2D mask for valid command-plane pairs
        """
        # Start with all commands available
        command_mask = np.ones(4, dtype=bool)  # 4 command types: NONE to GO_AROUND
        
        # Create plane mask 
        plane_id_mask = np.zeros(self.max_planes, dtype=bool)
        
        # Create 2D mask for command-plane combinations (4 commands x 10 planes)
        command_plane_mask = np.zeros((4, self.max_planes), dtype=bool)
        
        # Get all planes and their states
        planes_by_id = {}
        for plane in self.fs.plane_manager.planes:
            planes_by_id[plane.id] = plane
        
        # Get the first plane in queue for LINE_UP_AND_WAIT commands
        top_of_queue = self.fs.plane_manager.airport.get_top_of_queue()
        
        for plane_id in range(self.max_planes):
            if self.fs.plane_manager.id_to_callsign[plane_id] != '':  # Plane exists
                plane = planes_by_id.get(plane_id)
                if plane and plane.state not in [PlaneState.REALIGNING, PlaneState.TAKINGOFF, 
                                               PlaneState.LANDING, PlaneState.TAXIING]:
                    
                    # NONE command (0) - valid for any existing plane
                    command_plane_mask[0, plane_id] = True
                    plane_id_mask[plane_id] = True
                    
                    # LINE_UP_AND_WAIT command (1) - ONLY the first plane in queue (if queue exists)
                    if (plane.state == PlaneState.QUEUED and 
                        top_of_queue is not None and 
                        plane.id == top_of_queue):
                        command_plane_mask[1, plane_id] = True

            # lets the model access cleared_for_landing and go_around so that it can learn
            command_plane_mask[2, plane_id] = True
            command_plane_mask[3, plane_id] = True
        
        # Ensure at least one plane can be targeted for NONE command
        if not np.any(plane_id_mask):
            for plane_id in range(self.max_planes):
                if self.fs.plane_manager.id_to_callsign[plane_id] != '':
                    plane_id_mask[plane_id] = True
                    command_plane_mask[0, plane_id] = True
                    break
        
        return {
            'command': command_mask,
            'plane_id': plane_id_mask,
            'command_plane_combinations': command_plane_mask
        }
    
    def compute_external_reward(self, episode_stats):
        """Return the external reward in this case.
        """

        takeoff_rewards = 25 * episode_stats['planes_taken_off']
        landing_rewards = 50 * episode_stats['planes_landed']
        time_rewards = 0.05 * (2000 - episode_stats['ending_time'])
        crash_punishments = 30 * episode_stats['crashes']

        return takeoff_rewards + landing_rewards + time_rewards - crash_punishments

    def compute_max_reward(self, fs : FlightSimulator):
        """Return the maximum possible reward in this case.
        """

        # assumes that there are 3 queued, 7 flying
        #return len(self.fs.plane_manager.airport.queue) * 8 + (len(self.fs.plane_manager.planes) - len(self.fs.plane_manager.airport.queue)) * 50 + 75

        takeoff_rewards = 25 * fs.planes_taking_off
        landing_rewards = 50 * fs.landing_planes
        time_rewards = 0.05 * 1000

        return takeoff_rewards + landing_rewards + time_rewards