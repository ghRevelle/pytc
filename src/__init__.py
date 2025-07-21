from gymnasium.envs.registration import register

register(
    id='AirTrafficControl-v0',
    entry_point='air_traffic_env:AirTrafficControlEnv',
)