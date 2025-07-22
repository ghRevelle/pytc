from gymnasium.envs.registration import register

register(
    id='AirTrafficControl-v0',
    entry_point='DRL_env:AirTrafficControlEnv',
)