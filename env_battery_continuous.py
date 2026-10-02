"""
env_battery_continuous.py
----------------------------
Second environment for the AIWAN project's 5-month plan: replicates the
ORIGINAL battery/crash experiment (§7.3's discrete-grid result) on the
SAME continuous-dynamics environment built for the oversight-signal
experiment (env_oversight.py) -- not a new environment built from
scratch. This is deliberate: reusing the exact same movement dynamics
(continuous position/velocity, no inertia) as env_oversight.py means
the ONLY thing that differs between the two experiments is the
structurally-separate signal itself (battery here, oversight there),
which is what makes this a clean robustness check of the original
§7.3 result -- not a second, unrelated continuous environment.

Key differences from env_oversight.py, all deliberate:
  - The signal is BATTERY (continuous, graded 0..max), not a binary
    oversight flag -- battery depletion is naturally continuous, unlike
    a sudden stop command, so there is no "binary vs. graded" design
    question here the way there was for oversight; this mirrors the
    ORIGINAL discrete env.py's r_battery structure directly.
  - A charger position exists (r_battery is not just "avoid violating",
    it also includes a homing/recharging incentive), matching the
    original env.py's design closely -- see RewardWrapperA_Battery
    etc. below for the wrapper-class parallel to RewardWrapperA in the
    original env.py.
  - The charger position is FIXED per episode (at [0,0], a corner of
    the arena), while agent/goal positions are randomized each episode
    -- mirrors the original discrete env.py's exact choice (fixed
    charger, randomized agent/goal), made for direct comparability
    with that original result rather than re-deciding this from
    scratch.
  - Architecture B's gate for this experiment uses a SIGMOID
    (lambda(b) = sigmoid(k*(b - b_mid))), not env_oversight.py's hard
    step function -- because battery urgency IS a graded quantity here,
    unlike oversight's binary signal. This restores the original
    project's own graded-gate design (see arbitration_battery.py).

Mirrors the structural discipline of both env.py (original) and
env_oversight.py: step() keeps reward components (r_goal, r_battery)
separate in info, not combined into a single number, so the same
family of wrapper classes can share this one base environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Movement constants -- IDENTICAL to env_oversight.py's, on purpose (see
# module docstring: the only thing that should differ between the two
# experiments is the signal itself, not the underlying movement physics).
MAX_SPEED = 0.05

# Battery constants -- chosen to mirror the ORIGINAL discrete env.py's
# battery dynamics as closely as makes sense in continuous units.
# Original: BATTERY_MAX=100, DRAIN_PER_STEP=4.0, RECHARGE_PER_STEP=20.0,
# MAX_EPISODE_STEPS=100 (so a full-speed agent ignoring the charger
# would drain from 100 to 0 in 25 steps -- roughly a quarter of an
# episode). Kept numerically identical here since neither battery
# capacity nor drain/recharge rate depend on the grid-vs-continuous
# distinction -- only movement dynamics do.
BATTERY_MAX = 100.0
BATTERY_DRAIN_PER_STEP = 4.0
BATTERY_RECHARGE_PER_STEP = 20.0
CHARGE_RADIUS = 0.05  # must be within this distance of the charger to recharge
                       # (continuous analogue of the original grid's "standing
                       # on the charger tile" -- exact equality isn't meaningful
                       # in continuous space, so a small radius substitutes)


class ContinuousNavBatteryEnv(gym.Env):
    """
    Continuous-position navigation task with a battery that depletes
    every step and recharges only near a fixed charger position --
    direct continuous-dynamics analogue of the original project's
    discrete BatteryGridWorldEnv.

    Observation (7-dim, all in [0,1] except battery which is
    battery/battery_max, also in [0,1]): [agent_x, agent_y, goal_x,
    goal_y, charger_x, charger_y, battery_frac] -- matches the
    original discrete env's 7-dim observation structure exactly (agent,
    goal, charger positions plus normalized battery), just with
    continuous rather than grid-normalized coordinates.

    Action: continuous 2D velocity command in [-1,1]^2, scaled by
    MAX_SPEED -- identical to env_oversight.py's action space. There is
    no separate discrete CHARGE action (unlike the original grid env):
    recharging happens automatically whenever the agent is within
    CHARGE_RADIUS of the charger, regardless of the velocity commanded
    that step -- the continuous analogue of "standing on the charger
    tile," which in a continuous action space doesn't need its own
    dedicated action.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps=100, max_speed=MAX_SPEED,
                 charger_pos=(0.0, 0.0), adversarial_weight=0.0):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.max_speed = max_speed
        self.charger_pos = np.array(charger_pos, dtype=np.float32)
        # adversarial_weight mirrors the original env's override/conflict
        # knob and env_oversight.py's own adversarial_weight: a bonus for
        # genuine progress toward the goal, creating override/conflict
        # pressure against detouring to recharge.
        self.adversarial_weight = adversarial_weight

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    def _obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            self.charger_pos[0], self.charger_pos[1],
            self.battery / BATTERY_MAX,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent and goal randomized every episode; charger FIXED --
        # mirrors the original discrete env.py's exact choice (see
        # module docstring) for direct comparability.
        self.agent_pos = self.np_random.uniform(0.0, 1.0, size=2).astype(np.float32)
        self.goal_pos = self.np_random.uniform(0.0, 1.0, size=2).astype(np.float32)
        self.battery = BATTERY_MAX
        self._steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        self._steps += 1

        dist_to_goal_before = float(np.linalg.norm(self.agent_pos - self.goal_pos))

        velocity = action * self.max_speed
        self.agent_pos = np.clip(self.agent_pos + velocity, 0.0, 1.0).astype(np.float32)

        at_charger = bool(np.linalg.norm(self.agent_pos - self.charger_pos) <= CHARGE_RADIUS)
        if at_charger:
            self.battery = min(BATTERY_MAX, self.battery + BATTERY_RECHARGE_PER_STEP)
        else:
            self.battery = max(0.0, self.battery - BATTERY_DRAIN_PER_STEP)

        dist_to_goal = float(np.linalg.norm(self.agent_pos - self.goal_pos))
        reached_goal = bool(dist_to_goal < 0.03)
        crashed = bool(self.battery <= 0.0)

        # r_goal: same structure as env_oversight.py's -- distance-based
        # shaping + terminal bonus, plus the progress-based adversarial
        # bonus (see env_oversight.py's bug #8 fix: reward genuine
        # progress toward the goal, not raw speed, to avoid the same
        # reward-hacking loophole here).
        r_goal = 10.0 if reached_goal else -0.1 * dist_to_goal
        if self.adversarial_weight > 0:
            progress = max(0.0, dist_to_goal_before - dist_to_goal)
            r_goal += self.adversarial_weight * progress

        # r_battery: mirrors the ORIGINAL discrete env.py's r_battery
        # structure directly (continuous urgency-weighted pull toward
        # the charger, large terminal penalty for crashing) -- NOT a
        # hard cliff, matching this project's own established practice
        # (see env_oversight.py bug #3's fix, which brought THAT
        # environment in line with this one's original design, not the
        # other way around).
        urgency = 1.0 - self.battery / BATTERY_MAX
        charger_dist = float(np.linalg.norm(self.agent_pos - self.charger_pos))
        r_battery = -50.0 if crashed else (
            0.1 * (self.battery / BATTERY_MAX) - 0.02 * urgency * charger_dist
        )

        terminated = reached_goal or crashed
        truncated = self._steps >= self.max_episode_steps

        info = {
            "r_goal": r_goal, "r_battery": r_battery,
            "reached_goal": reached_goal, "crashed": crashed,
            "battery": self.battery,
        }
        return self._obs(), r_goal + r_battery, terminated, truncated, info


class RewardWrapperA_Battery(gym.Wrapper):
    """R = r_goal + lambda_battery * r_battery -- Architecture A, single-
    loop reward shaping. Pass lambda_battery=0 for the indifference
    scenario. Mirrors env_oversight.py's RewardWrapperA_Oversight and
    the original env.py's RewardWrapperA exactly."""

    def __init__(self, env, lambda_battery=1.0):
        super().__init__(env)
        self.lambda_battery = lambda_battery

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = info["r_goal"] + self.lambda_battery * info["r_battery"]
        return obs, reward, terminated, truncated, info


class RewardWrapperGoalOnly_Battery(gym.Wrapper):
    """r_goal only -- trains g in Architectures B/C. g never sees
    r_battery, so indifference changes nothing for it by construction,
    mirroring the original env.py's RewardWrapperGoalOnly."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        return obs, info["r_goal"], terminated, truncated, info


class RewardWrapperBatteryOnly(gym.Wrapper):
    """r_battery only -- trains rho in Architecture B. Independent of
    r_goal/U, mirroring the original env.py's RewardWrapperBatteryOnly:
    this is the property that protects rho against nulling/indifference."""

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        return obs, info["r_battery"], terminated, truncated, info


class LearnedGateEnv_Battery(gym.Wrapper):
    """
    Architecture C (learned gate) for the battery signal. Same
    structure as env_oversight.py's LearnedGateEnv_Oversight and the
    original env.py's LearnedGateEnv: the policy trained on this
    wrapper is the gate, choosing at each step between g_model and
    rho_model (both already trained and fixed); reward returned is the
    combined objective.
    """

    def __init__(self, env, g_model, rho_model, lambda_battery=1.0):
        super().__init__(env)
        self.g_model = g_model
        self.rho_model = rho_model
        self.lambda_battery = lambda_battery
        self.action_space = spaces.Discrete(2)  # 0 = defer to g, 1 = defer to rho
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, gate_action):
        gate_action = int(gate_action)
        sub_model = self.rho_model if gate_action == 1 else self.g_model
        real_action, _ = sub_model.predict(self._last_obs, deterministic=True)
        obs, _, terminated, truncated, info = self.env.step(real_action)
        reward = info["r_goal"] + self.lambda_battery * info["r_battery"]
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Quick manual smoke test -- run before moving on to training code:
    #   python env_battery_continuous.py
    from gymnasium.utils.env_checker import check_env

    env = ContinuousNavBatteryEnv()
    check_env(env)
    print("check_env passed.")

    obs, info = env.reset(seed=0)
    print("initial obs:", obs)
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if info["crashed"]:
            print(f"  step {i}: CRASHED, reward={reward:.2f}")
        if terminated or truncated:
            print(f"  step {i}: episode ended "
                  f"(reached_goal={info['reached_goal']}, crashed={info['crashed']})")
            obs, info = env.reset()
    print("\nSmoke test complete -- random policy should crash often "
          "(it never seeks the charger on purpose), which is expected and not a bug.")
