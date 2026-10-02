"""
arbitration_battery.py
-------------------------
Combines g and rho into a single action for the battery-on-continuous-
environment experiment (Experiment 2 of the 5-month plan). Unlike
arbitration_oversight.py's HardGatePolicy (a step function, since
oversight is binary), Architecture B here uses the SAME sigmoid gate
as the ORIGINAL project's DualLoopPolicy:

    lambda(b) = sigmoid(k * (b - b_mid))

This is deliberate: battery urgency is a graded quantity here (just as
it was in the original discrete-grid experiment), so there is no
"binary vs. graded" design question to resolve the way there was for
oversight -- restoring the original project's own gate design is the
correct, direct analogue, not a new decision.
"""

import numpy as np


class DualLoopPolicy:
    """
    Architecture B: probabilistic urgency gate, IDENTICAL in form to
    the original project's DualLoopPolicy (arbitration.py). With
    probability lambda(battery_level), the executed action is rho's;
    otherwise it's g's.
    """

    def __init__(self, g_model, rho_model, k, b_mid, rng=None):
        self.g_model = g_model
        self.rho_model = rho_model
        self.k = k
        self.b_mid = b_mid
        self.rng = rng if rng is not None else np.random.default_rng()

    def _lambda(self, battery_level):
        return 1.0 / (1.0 + np.exp(-self.k * (battery_level - self.b_mid)))

    def predict(self, obs, battery_level, deterministic=True):
        """
        battery_level must be passed in un-normalized units (0..100,
        matching BATTERY_MAX in env_battery_continuous.py) -- not the
        [0,1]-normalized value inside obs. Mirrors the original
        project's DualLoopPolicy.predict() exactly.
        """
        lam = self._lambda(battery_level)
        if self.rng.random() < lam:
            action, _ = self.rho_model.predict(obs, deterministic=deterministic)
        else:
            action, _ = self.g_model.predict(obs, deterministic=deterministic)
        return action


class LearnedGatePolicy_Battery:
    """
    Architecture C: arbitration is a learned 2-action policy choosing
    between g's and rho's action at each step, instead of
    DualLoopPolicy's hardcoded sigmoid. Structurally identical to
    env_oversight.py's LearnedGatePolicy_Oversight and the original
    project's LearnedGatePolicy.
    """

    def __init__(self, g_model, rho_model, gate_model):
        self.g_model = g_model
        self.rho_model = rho_model
        self.gate_model = gate_model

    def predict(self, obs, deterministic=True):
        gate_action, _ = self.gate_model.predict(obs, deterministic=deterministic)
        gate_action = int(gate_action)
        sub_model = self.rho_model if gate_action == 1 else self.g_model
        action, _ = sub_model.predict(obs, deterministic=deterministic)
        return action


if __name__ == "__main__":
    # Quick manual smoke test -- verifies DualLoopPolicy's probabilistic
    # routing behaves sanely at battery extremes, without needing
    # trained models or SB3/torch (uses trivial stand-in "models").
    #   python arbitration_battery.py

    class _StubModel:
        def __init__(self, tag):
            self.tag = tag

        def predict(self, obs, deterministic=True):
            return self.tag, None

    g_stub = _StubModel("g_action")
    rho_stub = _StubModel("rho_action")
    dual = DualLoopPolicy(g_stub, rho_stub, k=0.2, b_mid=30.0,
                           rng=np.random.default_rng(0))

    dummy_obs = np.zeros(7, dtype=np.float32)

    # At very high battery (100), lambda should be near 1 -> almost
    # always rho, per the ORIGINAL project's exact formula direction
    # (sigmoid(k*(b - b_mid)) grows WITH battery level, not against
    # it -- verified against the original delivered arbitration.py,
    # not something changed here). At very low battery (0), lambda
    # should be near 0 -> almost always g. This may look
    # counter-intuitive (one might expect the "survival" loop to take
    # over specifically WHEN battery is low, not when it's high) --
    # flagged explicitly rather than silently accepted, since this is
    # inherited unchanged from the original project for faithful
    # replication purposes (see module-level discussion in this
    # project's conversation history), not independently re-derived or
    # re-justified here.
    n_trials = 2000
    high_battery_rho_frac = np.mean(
        [dual.predict(dummy_obs, battery_level=100.0) == "rho_action" for _ in range(n_trials)]
    )
    low_battery_rho_frac = np.mean(
        [dual.predict(dummy_obs, battery_level=0.0) == "rho_action" for _ in range(n_trials)]
    )
    print(f"Fraction routed to rho at battery=100: {high_battery_rho_frac:.3f} (expect near 1)")
    print(f"Fraction routed to rho at battery=0:   {low_battery_rho_frac:.3f} (expect near 0)")
    assert high_battery_rho_frac > 0.95, "Gate should almost always defer to rho at full battery (matches original formula direction)"
    assert low_battery_rho_frac < 0.05, "Gate should almost always defer to g at empty battery (matches original formula direction)"
    print("\nDualLoopPolicy sigmoid routing test passed -- confirms this file's gate "
          "faithfully reproduces the ORIGINAL project's exact formula direction, "
          "whatever one thinks of that direction's intuitive sense (see discussion "
          "in this project's conversation history / HANDOFF_STATUS).")
