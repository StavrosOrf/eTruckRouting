"""The SB3 evaluation adapter must preserve masks and episode success."""

from types import SimpleNamespace

import numpy as np

from scripts.training.train_sb3_event_driven import MaskableEvalCallback


def test_masked_evaluation_records_success_for_each_episode():
    mask = np.array([False, True])
    predictions = []

    def predict(observation, *, action_masks, deterministic):
        np.testing.assert_array_equal(action_masks, mask)
        predictions.append(deterministic)
        return np.array([1]), None

    class EvaluationEnv:
        envs = [SimpleNamespace(action_masks=lambda: mask)]

        def __init__(self):
            self.episodes = 0

        def reset(self):
            self.episodes += 1
            return np.zeros((1, 2))

        def step(self, action):
            assert action.tolist() == [1]
            return (
                np.zeros((1, 2)),
                np.array([2.0]),
                np.array([True]),
                [{"successful": self.episodes == 1}],
            )

    callback = object.__new__(MaskableEvalCallback)
    callback.eval_env = EvaluationEnv()
    callback.model = SimpleNamespace(predict=predict)
    callback.n_eval_episodes = 2
    callback.deterministic = True
    callback.render = False
    callback._is_success_buffer = []

    rewards, lengths = callback._evaluate_maskable_policy()

    assert rewards == [2.0, 2.0]
    assert lengths == [1, 1]
    assert predictions == [True, True]
    assert callback._is_success_buffer == [True, False]


def test_success_logging_accepts_sb3_field_and_ignores_nonterminal_steps():
    callback = object.__new__(MaskableEvalCallback)
    callback._is_success_buffer = []
    callback._log_success_callback({"done": False, "info": {"is_success": True}}, {})
    callback._log_success_callback({"done": True, "info": {"is_success": True}}, {})
    assert callback._is_success_buffer == [True]
