"""Unified checkpoint retention: keep the last N checkpoints, epoch- or step-based.

Regression coverage for the retention overhaul: ``--save_last_n_checkpoints`` is a
checkpoint count for both cadences, the legacy ``--save_last_n_{steps,epochs}[_state]``
flags alias onto it, and states are pruned in lockstep with their checkpoints —
there is deliberately no separate state count.
"""

from types import SimpleNamespace

from musubi_tuner.utils import train_utils


def _args(**overrides):
    args = SimpleNamespace(
        save_last_n_checkpoints=None,
        save_last_n_epochs=None,
        save_last_n_epochs_state=None,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
        save_every_n_epochs=None,
        save_every_n_steps=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestIssueScenario:
    """save_every_n_steps=400 + keep 8 must keep 8 checkpoints, not only the newest."""

    def test_widely_spaced_step_checkpoints_survive(self):
        args = _args(save_last_n_steps=8, save_every_n_steps=400)

        # Before the fix this removed anything older than 8 steps: every save deleted
        # all prior checkpoints. Now step 3200 removes step 0 and keeps 800..3200.
        assert train_utils.get_remove_ckpt_no(args, 3200, 400) == 0
        assert train_utils.get_remove_ckpt_no(args, 2800, 400) is None  # nothing old enough yet

    def test_keep_window_is_count_not_steps(self):
        args = _args(save_last_n_steps=3, save_every_n_steps=400)

        for step in (400, 800, 1200):
            assert train_utils.get_remove_ckpt_no(args, 1200, 400) == 0
        assert train_utils.get_remove_ckpt_no(args, 1600, 400) == 400


class TestUnifiedOption:
    def test_new_flag_drives_step_retention(self):
        args = _args(save_last_n_checkpoints=2, save_every_n_steps=400)
        assert train_utils.get_remove_ckpt_no(args, 1600, 400) == 800
        # At 1200 the set is {400, 800, 1200}; removing 400 keeps the last 2.
        assert train_utils.get_remove_ckpt_no(args, 1200, 400) == 400
        # At 800 the formula targets step 0, whose checkpoint never existed — a
        # harmless no-op that the state saver / remove_model already tolerate.
        assert train_utils.get_remove_ckpt_no(args, 800, 400) == 0

    def test_new_flag_drives_epoch_retention(self):
        args = _args(save_last_n_checkpoints=2, save_every_n_epochs=5)
        assert train_utils.get_remove_ckpt_no(args, 11, 5) == 1
        assert train_utils.get_remove_ckpt_no(args, 6, 5) is None

    def test_epoch_and_step_cadence_share_one_formula(self):
        every, keep = 5, 3
        step_args = _args(save_last_n_checkpoints=keep, save_every_n_steps=every)
        epoch_args = _args(save_last_n_checkpoints=keep, save_every_n_epochs=every)
        for current in range(every, every * keep):
            assert train_utils.get_remove_ckpt_no(step_args, current, every) is None
            assert train_utils.get_remove_ckpt_no(epoch_args, current, every) is None
        for current in range(every * keep, every * 10):
            assert train_utils.get_remove_ckpt_no(step_args, current, every) == train_utils.get_remove_ckpt_no(
                epoch_args, current, every
            )


class TestLegacyAliases:
    def test_save_last_n_steps_aliases_count_semantics(self):
        args = _args(save_last_n_steps=4, save_every_n_steps=250)
        assert train_utils.get_remove_ckpt_no(args, 2000, 250) == 1000  # removes the 5th-newest slot

    def test_save_last_n_epochs_aliases_count_semantics(self):
        args = _args(save_last_n_epochs=4, save_every_n_epochs=2)
        assert train_utils.get_remove_ckpt_no(args, 10, 2) == 2

    def test_new_flag_wins_over_legacy_aliases(self):
        args = _args(save_last_n_checkpoints=2, save_last_n_steps=10, save_last_n_epochs=10, save_every_n_steps=100)
        assert train_utils.resolve_save_retention(args) == 2


class TestStateLockstep:
    """States and their checkpoints share one count — no separate state knob."""

    def test_states_use_the_checkpoint_count(self):
        args = _args(save_last_n_checkpoints=8, save_every_n_steps=400)
        # The state savers call the same get_remove_ckpt_no as checkpoint removal:
        # one flag drives both.
        assert train_utils.get_remove_ckpt_no(args, 3200, 400) == 0

    def test_legacy_state_flags_fold_into_the_shared_count(self):
        args = _args(save_last_n_steps_state=2, save_every_n_steps=400)
        assert train_utils.resolve_save_retention(args) == 2
        args = _args(save_last_n_epochs_state=3, save_every_n_epochs=1)
        assert train_utils.resolve_save_retention(args) == 3

    def test_new_flag_wins_over_legacy_state_flags(self):
        args = _args(save_last_n_checkpoints=6, save_last_n_steps_state=2, save_every_n_steps=400)
        assert train_utils.resolve_save_retention(args) == 6

    def test_no_retention_anywhere(self):
        args = _args()
        assert train_utils.resolve_save_retention(args) is None
        assert train_utils.get_remove_ckpt_no(args, 10**6, 100) is None


class TestStateSaverIntegration:
    def _accelerator(self, saved):
        class Accelerator:
            is_main_process = True

            def save_state(self, state_dir):
                from pathlib import Path

                Path(state_dir).mkdir(parents=True, exist_ok=True)
                saved.append(state_dir)

            def wait_for_everyone(self):
                pass

        return Accelerator()

    def test_stepwise_state_saver_applies_count_retention(self, tmp_path):
        saved = []
        args = _args(
            save_last_n_steps=2,
            save_every_n_steps=400,
            output_dir=str(tmp_path),
            output_name="run",
            save_state_to_huggingface=False,
            seed=None,
        )
        for step in (400, 800, 1200):
            train_utils.save_and_remove_state_stepwise(args, self._accelerator(saved), step, epoch=1, step_in_epoch=0)

        assert (tmp_path / "run-step00000400-state").exists() is False  # rotated out
        assert (tmp_path / "run-step00000800-state").exists()
        assert (tmp_path / "run-step00001200-state").exists()
        assert len(saved) == 3

    def test_epoch_end_state_saver_applies_count_retention(self, tmp_path):
        saved = []
        args = _args(
            save_last_n_epochs=2,
            save_every_n_epochs=1,
            output_dir=str(tmp_path),
            output_name="run",
            save_state_to_huggingface=False,
            seed=None,
        )
        for epoch in (1, 2, 3, 4):
            train_utils.save_and_remove_state_on_epoch_end(args, self._accelerator(saved), epoch)

        assert (tmp_path / "run-000001-state").exists() is False
        assert (tmp_path / "run-000002-state").exists() is False
        assert (tmp_path / "run-000003-state").exists()
        assert (tmp_path / "run-000004-state").exists()
        assert len(saved) == 4

    def test_stepwise_state_saver_can_skip_retention_for_request_saves(self, tmp_path):
        saved = []
        args = _args(
            save_last_n_steps=2,
            save_every_n_steps=400,
            output_dir=str(tmp_path),
            output_name="run",
            save_state_to_huggingface=False,
            seed=None,
        )
        # Ad-hoc save via request file at an off-cadence step: apply_retention=False must
        # leave every previously kept state in place.
        train_utils.save_and_remove_state_stepwise(
            args, self._accelerator(saved), 501, epoch=1, step_in_epoch=0, apply_retention=False
        )
        assert (tmp_path / "run-step00000501-state").exists()


class TestProjectConfigMigration:
    def test_legacy_retention_fields_fold_into_unified_option(self):
        from musubi_tuner.gui_dashboard.project_schema import ProjectConfig

        config = ProjectConfig.model_validate(
            {
                "version": 1,
                "training": {"save_last_n_steps": 5, "save_last_n_epochs_state": 2},
                "full_finetune": {"save_last_n_steps": 3},
            }
        )
        assert config.training.save_last_n_checkpoints == 5
        assert config.full_finetune.save_last_n_checkpoints == 3

    def test_unified_value_wins_over_legacy(self):
        from musubi_tuner.gui_dashboard.project_schema import ProjectConfig

        config = ProjectConfig.model_validate(
            {"version": 1, "training": {"save_last_n_steps": 5, "save_last_n_checkpoints": 9}}
        )
        assert config.training.save_last_n_checkpoints == 9
