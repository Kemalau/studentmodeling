import argparse
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.train_recurrent_student_m4_external_state_teacher import (
    ENV_PATH,
    ENV_SHA256,
    STUDENT_ADAPTER_NAME,
    TEACHER_ADAPTER_NAME,
    attach_teacher_config,
    capture_from_external_teacher,
    external_teacher_spec,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.active_adapter = STUDENT_ADAPTER_NAME
        self._studentmodeling_external_state_teacher = TEACHER_ADAPTER_NAME
        self._studentmodeling_student_adapter = STUDENT_ADAPTER_NAME
        self.events = []

    def set_adapter(self, name, inference_mode=False):
        self.events.append(("set", name, inference_mode))
        self.active_adapter = name


class ExternalStateTeacherTest(unittest.TestCase):
    def test_spec_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"teacher"
            (root / "adapter_model.safetensors").write_bytes(payload)
            (root / "adapter_config.json").write_text("{}")
            digest = hashlib.sha256(payload).hexdigest()
            with mock.patch.dict(os.environ, {ENV_PATH: str(root), ENV_SHA256: digest}):
                self.assertEqual(external_teacher_spec(), (root, digest))

    def test_config_rejects_non_layerwise_or_direct_replay(self):
        good = dict(
            state_architecture="layerwise",
            direct_replay_weight=0.0,
            consistency_weight=0.25,
        )
        args = argparse.Namespace(**good)
        attach_teacher_config(args, Path("/teacher"), "a" * 64)
        self.assertEqual(args.external_state_teacher_scope, "direct_layerwise_trajectory_only")
        with self.assertRaisesRegex(ValueError, "layerwise"):
            attach_teacher_config(
                argparse.Namespace(**{**good, "state_architecture": "scalar"}),
                Path("/teacher"),
                "a" * 64,
            )
        with self.assertRaisesRegex(ValueError, "direct replay"):
            attach_teacher_config(
                argparse.Namespace(**{**good, "direct_replay_weight": 0.1}),
                Path("/teacher"),
                "a" * 64,
            )

    def test_capture_restores_trainable_student_even_on_error(self):
        backbone = FakeBackbone()
        backbone.train(True)
        wrapper = argparse.Namespace(backbone=backbone)

        def original(_wrapper):
            self.assertEqual(backbone.active_adapter, TEACHER_ADAPTER_NAME)
            self.assertFalse(backbone.training)
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            capture_from_external_teacher(wrapper, original)
        self.assertEqual(backbone.active_adapter, STUDENT_ADAPTER_NAME)
        self.assertTrue(backbone.training)
        self.assertEqual(
            backbone.events,
            [
                ("set", TEACHER_ADAPTER_NAME, True),
                ("set", STUDENT_ADAPTER_NAME, False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
