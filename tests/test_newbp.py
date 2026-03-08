import unittest

import torch
from torch.autograd import gradcheck

from models.local_grouped_newbp import LocalGroupedZernikeNewBP


class TestLocalGroupedZernikeNewBP(unittest.TestCase):
    def _make_layer(self, implementation="native_autograd"):
        return LocalGroupedZernikeNewBP(
            implementation=implementation,
            local_joint_enabled=True,
            padding_mode="replicate",
            params={
                "special": {"alpha": 1.0, "amax": 0.35, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 4.0},
                "low": {"alpha": 1.0, "amax": 0.30, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 4.0},
                "mid": {"alpha": 1.0, "amax": 0.18, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 2.5},
                "high": {"alpha": 1.0, "amax": 0.10, "bias": 0.0, "eps": 1e-6, "gss": 1.0, "p_sat": 1.2},
            },
            learnable={"bias": True, "alpha": False, "amax": False, "gss": False, "p_sat": False},
        )

    def test_shape(self):
        layer = self._make_layer()
        x = torch.randn(2, 8, 8, 36)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (2, 8, 8, 36))

    def test_sign_preserving(self):
        layer = self._make_layer()
        x = torch.zeros(1, 4, 4, 36)
        x[..., 3] = 0.2
        x[..., 4] = -0.2
        y = layer(x)
        self.assertGreater(float(y[..., 3].mean().item()), 0.0)
        self.assertLess(float(y[..., 4].mean().item()), 0.0)

    def test_group_constraint(self):
        layer = self._make_layer()
        x = torch.full((1, 5, 5, 36), 2.0)
        y = layer(x)
        low_abs = y[..., 3:6].abs().mean().item()
        high_abs = y[..., 15:36].abs().mean().item()
        self.assertGreater(low_abs, high_abs)

    def test_locality(self):
        layer = self._make_layer()
        x = torch.zeros(1, 7, 7, 36, requires_grad=True)
        y = layer(x)
        # Probe one low-order output channel at center.
        center_scalar = y[0, 3, 3, 3]
        center_scalar.backward()
        grad_tensor = x.grad
        self.assertIsNotNone(grad_tensor)
        if grad_tensor is None:
            raise RuntimeError("Gradient is None in locality test")
        grad_map = grad_tensor[0, :, :, 3].abs()

        affected = (grad_map > 1e-10).nonzero(as_tuple=False)
        for rc in affected:
            r = int(rc[0].item())
            c = int(rc[1].item())
            self.assertTrue(abs(r - 3) <= 1 and abs(c - 3) <= 1)

    def test_native_gradcheck(self):
        layer = self._make_layer(implementation="native_autograd").double()
        x = torch.randn(1, 3, 3, 36, dtype=torch.double, requires_grad=True)

        def fn(inp):
            return layer(inp)

        self.assertTrue(gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3))

    def test_compatibility_switch(self):
        x = torch.randn(1, 4, 4, 36)
        layer_off = self._make_layer()
        layer_off.local_joint_enabled = False
        y_off = layer_off(x)
        # With local joint disabled, low/mid/high degenerate to special-only path and keep shape.
        self.assertEqual(tuple(y_off.shape), tuple(x.shape))


if __name__ == "__main__":
    unittest.main()
