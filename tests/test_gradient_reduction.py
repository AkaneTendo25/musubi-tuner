import torch

from musubi_tuner.training.trainer_base import reduce_trainable_gradients


class ReturningReducer:
    def __init__(self):
        self.calls = []

    def reduce(self, tensor, reduction):
        assert reduction == "mean"
        self.calls.append((tensor.device, tensor.dtype))
        return tensor.clone() * 0.5


def test_reduction_uses_returned_tensor_and_preserves_gradient_buffers():
    parameters = [torch.nn.Parameter(torch.ones(2, dtype=dtype)) for dtype in (torch.float32, torch.float64)]
    for parameter in parameters:
        parameter.grad = torch.tensor([2.0, 4.0], dtype=parameter.dtype)
    old_grads = [parameter.grad for parameter in parameters]
    reducer = ReturningReducer()

    reduce_trainable_gradients(reducer, parameters)

    assert len(reducer.calls) == 2
    for parameter, old_grad in zip(parameters, old_grads):
        assert parameter.grad is old_grad
        torch.testing.assert_close(parameter.grad, torch.tensor([1.0, 2.0], dtype=parameter.dtype))


def test_reduction_rejects_sparse_gradient():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.sparse_coo_tensor([[0]], [1.0], (2,))
    try:
        reduce_trainable_gradients(ReturningReducer(), [parameter])
    except ValueError as exc:
        assert "dense gradients" in str(exc)
    else:
        raise AssertionError("sparse gradient should be rejected")
