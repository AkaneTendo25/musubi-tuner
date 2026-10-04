import ast
from pathlib import Path


class _Loss:
    def __init__(self, value):
        self.value = value
        self.item_calls = 0

    def detach(self):
        return self

    def item(self):
        self.item_calls += 1
        return self.value


class _LossRecorder:
    def __init__(self):
        self.loss = None

    def add(self, *, epoch, step, loss):
        self.loss = loss


def _run_loss_recording(loss_metrics, fallback_loss):
    source_path = Path(__file__).parents[1] / "src/musubi_tuner/training/trainer_base.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    statements = next(
        body[index : index + 3]
        for node in ast.walk(tree)
        for body in (getattr(node, "body", None),)
        if isinstance(body, list)
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "loss_for_average" for target in statement.targets)
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(source_path), "exec")
    recorder = _LossRecorder()
    namespace = {
        "LOSS_FOR_AVERAGE_KEY": "_loss_for_average",
        "loss": fallback_loss,
        "loss_metrics": loss_metrics,
        "loss_recorder": recorder,
        "epoch": 1,
        "step": 2,
    }
    exec(code, namespace)
    return namespace["current_loss"], recorder.loss


def test_loss_recorder_uses_explicit_average_without_tensor_sync():
    loss = _Loss(9.0)

    current_loss, recorded_loss = _run_loss_recording({"_loss_for_average": 0.0}, loss)

    assert current_loss == 0.0
    assert recorded_loss == 0.0
    assert loss.item_calls == 0


def test_loss_recorder_falls_back_to_detached_loss():
    loss = _Loss(2.5)

    current_loss, recorded_loss = _run_loss_recording({}, loss)

    assert current_loss == 2.5
    assert recorded_loss == 2.5
    assert loss.item_calls == 1
