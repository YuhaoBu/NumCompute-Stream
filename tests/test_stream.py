import numpy as np
import pytest

from numcompute.stream import StreamTrainer


class DummyStreamingModel:
    def __init__(self):
        self.was_partial_fit_called = False

    def partial_fit(self, X, y):
        self.was_partial_fit_called = True
        return self

    def predict(self, X):
        return (X[:, 0] > 0.5).astype(int)


def test_stream_trainer_fit_chunk_calls_partial_fit():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    trainer.fit_chunk(X, y)

    assert model.was_partial_fit_called is True
    assert trainer.fit_chunks_ == 1


def test_stream_trainer_score_chunk_logs_metrics():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    log = trainer.score_chunk(X, y)

    assert log["chunk"] == 1
    assert log["chunk_accuracy"] == 1.0
    assert log["chunk_error"] == 0.0
    assert log["cumulative_accuracy"] == 1.0
    assert log["memory_bytes"] > 0


def test_stream_trainer_cumulative_accuracy_multiple_chunks():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 1])

    X2 = np.array([[0.0], [1.0]])
    y2 = np.array([1, 1])

    trainer.score_chunk(X1, y1)
    log = trainer.score_chunk(X2, y2)

    assert log["chunk_accuracy"] == 0.5
    assert log["cumulative_accuracy"] == 0.75


def test_stream_trainer_fit_score_chunk():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    log = trainer.fit_score_chunk(X, y)

    assert trainer.fit_chunks_ == 1
    assert trainer.score_chunks_ == 1
    assert log["chunk_accuracy"] == 1.0


def test_stream_trainer_get_and_reset_logs():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    trainer.score_chunk(X, y)

    assert len(trainer.get_logs()) == 1

    trainer.reset_logs()

    assert trainer.get_logs() == []
    assert trainer.score_chunks_ == 0
    assert trainer.metrics.accuracy() == 0.0


def test_stream_trainer_model_without_partial_fit_raises_error():
    class BadModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    trainer = StreamTrainer(BadModel())

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    with pytest.raises(AttributeError):
        trainer.fit_chunk(X, y)


def test_stream_trainer_model_without_predict_raises_error():
    class BadModel:
        def partial_fit(self, X, y):
            return self

    trainer = StreamTrainer(BadModel())

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    with pytest.raises(AttributeError):
        trainer.score_chunk(X, y)


def test_stream_trainer_shape_mismatch_raises_error():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    X = np.array([[0.0], [1.0]])
    y = np.array([0])

    with pytest.raises(ValueError):
        trainer.fit_chunk(X, y)