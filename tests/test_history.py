
# tests/test_history.py

from src.history import HistoryLogger

def test_logging_and_export():
    logger = HistoryLogger()
    logger.log("activation", {"psi": [1,0,0,0,0]}, info={"agent": "GabrielCell"})
    logger.log("decision", {"resonance": 0.97})
    assert len(logger.events) == 2
    assert logger.events[0].event_type == "activation"
    logger.clear()
    assert len(logger.events) == 0

def test_filter():
    logger = HistoryLogger()
    logger.log("feedback", {"val": 0.3})
    logger.log("feedback", {"val": 0.4})
    logger.log("decision", {"val": 1.0})
    feedback_events = logger.filter("feedback")
    assert len(feedback_events) == 2
