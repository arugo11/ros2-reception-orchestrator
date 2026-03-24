from ros2_reception_orchestrator.visitor_detection import DetectionObservation
from ros2_reception_orchestrator.visitor_detection import VisitorDetectionController


def test_controller_triggers_after_dwell_and_clears_after_absence() -> None:
    controller = VisitorDetectionController(
        roi_x=0.0,
        roi_y=0.0,
        roi_width=1.0,
        roi_height=1.0,
        dwell_sec=0.5,
        absence_clear_sec=1.0,
        cooldown_sec=0.0,
    )
    observation = [DetectionObservation(center_x=0.5, center_y=0.5, confidence=0.95)]

    assert controller.update(0.0, observation) == []
    events = controller.update(0.6, observation)
    assert [item.event_type for item in events] == ['VISITOR_TRIGGERED']
    assert controller.visitor_present is True

    events = controller.update(1.7, [])
    assert [item.event_type for item in events] == ['VISITOR_LEFT']
    assert controller.visitor_present is False
