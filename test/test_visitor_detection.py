from ros2_reception_orchestrator.visitor_detection import DetectionObservation
from ros2_reception_orchestrator.visitor_detection import VisitorDetectionNode
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


def test_controller_does_not_trigger_for_out_of_roi_detection() -> None:
    controller = VisitorDetectionController(
        roi_x=0.0,
        roi_y=0.0,
        roi_width=0.4,
        roi_height=0.4,
        dwell_sec=0.5,
        absence_clear_sec=1.0,
        cooldown_sec=0.0,
    )
    observation = [DetectionObservation(center_x=0.8, center_y=0.8, confidence=0.95)]

    assert controller.update(0.0, observation) == []
    assert controller.update(0.6, observation) == []
    assert controller.visitor_present is False


def test_face_detect_log_is_emitted_once_until_detections_clear() -> None:
    node = VisitorDetectionNode.__new__(VisitorDetectionNode)
    node._face_detect_logged = False
    node._detector_backend = 'opencv_face_detector_yn'
    logs: list[str] = []
    node.get_logger = lambda: type('Logger', (), {'info': lambda _, message: logs.append(message)})()

    observations = [DetectionObservation(center_x=0.5, center_y=0.5, confidence=0.91)]

    node._log_detection_observations(observations)
    node._log_detection_observations(observations)

    assert len(logs) == 1
    assert 'Face detect' in logs[0]

    node._face_detect_logged = False
    node._log_detection_observations(observations)

    assert len(logs) == 2
