from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image

from reception_interfaces.msg import VisitorDetectionEvent
from reception_interfaces.msg import VisitorDetectionState

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


@dataclass(slots=True)
class DetectionObservation:
    center_x: float
    center_y: float
    confidence: float


@dataclass(slots=True)
class DetectionEventData:
    event_type: str
    confidence: float = 0.0
    detail: str = ''


class VisitorDetectionController:
    def __init__(
        self,
        *,
        roi_x: float,
        roi_y: float,
        roi_width: float,
        roi_height: float,
        dwell_sec: float,
        absence_clear_sec: float,
        cooldown_sec: float,
    ) -> None:
        self._roi_x = max(0.0, min(1.0, float(roi_x)))
        self._roi_y = max(0.0, min(1.0, float(roi_y)))
        self._roi_width = max(0.0, min(1.0, float(roi_width)))
        self._roi_height = max(0.0, min(1.0, float(roi_height)))
        self._dwell_sec = max(0.0, float(dwell_sec))
        self._absence_clear_sec = max(0.0, float(absence_clear_sec))
        self._cooldown_sec = max(0.0, float(cooldown_sec))

        self.visitor_present = False
        self.trigger_armed = True
        self.confidence = 0.0
        self.health_state = 'ok'
        self.last_seen_monotonic = 0.0
        self._candidate_started_monotonic: float | None = None
        self._cooldown_until_monotonic = 0.0
        self._fault_detail = ''

    def update(self, now_monotonic: float, observations: list[DetectionObservation]) -> list[DetectionEventData]:
        if self.health_state != 'ok':
            return []

        best = self._best_roi_observation(observations)
        if best is not None:
            self.last_seen_monotonic = now_monotonic
            self.confidence = float(best.confidence)
            if self.visitor_present:
                return []
            if self._candidate_started_monotonic is None:
                self._candidate_started_monotonic = now_monotonic
                return []
            if now_monotonic - self._candidate_started_monotonic < self._dwell_sec:
                return []
            if not self.trigger_armed or now_monotonic < self._cooldown_until_monotonic:
                return []
            self.visitor_present = True
            self.trigger_armed = False
            self._candidate_started_monotonic = None
            self._cooldown_until_monotonic = now_monotonic + self._cooldown_sec
            return [DetectionEventData(event_type='VISITOR_TRIGGERED', confidence=float(best.confidence))]

        self._candidate_started_monotonic = None
        if self.visitor_present:
            if self.last_seen_monotonic and now_monotonic - self.last_seen_monotonic >= self._absence_clear_sec:
                self.visitor_present = False
                self.confidence = 0.0
                if now_monotonic >= self._cooldown_until_monotonic:
                    self.trigger_armed = True
                return [DetectionEventData(event_type='VISITOR_LEFT')]
            return []

        self.confidence = 0.0
        if now_monotonic >= self._cooldown_until_monotonic:
            self.trigger_armed = True
        return []

    def set_fault(self, detail: str) -> list[DetectionEventData]:
        detail_text = str(detail or 'camera_fault').strip() or 'camera_fault'
        if self.health_state == 'fault' and self._fault_detail == detail_text:
            return []
        self.health_state = 'fault'
        self._fault_detail = detail_text
        self.visitor_present = False
        self.trigger_armed = False
        self.confidence = 0.0
        self._candidate_started_monotonic = None
        return [DetectionEventData(event_type='CAMERA_FAULT', detail=detail_text)]

    def clear_fault(self, now_monotonic: float) -> list[DetectionEventData]:
        if self.health_state != 'fault':
            return []
        self.health_state = 'ok'
        self._fault_detail = ''
        if now_monotonic >= self._cooldown_until_monotonic:
            self.trigger_armed = True
        return [DetectionEventData(event_type='CAMERA_RECOVERED')]

    def _best_roi_observation(self, observations: list[DetectionObservation]) -> DetectionObservation | None:
        best: DetectionObservation | None = None
        for item in observations:
            if not self._in_roi(item.center_x, item.center_y):
                continue
            if best is None or item.confidence > best.confidence:
                best = item
        return best

    def _in_roi(self, center_x: float, center_y: float) -> bool:
        x2 = self._roi_x + self._roi_width
        y2 = self._roi_y + self._roi_height
        return self._roi_x <= center_x <= x2 and self._roi_y <= center_y <= y2


class VisitorDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__('visitor_detection_node')
        self._declare_parameters()
        self._load_parameters()

        self._controller = VisitorDetectionController(
            roi_x=self._roi_x,
            roi_y=self._roi_y,
            roi_width=self._roi_width,
            roi_height=self._roi_height,
            dwell_sec=self._dwell_ms / 1000.0,
            absence_clear_sec=self._absence_clear_ms / 1000.0,
            cooldown_sec=self._cooldown_sec,
        )

        self._detector_lock = threading.Lock()
        self._last_frame_monotonic = 0.0
        self._started_monotonic = time.monotonic()
        self._last_seen_stamp = self.get_clock().now().to_msg()
        self._last_state_fingerprint = ''

        self._state_publisher = self.create_publisher(
            VisitorDetectionState,
            self._state_topic,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._event_publisher = self.create_publisher(
            VisitorDetectionEvent,
            self._event_topic,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        self._subscription = self.create_subscription(
            Image,
            self._camera_image_topic,
            self._on_image,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        self._timer = self.create_timer(0.2, self._on_timer)

        self._sync_fault_state(detail=self._startup_fault_detail())
        self._publish_state(force=True)
        self.get_logger().info('visitor_detection_node ready')

    def _declare_parameters(self) -> None:
        self.declare_parameter('camera_image_topic', '/camera/image_raw')
        self.declare_parameter('detector_model_path', '')
        self.declare_parameter('detector_backend', 'opencv_haar_upperbody')
        self.declare_parameter('state_topic', '/visitor_detection/state')
        self.declare_parameter('event_topic', '/visitor_detection/events')
        self.declare_parameter('min_confidence', 0.8)
        self.declare_parameter('dwell_ms', 500)
        self.declare_parameter('absence_clear_ms', 2000)
        self.declare_parameter('cooldown_sec', 8.0)
        self.declare_parameter('frame_stale_sec', 2.0)
        self.declare_parameter('roi.x', 0.0)
        self.declare_parameter('roi.y', 0.0)
        self.declare_parameter('roi.width', 1.0)
        self.declare_parameter('roi.height', 1.0)

    def _load_parameters(self) -> None:
        self._camera_image_topic = str(self.get_parameter('camera_image_topic').value)
        self._detector_model_path = str(self.get_parameter('detector_model_path').value).strip()
        self._detector_backend = str(self.get_parameter('detector_backend').value).strip() or 'opencv_face_detector_yn'
        self._state_topic = str(self.get_parameter('state_topic').value)
        self._event_topic = str(self.get_parameter('event_topic').value)
        self._min_confidence = float(self.get_parameter('min_confidence').value)
        self._dwell_ms = int(self.get_parameter('dwell_ms').value)
        self._absence_clear_ms = int(self.get_parameter('absence_clear_ms').value)
        self._cooldown_sec = float(self.get_parameter('cooldown_sec').value)
        self._frame_stale_sec = float(self.get_parameter('frame_stale_sec').value)
        self._roi_x = float(self.get_parameter('roi.x').value)
        self._roi_y = float(self.get_parameter('roi.y').value)
        self._roi_width = float(self.get_parameter('roi.width').value)
        self._roi_height = float(self.get_parameter('roi.height').value)

    def _on_image(self, msg: Image) -> None:
        now_monotonic = time.monotonic()
        self._last_frame_monotonic = now_monotonic

        startup_fault = self._startup_fault_detail()
        if startup_fault:
            self._sync_fault_state(detail=startup_fault)
            self._publish_state()
            return

        try:
            image = self._image_to_bgr(msg)
            detections = self._detect_faces(image)
        except Exception as exc:  # noqa: BLE001
            self._sync_fault_state(detail=f'detector_runtime_error:{exc}')
            self._publish_state()
            return

        self._sync_fault_state(detail='')
        if detections:
            self._last_seen_stamp = self.get_clock().now().to_msg()
        events = self._controller.update(now_monotonic, detections)
        self._publish_events(events)
        self._publish_state()

    def _on_timer(self) -> None:
        now_monotonic = time.monotonic()
        startup_fault = self._startup_fault_detail()
        if startup_fault:
            self._sync_fault_state(detail=startup_fault)
            self._publish_state()
            return
        if now_monotonic - max(self._last_frame_monotonic, self._started_monotonic) >= self._frame_stale_sec:
            self._sync_fault_state(detail='camera_stale')
            self._publish_state()
            return
        self._sync_fault_state(detail='')
        self._publish_state()

    def _startup_fault_detail(self) -> str:
        if cv2 is None:
            return 'opencv_unavailable'
        if self._detector_backend == 'opencv_face_detector_yn':
            if not self._detector_model_path:
                return 'detector_model_path_missing'
            if not Path(self._detector_model_path).is_file():
                return 'detector_model_not_found'
        elif self._detector_backend in {'opencv_haar_frontalface', 'opencv_haar_upperbody'}:
            cascade_path = _cascade_classifier_path(self._detector_backend)
            if not cascade_path.is_file():
                return 'detector_cascade_not_found'
        else:
            return f'unsupported_detector_backend:{self._detector_backend}'
        return ''

    def _sync_fault_state(self, *, detail: str) -> None:
        now_monotonic = time.monotonic()
        if detail:
            self._publish_events(self._controller.set_fault(detail))
            return
        self._publish_events(self._controller.clear_fault(now_monotonic))

    def _publish_events(self, events: list[DetectionEventData]) -> None:
        for item in events:
            msg = VisitorDetectionEvent()
            msg.timestamp = self.get_clock().now().to_msg()
            msg.event_type = item.event_type
            msg.confidence = float(item.confidence)
            msg.detail = item.detail
            self._event_publisher.publish(msg)

    def _publish_state(self, *, force: bool = False) -> None:
        msg = VisitorDetectionState()
        msg.timestamp = self.get_clock().now().to_msg()
        msg.visitor_present = bool(self._controller.visitor_present)
        msg.trigger_armed = bool(self._controller.trigger_armed)
        msg.confidence = float(self._controller.confidence)
        msg.detector_backend = self._detector_backend
        msg.last_seen = self._last_seen_stamp
        msg.health_state = self._controller.health_state
        fingerprint = (
            f'{int(msg.visitor_present)}|{int(msg.trigger_armed)}|{msg.confidence:.3f}|'
            f'{msg.detector_backend}|{msg.health_state}|{msg.last_seen.sec}|{msg.last_seen.nanosec}'
        )
        if not force and fingerprint == self._last_state_fingerprint:
            return
        self._last_state_fingerprint = fingerprint
        self._state_publisher.publish(msg)

    def _detect_faces(self, image: np.ndarray) -> list[DetectionObservation]:
        with self._detector_lock:
            detector = self._get_detector()
            height, width = image.shape[:2]
            if self._detector_backend == 'opencv_face_detector_yn':
                detector.setInputSize((int(width), int(height)))
                _, faces = detector.detect(image)
            elif self._detector_backend in {'opencv_haar_frontalface', 'opencv_haar_upperbody'}:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            else:
                raise RuntimeError(f'unsupported detector backend: {self._detector_backend}')
        observations: list[DetectionObservation] = []
        if faces is None:
            return observations
        for row in np.asarray(faces):
            if self._detector_backend == 'opencv_face_detector_yn':
                if row.size < 15:
                    continue
                x, y, w, h = [float(value) for value in row[:4]]
                confidence = float(row[-1])
                if confidence < self._min_confidence or w <= 0.0 or h <= 0.0:
                    continue
            else:
                if row.size < 4:
                    continue
                x, y, w, h = [float(value) for value in row[:4]]
                confidence = 1.0
            observations.append(
                DetectionObservation(
                    center_x=(x + (w / 2.0)) / float(width),
                    center_y=(y + (h / 2.0)) / float(height),
                    confidence=confidence,
                )
            )
        return observations

    def _get_detector(self) -> Any:
        if cv2 is None:
            raise RuntimeError('opencv unavailable')
        if self._detector_backend == 'opencv_face_detector_yn':
            return cv2.FaceDetectorYN.create(
                self._detector_model_path,
                '',
                (320, 320),
                score_threshold=float(self._min_confidence),
            )
        if self._detector_backend in {'opencv_haar_frontalface', 'opencv_haar_upperbody'}:
            cascade_path = str(_cascade_classifier_path(self._detector_backend))
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                raise RuntimeError(f'failed to load cascade classifier: {cascade_path}')
            return detector
        raise RuntimeError(f'unsupported detector backend: {self._detector_backend}')

    @staticmethod
    def _image_to_bgr(msg: Image) -> np.ndarray:
        encoding = str(msg.encoding or '').lower()
        if not msg.width or not msg.height or not msg.step:
            raise ValueError('invalid image shape')
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if raw.size != int(msg.height) * int(msg.step):
            raise ValueError('image data size mismatch')
        rows = raw.reshape((int(msg.height), int(msg.step)))
        if encoding == 'bgr8':
            return rows[:, : int(msg.width) * 3].reshape((int(msg.height), int(msg.width), 3))
        if encoding == 'rgb8':
            rgb = rows[:, : int(msg.width) * 3].reshape((int(msg.height), int(msg.width), 3))
            return rgb[:, :, ::-1]
        if encoding == 'mono8':
            mono = rows[:, : int(msg.width)].reshape((int(msg.height), int(msg.width)))
            return np.repeat(mono[:, :, np.newaxis], 3, axis=2)
        if encoding == 'bgra8':
            bgra = rows[:, : int(msg.width) * 4].reshape((int(msg.height), int(msg.width), 4))
            return bgra[:, :, :3]
        if encoding == 'rgba8':
            rgba = rows[:, : int(msg.width) * 4].reshape((int(msg.height), int(msg.width), 4))
            return rgba[:, :, [2, 1, 0]]
        raise ValueError(f'unsupported image encoding: {msg.encoding}')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisitorDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _cascade_classifier_path(detector_backend: str) -> Path:
    filename = 'haarcascade_upperbody.xml' if detector_backend == 'opencv_haar_upperbody' else 'haarcascade_frontalface_default.xml'
    candidates: list[Path] = []
    if cv2 is not None and hasattr(cv2, 'data'):
        candidates.append(Path(cv2.data.haarcascades) / filename)
    candidates.extend(
        [
            Path.cwd() / '.venv' / 'lib' / 'python3.12' / 'site-packages' / 'cv2' / 'data' / filename,
            Path('/workspaces/ros2-workspace-template/.venv/lib/python3.12/site-packages/cv2/data') / filename,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


if __name__ == '__main__':
    main()
