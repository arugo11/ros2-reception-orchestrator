from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


class CameraImagePublisher(Node):
    def __init__(self) -> None:
        super().__init__('camera_image_publisher')
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)

        self._camera_device = str(self.get_parameter('camera_device').value).strip() or '/dev/video0'
        self._image_topic = str(self.get_parameter('image_topic').value).strip() or '/camera/image_raw'
        self._width = int(self.get_parameter('width').value)
        self._height = int(self.get_parameter('height').value)
        self._fps = float(self.get_parameter('fps').value)
        self._publisher = self.create_publisher(Image, self._image_topic, 10)

        if cv2 is None:
            raise RuntimeError('opencv unavailable')

        self._capture, backend_name = self._open_capture(self._camera_device)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        self._capture.set(cv2.CAP_PROP_FPS, float(self._fps))

        timer_period = 1.0 / max(self._fps, 1.0)
        self._timer = self.create_timer(timer_period, self._on_timer)
        self.get_logger().info(
            f'camera_image_publisher ready device={self._camera_device} backend={backend_name} topic={self._image_topic}'
        )

    def destroy_node(self) -> bool:
        capture = getattr(self, '_capture', None)
        if capture is not None:
            capture.release()
        return super().destroy_node()

    def _on_timer(self) -> None:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.get_logger().warning('camera read failed')
            return
        now = self.get_clock().now().to_msg()
        image = Image()
        image.header.stamp = now
        image.header.frame_id = 'camera'
        image.height = int(frame.shape[0])
        image.width = int(frame.shape[1])
        image.encoding = 'bgr8'
        image.is_bigendian = False
        image.step = int(frame.shape[1] * frame.shape[2])
        image.data = frame.tobytes()
        self._publisher.publish(image)

    @staticmethod
    def _open_capture(camera_device: str):
        attempts: list[tuple[object, int | None, str]] = []
        text = str(camera_device or '').strip()
        if text:
            attempts.append((text, None, f'path:{text}'))
            if hasattr(cv2, 'CAP_V4L2'):
                attempts.append((text, cv2.CAP_V4L2, f'path:{text}:CAP_V4L2'))
        if text.startswith('/dev/video'):
            suffix = text.removeprefix('/dev/video')
            if suffix.isdigit():
                index = int(suffix)
                if hasattr(cv2, 'CAP_V4L2'):
                    attempts.append((index, cv2.CAP_V4L2, f'index:{index}:CAP_V4L2'))
                attempts.append((index, None, f'index:{index}'))
        errors: list[str] = []
        for source, backend, label in attempts:
            capture = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
            if capture.isOpened():
                return capture, label
            capture.release()
            errors.append(label)
        raise RuntimeError(
            f'failed to open camera device {camera_device}; tried {", ".join(errors) or "no candidates"}'
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
