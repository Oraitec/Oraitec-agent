import io
import math
import queue
import random
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import librosa
import networkx as nx
import numpy as np
import pyaudio
import pygatt
import pyttsx3
import speech_recognition as sr
import torch
from collections import Counter
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DEFAULT_YOLO_MODEL_PATH = MODELS_DIR / "yolo11n.pt"

try:
    import clip
    _CLIP_OK = True
except Exception:
    clip = None
    _CLIP_OK = False


class VideoStream:
    def __init__(self, fps: int = 20, resolution_scale: float = 1.0, use_grayscale: bool = False):
        self.src = 0
        self.fps = max(int(fps), 1)
        self.frame_interval = 1.0 / float(self.fps)
        self.use_grayscale = use_grayscale
        self.scale_factor = resolution_scale

        self.video_capture = cv2.VideoCapture(self.src)
        if not self.video_capture.isOpened():
            raise ValueError("Could not open video device")

        # Reduce camera buffer latency when possible
        try:
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video_capture.set(cv2.CAP_PROP_FPS, self.fps)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        self._apply_resolution_scale()
        self.frame_queue = queue.Queue(maxsize=100)
        self.stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _apply_resolution_scale(self) -> None:
        original_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width = int(original_width * self.scale_factor)
        new_height = int(original_height * self.scale_factor)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    def start(self) -> "VideoStream":
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(self.src)
            if not self.video_capture.isOpened():
                raise ValueError("Could not open video device")
            self._apply_resolution_scale()
        if self._thread and self._thread.is_alive():
            return self
        self.stopped.clear()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        while not self.stopped.is_set():
            if self.video_capture is None:
                break
            grabbed, frame = self.video_capture.read()
            if not grabbed:
                time.sleep(self.frame_interval)
                continue
            if self.use_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = time.time()
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait((timestamp, frame))
            except queue.Full:
                pass
            time.sleep(self.frame_interval)

    def read(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self) -> None:
        self.stopped.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self._thread = None
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class HRVStream:
    HRV_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

    def __init__(self, ble_address: str = None):
        self.address = ble_address
        self.sensor_data_queue: queue.Queue[Tuple[float, Dict[str, Optional[float]]]] = queue.Queue(maxsize=200)
        self.stopped = threading.Event()
        self.adapter = None
        self.device = None
        self.sudden_change_triggered = threading.Event()
        # Persistent high-HR counter (stabilizes 4-of-5 above-threshold rule)
        self._sudden_counter = 0
        self._sudden_range = (120.0, 140.0)

        if self.address:
            self.adapter = pygatt.GATTToolBackend()
            threading.Thread(target=self.collect_sensor_data_ble, daemon=True).start()
        else:
            threading.Thread(target=self.dummy_stream, daemon=True).start()

    def _push_data(self, timestamp: float, payload: Dict[str, Optional[float]]) -> None:
        try:
            self.sensor_data_queue.put_nowait((timestamp, payload))
        except queue.Full:
            try:
                self.sensor_data_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.sensor_data_queue.put_nowait((timestamp, payload))
            except queue.Full:
                pass

    def collect_sensor_data_ble(self) -> None:
        device = None
        try:
            self.adapter.start()
            device = self.adapter.connect(self.address)
            device.subscribe(self.HRV_UUID, callback=self.handle_data)
            while not self.stopped.is_set():
                time.sleep(0.1)
        except Exception as exc:
            print(f"Error in BLE communication: {exc}")
        finally:
            if device:
                try:
                    device.disconnect()
                except Exception as exc:
                    print(f"Error disconnecting device: {exc}")
            if self.adapter:
                try:
                    self.adapter.stop()
                except Exception as exc:
                    print(f"Error stopping BLE adapter: {exc}")

    def handle_data(self, handle: int, value: bytes) -> None:
        try:
            if not value:
                return
            flags = value[0]
            idx = 1

            hr_16bit = flags & 0x01
            if hr_16bit:
                hr = int.from_bytes(value[idx:idx + 2], "little")
                idx += 2
            else:
                hr = value[idx]
                idx += 1

            rr_present = flags & 0x10
            rr_list = []
            if rr_present:
                while idx + 1 < len(value):
                    rr = int.from_bytes(value[idx:idx + 2], "little") / 1024.0
                    rr_list.append(rr)
                    idx += 2

            rmssd = None
            if len(rr_list) >= 2:
                rr_arr = np.array(rr_list, dtype=float)
                diffs = np.diff(rr_arr)
                rmssd = float(np.sqrt(np.mean(diffs ** 2)))

            timestamp = time.time()
            self._push_data(timestamp, {"hr": float(hr), "rmssd": rmssd})
        except Exception as exc:
            print(f"Error processing HR data: {exc}")


    def dummy_stream(self) -> None:
        while not self.stopped.is_set():
            hr = random.uniform(60, 85)  # 基础心率
            # 若处于“突发维持”阶段，则连续输出高心率样本
            if self._sudden_counter > 0:
                hr = random.uniform(*self._sudden_range)
                self._sudden_counter -= 1
            elif self.sudden_change_triggered.is_set():
                # 兼容旧触发：默认约 2 秒（10 个样本，0.2s/样本）
                self._sudden_range = (120.0, 140.0)
                self._sudden_counter = 10
                self.sudden_change_triggered.clear()
            rr = 60.0 / max(hr, 1e-3)
            rr_list = [rr + random.uniform(-0.05, 0.05) for _ in range(3)]
            rmssd = None
            if len(rr_list) >= 2:
                rr_arr = np.array(rr_list, dtype=float)
                diffs = np.diff(rr_arr)
                rmssd = float(np.sqrt(np.mean(diffs ** 2)))
            self._push_data(time.time(), {"hr": float(hr), "rmssd": rmssd})
            time.sleep(0.2)

    def trigger_sudden_change(self, duration_sec: float = 2.0,
                              target_range: Tuple[float, float] = (120.0, 140.0)) -> None:
        """模拟一次短暂心率升高，持续 duration_sec 秒，用于可靠唤醒。"""
        # dummy_stream 按 0.2s/样本生成
        self._sudden_range = target_range
        self._sudden_counter = max(1, int(duration_sec / 0.2))
        self.sudden_change_triggered.set()

    def read(self) -> Tuple[Optional[float], Optional[Dict[str, Optional[float]]]]:
        try:
            return self.sensor_data_queue.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self) -> None:
        self.stopped.set()
        if self.device:
            try:
                self.device.disconnect()
            except Exception:
                pass
            self.device = None
        if self.adapter:
            try:
                self.adapter.stop()
            except Exception:
                pass
            self.adapter = None


class AudioStream:
    DEFAULT_SAMPLING_RATE = 44100
    DEFAULT_CHANNELS = 1
    DEFAULT_FPB = 2048
    DEFAULT_DEVICE = None
    DEFAULT_FORMAT = pyaudio.paFloat32

    def __init__(
        self,
        sampling_rate: int = DEFAULT_SAMPLING_RATE,
        channels: int = DEFAULT_CHANNELS,
        fpb: int = DEFAULT_FPB,
        device: Optional[int] = DEFAULT_DEVICE,
        format: int = DEFAULT_FORMAT,
    ):
        self.sampling_rate = sampling_rate
        self.channel = channels
        self.fpb = fpb
        self.device = device
        self.format = format
        self.stopped = threading.Event()
        self.audio_queue: queue.Queue[Tuple[float, bytes]] = queue.Queue(maxsize=400)
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()

    def start(self) -> None:
        try:
            self.stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channel,
                rate=self.sampling_rate,
                input=True,
                input_device_index=self.device,
                frames_per_buffer=self.fpb,
                stream_callback=self.callback,
            )
            self.stream.start_stream()
        except Exception as exc:
            print(f"Failed to start audio stream: {exc}")
            raise

    def callback(self, in_data, frame_count, time_info, status):
        if self.stopped.is_set():
            return (None, pyaudio.paComplete)
        timestamp = time.time()
        if self.audio_queue.full():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.audio_queue.put_nowait((timestamp, in_data))
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def read(self) -> Tuple[Optional[float], Optional[bytes]]:
        try:
            timestamp, audio_data = self.audio_queue.get_nowait()
        except queue.Empty:
            return None, None
        samples = np.frombuffer(audio_data, dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        pcm16 = (samples * 32767).astype("<i2").tobytes()
        return timestamp, pcm16

    def stop(self) -> None:
        self.stopped.set()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.pyaudio_instance.terminate()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()


class IOUTracker:
    def __init__(self, max_age: int = 10, iou_thresh: float = 0.3):
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        self.max_age = max_age
        self.iou_thresh = iou_thresh

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, detections):
        assigned = set()
        for tid, tr in list(self.tracks.items()):
            tr["miss"] = tr.get("miss", 0) + 1
            best, best_iou, best_j = None, 0.0, None
            for j, det in enumerate(detections):
                if j in assigned:
                    continue
                iou = self._iou(tr["bbox"], det["bbox"])
                if iou > best_iou:
                    best, best_iou, best_j = det, iou, j
            if best is not None and best_iou >= self.iou_thresh:
                self.tracks[tid].update({"bbox": best["bbox"], "cls": best["cls"], "miss": 0})
                assigned.add(best_j)

        for j, det in enumerate(detections):
            if j in assigned:
                continue
            self.tracks[self.next_id] = {"bbox": det["bbox"], "cls": det["cls"], "miss": 0}
            self.next_id += 1

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["miss"] > self.max_age:
                del self.tracks[tid]

        out = []
        for tid, tr in self.tracks.items():
            out.append({"track_id": tid, "bbox": tr["bbox"], "cls": tr["cls"]})

        return out


# Optional DeepSORT integration (silent IOU fallback)
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    _DEEPSORT_OK = True
except Exception:
    DeepSort = None  # type: ignore
    _DEEPSORT_OK = False


class DeepSORTAdapter:
    """将检测结果对接 deep_sort_realtime；未安装时不要实例化。"""

    def __init__(self, max_age: int = 30):
        assert _DEEPSORT_OK and DeepSort is not None
        self.trk = DeepSort(max_age=max_age)
        self._last_frame = None

    def update(self, detections, frame=None):
        # detections: [{"bbox":[x1,y1,x2,y2], "cls": str, "conf": float}, ...]
        # frame: 用于特征提取的原始帧（BGR格式）
        if frame is not None:
            self._last_frame = frame
        dets = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            dets.append(([x1, y1, x2, y2], d.get("conf", 1.0), d.get("cls", "object")))
        tracks = []
        for t in self.trk.update_tracks(dets, frame=self._last_frame):
            if not t.is_confirmed() or t.to_tlbr() is None:
                continue
            x1, y1, x2, y2 = map(float, t.to_tlbr())
            tracks.append({"track_id": int(t.track_id), "bbox": [x1, y1, x2, y2], "cls": getattr(t, "det_class", "object")})
        return tracks


class AffectiveSceneAnalyzer:

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if _CLIP_OK:
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
            self.prompts = {
                "valence": ["a pleasant and comfortable environment", "an unpleasant and stressful environment"],
                "arousal": ["a calm and relaxing scene", "an exciting and highly stimulating scene"],
                "dominance": ["a spacious and open area", "a confined and crowded area"],
                "naturalness": ["a natural, plant-rich outdoor scene", "an artificial, screen-dominated indoor scene"],
                "social": ["a scene with many people", "a scene with few or no people"],
                "io": ["an outdoor scene", "an indoor room"],
            }
        else:
            self.model = None
            self.preprocess = None

    def score(self, frame_bgr):
        if not _CLIP_OK or self.model is None or self.preprocess is None:
            return {
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "naturalness": 0.0,
                "social": 0.0,
                "io": 0.0,
                "scene": "unknown",
                "scene_conf": 0.0,
            }
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(img)
            out = {}
            for dim, pair in self.prompts.items():
                text = clip.tokenize(pair).to(self.device)
                txt_feat = self.model.encode_text(text)
                sim = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].detach().cpu().numpy()
                out[dim] = float(sim[0] - sim[1])
            scene_labels = [
                "office workspace",
                "meeting room",
                "open-plan office",
                "home office",
                "library",
                "outdoor park",
                "bedroom",
                "living room",
                "kitchen",
                "gym/fitness center",
                "street/sidewalk",
                "track and field",
                "outdoor sports",
            ]
            txt = clip.tokenize(scene_labels).to(self.device)
            txt_feat = self.model.encode_text(txt)
            sim = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].detach().cpu().numpy()
            idx = int(sim.argmax())
            out["scene"] = scene_labels[idx]
            out["scene_conf"] = float(sim[idx])
            io_txt = clip.tokenize(self.prompts["io"]).to(self.device)
            io_feat = self.model.encode_text(io_txt)
            io_sim = (100.0 * img_feat @ io_feat.T).softmax(dim=-1)[0].detach().cpu().numpy()
            out["io"] = float(io_sim[0] - io_sim[1])
            return out


class SceneGraphConstructor:
    def __init__(self, proximity_norm: float = 0.08, contain_iou: float = 0.6):
        self.proximity_norm = proximity_norm
        self.contain_iou = contain_iou

    @staticmethod
    def _center(bbox):
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def build(self, dets, img_w, img_h):
        G = nx.DiGraph()
        diag = (img_w ** 2 + img_h ** 2) ** 0.5
        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            area = (x2 - x1) * (y2 - y1)
            cx, cy = self._center(det["bbox"])
            G.add_node(
                i,
                cls=det["cls"],
                track_id=det.get("track_id", -1),
                bbox=det["bbox"],
                area=float(area),
                center=(float(cx), float(cy)),
            )
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                bi, bj = dets[i]["bbox"], dets[j]["bbox"]
                c1, c2 = self._center(bi), self._center(bj)
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if diag and (dist / diag) < self.proximity_norm:
                    G.add_edge(i, j, relation="near", conf=1.0 - (dist / diag) / self.proximity_norm)
                iou = self._iou(bi, bj)
                if iou > self.contain_iou:
                    if G.nodes[i]["area"] >= G.nodes[j]["area"]:
                        G.add_edge(i, j, relation="contains", conf=iou)
                    else:
                        G.add_edge(j, i, relation="contains", conf=iou)
        return G

    @staticmethod
    def extract_features(G):
        feats = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
            "person_count": len([n for n in G.nodes if G.nodes[n]["cls"] == "person"]),
        }
        if G.number_of_nodes() > 0:
            near_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "near")
            feats["crowding"] = near_edges / G.number_of_nodes()
        else:
            feats["crowding"] = 0.0
        return feats




class VisualPerceptionPipeline:
    def __init__(
        self,
        yolo_model_path: Optional[Union[str, Path]] = DEFAULT_YOLO_MODEL_PATH,
        keyframe_min_interval: float = 0.8,
        device: Optional[str] = None,
        use_stream_predict: bool = True,
        tracker_type: str = "iou",
        smoothing_alpha: float = 0.2,
        yolo_imgsz: int = 400,
    ):
        weights_path = Path(yolo_model_path) if yolo_model_path is not None else DEFAULT_YOLO_MODEL_PATH
        if not weights_path.is_absolute():
            weights_path = (MODELS_DIR / weights_path).resolve()
        self.model = YOLO(str(weights_path))
        self.yolo_imgsz = self._resolve_imgsz(yolo_imgsz)
        # 选择跟踪后端：优先 DeepSORT（若安装且配置指定），否则 IOU
        self.tracker_name = tracker_type.lower().strip()
        if self.tracker_name == "deepsort" and _DEEPSORT_OK:
            self.tracker = DeepSORTAdapter()
        else:
            if self.tracker_name in ("deepsort", "bytetrack"):
                print(f"[WARN] Tracker '{self.tracker_name}' unavailable, fallback to IOU.")
            self.tracker = IOUTracker()
        self.clip = AffectiveSceneAnalyzer(device=device)
        self.graph = SceneGraphConstructor()
        # Timestamp of the last accepted keyframe; None means not set yet
        self.last_key_ts: Optional[float] = None
        self.cache: List[Dict[str, Any]] = []
        self.keyframe_min_interval = keyframe_min_interval
        self.screen_classes = {"laptop", "tv", "cell phone", "keyboard", "mouse"}
        self.nature_classes = {"potted plant"}
        # Rolling buffers for crowd metrics
        from collections import deque

        self.crowd_series = deque(maxlen=256)  # (ts, crowding)
        self.alpha = float(max(0.0, min(1.0, smoothing_alpha)))
        self.crowd_ewm: Optional[float] = None
        # Stream inference toggle
        self.use_stream_predict = bool(use_stream_predict)
        self.sports_cues = {"sports ball", "tennis racket", "baseball bat", "frisbee", "skateboard", "bicycle", "snowboard", "surfboard"}
        self.home_cues = {"bed", "sofa", "dining table", "tv"}
        self._prev_key_gray: Optional[np.ndarray] = None

    @staticmethod
    def _resolve_imgsz(value: int, stride: int = 32) -> int:
        value = max(int(value), stride)
        q, r = divmod(value, stride)
        if r == 0:
            return value
        down = q * stride
        up = (q + 1) * stride
        if (value - down) >= (up - value):
            return up
        return down

    def _fallback_scene(self, people: float, screen_dom: float, natural_cnt: float, density: float) -> Tuple[str, float]:
        if people >= 3:
            base = "busy collaborative workspace"
        elif people >= 1.5:
            base = "small team workspace"
        elif people >= 1.0:
            base = "individual workspace"
        else:
            base = "quiet workspace"
        if screen_dom >= 0.25:
            base = "screen-focused workspace"
        elif natural_cnt >= 1:
            base = "plant-rich workspace"
        elif density >= 0.3 and people >= 1.0:
            base = "compact collaborative area"
        return base, 0.25

    def _is_keyframe(self, ts: float) -> bool:
        # First frame is always a keyframe
        if self.last_key_ts is None:
            self.last_key_ts = ts
            return True
        # Enforce minimum spacing between keyframes by timestamp
        if (ts - self.last_key_ts) > self.keyframe_min_interval:
            self.last_key_ts = ts
            return True
        return False

    def _consume_frame_and_dets(self, frame_bgr, ts: float, detections: List[Dict[str, Any]]) -> None:
        h, w = frame_bgr.shape[:2]
        # 为 DeepSORT 传递原始帧用于特征提取
        if hasattr(self.tracker, 'update') and 'frame' in self.tracker.update.__code__.co_varnames:
            tracks = self.tracker.update(detections, frame=frame_bgr)
        else:
            tracks = self.tracker.update(detections)
        if self._is_keyframe(ts):
            clip_scores = self.clip.score(frame_bgr)
            G = self.graph.build(tracks, w, h)
            graph_feats = self.graph.extract_features(G)
            # Exponential smoothing of crowding metric
            curr_crowd = float(graph_feats.get("crowding", 0.0))
            if self.crowd_ewm is None:
                self.crowd_ewm = curr_crowd
            else:
                self.crowd_ewm = self.alpha * curr_crowd + (1 - self.alpha) * self.crowd_ewm
            self.crowd_series.append((ts, curr_crowd))
            screen_area = 0.0
            for t in tracks:
                if t["cls"] in self.screen_classes:
                    x1, y1, x2, y2 = t["bbox"]
                    screen_area += (x2 - x1) * (y2 - y1)
            screen_dom = screen_area / (w * h + 1e-6)
            natural_cnt = sum(1 for t in tracks if t["cls"] in self.nature_classes)
            sports_hits = sum(1 for t in tracks if t["cls"] in self.sports_cues)
            home_hits = sum(1 for t in tracks if t["cls"] in self.home_cues)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            motion_intensity = 0.0 if self._prev_key_gray is None else float(np.mean(cv2.absdiff(gray, self._prev_key_gray)))
            self._prev_key_gray = gray
            scene_label = clip_scores.get("scene", "unknown")
            scene_conf = clip_scores.get("scene_conf", 0.0)
            if scene_label == "unknown" or scene_conf < 0.2:
                fallback_label, fallback_conf = self._fallback_scene(
                    graph_feats.get("person_count", 0.0),
                    float(screen_dom),
                    float(natural_cnt),
                    graph_feats.get("crowding", 0.0),
                )
                scene_label = fallback_label
                scene_conf = max(scene_conf, fallback_conf)
            people_ids = [t["track_id"] for t in tracks if t["cls"] == "person"]
            summary = {
                "ts": float(ts),
                "person_count": graph_feats["person_count"],
                "social_density": graph_feats["crowding"],
                "social_density_ewm": float(self.crowd_ewm if self.crowd_ewm is not None else graph_feats["crowding"]),
                "screen_dominance": float(screen_dom),
                "naturalness_index": float(natural_cnt),
                "valence": clip_scores.get("valence", 0.0),
                "arousal": clip_scores.get("arousal", 0.0),
                "scene_type": scene_label,
                "scene_conf": float(scene_conf),
                "people_ids": people_ids,
                "io_score": float(clip_scores.get("io", 0.0)),
                "sports_hits": int(sports_hits),
                "home_hits": int(home_hits),
                "motion_intensity": float(motion_intensity),
            }
            self.cache.append(summary)

    def update_with_frame(self, frame_bgr, ts: float) -> None:
        result = self.model.predict(frame_bgr, imgsz=self.yolo_imgsz, verbose=False)[0]
        boxes = result.boxes
        boxes_iter = boxes if boxes is not None else []
        detections: List[Dict[str, Any]] = []
        for box in boxes_iter:
            cls = result.names[int(box.cls)]
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            detections.append({"bbox": [x1, y1, x2, y2], "cls": cls, "conf": float(box.conf)})
        self._consume_frame_and_dets(frame_bgr, ts, detections)

    def process_stream(self, frames: List[np.ndarray], ts_list: List[float]) -> None:
        """对一批帧用 Ultralytics 的 stream=True 推理路径，降低开销。"""
        if not frames:
            return
        for res, frame_bgr, ts in zip(
            self.model.predict(source=frames, stream=True, imgsz=self.yolo_imgsz, verbose=False), frames, ts_list
        ):
            boxes = getattr(res, "boxes", None)
            boxes_iter = boxes if boxes is not None else []
            detections: List[Dict[str, Any]] = []
            for box in boxes_iter:
                cls = res.names[int(box.cls)]
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append({"bbox": [x1, y1, x2, y2], "cls": cls, "conf": float(box.conf)})
            self._consume_frame_and_dets(frame_bgr, ts, detections)

    def summarize_window(self, t0: float, t1: float):
        if not self.cache:
            return "No significant visual context.", {}
        recs = [r for r in self.cache if t0 <= r["ts"] <= t1]
        if not recs:
            recs = self.cache

        def mean(key: str) -> float:
            return float(np.mean([r[key] for r in recs]))

        def mean_or_zero(key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in recs if key in r]
            return float(np.mean(vals)) if vals else 0.0

        def maxv(key: str) -> float:
            return float(max(r[key] for r in recs))

        def mode_scene() -> str:
            scenes = [r["scene_type"] for r in recs]
            if scenes:
                cnt = Counter(scenes)
                return cnt.most_common(1)[0][0]
            return "unknown"

        uniq_ids = set()
        for r in recs:
            for pid in r.get("people_ids", []):
                uniq_ids.add(pid)
        person_seconds = 0.0
        for i in range(1, len(recs)):
            dt = max(0.0, recs[i]["ts"] - recs[i - 1]["ts"])
            person_seconds += dt * float(recs[i - 1]["person_count"])
        slope = 0.0
        if len(recs) >= 2:
            xs = np.array([r["ts"] - recs[0]["ts"] for r in recs], dtype=float)
            ys = np.array([r["social_density"] for r in recs], dtype=float)
            if not np.allclose(xs, xs[0]):
                slope = float(np.polyfit(xs, ys, 1)[0])

        sports_hits_total = int(sum(int(r.get("sports_hits", 0)) for r in recs))
        home_hits_total = int(sum(int(r.get("home_hits", 0)) for r in recs))
        feats_extra = {
            "io_score_mean": mean_or_zero("io_score"),
            "motion_intensity_mean": mean_or_zero("motion_intensity"),
            "sports_hits_total": sports_hits_total,
            "home_hits_total": home_hits_total,
            "keyframe_count": int(len(recs)),
        }

        feats = {
            "person_count_mean": mean("person_count"),
            "social_density_mean": mean("social_density"),
            "social_density_ewm_last": float(recs[-1].get("social_density_ewm", recs[-1]["social_density"])),
            "screen_dominance_mean": mean("screen_dominance"),
            "naturalness_index_mean": mean("naturalness_index"),
            "valence_mean": mean("valence"),
            "arousal_mean": mean("arousal"),
            "scene_type_mode": mode_scene(),
            "screen_dominance_peak": maxv("screen_dominance"),
            "unique_person_ids": int(len(uniq_ids)),
            "person_seconds": float(person_seconds),
            "crowding_trend_slope": slope,
        }
        feats.update(feats_extra)
        trend_symbol = "up" if feats["crowding_trend_slope"] > 0.0 else ("down" if feats["crowding_trend_slope"] < 0.0 else "flat")
        text = (
            f"In this window: scene={feats['scene_type_mode']}, "
            f"people~{feats['person_count_mean']:.1f} (uniq={feats['unique_person_ids']}), "
            f"social_density~{feats['social_density_mean']:.2f}/{feats['social_density_ewm_last']:.2f}{trend_symbol}, "
            f"screen_dominance~{feats['screen_dominance_mean']:.2f}, "
            f"naturalness~{feats['naturalness_index_mean']:.1f}, "
            f"valence~{feats['valence_mean']:.2f}, arousal~{feats['arousal_mean']:.2f}, "
            f"person_seconds~{feats['person_seconds']:.1f}."
        )
        return text, feats
    def reset_cache(self) -> None:
        self.cache.clear()
        self.crowd_series.clear()
        self._prev_key_gray = None

class CausalSceneReasoner:
    """Hybrid logistic scorer with causal attributions and counterfactuals."""

    def __init__(self):
        self.weights: Dict[str, Dict[str, float]] = {
            "bedroom": {
                "home_ratio": 2.0,
                "io_outdoor": -1.5,
                "person": -0.8,
                "crowd": -0.6,
                "speech": -0.8,
                "screen": -0.5,
                "motion": -0.4,
            },
            "outdoor sports": {
                "sports_ratio": 2.0,
                "io_outdoor": 1.2,
                "motion": 1.0,
                "crowd": 0.4,
                "person": 0.3,
                "screen": -0.2,
            },
            "office workspace": {
                "screen": 1.2,
                "io_outdoor": -0.8,
                "person": 0.3,
                "crowd": 0.2,
                "home_ratio": -0.5,
            },
            "home office": {
                "screen": 0.9,
                "home_ratio": 0.6,
                "io_outdoor": -0.7,
                "person": 0.2,
            },
            "library": {
                "speech": -1.0,
                "crowd": -0.4,
                "person": -0.3,
                "screen": -0.2,
                "io_outdoor": -0.3,
            },
            "outdoor park": {
                "io_outdoor": 1.0,
                "natural": 0.5,
                "crowd": 0.2,
                "motion": 0.2,
                "screen": -0.3,
            },
        }
        self.bias: Dict[str, float] = {
            "bedroom": -0.5,
            "outdoor sports": -0.6,
            "office workspace": -0.2,
            "home office": -0.3,
            "library": -0.4,
            "outdoor park": -0.3,
        }
        self.baseline = {
            "home_ratio": 0.0,
            "sports_ratio": 0.0,
            "io_outdoor": 0.0,
            "motion": 0.0,
            "person": 0.33,
            "crowd": 0.1,
            "screen": 0.1,
            "speech": 0.2,
            "natural": 0.2,
        }
        self.feature_text = {
            "home_ratio": "home cues dominate",
            "sports_ratio": "sports gear or activity cues",
            "io_outdoor": "outdoor visual context",
            "motion": "strong motion cues",
            "person": "people count",
            "crowd": "perceived crowding",
            "screen": "screen dominance",
            "speech": "speech activity",
            "natural": "natural elements",
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _normalize(self, feats: Dict[str, Any], audio_feats: Dict[str, Any]) -> Dict[str, float]:
        kf = max(1, int(feats.get("keyframe_count", 1)))
        home_ratio = self._clip01(float(feats.get("home_hits_total", 0)) / float(kf))
        sports_ratio = self._clip01(float(feats.get("sports_hits_total", 0)) / float(kf))
        io_outdoor = self._clip01(0.5 + 0.5 * float(feats.get("io_score_mean", 0.0)))
        motion = self._clip01(float(feats.get("motion_intensity_mean", 0.0)) / 10.0)
        person = self._clip01(float(feats.get("person_count_mean", 0.0)) / 3.0)
        crowd = self._clip01(float(feats.get("social_density_mean", 0.0)))
        screen = self._clip01(float(feats.get("screen_dominance_mean", 0.0)))
        natural = self._clip01(float(feats.get("naturalness_index_mean", 0.0)) / 3.0)
        speech = self._clip01(float(audio_feats.get("A_speech_ratio", 0.0)))
        return {
            "home_ratio": home_ratio,
            "sports_ratio": sports_ratio,
            "io_outdoor": io_outdoor,
            "motion": motion,
            "person": person,
            "crowd": crowd,
            "screen": screen,
            "natural": natural,
            "speech": speech,
        }

    def explain_feature(self, name: str) -> str:
        return self.feature_text.get(name, name)

    def predict_with_explanations(self, feats: Dict[str, Any], audio_feats: Dict[str, Any]) -> Dict[str, Any]:
        f = self._normalize(feats, audio_feats)
        best_scene: Optional[str] = None
        best_prob = -1.0
        best_contrib: Dict[str, Dict[str, float]] = {}
        best_weights: Dict[str, float] = {}
        all_scores: Dict[str, Any] = {}
        for scene, weights in self.weights.items():
            score = float(self.bias.get(scene, 0.0))
            contrib: Dict[str, Dict[str, float]] = {}
            for feat_name, w in weights.items():
                val = f.get(feat_name, 0.0)
                delta = w * val
                score += delta
                contrib[feat_name] = {"delta": delta, "value": val, "weight": w}
            prob = self._sigmoid(score)
            all_scores[scene] = {"score": score, "prob": prob, "contrib": contrib}
            if prob > best_prob:
                best_scene = scene
                best_prob = prob
                best_contrib = contrib
                best_weights = weights
        if best_scene is None:
            return {
                "scene": "unknown",
                "prob": 0.0,
                "top_causes": [],
                "counterfactuals": [],
                "all_scores": all_scores,
            }
        sorted_items = sorted(best_contrib.items(), key=lambda item: -item[1]["delta"])
        top_causes: List[Dict[str, Any]] = []
        for name, info in sorted_items:
            if info["delta"] <= 0:
                continue
            top_causes.append({
                "feature": name,
                "description": self.explain_feature(name),
                "delta": float(info["delta"]),
                "value": float(info["value"]),
                "weight": float(info["weight"]),
            })
            if len(top_causes) >= 3:
                break
        counterfactuals: List[Dict[str, Any]] = []
        for cause in top_causes:
            feat_name = cause["feature"]
            f_cf = dict(f)
            f_cf[feat_name] = self.baseline.get(feat_name, 0.0)
            score_cf = float(self.bias.get(best_scene, 0.0))
            for kk, wk in best_weights.items():
                score_cf += wk * f_cf.get(kk, 0.0)
            prob_cf = self._sigmoid(score_cf)
            counterfactuals.append({
                "feature": feat_name,
                "description": self.explain_feature(feat_name),
                "prob_after": float(prob_cf),
                "delta": float(prob_cf - best_prob),
            })
        return {
            "scene": best_scene,
            "prob": float(best_prob if best_prob >= 0 else 0.0),
            "top_causes": top_causes,
            "counterfactuals": counterfactuals,
            "all_scores": all_scores,
        }


def extract_audio_features_from_wav_bytes(wav_bytes: bytes) -> Dict[str, Any]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_reader:
        sr = wav_reader.getframerate()
        ch = wav_reader.getnchannels()
        data = wav_reader.readframes(wav_reader.getnframes())
    y = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        y = y.reshape(-1, 2).mean(axis=1)
    rms = float(librosa.feature.rms(y=y, frame_length=2048, hop_length=512).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf_ratio = float(S[freqs > 4000].mean() / (S[freqs <= 4000].mean() + 1e-9))
    zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512).mean())
    frame_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    # 提高 VAD 阈值以避免将噪音误判为语音
    # 1. 提高最小阈值从 1e-4 (0.0001) 到 0.01
    # 2. 提高自适应系数从 0.6 到 0.8
    # 3. 增加绝对能量门槛 0.005，避免极低音量被判定为语音
    vad_thresh = max(0.01, frame_rms.mean() * 0.8)
    speech_frames = (frame_rms > vad_thresh) & (frame_rms > 0.005)
    speech_ratio = float(speech_frames.mean())
    spectral_flux = np.diff(S, axis=1).clip(min=0).mean(axis=0)
    events = int((spectral_flux > spectral_flux.mean() + 2 * spectral_flux.std()).sum())
    return {
        "A_loudness_rms": rms,
        "A_spectral_centroid": centroid,
        "A_hf_ratio": hf_ratio,
        "A_zcr": zcr,
        "A_speech_ratio": speech_ratio,
        "A_transient_events": events,
        "sr": sr,
    }


class TarsAgent:
    def __init__(self, hrv_BLE_ADDRESS: str = None, batch_time: int = 10, hrv_threshold: int = 110):
        self.hrv_ble = hrv_BLE_ADDRESS
        self.batch_time = batch_time
        self.video_stream = VideoStream(fps=20, resolution_scale=1, use_grayscale=False)
        self.audio_stream = AudioStream()
        self.hrv_stream = HRVStream(self.hrv_ble)
        self.visual = VisualPerceptionPipeline(use_stream_predict=True, keyframe_min_interval=0.8, yolo_imgsz=400)
        self.causal = CausalSceneReasoner()
        self.speech_recognizer = sr.Recognizer()
        self.text_to_speech = pyttsx3.init()
        self.hrv_threshold = hrv_threshold
        self.stopped = threading.Event()
        self.wake_up_event = threading.Event()

    def monitor_hrv(self) -> None:
        history: List[float] = []
        while not self.stopped.is_set():
            timestamp, payload = self.hrv_stream.read()
            if payload is None:
                time.sleep(0.05)
                continue
            hr = payload.get("hr")
            if hr is None:
                time.sleep(0.05)
                continue
            history.append(hr)
            history = history[-5:]
            if len(history) == 5 and sum(h > self.hrv_threshold for h in history) >= 4:
                print(f"HR threshold crossed: {history[-1]:.1f}")
                self.wake_up_event.set()
            time.sleep(0.05)

    def analyze_environment(self, frames, timestamps, wav_bytes: bytes):
        if timestamps:
            t0, t1 = timestamps[0], timestamps[-1]
        else:
            t0, t1 = 0.0, 0.0
        video_text, video_feats = self.visual.summarize_window(t0, t1)
        audio_text = "No significant audio context."
        audio_feats: Dict[str, Any] = {}
        if wav_bytes:
            try:
                audio_feats = extract_audio_features_from_wav_bytes(wav_bytes)
                audio_text = (
                    f"loudness~{audio_feats['A_loudness_rms']:.4f}, "
                    f"speech~{audio_feats['A_speech_ratio']:.2f}, "
                    f"hf~{audio_feats['A_hf_ratio']:.2f}, "
                    f"events~{audio_feats['A_transient_events']}"
                )
            except Exception as exc:
                audio_text = f"Audio feature extraction failed: {exc}"
        asr_text = ""
        if wav_bytes:
            try:
                with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
                    audio = self.speech_recognizer.record(source)
                    asr_text = self.speech_recognizer.recognize_google(audio)
            except Exception:
                pass
        causal_out = self.causal.predict_with_explanations(video_feats, audio_feats)
        causal_scene = causal_out.get("scene", "unknown")
        causal_prob = causal_out.get("prob", 0.0)
        cause_bits = []
        for cause in causal_out.get("top_causes", []):
            desc = cause.get("description", cause.get("feature"))
            cause_bits.append(f"{desc} (+{cause.get('delta', 0.0):.2f})")
        cf_bits = []
        for cf in causal_out.get("counterfactuals", [])[:3]:
            desc = cf.get("description", cf.get("feature"))
            cf_bits.append(f"without {desc} -> p={cf.get('prob_after', 0.0):.2f} (delta={cf.get('delta', 0.0):.2f})")
        causal_txt = f"Scene(causal)={causal_scene}@{causal_prob:.2f}"
        if cause_bits:
            causal_txt += "; drivers: " + ", ".join(cause_bits)
        if cf_bits:
            causal_txt += " | CF: " + "; ".join(cf_bits)
        summary_text = f"[Video] {video_text}  |  [Audio] {audio_text}  |  [Causal] {causal_txt}"
        if asr_text:
            summary_text += f"  |  [ASR] {asr_text}"
        payload = {"video": video_feats, "audio": audio_feats, "asr": asr_text, "causal": causal_out}
        return summary_text, payload
    def respond(self, environment_analysis: str) -> None:
        self.text_to_speech.say("Hello, I'm Tars. " + environment_analysis)
        self.text_to_speech.runAndWait()


    def run(self) -> None:
        self.video_stream.start()
        self.audio_stream.start()
        self.hrv_stream.stopped.clear()
        threading.Thread(target=self.monitor_hrv, daemon=True).start()
        # In simulation mode trigger a synthetic spike to exercise the loop
        if self.hrv_ble is None:
            threading.Timer(1.0, lambda: self.hrv_stream.trigger_sudden_change(2.0)).start()

        while not self.stopped.is_set():
            if self.wake_up_event.wait(timeout=0.1):
                self.wake_up_event.clear()
                frames: List[np.ndarray] = []
                timestamps: List[float] = []
                pcm_frames: List[bytes] = []
                micro_frames: List[np.ndarray] = []
                micro_ts: List[float] = []
                last_flush = time.time()
                MICRO_BATCH = 4
                start_time = time.time()
                while time.time() - start_time < self.batch_time:
                    ts_frame, frame = self.video_stream.read()
                    if frame is not None:
                        ts_val = ts_frame if ts_frame is not None else time.time()
                        frames.append(frame)
                        timestamps.append(ts_val)
                        micro_frames.append(frame)
                        micro_ts.append(ts_val)
                        if len(micro_frames) >= MICRO_BATCH or (time.time() - last_flush) > 0.25:
                            if getattr(self.visual, "use_stream_predict", False):
                                self.visual.process_stream(micro_frames, micro_ts)
                            else:
                                for f, ts in zip(micro_frames, micro_ts):
                                    self.visual.update_with_frame(f, ts)
                            micro_frames, micro_ts = [], []
                            last_flush = time.time()
                    ts_audio, pcm16 = self.audio_stream.read()
                    if pcm16 is not None:
                        pcm_frames.append(pcm16)
                    time.sleep(0.01)
                if micro_frames:
                    if getattr(self.visual, "use_stream_predict", False):
                        self.visual.process_stream(micro_frames, micro_ts)
                    else:
                        for f, ts in zip(micro_frames, micro_ts):
                            self.visual.update_with_frame(f, ts)
                wav_bytes = b""
                if pcm_frames:
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as wav_writer:
                        wav_writer.setnchannels(self.audio_stream.channel)
                        wav_writer.setsampwidth(2)
                        wav_writer.setframerate(self.audio_stream.sampling_rate)
                        wav_writer.writeframes(b"".join(pcm_frames))
                    wav_bytes = buffer.getvalue()
                summary_text, payload = self.analyze_environment(frames, timestamps, wav_bytes)
                print(summary_text)
                print(payload)
                self.respond(summary_text)
                self.visual.reset_cache()
    def stop(self) -> None:
        self.stopped.set()
        self.video_stream.stop()
        self.audio_stream.stop()
        self.hrv_stream.stop()


if __name__ == "__main__":
    tars_agent = TarsAgent(hrv_BLE_ADDRESS=None, batch_time=10)
    try:
        # 直接运行；模拟模式会在 run() 内自动触发一次“突发”
        tars_agent.run()
    except KeyboardInterrupt:
        tars_agent.stop()


