import cv2, pyaudio, wave, librosa, random, logging, pickle, time, struct
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, pipeline
from torchvision import models, transforms
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from PIL import Image
import threading
from threading import Thread
import queue
import asyncio
from bleak import BleakClient
import pyttsx3
import pyaudio
import sounddevice as sd
import pygatt  # New import for synchronous BLE operations
from collections import Counter
import re
from typing import Tuple, Optional
import speech_recognition as sr
import io

# ---------------- > Recorder
class VideoStream:
    def __init__(self, fps: int = 2, resolution_scale: float = 1, use_grayscale: bool = False):
        """
        Initialize video stream.

        Args:
        - fps (int): Frames per second.
            - Low: 1-5 FPS (suitable for static scenes or low-motion applications)
            - Medium: 5-15 FPS (suitable for general-purpose applications)
            - High: 15-30 FPS (suitable for fast-paced or high-motion applications)
            - Very High: 30+ FPS (suitable for specialized applications, e.g., sports analysis)

        - resolution_scale (float): Scaling factor for camera resolution.
        - use_grayscale (bool): Convert frames to grayscale.
        """
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise ValueError("Could not open video device")
        
        self.frame_interval = 1 / fps
        self.stopped = threading.Event()
        self.frame_queue = queue.Queue() # maxsize=100 >>> Set maximum queue size
        self.use_grayscale = use_grayscale
        self.scale_factor = resolution_scale

        # Retrieve the camera's default resolution
        original_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new resolution based on the scaling factor
        new_width = int(original_width * self.scale_factor)
        new_height = int(original_height * self.scale_factor)

        # Set new resolution
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    def start(self)-> None:
        """Start video stream thread."""
        threading.Thread(target=self.update, daemon=True).start()

    def update(self)-> None:
        """Update video stream."""
        while not self.stopped.is_set():
            ret, frame = self.video_capture.read()
            if not ret:
                continue
            if self.use_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.frame_queue.put((time.time(), frame))
            time.sleep(self.frame_interval)

    def read(self) -> Optional[Tuple[float, np.ndarray]]:
        """Read frame from queue."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self) -> None:
        """Stop video stream."""
        self.stopped.set()
        self.video_capture.release()
        cv2.destroyAllWindows()
        # Join thread to ensure clean termination
        threading.current_thread().join()


class HRVStream:
    def __init__(self, ble_address: str = None):
        self.address = ble_address
        self.sensor_data_queue = queue.Queue()
        self.HRV_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
        self.stopped = threading.Event()
        self.adapter = None
        self.device = None
        self.sudden_change_triggered = threading.Event()  # Event to trigger sudden HRV change

        # Initialize the BLE adapter
        if self.address:
            # Initialize the BLE adapter only if an address is provided
            self.adapter = pygatt.GATTToolBackend()
            self.connect_device()
            threading.Thread(target=self.collect_sensor_data_ble, daemon=True).start()
        else:
            threading.Thread(target=self.dummy_stream, daemon=True).start()

    def connect_device(self) -> None:
        """Connect to BLE device."""
        try:
            self.adapter.start()
            self.device = self.adapter.connect(self.address)
        except Exception as e:
            print(f"Error connecting to BLE device: {e}")
            self.adapter.stop()

    def collect_sensor_data_ble(self) -> None:
        """Collect sensor data from BLE device."""
        try:
            self.adapter.start()
            device = self.adapter.connect(self.address)
            # Subscribe to the HRV characteristic
            device.subscribe(self.HRV_UUID, callback=self.handle_data)
            while not self.stopped.is_set():
                time.sleep(0.1)  # Adjust sleep time as needed
        except Exception as e:
            print(f"Error in BLE communication: {e}")
        finally:
            try:
                device.disconnect()
            except Exception as e:
                print(f"Error disconnecting device: {e}")
            self.adapter.stop()

    def handle_data(self, handle: int, value: bytes) -> None:
        """Callback function to handle incoming HRV data."""
        try:
            hrv = int.from_bytes(value, byteorder='little') / 100.0
            timestamp = time.time()
            self.sensor_data_queue.put((timestamp, hrv))
        except Exception as e:
            print(f"Error processing HRV data: {e}")

    def dummy_stream(self) -> None:
        """Generate dummy HRV data."""
        while not self.stopped.is_set():
            # Normal HRV range
            if self.sudden_change_triggered.is_set():
                # Simulate a sudden change in HRV (e.g., high stress event)
                hrv = random.uniform(160, 200)  # Sudden high HRV value
                print("Sudden change in HRV triggered.")
                # Clear the sudden change trigger after generating one higher value
                self.sudden_change_triggered.clear()
            else:
                # Normal HRV value between 50 and 150
                hrv = random.uniform(50, 150)

            timestamp = time.time()
            self.sensor_data_queue.put((timestamp, hrv))
            time.sleep(0.1)  # Adjust sleep time as needed

    def trigger_sudden_change(self) -> None:
        """Trigger a sudden change in HRV to simulate stress or excitement."""
        self.sudden_change_triggered.set()

    def read(self) -> tuple:
        """Read latest HRV value from queue.

        :return: (timestamp, hrv) or (None, None) if queue is empty.
        """
        try:
            return self.sensor_data_queue.get_nowait()
        except queue.Empty:
            return None, None

    def stop(self) -> None:
        """Stop HRV data streaming."""
        self.stopped.set()


class AudioStream:
    DEFAULT_SAMPLING_RATE = 44100
    DEFAULT_CHANNELS = 1
    DEFAULT_FPB = 2048
    DEFAULT_DEVICE = 1
    DEFAULT_FORMAT = pyaudio.paFloat32

    def __init__(self, sampling_rate: int=DEFAULT_SAMPLING_RATE,
                       channels: int=DEFAULT_CHANNELS, fpb:int=DEFAULT_FPB,
                       device:int=DEFAULT_DEVICE, format=DEFAULT_FORMAT):
        """
        Initialize audio stream.
            :param sampling_rate: Sampling rate.
            :param channels: Number of channels.
            :param fpb: Frames per buffer.
            :param device: Device index.
            :param format: Audio format.
        """
        self.sampling_rate = sampling_rate
        self.channel = channels
        self.fpb = fpb
        self.device = device
        self.format = format
        self.stopped = threading.Event()
        self.audio_queue = queue.Queue()
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()

    def start(self) -> None:
        """
        Start audio stream.
        """
        try:
            self.stream = self.pyaudio_instance.open(
                format=self.format,
                channels=self.channel,
                rate=self.sampling_rate,
                input=True,
                input_device_index=self.device,
                frames_per_buffer=self.fpb,
                stream_callback=self.callback
            )
            self.stream.start_stream()
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            raise

    def callback(self, in_data, frame_count, time_info, status) -> tuple:
        """
        Audio stream callback.

        :param in_data: Input audio data.
        :param frame_count: Frame count.
        :param time_info: Time information.
        :param status: Status.
        :return: (None, pyaudio.paContinue) or (None, pyaudio.paComplete)
        """
        if self.stopped.is_set():
            return (None, pyaudio.paComplete)
        timestamp = time.time()
        audio_data = in_data
        self.audio_queue.put((timestamp, audio_data))
        return (None, pyaudio.paContinue)

    # def read(self) -> tuple:
    #     """
    #     Read audio data from queue.

    #     :return: (timestamp, audio_data) or (None, None) if queue is empty.
    #     """
    #     try:
    #         return self.audio_queue.get_nowait()
    #     except queue.Empty:
    #         return None, None

    def read(self) -> tuple:
        """
        Read audio data from queue and convert to WAV format.

        :return: (timestamp, audio_data) or (None, None) if queue is empty.
        """
        try:
            timestamp, audio_data = self.audio_queue.get_nowait()

            # Clip the audio samples to ensure they are within the valid range for 16-bit integers
            audio_samples = np.frombuffer(audio_data, dtype=np.float32)
            audio_samples = np.clip(audio_samples, -1.0, 1.0)
            
            # Convert audio data to WAV format
            wav_data = io.BytesIO()
            with wave.open(wav_data, 'wb') as wav_writer:
                wav_writer.setnchannels(self.channel)
                wav_writer.setsampwidth(2)  # 16-bit
                wav_writer.setframerate(self.sampling_rate)
                # wav_writer.writeframes(b''.join([bytes(struct.pack('f', sample)) for sample in audio_data]))
                wav_writer.writeframes(b''.join([struct.pack('<h', int(sample * 32767)) for sample in audio_samples]))
                # wav_writer.close()
            
            return timestamp, wav_data.getvalue()
        except queue.Empty:
            return None, None
    
    def stop(self) -> None:
        self.stopped.set()
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()

# --------------- > VLM

class VLMBlock:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    def analyze_frame(self, frame):
        if frame is None:
            return "No visual context available."
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

# --------------------> context analysis

class EnvironmentalAnalysis:
    def __init__(self, vlm_block):
        self.vlm_block = vlm_block
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def analyze_video(self, frames):
        captions = [self.vlm_block.analyze_frame(frame) for frame in frames]
        input_text = ' '.join(captions)
        inputs = self.t5_tokenizer.encode_plus(input_text, 
                                                add_special_tokens=True, 
                                                max_length=512, 
                                                return_attention_mask=True, 
                                                return_tensors='pt')
        
        outputs = self.t5_model.generate(inputs['input_ids'], 
                                          attention_mask=inputs['attention_mask'], 
                                          num_beams=4, 
                                          no_repeat_ngram_size=3, 
                                          early_stopping=True)
        summary = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

# --------------------> Tars

class TarsAgent:
    def __init__(self, hrv_BLE_ADDRESS: str = None, batch_time: int = 10, hrv_threshold:int =160):
        self.hrv_ble = hrv_BLE_ADDRESS
        self.batch_time = batch_time
        self.video_stream = VideoStream(fps=2, resolution_scale=1, use_grayscale=False)
        self.audio_stream = AudioStream()
        self.hrv_stream = HRVStream(self.hrv_ble)
        self.vlm_block = VLMBlock()
        self.environmental_analysis = EnvironmentalAnalysis(self.vlm_block)
        self.speech_recognizer = sr.Recognizer()
        self.text_to_speech = pyttsx3.init()
        self.hrv_threshold = hrv_threshold  # Adjust this value based on your HRV device
        self.stopped = threading.Event()
        self.wake_up_event = threading.Event()  # Event to indicate wake-up condition

    def monitor_hrv(self):
        """Continuously monitor HRV and set wake-up event if threshold is crossed."""
        while not self.stopped.is_set():
            timestamp, hrv = self.hrv_stream.read()
            if hrv is not None and hrv > self.hrv_threshold:
                print(f"HRV threshold crossed: {hrv}")
                self.wake_up_event.set()
            time.sleep(0.1)

    def analyze_environment(self, frames, audio_data) -> str:
        """Analyze video frames and audio data to summarize the environment."""
        # Video Analysis
        video_summary = "No significant visual data available."
        if len(frames) > 0:
            video_summary = self.environmental_analysis.analyze_video(frames)

        # Audio Analysis
        audio_summary = "No significant audio data available."
        if audio_data is not None:
            try:
                with sr.AudioFile(io.BytesIO(audio_data)) as source:
                    audio = self.speech_recognizer.record(source)
                    audio_summary = self.speech_recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                audio_summary = "Could not understand the audio."
            except sr.RequestError as e:
                audio_summary = f"Audio recognition failed; {e}"

        # Combine summaries
        overall_summary = f"Video Analysis: {video_summary}\nAudio Analysis: {audio_summary}\n"
        return overall_summary

    def respond(self, environment_analysis: str) -> None:
        """Respond using text-to-speech."""
        self.text_to_speech.say("Hello, I'm Tars. " + environment_analysis)
        self.text_to_speech.runAndWait()

    def run(self) -> None:
        """Start all streams and initiate the agent's main behavior."""
        self.video_stream.start()
        self.audio_stream.start()
        self.hrv_stream.stopped.clear()

        # Start HRV monitoring in a separate thread
        threading.Thread(target=self.monitor_hrv, daemon=True).start()

        while not self.stopped.is_set():
            # Wait for HRV threshold to trigger the wake-up event
            if self.wake_up_event.wait(timeout=0.1):  # Wait for the event to be set
                # Reset the wake-up event for the next cycle
                self.wake_up_event.clear()

                # Record for T=10 seconds after HRV threshold is crossed
                frames = []
                audio_data = io.BytesIO()
                start_time = time.time()

                # Start collecting video frames and audio for T seconds
                while time.time() - start_time < self.batch_time:
                    # Video Frames Collection
                    timestamp, frame = self.video_stream.read()
                    if frame is not None:
                        frames.append(frame)

                    # Audio Data Collection
                    timestamp, audio_chunk = self.audio_stream.read()
                    if audio_chunk is not None:
                        audio_data.write(audio_chunk)

                    time.sleep(self.video_stream.frame_interval)

                # Analyze the recorded environment
                environment_analysis = self.analyze_environment(frames, audio_data.getvalue())

                # Respond with the analysis summary
                self.respond(environment_analysis)

    def stop(self) -> None:
        """Stop the agent and all streams."""
        self.stopped.set()
        self.video_stream.stop()
        self.audio_stream.stop()
        self.hrv_stream.stop()


# Usage
if __name__ == "__main__":
    tars_agent = TarsAgent(hrv_BLE_ADDRESS=None, batch_time=10)
    try:
        # Simulate a sudden change after 5 seconds
        time.sleep(5)
        tars_agent.hrv_stream.trigger_sudden_change()

        tars_agent.run()
    except KeyboardInterrupt:
        tars_agent.stop()



