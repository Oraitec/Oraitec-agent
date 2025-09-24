import cv2, pyaudio, wave, librosa, random, logging, pickle, time
import numpy as np
import torch
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
from torchvision import models, transforms
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from PIL import Image
import threading
import queue
import asyncio
from bleak import BleakClient
import pyttsx3
import sounddevice as sd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


# text_to_speech agent
class TTSAgent:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

    def provide_feedback(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

## Language Agent
class LLMIntegrationAgent:
    def __init__(self):
        # Use a more capable model to avoid token length issues
        self.model = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
        # hf_pipeline = pipeline("text-generation", model="microsoft/BioGPT", device=0 if torch.cuda.is_available() else -1)  # Models to use: "distilgpt2". alternative models to use: GPT-J ("EleutherAI/gpt-j-6B")(24G) Or BioGPT("microsoft/BioGPT")
        
        # llm = HuggingFacePipeline(pipeline=self.model, max_new_tokens=150,num_return_sequences=1)
    def run_llm(self, prompt):
        
        # Call the generate function with explicit parameters to avoid being overridden
        response = self.model.model.generate(
            self.model.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device),
            max_length=300,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        generated_text = self.model.tokenizer.decode(response[0], skip_special_tokens=True)
        return generated_text


class VisualAgent:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_visual_context(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            image_features = self.model(input_tensor)
        return image_features.cpu().numpy()


class SensorAgent:
    def __init__(self, ble_address):
        self.address = ble_address
        self.sensor_data_queue = queue.Queue()
        self.HRV_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
        self.TEMPERATURE_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"
        self.ACCELEROMETER_UUID = "00002a73-0000-1000-8000-00805f9b34fb"
        threading.Thread(target=self.collect_sensor_data_ble, daemon=True).start()

    def collect_sensor_data_ble(self):
        async def run():
            async with BleakClient(self.address) as client:
                logger.info(f"Connected to BLE device at {self.address}")
                while True:
                    hrv_raw = await client.read_gatt_char(self.HRV_UUID)
                    hrv = int.from_bytes(hrv_raw, byteorder='little') / 100.0

                    temp_raw = await client.read_gatt_char(self.TEMPERATURE_UUID)
                    temperature = int.from_bytes(temp_raw, byteorder='little') / 100.0

                    accel_raw = await client.read_gatt_char(self.ACCELEROMETER_UUID)
                    accelerometer = {
                        "x": int.from_bytes(accel_raw[0:2], byteorder='little') / 100.0,
                        "y": int.from_bytes(accel_raw[2:4], byteorder='little') / 100.0,
                        "z": int.from_bytes(accel_raw[4:6], byteorder='little') / 100.0
                    }

                    gyroscope = {
                        "x": random.uniform(-180, 180),
                        "y": random.uniform(-180, 180),
                        "z": random.uniform(-180, 180)
                    }

                    sensor_data = {
                        "hrv": hrv,
                        "temperature": temperature,
                        "gyroscope": gyroscope,
                        "accelerometer": accelerometer
                    }
                    self.sensor_data_queue.put(sensor_data)
                    await asyncio.sleep(1)

        try:
            asyncio.run(run())
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run())

    def get_sensor_data(self):
        if not self.sensor_data_queue.empty():
            return self.sensor_data_queue.get()
        # Generate random sensor data if real sensor data is not available
        return {
            "hrv": random.uniform(50, 70),
            "temperature": random.uniform(20, 35),
            "gyroscope": {
                "x": random.uniform(-180, 180),
                "y": random.uniform(-180, 180),
                "z": random.uniform(-180, 180)
            },
            "accelerometer": {
                "x": random.uniform(-10, 10),
                "y": random.uniform(-10, 10),
                "z": random.uniform(-10, 10)
            },
            "generated": True
        }


class VideoAgent:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.video_capture.release()
        cv2.destroyAllWindows()


# class AudioAgent:
#     def __init__(self, sampling_rate=44100, duration=1):
#         self.sampling_rate = sampling_rate
#         self.duration = duration
#         self.audio_queue = queue.Queue()
#         threading.Thread(target=self.record_audio, daemon=True).start()

#     def record_audio(self):
#         while True:
#             audio = sd.rec(int(self.sampling_rate * self.duration), samplerate=self.sampling_rate, channels=1, dtype='float64')
#             sd.wait()
#             audio_volume = np.linalg.norm(audio) / len(audio)
#             self.audio_queue.put((audio, audio_volume))

#     def get_audio_data(self):
#         if not self.audio_queue.empty():
#             return self.audio_queue.get()
#         return None, 0.5  # Default value if no audio is available

# # Speech Analysis Agent
# class SpeechAnalysisAgent:
#     def __init__(self):
#         self.model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0 if torch.cuda.is_available() else -1)

#     def analyze_speech(self, audio):
#         if audio is not None:
#             audio_flat = audio.flatten()
#             transcription = self.model(audio_flat, return_timestamps=False)['text']
#             return transcription
#         return ""

# # Audio Classification Agent
# class AudioClassificationAgent:
#     def __init__(self):
#         # Using a simple heuristic-based approach for now; can be replaced with a more sophisticated model.
#         self.noise_threshold = 0.05

#     def classify_audio(self, audio_volume):
#         if audio_volume < self.noise_threshold:
#             return "ambient noise"
#         return "speech"



class AudioAgent:
    def __init__(self, sampling_rate=44100, duration=1):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.audio_queue = queue.Queue()
        threading.Thread(target=self.record_audio, daemon=True).start()

    def record_audio(self):
        while True:
            audio = sd.rec(int(self.sampling_rate * self.duration), samplerate=self.sampling_rate, channels=1, dtype='float64')
            sd.wait()
            audio_volume = np.linalg.norm(audio) / len(audio)
            self.audio_queue.put((audio, audio_volume))

    def get_audio_data(self):
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None, 0.5  # Default value if no audio is available

# Speech Analysis Agent
class SpeechAnalysisAgent:
    def __init__(self):
        self.model = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0 if torch.cuda.is_available() else -1)

    def analyze_speech(self, audio):
        if audio is not None:
            audio_flat = audio.flatten()
            transcription = self.model(audio_flat, return_timestamps=False)['text']
            return transcription
        return ""

# Audio Classification Agent
class AudioClassificationAgent:
    def __init__(self):
        # Using a simple heuristic-based approach for now; can be replaced with a more sophisticated model.
        self.noise_threshold = 0.05

    def classify_audio(self, audio_volume):
        if audio_volume < self.noise_threshold:
            return "ambient noise"
        return "speech"





class StressEstimationAgent:
    def estimate_stress_level(self, hrv, temperature, gyroscope, accelerometer, audio_volume):
        
        stress_score = 0
        if hrv < 60:
            stress_score += 3 if hrv < 50 else 2
        if temperature > 30 or temperature < 20:
            stress_score += 2
        movement_magnitude = np.sqrt(gyroscope["x"] ** 2 + gyroscope["y"] ** 2 + gyroscope["z"] ** 2)
        if movement_magnitude > 250:
            stress_score += 2
        if audio_volume > 0.7:
            stress_score += 2

        if stress_score >= 8:
            return "High stress"
        elif stress_score >= 4:
            return "Moderate stress"
        else:
            return "Low stress"


class EnvironmentAnalysisAgent:
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent

    def generate_analysis_summary(self, visual_context, stress_level, temperature, hrv, gyroscope, accelerometer):
        # Improved prompt to avoid length issues and focus on essential information
        prompt = (
            f"Scene detected: visual features extracted by AR glasses. The user's HRV is {hrv}, "
            f"temperature is {temperature}°C. Movement data from the gyroscope: x={gyroscope['x']}, y={gyroscope['y']}, z={gyroscope['z']}. "
            f"Accelerometer data: x={accelerometer['x']}, y={accelerometer['y']}, z={accelerometer['z']}. "
            f"Stress level: {stress_level}. "
            f"Please provide 2-3 brief and practical recommendations to help the user reduce stress and improve well-being."
        )
        # prompt = (
            # f"The AR glasses have detected the following scene: {visual_context}. The HRV is {hrv}, the user's body temperature is {temperature}°C. "
            # f"Gyroscope data indicates movement with values {gyroscope}, and accelerometer data indicates values {accelerometer}. "
            # f"Stress level is determined as {stress_level}. Give specific recommendations to help the user improve their well-being."
        # )
        
        return self.llm_agent.run_llm(prompt)


# Initialize agents
tts_agent = TTSAgent()
llm_agent = LLMIntegrationAgent()
visual_agent = VisualAgent()
sensor_agent = SensorAgent("XX:XX:XX:XX:XX:XX")
stress_agent = StressEstimationAgent()
env_analysis_agent = EnvironmentAnalysisAgent(llm_agent)
video_agent = VideoAgent()
audio_agent = AudioAgent()
speech_agent = SpeechAnalysisAgent()
audio_classification_agent = AudioClassificationAgent()

previous_accelerometer = None

# Main loop for real-time analysis
while True:
    frame = video_agent.get_frame()
    if frame is None:
        break

    visual_context = visual_agent.get_visual_context(frame)
    sensor_data = sensor_agent.get_sensor_data()
    audio, audio_volume = audio_agent.get_audio_data()
    audio_classification = audio_classification_agent.classify_audio(audio_volume)
    speech_context = ""

    if audio_classification == "speech":
        speech_context = speech_agent.analyze_speech(audio) if audio is not None else ""

    if sensor_data:
        hrv = sensor_data["hrv"]
        temperature = sensor_data["temperature"]
        gyroscope = sensor_data["gyroscope"]
        accelerometer = sensor_data["accelerometer"]
        generated = sensor_data.get("generated", False)

        stress_level = stress_agent.estimate_stress_level(hrv, temperature, gyroscope, accelerometer, audio_volume)
        suggestion = "Take a break." if stress_level == "High stress" else "You are doing well."
        analysis_summary = env_analysis_agent.generate_analysis_summary(visual_context, stress_level, temperature, hrv, gyroscope, accelerometer)

        tts_agent.provide_feedback(suggestion)

        # Overlay information on the frame
        cv2.putText(frame, f"HRV: {hrv} {'(Generated)' if generated else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Temperature: {temperature}C {'(Generated)' if generated else ''}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Stress Level: {stress_level}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Suggestion: {suggestion}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Speech Context: {speech_context}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Audio Volume: {audio_volume:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Audio Classification: {audio_classification}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if speech_context:
            cv2.putText(frame, f"Conversation: {speech_context}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

    # Display frame
    cv2.imshow('Real-Time Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_agent.release()
