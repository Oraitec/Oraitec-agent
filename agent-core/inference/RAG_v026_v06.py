import cv2, pyaudio, wave, librosa, random, logging, pickle, time, struct, os, io, json
import numpy as np
import torch
from transformers import (VisionEncoderDecoderModel, ViTImageProcessor,
                          AutoTokenizer, pipeline)
from PIL import Image
import threading
import queue
import asyncio
from bleak import BleakClient
import pyttsx3
import pyaudio
import sounddevice as sd
import pygatt
from typing import Tuple, Optional, List
import speech_recognition as sr
import openai
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"NLTK download warning (non-critical): {e}")

# ---------------- > Recorder
class VideoStream:
    def __init__(self, fps: int = 2, resolution_scale: float = 1, use_grayscale: bool = False):
        self.video_capture = cv2.VideoCapture(index=0)
        if not self.video_capture.isOpened():
            raise ValueError("Could not open video device")
        
        self.frame_interval = 1 / fps
        self.stopped = threading.Event()
        self.frame_queue = queue.Queue()
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

    def start(self) -> None:
        """Start video stream thread."""
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self) -> None:
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
        self.thread.join()

class CaptionSummarizer:
    def __init__(self):
        # Load the pre-trained image captioning model
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_generator = pipeline('text-generation', model='gpt2')

        self.device = torch.device("cpu")
        self.caption_model.to(self.device)

        # Store captions with their timestamps
        self.captions_with_timestamps = []

    def generate_caption(self, image: np.ndarray, timestamp: float) -> str:
        """
        Generate a caption for a given image frame and store it with its timestamp.

        Args:
        - image (np.ndarray): The image frame in BGR format.
        - timestamp (float): The timestamp of the frame.

        Returns:
        - str: Generated caption.
        """
        # Convert the OpenCV image (BGR) to PIL Image (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Preprocess image
        pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate caption
        with torch.no_grad():
            output_ids = self.caption_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Store the caption with its timestamp
        self.captions_with_timestamps.append((timestamp, caption))

        return caption

    def generate_narrative_local_v1(self) -> str:
        """
        Generate a narrative based on the sequence of captions and their timestamps.

        Returns:
        - str: Generated narrative.
        """

        # Sort captions by timestamp to ensure temporal order
        sorted_captions = sorted(self.captions_with_timestamps, key=lambda x: x[0])

        # Remove duplicates or very similar captions
        unique_captions = []
        for timestamp, caption in sorted_captions:
            if not unique_captions:
                unique_captions.append((timestamp, caption))
            else:
                # Check similarity with the last caption
                last_caption = unique_captions[-1][1]
                score = sentence_bleu([last_caption.split()], caption.split(), weights=(0.5, 0.5))
                if score < 0.7:  # Adjust threshold as needed
                    unique_captions.append((timestamp, caption))

        # Build the narrative by combining captions
        narrative_sentences = []
        for idx, (timestamp, caption) in enumerate(unique_captions):
            # Optionally, include time information
            # formatted_time = time.strftime("%H:%M:%S", time.gmtime(timestamp))
            # sentence = f"At {formatted_time}, {caption}."
            sentence = f"{caption.capitalize()}."
            narrative_sentences.append(sentence)

        # Combine sentences into a paragraph
        narrative = ' '.join(narrative_sentences)

        # Analyze for possible causes of excitement or mental disturbance
        excitement_causes = self.analyze_captions_for_excitation()

        # Construct the final narrative
        final_narrative = "Neutral description of the scene and main elements:\n"
        final_narrative += narrative

        if excitement_causes:
            final_narrative += "\n\nPossible causes of excitement or mental disturbance detected:\n"
            final_narrative += '\n'.join(excitement_causes)
        else:
            final_narrative += "\n\nNo specific causes of excitement or mental disturbance were detected."

        return final_narrative

    def analyze_captions_for_excitation(self) -> List[str]:
        """
        Analyze captions to identify possible causes of excitement or mental disturbance.

        Returns:
        - List[str]: List of findings related to excitement or disturbance.
        """
        excitement_causes = []
        # Define keywords that might indicate excitement or disturbance
        keywords = [
            'fight', 'explosion', 'crying', 'shouting', 'screaming',
            'danger', 'fire', 'accident', 'running', 'laughing', 'applause',
            'weapon', 'blood', 'injury', 'police', 'emergency', 'violence',
            'crowd', 'angry', 'storm', 'protest', 'panic', 'fear', 'chaos',
            'gun', 'crowded', 'messy',
        ]

        for timestamp, caption in self.captions_with_timestamps:
            lower_caption = caption.lower()
            for keyword in keywords:
                if keyword in lower_caption:
                    formatted_time = time.strftime("%H:%M:%S", time.gmtime(timestamp))
                    excitement_causes.append(
                        f"At {formatted_time}, detected '{keyword}' in caption: '{caption}'."
                    )
                    break  # Stop checking other keywords for this caption

        return excitement_causes


class AIAssistant:
    def __init__(self, batch_time: int = 10):
        self.video_stream = VideoStream(fps=2, resolution_scale=0.5)
        self.caption_summarizer = CaptionSummarizer()
        self.batch_time = batch_time

    def run(self):
        # Start the video stream
        self.video_stream.start()
        start_time = time.time()

        # Capture video frames for the specified batch time
        while time.time() - start_time < self.batch_time:
            timestamp, frame = self.video_stream.read()
            if frame is not None:
                caption = self.caption_summarizer.generate_caption(frame, timestamp)
                print(f"Caption at {time.strftime('%H:%M:%S', time.gmtime(timestamp))}: {caption}")

        # Stop the video stream
        self.video_stream.stop()

        # Generate a narrative from the collected captions
        narrative = self.caption_summarizer.generate_narrative_local_v1()
        print("\nGenerated Narrative:\n")
        print(narrative)

def wake_up_and_run():
    user_input = input("Enter 'N' to wake up the AI assistant: ").strip().upper()
    if user_input == 'N':
        assistant = AIAssistant(batch_time=10)  # Record for 10 seconds
        assistant.run()

if __name__ == "__main__":
    wake_up_and_run()

# class ConversationalAIAssistant:
#     def __init__(self, batch_time: int = 10):
#         self.video_stream = VideoStream(fps=2, resolution_scale=0.5)
#         self.caption_summarizer = CaptionSummarizer()
#         self.batch_time = batch_time
        
#         # Initialize text-to-speech engine
#         self.tts_engine = pyttsx3.init()
#         self.tts_engine.setProperty('rate', 150)
        
#         # Initialize speech recognition
#         self.recognizer = sr.Recognizer()
#         self.microphone = sr.Microphone()
        
#         # Conversation state
#         self.current_context = None
#         self.conversation_history = []

#     def speak(self, text: str) -> None:
#         """Convert text to speech and speak it."""
#         print(f"Assistant: {text}")
#         self.tts_engine.say(text)
#         self.tts_engine.runAndWait()

#     def listen(self) -> Optional[str]:
#         """Listen for user input and convert speech to text."""
#         with self.microphone as source:
#             print("\nListening...")
#             self.recognizer.adjust_for_ambient_noise(source)
#             try:
#                 audio = self.recognizer.listen(source, timeout=5)
#                 text = self.recognizer.recognize_google(audio)
#                 print(f"User: {text}")
#                 return text.lower()
#             except sr.WaitTimeoutError:
#                 return None
#             except sr.UnknownValueError:
#                 self.speak("I didn't catch that. Could you please repeat?")
#                 return None
#             except sr.RequestError:
#                 self.speak("I'm having trouble with speech recognition. Could you type your response instead?")
#                 return input("Your response: ").lower()

#     def split_into_sentences(self, text: str) -> List[str]:
#         """Split text into sentences using a simple but robust approach."""
#         # First try NLTK's sentence tokenizer
#         try:
#             return sent_tokenize(text)
#         except Exception as e:
#             # Fallback to simple period-based splitting if NLTK fails
#             simple_sentences = [s.strip() for s in text.split('.') if s.strip()]
#             return [s + '.' for s in simple_sentences]

#     def generate_response(self, narrative: str, user_input: Optional[str] = None) -> str:
#         """Generate contextual response based on narrative and user input."""
#         # Extract key information from narrative
#         excitement_mentioned = any(keyword in narrative.lower() for keyword in 
#                                  ['excitement', 'disturbance', 'detected'])
        
#         if user_input is None:
#             # Initial response after environment analysis
#             if excitement_mentioned:
#                 return ("I've noticed some interesting activity in your surroundings. "
#                        "Would you like to talk about what's happening? "
#                        "I'm here to listen and help if needed.")
#             else:
#                 return ("Everything looks calm and normal in your surroundings. "
#                        "How are you feeling? Is there anything specific you'd like to discuss?")
        
#         # Process user input and generate appropriate response
#         user_input = user_input.lower()
#         if 'what did you see' in user_input or "what's happening" in user_input:
#             return self._summarize_narrative(narrative)
#         elif any(word in user_input for word in ['anxious', 'worried', 'scared']):
#             return self._generate_comfort_response(narrative)
#         elif 'bye' in user_input or 'goodbye' in user_input:
#             return "Take care! Remember, I'm here if you need me. Just wake me up with 'N'."
#         else:
#             return ("I'm here to help. Would you like me to describe what I observed, "
#                    "or is there something specific you'd like to discuss?")

#     def _summarize_narrative(self, narrative: str) -> str:
#         """Create a conversational summary of the narrative."""
#         sentences = self.split_into_sentences(narrative)
#         # Filter out timestamps and system messages
#         key_observations = [sent for sent in sentences 
#                           if not sent.startswith(('At ', 'No specific', 'Neutral description'))]
        
#         if not key_observations:
#             return "I observed a relatively calm environment. Would you like more specific details?"
        
#         summary = " ".join(key_observations[:3])
#         return f"{summary} Would you like me to elaborate on anything specific?"

#     def _generate_comfort_response(self, narrative: str) -> str:
#         """Generate a comforting response based on the narrative context."""
#         if 'No specific causes of excitement or mental disturbance' in narrative:
#             return ("From what I can see, your environment is actually quite peaceful right now. "
#                    "Would you like to talk about what's making you feel this way?")
#         else:
#             return ("I understand why you might be feeling this way. Let's talk about what's "
#                    "happening and figure out how to help you feel more at ease.")

#     def run(self):
#         # Initial greeting
#         self.speak("Hey! I noticed you might be a little concerned. "
#                   "Let me take a quick look around to see what's happening. "
#                   "If you have any questions, just ask!")

#         # Start the video stream
#         self.video_stream.start()
#         start_time = time.time()

#         # Capture and process video frames
#         while time.time() - start_time < self.batch_time:
#             timestamp, frame = self.video_stream.read()
#             if frame is not None:
#                 caption = self.caption_summarizer.generate_caption(frame, timestamp)

#         # Stop the video stream
#         self.video_stream.stop()

#         # Generate narrative and initial response
#         narrative = self.caption_summarizer.generate_narrative_local_v1()
#         self.current_context = narrative
        
#         # Start conversation loop
#         initial_response = self.generate_response(narrative)
#         self.speak(initial_response)

#         while True:
#             user_input = self.listen()
#             if user_input is None:
#                 continue
            
#             if 'bye' in user_input or 'goodbye' in user_input:
#                 self.speak(self.generate_response(narrative, user_input))
#                 break
                
#             response = self.generate_response(narrative, user_input)
#             self.speak(response)
#             self.conversation_history.append(("user", user_input))
#             self.conversation_history.append(("assistant", response))

# def wake_up_and_run():
#     user_input = input("Enter 'N' to wake up the AI assistant: ").strip().upper()
#     if user_input == 'N':
#         assistant = ConversationalAIAssistant(batch_time=10)
#         assistant.run()

# if __name__ == "__main__":
#     wake_up_and_run()