"""
ww_model.py
-------------

Description:
    This module loads a pretrained audio classification model and classifies wake word.

Author:
    Umair Cheema <cheemzgpt@gmail.com>

Version:
    1.0.0

License:
    Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Date Created:
    2024-11-30

Last Modified:
    2024-11-30

Python Version:
    3.8+

Usage:
    Instantiate this model and listen for wake word in a separate.
    Example:
        from ww_model import WakeWordClassifier

Dependencies:

"""
import torch
import torchaudio
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from vragconfig import VRAGConfig

class WakeWordClassifier():

    def __init__(self):
        self.config = VRAGConfig(file_path='./vrag.yaml').read()
        self.model_path = self.config['ww_model_path']
        self.model_device = self.config['ww_model_device']
        self.wake_word = self.config['ww_wake_word']
        self.prob_threshold = self.config['ww_prob_threshold']
        self.chunk_length = self.config['ww_chunk_length']
        self.stream_chunk = self.config['ww_stream_chunk']
        self.debug = self.config['ww_debug']

    def load_wakeword_classifier (self):
        self.classifier = pipeline(
                          "audio-classification", model=self.model_path, device=self.model_device
                        )
        
    def detect_wakeword(self):
        #check if wake word is available in the model
        if self.wake_word not in self.classifier.model.config.label2id.keys():
            print(f"The configured wake word {self.wake_word} is not available in the model at {self.model_path}.")
            return False
        
        #Get sampling rate from the model
        audio_sampling_rate = self.classifier.feature_extractor.sampling_rate

        #Get sequence of audio chunks from mic
        audio_chunks = ffmpeg_microphone_live(sampling_rate=audio_sampling_rate,chunk_length_s=self.chunk_length,
                                              stream_chunk_s=self.stream_chunk,
                       )
        
        #Classify audio chunks
        for prediction in self.classifier(audio_chunks):
            prediction = prediction[0]
            if self.debug:
                print(f'Word detected:{prediction}')
            if prediction["label"] == self.wake_word:
                if prediction["score"] > self.prob_threshold:
                    return True
    

if __name__ == "__main__":
    wake_word_classifier = WakeWordClassifier()
    print(f"Loading the wake word classification model")
    wake_word_classifier.load_wakeword_classifier()
    print(f"Listening for the wakeword {wake_word_classifier.wake_word}")
    wake_word_classifier.detect_wakeword()


