"""
tts_model.py
-------------

Description:
    This module instantiates the Text to Speech model.

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
    Instantiate this model and convert text into speech
    Example:
        from tts_model import TextToSpeechModel

Dependencies:

"""
import re
import torch
import sounddevice as sd
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from vragconfig import VRAGConfig


class TextToSpeechModel():

    def __init__(self):
        self.config = VRAGConfig(file_path='./vrag.yaml').read()
        self.tts_model_path = self.config['tts_model_path']
        self.tts_hifigan_path = self.config['tts_hifigan_path']
        self.tts_speaker_embeddings = self.config['tts_speaker_embeddings']
        self.tts_sampling_rate = self.config['tts_sampling_rate']
        self.tts_word_per_utterance = self.config['tts_word_per_utterance']

    def load_tts_model(self):
        self.processor = SpeechT5Processor.from_pretrained(self.tts_model_path)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.tts_model_path)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.tts_hifigan_path)
        self.speaker_embeddings = torch.load(self.tts_speaker_embeddings)

    def split_sentences(self,text, max_words=8):
        sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
        result = []
    
        for sentence in sentences:
            words = sentence.split()
            while len(words) > max_words:
                result.append(' '.join(words[:max_words]) + '')
                words = words[max_words:]
            result.append(' '.join(words) + '')
        return result
    
    
    def convert_text_to_speech(self,text):
        sentence_parts = self.split_sentences(text, max_words=self.tts_word_per_utterance)
        for sentence_part in sentence_parts:
            inputs = self.processor(text=sentence_part, return_tensors="pt")
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
            sd.play(speech.numpy(),self.tts_sampling_rate)
            sd.wait()


        

    


