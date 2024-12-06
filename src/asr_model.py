"""
asr_model.py
-------------

Description:
    This module instantiates the Speech Recognition model.

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
    Instantiate this model and convert speech into text
    Example:
        from asr_model import AutomaticSpeechRecognition

Dependencies:

"""

import sys
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from vragconfig import VRAGConfig

class AutomaticSpeechRecognition():
    
    def __init__(self):
        self.config = VRAGConfig(file_path='./vrag.yaml').read()
        self.model_path = self.config['asr_model_path']
        self.model_device = self.config['asr_model_device']
        self.chunk_length = self.config['asr_chunk_length']
        self.stream_chunk = self.config['asr_stream_chunk_s']
        self.max_new_tokens = self.config['asr_max_new_tokens']

    def load_asr_model(self):
        self.model = pipeline(
                           "automatic-speech-recognition", model=self.model_path, device=self.model_device,
                    )
        
    def convert_speech_to_text(self, out=False):
        #Get sampling rate from the model
        audio_sampling_rate = self.model.feature_extractor.sampling_rate

         #Get sequence of audio chunks from mic
        audio_chunks = ffmpeg_microphone_live(sampling_rate=audio_sampling_rate,chunk_length_s=self.chunk_length,
                                              stream_chunk_s=self.stream_chunk)
        #Convert audio chuncks to text
        for item in self.model(audio_chunks, generate_kwargs={"max_new_tokens": self.max_new_tokens}):
            if out:
                sys.stdout.write("\033[K")
                print(item["text"], end="\r")
                if not item["partial"][0]:
                    break

        return item["text"]
    

if __name__ == "__main__":

    asr = AutomaticSpeechRecognition()
    print('Loading Automatic Speech Recognition Model')
    asr.load_asr_model()
    print(f'Speak for less than {asr.chunk_length} seconds to convert speech to text')
    text = asr.convert_speech_to_text(out=True)
    print(f'Speaker said: {text}')
