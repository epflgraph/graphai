from graphai.core.video.video import NLPModels
from graphai.core.video.transcribe import WhisperTranscriptionModel
from graphai.core.interfaces.caching import VideoConfig

file_management_config = VideoConfig()
transcription_model = WhisperTranscriptionModel()
local_ocr_nlp_models = NLPModels()
