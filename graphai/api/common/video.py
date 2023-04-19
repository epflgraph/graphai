from graphai.core.common.video import WhisperTranscriptionModel, NLPModels, GoogleOCRModel
from graphai.core.common.caching import AudioDBCachingManager, SlideDBCachingManager, VideoConfig

file_management_config = VideoConfig()
audio_db_manager = AudioDBCachingManager()
slide_db_manager = SlideDBCachingManager()
transcription_model = WhisperTranscriptionModel()
local_ocr_nlp_models = NLPModels()
google_ocr_model = GoogleOCRModel()