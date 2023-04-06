from graphai.core.common.video import WhisperTranscriptionModel
from graphai.core.common.caching import AudioDBCachingManager, VideoConfig

file_management_config = VideoConfig()
audio_db_manager = AudioDBCachingManager()
transcription_model = WhisperTranscriptionModel('medium')
