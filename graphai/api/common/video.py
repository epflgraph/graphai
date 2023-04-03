from graphai.core.common.video import VideoConfig, DBCachingManager, load_model_whisper

video_config = VideoConfig()
video_db_manager = DBCachingManager()
transcription_model = load_model_whisper('medium')
