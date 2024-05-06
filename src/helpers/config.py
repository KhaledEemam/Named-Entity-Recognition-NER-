from pydantic_settings import BaseSettings , SettingsConfigDict
      
class Settings(BaseSettings) :

    RAW_DATA_PATH : str
    PRETRAINED_MODEL_NAME : str
    PREPROCESSED_DATA_PATH : str
    TAGS_COUNT :  int
    MAX_LENGTH : int

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()
