from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PrivateAttr
from pathlib import Path
from typing import Any
import yaml

from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Anthropic
    ANTHROPIC_API_KEY: str
    
    # OpenAI
    OPENAI_API_KEY: str

    # Cohere
    COHERE_API_KEY: str

    # Qdrant
    QDRANT_URL: str

    # Qdrant Collection Name
    QDRANT_COLLECTION_NAME: str

    _yaml_config: dict[str , Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context:Any) -> None:
        yaml_path = Path(__file__).parent.parent / "configs" / "settings.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                self._yaml_config = yaml.safe_load(f)

    @property
    def ingestion(self) -> dict[str , Any]:
        return self._yaml_config.get("ingestion", {})
    
    @property
    def retrieval(self) -> dict[str , Any]:
        return self._yaml_config.get("retrieval", {})
    
    @property
    def generation(self) -> dict[str , Any]:
        return self._yaml_config.get("generation", {})
    
    @property
    def evaluation(self) -> dict[str , Any]:
        return self._yaml_config.get("evaluation", {})
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]



    