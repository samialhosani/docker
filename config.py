import json
import os
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError

class LocalConfig(BaseModel):
    provider: str = Field(..., description="e.g., ollama, llama.cpp, huggingface")
    model_name: str
    api_base: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024)
    context_window: int = Field(default=4096)

class RemoteConfig(BaseModel):
    provider: str = Field(..., description="e.g., openai, anthropic, azure")
    model_name: str
    api_base: str
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048)
    env_api_key_name: str = Field(
        ..., description="Name of the environment variable holding the API key"
    )

    @property
    def api_key(self) -> str:
        key = os.getenv(self.env_api_key_name)
        if not key:
            raise ValueError(f"Environment variable '{self.env_api_key_name}' is not set!")
        return key

class AppConfig(BaseModel):
    mode: Literal["local", "remote"]
    local_config: LocalConfig
    remote_config: RemoteConfig
    
    # --- NEW: Centralized File & Path Management ---
    db_path: str = Field(default="education_platform.db")
    vector_db_path: str = Field(default="education_vectors.db")
    data_dir: str = Field(default="./data")
    materials_dir: str = Field(default="./materials")

    @property
    def active_llm_config(self) -> LocalConfig | RemoteConfig:
        if self.mode == "local":
            return self.local_config
        return self.remote_config

def load_config(file_path: str = "config.json") -> AppConfig:
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return AppConfig(**data)
    except FileNotFoundError:
        print(f"⚠️ Config not found at {file_path}. Using default settings.")
        return AppConfig(
            mode="local",
            local_config=LocalConfig(provider="ollama", model_name="llama3"),
            remote_config=RemoteConfig(provider="openai", model_name="gpt-4o", api_base="", env_api_key_name="OPENAI_API_KEY")
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")