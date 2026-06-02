from typing import Union
from langchain_core.language_models.chat_models import BaseChatModel


from config import LocalConfig, RemoteConfig


class LLMProviderFactory:
    """
    A Factory class that abstracts the initialization of different LangChain
    Chat Models based on the provided configuration.
    """

    @classmethod
    def create_llm(cls, config: Union[LocalConfig, RemoteConfig]) -> BaseChatModel:
        """
        Takes a loaded configuration object and returns a standardized
        LangChain BaseChatModel ready to be used in RAG or Agents.
        """
        provider = config.provider.lower()
        if provider == "groq":
            return cls._build_groq(config)
        elif provider == "openai":
            return cls._build_openai(config)
        elif provider == "anthropic":
            return cls._build_anthropic(config)
        elif provider == "ollama":
            return cls._build_ollama(config)
        elif provider in ["llama.cpp", "vllm", "local_openai"]:

            return cls._build_openai_compatible(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
    @staticmethod
    def _build_groq(config) -> BaseChatModel:
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Please install langchain-groq: pip install langchain-groq")

        return ChatGroq(
            api_key=config.api_key,
            model_name=config.model_name, 
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

    @staticmethod
    def _build_openai(config: RemoteConfig) -> BaseChatModel:
        """Initializes OpenAI models (e.g., GPT-4o)"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Please install langchain-openai: pip install langchain-openai"
            )

        return ChatOpenAI(
            api_key=config.api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=getattr(config, "api_base", None),
        )

    @staticmethod
    def _build_anthropic(config: RemoteConfig) -> BaseChatModel:
        """Initializes Anthropic models (e.g., Claude 3.5 Sonnet)"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "Please install langchain-anthropic: pip install langchain-anthropic"
            )

        return ChatAnthropic(
            api_key=config.api_key,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    @staticmethod
    def _build_ollama(config: LocalConfig) -> BaseChatModel:
        """Initializes local Ollama models (e.g., Llama 3, Mistral)"""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Please install langchain-ollama: pip install langchain-ollama"
            )

        return ChatOllama(
            base_url=config.api_base,
            model=config.model_name,
            temperature=config.temperature,
            num_predict=config.max_tokens,
            num_ctx=config.context_window,
        )

    @staticmethod
    def _build_openai_compatible(config: LocalConfig) -> BaseChatModel:
        """
        Initializes local setups that mimic the OpenAI API,
        such as LM Studio, Llama.cpp web server, or vLLM.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "Please install langchain-openai: pip install langchain-openai"
            )

        return ChatOpenAI(
            api_key="lm-studio",
            base_url=config.api_base,
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )


if __name__ == "__main__":
    from config import load_config
    import os

    os.environ["OPENAI_API_KEY"] = "sk-dummy-test-key"

    app_config = load_config("config.json")
    active_config = app_config.active_llm_config

    print(f"Initializing LLM for provider: {active_config.provider}...")

    llm = LLMProviderFactory.create_llm(active_config)

    print("\n✅ Successfully initialized LangChain model!")
    print(f"Model Class: {type(llm).__name__}")
