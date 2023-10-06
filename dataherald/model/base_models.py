import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://testopenaiaerovision.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "9fb24ce358204ea78dc0b3ef4e2e7e38"

from typing import Any

from langchain.llms import AlephAlpha, Anthropic, Cohere, AzureOpenAI
from overrides import override

from dataherald.model import LLMModel
from dataherald.sql_database.models.types import DatabaseConnection
from dataherald.utils.encrypt import FernetEncrypt


class BaseModel(LLMModel):
    def __init__(self, system):
        super().__init__(system)
        self.model_name = os.environ.get("LLM_MODEL", "text-davinci-003")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.aleph_alpha_api_key = os.environ.get("ALEPH_ALPHA_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.cohere_api_key = os.environ.get("COHERE_API_KEY")

    @override
    def get_model(
        self,
        database_connection: DatabaseConnection,
        model_family="openai",
        **kwargs: Any
    ) -> Any:
        if database_connection.llm_credentials is not None:
            fernet_encrypt = FernetEncrypt()
            api_key = fernet_encrypt.decrypt(
                database_connection.llm_credentials.api_key
            )
            if model_family == "openai":
                self.openai_api_key = api_key
            elif model_family == "anthropic":
                self.anthropic_api_key = api_key
            elif model_family == "google":
                self.google_api_key = api_key
        if self.openai_api_key:
            self.model = AzureOpenAI(deployment_name='id-ai-gpt4', model_name=self.model_name, **kwargs)
        elif self.aleph_alpha_api_key:
            self.model = AlephAlpha(model=self.model_name, **kwargs)
        elif self.anthropic_api_key:
            self.model = Anthropic(model=self.model, **kwargs)
        elif self.cohere_api_key:
            self.model = Cohere(model=self.model, **kwargs)
        else:
            raise ValueError("No valid API key environment variable found")
        return self.model
