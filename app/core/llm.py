"""The LLM provider layer.

Everything in the app gets its chat model from `get_llm()`. Nobody else
constructs a model or reads provider config. Swapping OpenAI <-> local is a
.env change, never a code change.
"""


from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from app.core.config import get_settings
from app.core.pricing import cost_usd
from langchain_core.messages import SystemMessage, HumanMessage 

def get_llm(*, streaming: bool = False, temperature: float | None = None) -> ChatOpenAI:
    settings = get_settings()
    # Ollama (and other local servers) ignore the key but require the field
    api_key = settings.openai_api_key if settings.llm_provider == "openai" else "ollama"
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=streaming,
        max_retries=3,
    )
@dataclass
class CallRequest:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def invoke_tracked(prompt: str, *, system: str | None = None) -> CallRequest:
    settings = get_settings()
    llm = get_llm(treaming=True)

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)

    usage = response.usage_metadata or {}
    in_tok = usage.get("input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)

    return CallRequest(
        text=response.content,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost_usd(settings.llm_model, in_tok, out_tok),
    )
  




