"""
bitnet_client.py — Cliente HTTP para o llama-server.exe (BitNet 1-bit local)

Centraliza chamadas à API REST do servidor local na porta 8080.
O servidor é iniciado automaticamente pelo bitnetLLMWrapper.py da Lira,
ou pode ser iniciado manualmente via llama-server.exe.
"""

import requests
import json
import logging

logger = logging.getLogger(__name__)

SERVER_URL = "http://127.0.0.1:8080"


def is_server_alive() -> bool:
    """Verifica se o llama-server está rodando."""
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def generate_content(prompt: str,
                     system_instruction: str = "You are an expert AI.",
                     as_json: bool = False,
                     temperature: float = 0.4,
                     max_tokens: int = 512) -> str | None:
    """
    Envia uma query ao BitNet local e retorna o texto gerado.

    Args:
        prompt:             Mensagem do usuário
        system_instruction: System prompt
        as_json:            Se True, força output em JSON
        temperature:        Temperatura de amostragem (padrão: 0.4 para estabilidade)
        max_tokens:         Máximo de tokens gerados

    Returns:
        Texto gerado ou None em caso de erro.
    """
    payload = {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    if as_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error("[BitNet] HTTP %d: %s", response.status_code, response.text[:200])
    except Exception as e:
        logger.error("[BitNet] Erro de conexão: %s", e)

    return None
