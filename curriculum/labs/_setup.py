"""Shared notebook setup, injected by the builder after package installation."""
import importlib.metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

# Change a switch only when you choose an optional route. Enable one API at most.
USE_GPU = False
USE_GEMINI = False
USE_OPENAI = False
GEMINI_MODEL = "gemini-3.5-flash-lite"
OPENAI_MODEL = "gpt-4.1-mini"
# Local users may change this path to their private .env file; never upload that file.
LOCAL_ENV_FILE = ".env"
SEED = 42
SMOKE = os.environ.get("GENAI_LAB_SMOKE") == "1"
try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
# The instructor runner's explicit --device cuda option is also an opt-in.
_gpu_opt_in = USE_GPU or os.environ.get("GENAI_RUNNER_GPU") == "1"
DEVICE = "cuda" if _gpu_opt_in and torch.cuda.is_available() and os.environ.get("GENAI_FORCE_CPU") != "1" else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Small CPU jobs often run faster without oversubscribing the machine.
torch.set_num_threads(min(4, os.cpu_count() or 1))
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

def check(condition, ok_message, hint):
    """Give feedback without hiding what needs to be corrected."""
    print("✅ " + ok_message if bool(condition) else "Try again: " + hint)
    return bool(condition)

def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"

RUNTIME_INFO = {
    "environment": "Colab" if IN_COLAB else "local",
    "python": platform.python_version(), "device": DEVICE, "seed": SEED,
    "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
    "cpu_threads": torch.get_num_threads(),
    "packages": {name: _package_version(name) for name in (
        "torch", "transformers", "diffusers", "peft", "trl",
        "sentence-transformers", "google-genai", "openai", "python-dotenv", "ipywidgets")},
}
print("Runtime:", json.dumps(RUNTIME_INFO, sort_keys=True))
print("CPU is the default. Complete the core tasks before trying optional GPU or API work.")

_gemini_calls = 0
_openai_calls = 0
_gemini_cache = Path.cwd() / ".genai_cache"

def _optional_key(name):
    """Read a secret only after the student opts in; never display it."""
    if IN_COLAB:
        try:
            from google.colab import userdata
            key = userdata.get(name)
            if key:
                return key
        except Exception:
            pass
    key = os.environ.get(name)
    if key or IN_COLAB:
        return key
    # Read this one file only after explicit API opt-in. Do not change the environment.
    try:
        from dotenv import dotenv_values
        return dotenv_values(LOCAL_ENV_FILE).get(name)
    except Exception:
        return None

def _gemini_key():
    return _optional_key("GEMINI_API_KEY")

def ask_gemini(prompt, *, temperature=0.2, max_output_tokens=256):
    """Return optional synthetic-data text, or None to use the local model.

    Requests, including retries, are capped at 40 per notebook session.
    Actual free quotas and eligibility depend on the student's account.
    """
    global _gemini_calls
    if not USE_GEMINI or SMOKE:
        return None
    key = _gemini_key()
    if not key:
        print("Gemini key unavailable; continue with the local result.")
        return None
    request = {"model": GEMINI_MODEL, "prompt": prompt,
               "temperature": temperature, "max_output_tokens": max_output_tokens}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    cache_file = _gemini_cache / (digest + ".json")
    try:
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
        for attempt in range(4):
            if _gemini_calls + _openai_calls >= 40:
                print("Notebook request ceiling reached; use the local result.")
                return None
            _gemini_calls += 1
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature, max_output_tokens=max_output_tokens),
                )
                text = response.text
                if not isinstance(text, str) or not text.strip():
                    return None
                try:
                    _gemini_cache.mkdir(exist_ok=True)
                    cache_file.write_text(json.dumps({"text": text}), encoding="utf-8")
                except OSError:
                    pass
                return text
            except Exception as error:
                # Error messages can contain request details, so display no raw exception.
                if getattr(error, "code", None) != 429 or attempt == 3:
                    print("Gemini unavailable; continue with the local result.")
                    return None
                time.sleep(2 ** attempt)
    except Exception:
        print("Gemini setup unavailable; continue with the local result.")
    finally:
        if "client" in locals() and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass
    return None

def ask_openai(prompt, *, temperature=0.2, max_output_tokens=256):
    """Return optional OpenAI API text, or None for the complete local fallback.

    API usage may incur charges and is separate from a ChatGPT subscription.
    The notebook ceiling counts both providers' attempts, including retries.
    """
    global _openai_calls
    if not USE_OPENAI or SMOKE:
        return None
    key = _optional_key("OPENAI_API_KEY")
    if not key:
        print("OpenAI key unavailable; continue with the local result.")
        return None
    request = {"provider": "openai", "model": OPENAI_MODEL, "prompt": prompt,
               "temperature": temperature, "max_output_tokens": max_output_tokens}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    cache_file = _gemini_cache / (digest + ".json")
    try:
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        from openai import OpenAI
        # The helper owns retries so each network attempt counts towards the ceiling.
        client = OpenAI(api_key=key, timeout=30.0, max_retries=0)
        for attempt in range(4):
            if _gemini_calls + _openai_calls >= 40:
                print("Notebook request ceiling reached; use the local result.")
                return None
            _openai_calls += 1
            try:
                response = client.responses.create(
                    model=OPENAI_MODEL, input=prompt, temperature=temperature,
                    max_output_tokens=max_output_tokens, store=False,
                )
                text = response.output_text
                if not isinstance(text, str) or not text.strip():
                    return None
                try:
                    _gemini_cache.mkdir(exist_ok=True)
                    cache_file.write_text(json.dumps({"text": text}), encoding="utf-8")
                except OSError:
                    pass
                return text
            except Exception as error:
                if getattr(error, "status_code", None) != 429 or attempt == 3:
                    print("OpenAI unavailable; continue with the local result.")
                    return None
                time.sleep(2 ** attempt)
    except Exception:
        print("OpenAI setup unavailable; continue with the local result.")
    finally:
        if "client" in locals() and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass
    return None

def ask_model(prompt, *, temperature=0.2, max_output_tokens=256):
    """Use the student's one selected API, otherwise return None for local work."""
    if SMOKE:
        return None
    if USE_GEMINI and USE_OPENAI:
        print("Choose one API switch; keep the other False. Use the local result for now.")
        return None
    if USE_GEMINI:
        return ask_gemini(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    if USE_OPENAI:
        return ask_openai(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    return None
