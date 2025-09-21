# server.py
# Minimal HTTP server that loads Qwen/Qwen2.5-7B-Instruct once (4-bit by default)
# and serves a /generate endpoint for multiple "agents" (personas) without
# duplicating the model in VRAM.

import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "./qwen2.5-7b-instruct")  # local dir or HF repo id
LOAD_4BIT = os.getenv("LOAD_4BIT", "1") == "1"                # set to 0 to use 8-bit
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "2048"))
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.9"))
TOP_P_DEFAULT = float(os.getenv("TOP_P", "0.95"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "100"))

# -----------------------------
# Global model singletons
# -----------------------------
_tokenizer = None
_model = None
_model_lock = asyncio.Lock()  # serialize generations to avoid GPU OOM & contention


def load_model() -> None:
    global _tokenizer, _model
    if _model is not None and _tokenizer is not None:
        return

    quant_cfg = None
    if LOAD_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=quant_cfg,
    )


# -----------------------------
# Prompt utilities
# -----------------------------
SYSTEM_DEFAULT = (
    "You are Qwen2.5-7B-Instruct, a concise analyst. "
    "If context is provided, ground answers in it. If unsure, say you don't know."
)


def build_prompt(system: Optional[str], context: Optional[str], messages: List[Dict[str, str]]) -> str:
    sys_txt = system.strip() if system else SYSTEM_DEFAULT
    parts = [f"<|system|>\n{sys_txt}\n"]
    if context:
        parts.append(f"<|context|>\n{context.strip()}\n")
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "user"
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# -----------------------------
# FastAPI app & schemas
# -----------------------------
app = FastAPI(title="Qwen2.5-7B Instruct Server", version="1.0")


class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class GenerateRequest(BaseModel):
    agent_id: Optional[str] = None
    system: Optional[str] = None
    context: Optional[str] = None  # RAG snippets if any
    messages: List[Message]
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str
    usage: Dict[str, Any]


@app.on_event("startup")
async def on_startup():
    load_model()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    load_model()
    tok = _tokenizer
    model = _model

    prompt = build_prompt(req.system, req.context, [m.model_dump() for m in req.messages])

    max_new = req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
    temperature = req.temperature if req.temperature is not None else TEMPERATURE_DEFAULT
    top_p = req.top_p if req.top_p is not None else TOP_P_DEFAULT
    top_k = req.top_k if req.top_k is not None else TOP_K_DEFAULT

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    async with _model_lock:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    text = tok.decode(output_ids[0], skip_special_tokens=True)
    # Heuristic: return only the assistant part after the last tag
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1].strip()

    return GenerateResponse(
        text=text,
        usage={
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(output_ids.shape[1] - inputs["input_ids"].shape[1]),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new,
        },
    )

@app.post("/generate_stream")
async def generate_stream(req: GenerateRequest):
    load_model()
    tok = _tokenizer
    model = _model

    prompt = build_prompt(req.system, req.context, [m.model_dump() for m in req.messages])

    max_new = req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
    temperature = req.temperature if req.temperature is not None else TEMPERATURE_DEFAULT
    top_p = req.top_p if req.top_p is not None else TOP_P_DEFAULT
    top_k = req.top_k if req.top_k is not None else TOP_K_DEFAULT

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # Hugging Face streamer for real-time tokens
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        streamer=streamer,
    )

    # Run generate in background so streamer yields
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    def token_stream():
        for new_text in streamer:
            yield new_text

    return StreamingResponse(token_stream(), media_type="text/plain")



# Health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "four_bit": LOAD_4BIT}
