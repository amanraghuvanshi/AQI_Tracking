import json
import gradio as gr
from agno.agent import Agent
from dataclasses import dataclass
from typing import Dict, Optional
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from agno.models.openai import OpenAIChat


class AQIResponse(BaseModel):
    success: bool
    data: Dict[str, float]
    status: str
    expiresAt: str