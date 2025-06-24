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
    
class ExtractSchema(BaseModel):
    aqi: float = Field(description = "Air Quality Index")
    temperature: float = Field(description = "Temperature in Degree Celsius")
    humidity: float = Field(description = "Humidity Percentage")
    wind_speed: float = Field(description = "")
    pm25:float = Field(description = "Particulate Matter 2.5 micrometers")
    pm10:float = Field(description = "Particulate Matter 10 micrometers")
    co: float = Field(description = "Carbon Monoxide Level")
    
@dataclass
class UserInput:
    city: str
    state: str
    country: str
    medical_conditions: Optional[str]
    planned_activity: str