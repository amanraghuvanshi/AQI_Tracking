import os
import json
import requests
import gradio as gr
from agno.agent import Agent
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Optional
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from agno.models.openai import OpenAIChat

load_dotenv()

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
    
class AQIAnalyzer:
    
    def __init__(self, firecrawl_key : str) -> None:
        self.firecrawl = FirecrawlApp(api_key = os.getenv("firecrawl_key"))
    
    def _format_url(self, country : str, state: str, city: str) -> str:
        """Format URLs based on location, handling cases with and without state
        """
        country_clean = country.lower().replace(" ", "-")
        city_clean = city.lower().replace(" ", "-")
        
        if not state or state.lower().replace(" ","-"):
            return f"https://www.aqi.in/dashboard/{country_clean}/{city_clean}"
        
        state_clean = state.lower().replace(" ", "-")
        return f"https://www.aqi.in/dashboard/{country_clean}/{state_clean}/{city_clean}"
        
    
    def fetch_aqi_data(self, city: str, state: str, country: str) -> tuple[Dict[str, float], str]:
        """Fetch API data using Firecrawl"""
        url = self._format_url(country, state, city)
        info_msg = f"Accessing URL: {url}"
        
        resp = self.firecrawl.extract(
            urls = [f"{url}/*"],
            params = {
                "prompt" : "Extract the current real-time AQI, temperature, humidity, wind speed, PM2.5, PM10 and CO Levels from the page. Also extract the timestamp of the data.",
                "schema": ExtractSchema.model_json_schema()
            }
        )
        
        aqi_response = AQIResponse(**resp)
        
        if not aqi_response.success:
            raise requests.HTTPError(f"Failed to fetch AQI Data: {aqi_response.status}")
        
        return aqi_response.data, info_msg