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
        self.firecrawl = FirecrawlApp(api_key = firecrawl_key)
    
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
        try:
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
        
        except Exception as e:
            error_msg = f"Error Fetching AQI Data: {str(e)}"
            return {
                "api": 0,
                "temperature": 0,
                "humidity": 0,
                "wind_speed": 0,
                "pm25": 0,
                "pm10": 0,
                "co": 0
            }, error_msg

class HealthRecommendationAgent:
    
    def __init__(self, openai_key: str) -> Agent:
        self.agent = Agent(
            model = OpenAIChat(
            id = "gpt-4.1-nano",
            name = "Health Recommendation Agent",
            api_key = openai_key
            )
        )
        
    def _create_prompt(self, aqi_data: Dict[str, float], user_input: UserInput) -> str:
        return f"""
        Based on the following air quality condition in {user_input.city}, {user_input.state}, {user_input.country}:
        - Overall AQI: {aqi_data["aqi"]}
        - PM2.5 Level: {aqi_data["pm25"]} µg/m³
        - PM10 Level: {aqi_data["pm10"]} µg/m³
        - CO Level: {aqi_data["co"]} ppb
        
        Weather Conditions:
        - Temperature: {aqi_data["temperature"]}°C
        - Humidity: {aqi_data["humidity"]}%
        - Wind Speed: {aqi_data["co"]} ppb
    """ 
    
    def get_recommendation(self, aqi_data: Dict[str, float], user_input: UserInput) -> str:
        prompt = self._create_prompt(prompt)
        resp = self.agent.run(prompt)
        
        return resp.content

def analyze_conditions(city: str, state: str, country: str, medical_condition: str, planned_activity: str, firecrawl_key: str, openai_key: str) -> tuple[str, str, str, str]:
    """Analyze condition and return AQI data, recommendation, and status messages"""
    try:
        # initialize the analyzer
        aqi_analyzer = AQIAnalyzer(firecrawl_key=firecrawl_key)
        health_agent = HealthRecommendationAgent(openai_key = openai_key)
        
        # Create user input
        user_input = UserInput(
            city = city, 
            state = state,
            country = country,
            medical_conditions = medical_condition,
            planned_activity = planned_activity
        )
        
        # Get AQI Data
        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(
            city = user_input.city,
            state = user_input.state,
            country = user_input.country
        )
        
        # Format AQI data for display
        aqi_json = json.dumps({
            "Air Quality Index (AQI): ": aqi_data["aqi"],
            "PM2.5: ":f"{aqi_data["pm25"]} µg/m³",
            "PM10: ": f"{aqi_data["pm10"]} µg/m³",
            "Carbon Monoxide (CO): " : f"{aqi_data["co"]} ppb",
            "Temperature": f"{aqi_data['temperature']}°C",
            "Humidity": f"{aqi_data['humidity']}%",
            "Wind Speed": f"{aqi_data['wind_speed']} km/h"
        }, indent=2)
        
        # Get Recommendations
        recommendations = health_agent.get_recommendation(aqi_data, user_input)
        
        warning_msg = """
        Note: The data shown may not match real-time values on the website.
        This could be due to:
        - Cached data in Firecrawl
        - Rate Limiting
        - Website updates not being captured
        
        Consider refreshing or checking the website directly for real-time values
        """
        
        return aqi_json, recommendations, info_msg, warning_msg
    
    except Exception as e:
        error_msg = f"Error Occured: {str(e)}"
        return "", "Analysis Failed", error_msg, ""
    
def create_demo() -> gr.Blocks:
    """Create and configure the gradio interface"""
    
    with gr.Blocks(title = "AQL Analysis and Recommendation Agent") as Demo:
        gr.Markdown(
            """
            AQI Analysis Agent
            Get personalized health recommendations based on air quality conditions.
            """
        )
        
        # API Configurations
        with gr.Accordion("API Configuration", open=False):
            firecrawl_key = gr.Textbox(
                label="Firecrawl API Key",
                type="password",
                placeholder="Enter your Firecrawl API Key"
            )
            
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                type = "password",
                placeholder="Enter your OpenAI API Key"
            )
            
        # Location Details
        with gr.Row():
            with gr.Column():
                city = gr.Textbox(label="City", placeholder="eg. Mumbai")
                state = gr.Textbox(
                    label="State",
                    placeholder="Leave blank for UT or US Cities",
                    value = ""
                )
                country = gr.Textbox(label="Country", value = "India")
        # Personal Details
        with gr.Row():
            with gr.Column():
                medical_conditions = gr.Textbox(
                    label="Medical Conditions (optional)",
                    placeholder="e.g., asthma, allergies",
                    lines=2
                )
                planned_activity = gr.Textbox(
                    label="Planned Activity",
                    placeholder="e.g., morning jog for 2 hours",
                    lines=2
                )
        
        # Status Messages
        info_box = gr.Textbox(label="ℹ️ Status", interactive=False)
        warning_box = gr.Textbox(label="⚠️ Warning", interactive=False)
        
        # Output Areas
        aqi_data_json = gr.JSON(label="Current Air Quality Data")
        recommendations = gr.Markdown(label="Health Recommendations")
        
        # Analyze Button
        analyze_btn = gr.Button("🔍 Analyze & Get Recommendations", variant="primary")
        analyze_btn.click(
            fn=analyze_conditions,
            inputs=[
                city,
                state,
                country,
                medical_conditions,
                planned_activity,
                firecrawl_key,
                openai_key
            ],
            outputs=[aqi_data_json, recommendations, info_box, warning_box]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Mumbai", "Maharashtra", "India", "asthma", "morning walk for 30 minutes"],
                ["Delhi", "", "India", "", "outdoor yoga session"],
                ["New York", "", "United States", "allergies", "afternoon run"],
                ["Kakinada", "Andhra Pradesh", "India", "none", "Tennis for 2 hours"]
            ],
            inputs=[city, state, country, medical_conditions, planned_activity]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)