from typing import Dict, Any, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import base64
import io
from PIL import Image

from hack.agents import aerial_photo_analysis

class AgentState(TypedDict):
    """State for the aerial analysis agent."""
    image: Optional[str]  # Base64 encoded image
    image_key: Optional[str]  # S3 image key
    verification_status: bool
    carbon_credits: float
    analysis_result: Dict[str, Any]
    messages: List[BaseMessage]

def verify_image(state: AgentState) -> AgentState:
    """Verify if the provided image is valid for analysis."""
    if state.get("image"):
        try:
            # Decode base64 image
            image_data = base64.b64decode(state["image"])
            Image.open(io.BytesIO(image_data))
            state["verification_status"] = True
        except Exception:
            state["verification_status"] = False
    else:
        # If using S3 key directly (assuming it exists)
        state["verification_status"] = True
    
    return state

def perform_analysis(state: AgentState, llm: ChatOpenAI) -> AgentState:
    """Perform aerial analysis on the image."""
    if not state["verification_status"]:
        state["analysis_result"] = {"error": "Image verification failed"}
        state["carbon_credits"] = 0.0
        return state
    
    # Use existing image key or store the uploaded image (simplified)
    image_key = state.get("image_key", "uploaded_image.tif")
    
    # Call existing analysis function
    result = aerial_photo_analysis(llm, image_key)
    
    # Process the result to extract carbon credits
    state["analysis_result"] = result
    # Assuming the result contains carbon credit info or calculate based on result
    state["carbon_credits"] = float(result.get("carbon_credits", 0.0))
    
    return state

def create_aerial_analysis_graph(llm: ChatOpenAI) -> StateGraph:
    """Create a graph for aerial photo analysis."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("verify_image", verify_image)
    workflow.add_node("perform_analysis", lambda state: perform_analysis(state, llm))
    
    # Add edges
    workflow.add_edge("verify_image", "perform_analysis")
    workflow.add_edge("perform_analysis", END)
    
    # Set entry point
    workflow.set_entry_point("verify_image")
    
    return workflow.compile()
