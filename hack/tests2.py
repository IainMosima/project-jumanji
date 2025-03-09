from langchain_openai import ChatOpenAI
import requests
import json
import base64
from pathlib import Path

from hack.agents import aerial_photo_analysis
from hack.graph_agent import create_aerial_analysis_graph

def test_original_method():
    # Instantiate your LLM (ensure your API key is configured appropriately)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Provide the S3 image key to be analyzed
    image_key = "20180627_seq_50m_NC.tif"
    
    # Call the aerial photo analysis pipeline function
    result = aerial_photo_analysis(llm, image_key)
    
    # Print the final agent decision
    print("Final agent decision (original method):", result)

def test_graph_method():
    # Instantiate your LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create the analysis graph
    analysis_graph = create_aerial_analysis_graph(llm)
    
    # Set initial state with S3 key
    initial_state = {
        "image": None,
        "image_key": "20180627_seq_50m_NC.tif",
        "verification_status": False,
        "carbon_credits": 0.0,
        "analysis_result": {},
        "messages": []
    }
    
    # Run the graph
    result = analysis_graph.invoke(initial_state)
    
    # Print the result
    print("Final agent decision (graph method):")
    print(f"  Verification status: {result['verification_status']}")
    print(f"  Carbon credits: {result['carbon_credits']}")
    print(f"  Analysis details: {json.dumps(result['analysis_result'], indent=2)}")

def test_api(api_url="http://localhost:8000"):
    """Test the API by sending a request with an image."""
    # Replace with your actual image path
    image_path = Path("/path/to/your/image.tif")
    
    if image_path.exists():
        # Test with file upload
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/tiff")}
            response = requests.post(f"{api_url}/analyze-aerial-photo", files=files)
            print("API response (file upload):", response.json())
    
    # Test with S3 key
    data = {"s3_key": "20180627_seq_50m_NC.tif"}
    response = requests.post(f"{api_url}/analyze-aerial-photo", data=data)
    print("API response (S3 key):", response.json())

if __name__ == "__main__":
    # Run tests
    test_original_method()
    test_graph_method()
    
    # Uncomment to test the API (ensure API is running)
    # test_api()
