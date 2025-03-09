from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
import os
import boto3
import io
from typing import Optional
from tempfile import NamedTemporaryFile

from hack.agents import aerial_photo_analysis
from hack.graph_agent import create_aerial_analysis_graph

app = FastAPI(title="Aerial Photo Analysis API")

# Configure environment
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
USE_GRAPH_METHOD = os.getenv("USE_GRAPH_METHOD", "true").lower() == "true"

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET", "your-default-bucket-name")
s3_client = boto3.client('s3')

@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy"}

@app.post("/analyze-aerial-photo")
async def analyze_photo(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    s3_key: Optional[str] = Form(None)
):
    """
    Analyze aerial photos for environmental verification.
    Accepts either a direct file upload or an S3 key reference.
    """
    if not file and not s3_key:
        return JSONResponse(
            status_code=400,
            content={"error": "Either a file or an S3 key must be provided"}
        )

    try:
        # Initialize the LLM
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
        
        # Process file upload or S3 key
        if file:
            # For direct file uploads
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(await file.read())
                temp_path = temp_file.name
            
            # Upload to S3 and get key
            s3_key = f"uploads/{file.filename}"
            s3_client.upload_file(temp_path, S3_BUCKET, s3_key)
            os.unlink(temp_path)  # Delete the temporary file
        
        # Process with appropriate method
        if USE_GRAPH_METHOD:
            # Use graph method
            analysis_graph = create_aerial_analysis_graph(llm)
            initial_state = {
                "image": None,
                "image_key": s3_key,
                "verification_status": False,
                "carbon_credits": 0.0,
                "analysis_result": {},
                "messages": []
            }
            result = analysis_graph.invoke(initial_state)
            
            return {
                "verification_status": result["verification_status"],
                "carbon_credits": result["carbon_credits"],
                "analysis_result": result["analysis_result"],
                "s3_key": s3_key
            }
        else:
            # Use original method
            result = aerial_photo_analysis(llm, s3_key)
            
            return {
                "result": result,
                "s3_key": s3_key
            }
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
