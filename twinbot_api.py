from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
import traceback
from datetime import datetime
import os
from typing import Optional
import uvicorn
from dotenv import load_dotenv
load_dotenv()

# Import your existing TwinBot workflow
from twinbot_workflow import run_twinbot

# Initialize FastAPI app
app = FastAPI(
    title="TwinBot API",
    description="SCADA System Query Processing API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twinbot_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query about the SCADA system")
    user_id: Optional[str] = Field(None, max_length=100, description="Optional user identifier")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or just whitespace')
        return v.strip()

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully")
    response: str = Field(..., description="Natural language response from TwinBot")
    timestamp: str = Field(..., description="ISO timestamp of the response")
    query: str = Field(..., description="Original user query")
    processing_time_seconds: float = Field(..., description="Time taken to process the query")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code for programmatic handling")
    timestamp: str = Field(..., description="ISO timestamp of the error")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="ISO timestamp")
    service: str = Field(..., description="Service name")

class StatusResponse(BaseModel):
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="ISO timestamp")
    endpoints: dict = Field(..., description="Available endpoints")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        service="TwinBot API"
    )

@app.post("/api/twinbot/query", 
          response_model=QueryResponse,
          responses={
              400: {"model": ErrorResponse, "description": "Bad Request"},
              500: {"model": ErrorResponse, "description": "Internal Server Error"}
          },
          tags=["TwinBot"])
async def process_query(request: QueryRequest):
    """
    Process user queries and return natural language responses
    
    This endpoint processes queries about the SCADA system and returns
    human-readable responses based on real-time telemetry data.
    
    Example queries:
    - "What is the current tail-end pressure?"
    - "Which pump is currently running?"
    - "What is the water level in the sump tank?"
    - "Show me the power consumption of Pump 1"
    """
    try:
        # Log the incoming request
        logger.info(f"Processing query from user {request.user_id or 'anonymous'}: {request.query}")
        
        # Process the query using your TwinBot workflow
        start_time = datetime.utcnow()
        natural_response = run_twinbot(request.query)
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log successful processing
        logger.info(f"Query processed successfully in {processing_time:.2f}s for user {request.user_id or 'anonymous'}")
        
        # Return the natural language response
        return QueryResponse(
            success=True,
            response=natural_response,
            timestamp=datetime.utcnow().isoformat(),
            query=request.query,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        # Log the error
        error_trace = traceback.format_exc()
        logger.error(f"Error processing query: {str(e)}\n{error_trace}")
        
        # Return error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Internal server error occurred while processing your query",
                "code": "PROCESSING_ERROR",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/api/twinbot/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    """Get API status and basic information"""
    return StatusResponse(
        status="active",
        service="TwinBot Query Processor",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        endpoints={
            "query": "POST /api/twinbot/query",
            "status": "GET /api/twinbot/status",
            "health": "GET /health",
            "docs": "GET /docs",
            "redoc": "GET /redoc"
        }
    )

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "success": False,
        "error": exc.detail if isinstance(exc.detail, str) else exc.detail.get("error", "HTTP error"),
        "code": exc.detail.get("code", "HTTP_ERROR") if isinstance(exc.detail, dict) else "HTTP_ERROR",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return {
        "success": False,
        "error": "An unexpected error occurred",
        "code": "INTERNAL_ERROR",
        "timestamp": datetime.utcnow().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    # Check if required environment variables are set
    required_env_vars = ['GOOGLE_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    logger.info("TwinBot API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("TwinBot API shutting down")

if __name__ == '__main__':
    # Get configuration from environment variables
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    reload = os.getenv('API_RELOAD', 'False').lower() == 'true'
    
    logger.info(f"Starting TwinBot API server on {host}:{port}")
    logger.info(f"Reload mode: {reload}")
    logger.info(f"API Documentation available at: http://{host}:{port}/docs")
    
    # Start the FastAPI application
    uvicorn.run(
        "twinbot_api:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    )