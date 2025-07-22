from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# Import all routers
from .routers import programs, connections, discovery, system, budgets, automation, settings

# Create FastAPI application
app = FastAPI(
    title="Affiliate Matrix API",
    description="API for the Affiliate Matrix automated affiliate marketing system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure with appropriate origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(programs.router, prefix="/api")
app.include_router(connections.router, prefix="/api")
app.include_router(discovery.router, prefix="/api")
app.include_router(system.router, prefix="/api")
app.include_router(budgets.router, prefix="/api")
app.include_router(automation.router, prefix="/api")
app.include_router(settings.router, prefix="/api")

# Add middleware for authentication, database sessions, etc.
# TODO: Add authentication middleware
# TODO: Add database session middleware

# Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Initialize resources on startup.
    
    TODO: Initialize database connection
    TODO: Initialize caching
    TODO: Initialize background task scheduler
    """
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown.
    
    TODO: Close database connection
    TODO: Clean up any other resources
    """
    pass

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint that returns basic API information.
    """
    return {
        "name": "Affiliate Matrix API",
        "version": "1.0.0",
        "status": "online"
    }
