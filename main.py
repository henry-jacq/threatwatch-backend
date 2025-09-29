from fastapi import FastAPI
from app.api.routes import agents, insights, simulation

app = FastAPI(title="Network Admin Backend")

# Register routers
app.include_router(agents.router, prefix="/agents", tags=["Agents"])
app.include_router(insights.router, prefix="/insights", tags=["Insights"])
app.include_router(simulation.router, prefix="/simulate", tags=["Simulation"])

@app.get("/")
def healthcheck():
    return {"status": "ok"}
