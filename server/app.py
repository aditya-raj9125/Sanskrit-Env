import sys
from pathlib import Path

# Ensure the project root is on sys.path so models/graders imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server import create_fastapi_app
from server.environment import SanskritEnvironment
from models import ManuscriptAction, ManuscriptObservation

app = create_fastapi_app(
    SanskritEnvironment,       # callable factory — calling it returns an Environment instance
    ManuscriptAction,          # action type class
    ManuscriptObservation,     # observation type class
)
