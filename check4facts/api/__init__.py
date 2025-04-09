from contextlib import asynccontextmanager
import os
import jwt
import time
import yaml
import base64
from celery import Celery
from dotenv import load_dotenv
from check4facts.api.tasks import *
from check4facts.config import DirConf
from check4facts.database import DBHandler
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
)

"""
This is responsible for creating the API layer app for our python module Check4Facts
"""

load_dotenv(dotenv_path="../../.env")


db_path = os.path.join(DirConf.CONFIG_DIR, "db_config.yml")  # while using uwsgi
with open(db_path, "r") as db_f:
    db_params = yaml.safe_load(db_f)
dbh = DBHandler(**db_params)


@asynccontextmanager
async def lifespan(app: FastAPI):
    dbh.connect()

    async def handle_notify(payload):
        task_id = payload.get("task_id")
        if task_id in connections:
            ws = connections[task_id]
            await ws.send_json(payload)

            # ✅ If task is done, close socket and remove it
            if payload.get("status") == "completed":
                await ws.close(code=1000)
                connections.pop(task_id, None)

    dbh.listen("task_progress", handle_notify)
    print("Listening on channel: task_progress")  # ← This too
    yield
    dbh.disconnect()

BASE64_JWT_KEY = os.getenv("JWT_SECRET_KEY")
if not BASE64_JWT_KEY:
    raise ValueError("JWT_SECRET_KEY is not set in the environment variables.")

try:
    SECRET_KEY = base64.b64decode(BASE64_JWT_KEY)
except Exception as e:
    raise ValueError(
        "Failed to decode JWT_SECRET_KEY. Ensure it is a valid base64-encoded string."
    ) from e


# Custom middleware for adding Content Security Policy (CSP) header
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Set Content Security Policy header
        # response.headers["Content-Security-Policy"] = (
        #     "default-src 'self'; "
        #     "script-src 'self' http://localhost:8080; "  # Allow React app at localhost:8080
        #     "style-src 'self' 'unsafe-inline'; "
        #     "connect-src 'self' http://localhost:9090; "  # Allow FastAPI backend at localhost:9090
        #     "img-src 'self' data:;"
        # )
        response.headers["Content-Security-Policy"] = (
            "default-src *; script-src *; connect-src *;"
        )

        return response


def validate_jwt(token):
    try:
        decoded = jwt.decode(
            token, SECRET_KEY, algorithms=["HS512"]
        )  # Adjust algorithm if needed
        return decoded  # Return decoded data if valid
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


def celery_init_app(app: FastAPI) -> Celery:
    celery_app = Celery(app.title)
    celery_app.conf.broker_connection_retry_on_startup = True
    celery_app.conf.broker_url = os.getenv("CELERY_BROKER_URL")
    celery_app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND")
    return celery_app


app = FastAPI(lifespan=lifespan)

# Add CORS middleware (update origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "x-xsrf-token"],
)


# Add CSPMiddleware
app.add_middleware(CSPMiddleware)


# JWT validation dependency for FastAPI
async def get_current_user(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    token = auth.split(" ")[1]
    decoded = validate_jwt(token)
    if not decoded:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    return decoded

# JWT validation for WebSocket connections
async def get_current_user_from_ws(websocket: WebSocket):
    auth_header = websocket.headers.get("Authorization")    
    if not auth_header or not auth_header.startswith("Bearer "):
        # You can use WebSocketDisconnect or raise an HTTPException here, 
        # but raising an exception during the handshake will cause a connection error.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    token = auth_header.split(" ")[1]
    decoded = validate_jwt(token)
    if not decoded:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    return decoded


# REST endpoints (converted to async functions)
@app.post("/analyze")
async def analyze_endpoint(
    statement: dict, current_user: dict = Depends(get_current_user)
):
    task = analyze_task.apply_async(kwargs={"statement": statement})
    return {
        "status": "PROGRESS",
        "taskId": task.task_id,
        "taskInfo": {"current": 1, "total": 4, "type": statement.get("id")},
    }


@app.post("/train")
async def train_endpoint(current_user: dict = Depends(get_current_user)):
    task = train_task.apply_async(
        task_id=f"train_task_on_{time.strftime('%Y-%m-%d-%H:%M')}"
    )
    return {
        "status": "PROGRESS",
        "taskId": task.task_id,
        "taskInfo": {"current": 1, "total": 2, "type": "TRAIN"},
    }


@app.get("/intial-train")
async def initial_train_endpoint(current_user: dict = Depends(get_current_user)):
    total = dbh.count_statements()
    task = intial_train_task.apply_async()
    return {
        "status": "PROGRESS",
        "taskId": task.task_id,
        "taskInfo": {"current": 1, "total": (4 * total) + 1, "type": "INITIAL_TRAIN"},
    }


@app.get("/task-status/{task_id}")
async def task_status_endpoint(
    task_id: str, current_user: dict = Depends(get_current_user)
):
    result = status_task(task_id)
    return {"taskId": task_id, "status": result.status, "taskInfo": result.info}


@app.post("/batch-task-status")
async def batch_task_status_endpoint(
    json: list = [], current_user: dict = Depends(get_current_user)
):
    response = []
    for j in json:
        result = status_task(j["id"])
        response.append(
            {"taskId": j["id"], "status": result.status, "taskInfo": result.info}
        )
    return response


@app.get("/fetch-active-tasks")
async def fetch_active_tasks_endpoint(current_user: dict = Depends(get_current_user)):
    try:
        task_ids = dbh.fetch_active_tasks_ids()
        response = []
        for task_id in task_ids:
            result = status_task(task_id)
            response.append(
                {"taskId": task_id, "status": result.status, "taskInfo": result.info}
            )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetch active celery tasks from database: {e}",
        )


@app.post("/summarize/{article_id}")
async def summ_endpoint(
    article_id: str, current_user: dict = Depends(get_current_user)
):
    print(current_user)
    task = summarize_text.apply_async(kwargs={"article_id": article_id})
    return {"taskId": task.id, "status": task.status, "taskInfo": task.info}


@app.post("/batch-summarize")
async def batch_summ_endpoint(current_user: dict = Depends(get_current_user)):
    task = batch_summarize_text.apply_async(kwargs={})
    return {"taskId": task.id, "status": task.status, "taskInfo": task.info}


# Test endpoints
@app.post("/test/summarize")
async def test_get_summ_endpoint(
    json: dict, current_user: dict = Depends(get_current_user)
):
    article_id = json["article_id"]
    text = json["text"]
    result = test_summarize_text.apply_async(
        kwargs={"article_id": article_id, "text": text}
    )
    return {"task_id": result.id, "status": result.status}


@app.post("/start-dummy-task")
async def start_dummy_task_endpoint(current_user: dict = Depends(get_current_user)):
    task = dummy_task.apply_async(kwargs={})
    return {"taskId": task.id, "status": task.status, "taskInfo": task.info}

# Dictionary to store WebSocket connections for each task
connections = {}

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    try:
        # Validate JWT during connection establishment
        user = await get_current_user_from_ws(websocket)
        print(f"User connected: {user}")

        await websocket.accept()
        connections[task_id] = websocket  # Assume you already have a connections dict
        try:
            while True:
                data = await websocket.receive_text()  # Keep the connection alive or process incoming messages
                print(f"Received data: {data}")
        except WebSocketDisconnect:
            print(f"WebSocket disconnected for task {task_id}")
        finally:
            connections.pop(task_id, None)
    except HTTPException as exc:
        # The connection will not be accepted if token is invalid/expired
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        print(f"Connection closed due to authentication error: {exc.detail}")
