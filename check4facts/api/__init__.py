import os
import time
from flask.cli import load_dotenv
import yaml
from celery import Celery, Task
from flask_cors import CORS
from flask import Flask, request, jsonify
import jwt
import base64
from check4facts.api.tasks import (
    status_task,
    analyze_task,
    train_task,
    intial_train_task,
    summarize_text,
    test_summarize_text,
    batch_summarize_text,
)
from check4facts.config import DirConf
from check4facts.database import DBHandler

"""
This is responsible for creating the API layer app for our python module Check4Facts
"""

load_dotenv(path="../../.env")


db_path = os.path.join(DirConf.CONFIG_DIR, "db_config.yml")  # while using uwsgi
with open(db_path, "r") as db_f:
    db_params = yaml.safe_load(db_f)
dbh = DBHandler(**db_params)


BASE64_JWT_KEY = os.getenv("JWT_SECRET_KEY")
if not BASE64_JWT_KEY:
    raise ValueError("JWT_SECRET_KEY is not set in the environment variables.")

try:
    SECRET_KEY = base64.b64decode(BASE64_JWT_KEY)
except Exception as e:
    raise ValueError(
        "Failed to decode JWT_SECRET_KEY. Ensure it is a valid base64-encoded string."
    ) from e


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


def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(
        app,
        resources={
            r"/*": {"origins": "*", "allow_headers": ["Authorization", "Content-Type", "x-xsrf-token"]}
        },
        supports_credentials=True,
    )
    app.config.from_mapping(
        CELERY=dict(
            broker_url=os.getenv("CELERY_BROKER_URL"),
            result_backend=os.getenv("CELERY_RESULT_BACKEND"),
            task_ignore_result=True,
        ),
    )
    app.config.from_prefixed_env()
    celery_init_app(app)

    @app.before_request
    def handle_preflight_and_auth():
        """Handle preflight requests and authenticate requests."""
        if request.method == "OPTIONS":
            # Handle CORS preflight request
            return "", 204  # No Content response for preflight

        # Authentication check
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401

        token = auth_header.split(" ")[1]  # Extract token
        decoded = validate_jwt(token)  # Assuming this is your function to validate JWT
        if not decoded:
            return jsonify({"error": "Invalid or expired token"}), 401

        request.user = decoded  # Store decoded user data in request object

    @app.route("/analyze", methods=["POST"])
    def analyze():
        statement = request.json

        task = analyze_task.apply_async(kwargs={"statement": statement})

        return jsonify(
            {
                "status": "PROGRESS",
                "taskId": task.task_id,
                "taskInfo": {
                    "current": 1,
                    "total": 4,
                    "type": f'{statement.get("id")}',
                },
            }
        )

    @app.route("/train", methods=["POST"])
    def train():
        task = train_task.apply_async(
            task_id=f"train_task_on_{time.strftime('%Y-%m-%d-%H:%M')}"
        )

        return jsonify(
            {
                "status": "PROGRESS",
                "taskId": task.task_id,
                "taskInfo": {"current": 1, "total": 2, "type": "TRAIN"},
            }
        )

    @app.route("/intial-train", methods=["GET"])
    def initial_train():
        total = dbh.count_statements()

        task = intial_train_task.apply_async()

        return jsonify(
            {
                "status": "PROGRESS",
                "taskId": task.task_id,
                "taskInfo": {
                    "current": 1,
                    "total": (4 * total) + 1,
                    "type": "INITIAL_TRAIN",
                },
            }
        )

    @app.route("/task-status/<task_id>", methods=["GET"])
    def task_status(task_id):
        result = status_task(task_id)

        return jsonify(
            {"taskId": task_id, "status": result.status, "taskInfo": result.info}
        )

    @app.route("/batch-task-status", methods=["POST"])
    def batch_task_status():
        json = request.json

        response = []
        for j in json:
            result = status_task(j["id"])
            response.append(
                {"taskId": j["id"], "status": result.status, "taskInfo": result.info}
            )

        return jsonify(response)

    @app.route("/fetch-active-tasks", methods=["GET"])
    def fetch_active_tasks():
        try:
            task_ids = dbh.fetch_active_tasks_ids()
            response = []
            for task_id in task_ids:
                result = status_task(task_id)
                response.append(
                    {
                        "taskId": task_id,
                        "status": result.status,
                        "taskInfo": result.info,
                    }
                )
            return jsonify(response)
        except Exception as e:
            return (
                jsonify(
                    {
                        "status": "ERROR",
                        "message": f"Error fetch active celery tasks from database: {e}",
                    }
                ),
                400,
            )

    @app.route("/summarize/<article_id>", methods=["POST"])
    def summ(article_id):

        task = summarize_text.apply_async(kwargs={"article_id": article_id})
        return (
            jsonify({"taskId": task.id, "status": task.status, "taskInfo": task.info}),
            202,
        )

    @app.route("/batch-summarize", methods=["POST"])
    def batch_summ():

        task = batch_summarize_text.apply_async(kwargs={})

        return (
            jsonify({"taskId": task.id, "status": task.status, "taskInfo": task.info}),
            202,
        )

    # Test endpoints

    @app.route("/test/summarize", methods=["POST"])
    def test_get_summ():
        json = request.json
        article_id = json["article_id"]
        text = json["text"]

        result = test_summarize_text.apply_async(
            kwargs={"article_id": article_id, "text": text}
        )
        return jsonify({"task_id": result.id, "status": result.status}), 200

    return app


flask_app = create_app()
celery_app = flask_app.extensions["celery"]
