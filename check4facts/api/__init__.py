import os
import time
from flask.cli import load_dotenv
import yaml
from celery import Celery, Task
from flask_cors import CORS
from flask import Flask, request, jsonify
from check4facts.api.tasks import *
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
    CORS(app)
    app.config.from_mapping(
        CELERY=dict(
            broker_url=os.getenv("CELERY_BROKER_URL"),
            result_backend=os.getenv("CELERY_RESULT_BACKEND"),
            task_ignore_result=True,
        ),
    )
    app.config.from_prefixed_env()
    celery_init_app(app)

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

    @app.route("/justify", methods=["POST"])
    def justify():

        req = request.json

        statement_id = req.get("id")
        n = req.get("n")
        task = justify_task.apply_async(kwargs={"statement_id": statement_id, "n": n})

        return (
            jsonify({"taskId": task.id, "status": task.status, "taskInfo": task.info}),
            202,
        )

    @app.route("/batch-justify", methods=["POST"])
    def batch_justify():

        req = request.json

        n = req.get("n")
        task = batch_justify_task.apply_async(kwargs={"n": n})

        return (
            jsonify({"taskId": task.id, "status": task.status, "taskInfo": task.info}),
            202,
        )

    # Test endpoints

    @app.route("/rag-test", methods=["POST"])
    def get_rag():

        data = request.get_json()
        claim = data.get("text", "")
        n = data.get("n", 3)
        article_id = data.get("article_id", 9999)
        print(f"article_id: {article_id}, claim: {claim}, n: {n}")  # Debugging

        task = run_rag.apply_async(
            kwargs={"article_id": article_id, "claim": claim, "n": n}
        )
        return (
            jsonify({"taskId": task.id, "status": task.status, "taskInfo": task.info}),
            202,
        )

    @app.route("/rag-batch", methods=["GET"])
    def get_rag_2():

        task = run_batch_rag.apply_async(kwargs={})
        return (
            jsonify({"taskId": task.id, "status": task.status, "taskInfo": task.info}),
            202,
        )

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
