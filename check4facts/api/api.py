import os
import time
import yaml
from celery import Celery
from flask_cors import CORS
from flask import request, jsonify
from check4facts.api.init import create_app
from check4facts.api.tasks import status_task, analyze_task, train_task, intial_train_task
from check4facts.config import DirConf
from check4facts.database import DBHandler

#text summarization imports
from check4facts.api.tasks import summarize_text, celery_get_task_result, celery_insert

app = create_app()
app.config['CELERY_BROKER_URL'] = 'sqla+postgresql://check4facts@localhost:5432/check4facts'
app.config['result_backend'] = 'db+postgresql://check4facts@localhost:5432/check4facts'

client = Celery(app.name, backend=app.config['result_backend'], broker=app.config['CELERY_BROKER_URL'])
client.conf.update(app.config)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze():
    statement = request.json

    task = analyze_task.apply_async(kwargs={'statement': statement})

    return jsonify({'status': 'PROGRESS',
                    'taskId': task.task_id,
                    'taskInfo': {'current': 1, 'total': 4,
                                 'type': f'{statement.get("id")}'}
                    })


@app.route('/train', methods=['POST'])
def train():
    task = train_task.apply_async(task_id=f"train_task_on_{time.strftime('%Y-%m-%d-%H:%M')}")

    return jsonify({'status': 'PROGRESS',
                    'taskId': task.task_id,
                    'taskInfo': {'current': 1, 'total': 2, 'type': 'TRAIN'}
                    })


@app.route('/intial-train', methods=['GET'])
def initial_train():
    db_path = os.path.join(DirConf.CONFIG_DIR, 'db_config.yml')  # while using uwsgi
    with open(db_path, 'r') as db_f:
        db_params = yaml.safe_load(db_f)
    dbh = DBHandler(**db_params)
    total = dbh.count_statements()

    task = intial_train_task.apply_async()

    return jsonify({
        'status': 'PROGRESS',
        'taskId': task.task_id,
        'taskInfo': {
            'current': 1,
            'total': (4 * total) + 1,
            'type': 'INITIAL_TRAIN'
        }
    })


@app.route('/task-status/<task_id>', methods=['GET'])
def task_status(task_id):
    result = status_task(task_id)

    return jsonify({'taskId': task_id, 'status': result.status, 'taskInfo': result.info})


@app.route('/batch-task-status', methods=['POST'])
def batch_task_status():
    json = request.json

    response = []
    for j in json:
        result = status_task(j['id'])
        response.append({'taskId': j['id'], 'status': result.status, 'taskInfo': result.info})

    return jsonify(response)


#Text summarization api functions

@app.route("/")
def index():
    return "Text summarization with the use of large language models"

@app.route("/summarize", methods=["POST"])
def get_sum():
    data = request.get_json()
    user_input = data.get('text', '').strip()
    article_id = data.get('article_id',9999)

    if not user_input:
        return jsonify({
            'status': 'error',
            'message': 'Text is empty. Please provide the "text" field in your JSON.'
        }), 400  

    task = summarize_text.delay(user_input, article_id)
    return jsonify({"task_id": task.id}), 202

#mainly for debugging purposes. Not to be used in production
@app.route('/db_fetch/<task_id>', methods=["GET"])
def fetch(task_id):
    task = celery_get_task_result.delay(task_id)

    return jsonify({'task_id': task.id, 'status': 'Task created, check status later.'})


@app.route('/db_insert/<task_id>', methods=["GET"])
def insert(task_id):

    task_result = client.AsyncResult(task_id, app=client)
    if task_result.state == 'PENDING':  
        return jsonify("The summary generation task is still pending. Please try again later.")
    elif task_result.state == 'STARTED':
        return jsonify("Summary is being generated. Please try again after some time.")
    elif task_result.state == 'FAILURE':
        return jsonify(f"The summary generation task failed: {task_result.info}")
    elif task_result.state == 'SUCCESS':
        task  = celery_insert.delay(task_id)
        return jsonify(f"Summary was successfully inserted into the database: {task_result.info}")
    else:
        return jsonify(f"Unexpected task state: {task_result.state}")
    

@app.route("/task_state/<task_id>", methods=["GET"])
def get_result(task_id):
    from celery.result import AsyncResult

    task_result = client.AsyncResult(task_id, app=client)

    if task_result.state == "PENDING":
        return jsonify({"status": "Task is still in progress"}), 202
    elif task_result.state == "FAILURE":
        return jsonify({"status": "Task failed", "error": str(task_result.info)}), 500
    else:
        if task_result.result:
            return jsonify({"status": "Task completed", "result": task_result.result}), 200
        else:
            return jsonify({"status": "Task completed"})










if __name__ == '__main__':
    app.run(debug=True)
