import json
import os
import time
import yaml
import numpy as np
from check4facts.train import Trainer
from celery import result, shared_task
from check4facts.config import DirConf
from check4facts.predict import Predictor
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.search import SearchEngine
from check4facts.scripts.features import FeaturesExtractor
from check4facts.database import task_channel_name
from check4facts.logging import get_logger

# imports for text summarization
from check4facts.scripts.text_sum.local_llm import invoke_local_llm, google_llm
from check4facts.scripts.text_sum.groq_api import groq_api
from check4facts.scripts.text_sum.text_process import (
    extract_text_from_html,
)

# imports for rag
from check4facts.scripts.rag.pipeline import run_pipeline
from check4facts.scripts.rag.batch_process import testing

log = get_logger()

@shared_task(bind=True, ignore_result=False)
def status_task(self, task_id):
    return result.AsyncResult(task_id)


@shared_task(bind=True, ignore_result=False)
def analyze_task(self, statement):
    from check4facts.api import dbh
    progress = {
        "taskId": self.request.id,
        "progress": 0,
        "status": "PROGRESS",
        "type": "analyze",
    }
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

    statement_id = statement.get("id")
    statement_text = statement.get("text")

    log.info(
        f'[Worker: {os.getpid()}] Started analyze procedure for statement id: "{statement_id}"'
    )

    path = os.path.join(DirConf.CONFIG_DIR, "search_config.yml")  # when using uwsgi.
    with open(path, "r") as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    statements = [statement_text]

    progress["progress"] = 20
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(statements)[0]

    path = os.path.join(DirConf.CONFIG_DIR, "harvest_config.yml")  # while using uwsgi.
    with open(path, "r") as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [
        {"s_id": statement_id, "s_text": statement_text, "s_resources": search_results}
    ]

    progress["progress"] = 40
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]

    if harvest_results.empty:
        log.info(
            f'[Worker: {os.getpid()}] No resources found for statement id: "{statement_id}"'
        )
    #     return

    path = os.path.join(DirConf.CONFIG_DIR, "features_config.yml")  # while using uwsgi
    with open(path, "r") as f:
        features_params = yaml.safe_load(f)
    fe = FeaturesExtractor(**features_params)
    statement_dicts = [
        {"s_id": statement_id, "s_text": statement_text, "s_resources": harvest_results}
    ]

    progress["progress"] = 60
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
    features_results = fe.run(statement_dicts)[0]

    if harvest_results.empty:
        predict_result = np.array([-1.0, -1.0])
    else:
        path = os.path.join(DirConf.CONFIG_DIR, "predict_config.yml")
        with open(path, "r") as f:
            predict_params = yaml.safe_load(f)
        p = Predictor(**predict_params)

        progress["progress"] = 80
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        predict_result = p.run([features_results]).loc[0, ["pred_0", "pred_1"]].values

    resource_records = harvest_results.to_dict("records")
    dbh.insert_statement_resources(statement_id, resource_records)
    log.info(
        f'[Worker: {os.getpid()}] Finished storing harvest results for statement id: "{statement_id}"'
    )
    dbh.insert_statement_features(statement_id, features_results, predict_result, None)
    log.info(
        f'[Worker: {os.getpid()}] Finished storing features results for statement id: "{statement_id}"'
    )

    progress["progress"] = 100
    progress["status"] = "SUCCESS"
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))


@shared_task(bind=True, ignore_result=False)
def train_task(self):
    from check4facts.api import dbh
    progress = {
        "taskId": self.request.id,
        "progress": 0,
        "status": "PROGRESS",
        "type": "train",
    }
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
    
    path = os.path.join(DirConf.CONFIG_DIR, "train_config.yml")
    with open(path, "r") as f:
        train_params = yaml.safe_load(f)
    t = Trainer(**train_params)
    
    progress["progress"] = 33
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

    features_records = dbh.fetch_statement_features(train_params["features"])
    features = np.vstack([np.hstack(f) for f in features_records])
    labels = dbh.fetch_statement_labels()
    t.run(features, labels)
    
    progress["progress"] = 66
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

    if not os.path.exists(DirConf.MODELS_DIR):
        os.mkdir(DirConf.MODELS_DIR)
    fname = t.best_model["clf"] + "_" + time.strftime("%Y-%m-%d-%H:%M") + ".joblib"
    path = os.path.join(DirConf.MODELS_DIR, fname)
    t.save_best_model(path)
    
    progress["progress"] = 100
    progress["status"] = "SUCCESS"
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))


@shared_task(bind=True, ignore_result=False)
def intial_train_task(self):
    from check4facts.api import dbh
    progress = {
        "taskId": self.request.id,
        "progress": 0,
        "status": "PROGRESS",
        "type": "train",
    }
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

    # Initialize all python modules.
    path = os.path.join(DirConf.CONFIG_DIR, "search_config.yml")  # when using uwsgi.
    with open(path, "r") as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)

    path = os.path.join(DirConf.CONFIG_DIR, "harvest_config.yml")  # while using uwsgi.
    with open(path, "r") as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)

    path = os.path.join(DirConf.CONFIG_DIR, "features_config.yml")  # while using uwsgi
    with open(path, "r") as f:
        features_params = yaml.safe_load(f)
    fe = FeaturesExtractor(**features_params)

    # Get all statements from database.
    statements = dbh.fetch_statements()
    total_count = len(statements)
    counter = 0

    # Execute all steps for each statement.
    for statement in statements:
        statement_id, text, true_label = statement[0], statement[1], statement[2]
        counter += 1
        
        progress["progress"] = counter / total_count * 100
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

        print(f"Starting search for {statement_id}")

        search_results = se.run([text])[0]

        print(f"Starting harvest for {statement_id}")
        articles = [
            {"s_id": statement_id, "s_text": text, "s_resources": search_results}
        ]
        harvest_results = h.run(articles)[0]

        print(f"Saving Harvest Results to db  for {statement_id}")
        resource_records = harvest_results.to_dict("records")
        dbh.insert_statement_resources(statement_id, resource_records)

        print(f"Starting feature for {statement_id}")
        statement_dicts = [
            {"s_id": statement_id, "s_text": text, "s_resources": harvest_results}
        ]
        features_results = fe.run(statement_dicts)[0]

        print(f"Saving Feature Results to db for {statement_id}")
        dbh.insert_statement_features(statement_id, features_results, None, true_label)

    print(f"Initiating model training.")
    path = os.path.join(DirConf.CONFIG_DIR, "train_config.yml")
    with open(path, "r") as f:
        train_params = yaml.safe_load(f)
    t = Trainer(**train_params)

    features_records = dbh.fetch_statement_features(train_params["features"])
    features = np.vstack([np.hstack(f) for f in features_records])
    labels = dbh.fetch_statement_labels()
    t.run(features, labels)

    if not os.path.exists(DirConf.MODELS_DIR):
        os.mkdir(DirConf.MODELS_DIR)
    fname = t.best_model["clf"] + "_" + time.strftime("%Y-%m-%d-%H:%M") + ".joblib"
    path = os.path.join(DirConf.MODELS_DIR, fname)
    t.save_best_model(path)
    print(f"Successfully saved the best model.")
        
    progress["status"] = "SUCCESS"
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))


# Tasks for text summarization


@shared_task(bind=True, ignore_result=False)
def summarize_text(self, article_id):
    from check4facts.api import dbh

    try:
        progress = {
            "taskId": self.request.id,
            "progress": 0,
            "status": "PROGRESS",
            "type": "summarize",
        }
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        content = dbh.fetch_article_content(article_id)

        progress["progress"] = 20
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

        # Keep only the actual text from the article's content
        content = extract_text_from_html(content)
        progress["progress"] = 40
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

        # invoke Gemini llm
        log.debug("Trying to invoke gemini llm....")
        answer = google_llm(article_id, content)
        
        progress["progress"] = 60
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        if answer:
            result = {
                "summarization": answer["summarization"],
                "time": answer["elapsed_time"],
                "article_id": article_id,
                "timestamp": answer["timestamp"],
            }
        else:
            # try invoking the groq llm
            log.info("Trying to invoke groq llm....")
            api = groq_api()
            if api:
                answer = api.run(content)
                if answer is not None:
                    result = {
                        "summarization": answer["response"],
                        "time": answer["elapsed_time"],
                        "article_id": article_id,
                        "timestamp": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime()
                        ),
                    }

            # if everything fails, invoke the local model
            if (not api) or (answer is None):
                result = invoke_local_llm(content, article_id)

        log.info(
            f"Finished generating summary in: {result['time']} seconds. Storing in database..."
        )
        
        progress["progress"] = 80
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

        dbh.insert_summary(article_id, result["summarization"])
        progress["progress"] = 100
        progress["status"] = "SUCCESS"
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        
        dbh.disconnect()
    except Exception as e:
        progress["status"] = "FAILURE"
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        log.error(f"Error generating summary for article with id {article_id}: {e}")


@shared_task(bind=True, ignore_result=False)
def batch_summarize_text(self):
    from check4facts.api import dbh

    try:

        # Fetch the id and content (extract_text_from_html function used)
        # from all published articles and null summary
        articles = dbh.fetch_articles_without_summary()

        log.info(f"Total articles fetched: {len(articles)}")
        for index, article in enumerate(articles):
            log.info(
                f"Processing article {index + 1}/{len(articles)} with id: {article[0]}"
            )
            article_id, content = article[0], article[1]

            self.update_state(
                state="PROGRESS",
                meta={
                    "current": index + 1,
                    "total": len(articles),
                    "type": "BATCH_SUMMARIZE",
                },
            )
            # invoke Gemini llm
            log.info("Trying to invoke gemini llm....")
            answer = google_llm(article_id, content)
            if answer:
                result = {
                    "summarization": answer["summarization"],
                    "time": answer["elapsed_time"],
                    "article_id": article_id,
                    "timestamp": answer["timestamp"],
                }
            else:
                # if Gemini Fails, call Groq
                answer = groq_api().run(content)
                if answer is not None:
                    result = {
                        "summarization": answer["response"],
                        "time": answer["elapsed_time"],
                        "article_id": article_id,
                        "timestamp": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime()
                        ),
                    }

            log.info(result["summarization"])
            dbh.insert_summary(article_id, result["summarization"])
            time.sleep(5)

        dbh.disconnect()

    except Exception as e:
        log.error(f"Error generating summary for published articles: {e}")


@shared_task(bind=True, ignore_result=False)
def justify_task(self, statement_id, n):
    from check4facts.api import dbh
    progress = {
        "taskId": self.request.id,
        "progress": 0,
        "status": "PROGRESS",
        "type": "justify",
    }
    dbh.notify(task_channel_name(self.request.id), json.dumps(progress))

    text = dbh.fetch_statement_text(statement_id)
    try:

        progress["progress"] = 40
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        answer = run_pipeline(statement_id, text, n, None)
        if answer:
            
            progress["progress"] = 80
            dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
            log.debug("FINAL ANSWER: ")
            for key, value in answer.items():
                log.debug(f"{key}: {value}")

            # Store to Database
            dbh.insert_justification(
                statement_id,
                answer["justification"],
                answer["timestamp"],
                answer["elapsed_time"],
                answer["label"],
                answer["model"],
                answer["sources"],
            )
            
            progress["progress"] = 100
            progress["status"] = "SUCCESS"
            dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        else:
            raise Exception("Pipeline returned empty result")

    except Exception as e:
        progress["status"] = "FAILURE"
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        log.error(f"Error during rag run: {e}")

    return


@shared_task(bind=True, ignore_result=False)
def batch_justify_task(self, n):
    from check4facts.api import dbh

    # Each row is statement_id, statement_text
    rows = dbh.fetch_all_statement_texts()
    for row in rows:
        # already_justified = []
        # if int(row[0]) in already_justified:
        #     print(f"Statement with id: {row[0]} has already a justification. Moving on...")
        #     continue
        try:

            answer = run_pipeline(row[0], row[1], n, None)
            if answer:
                log.debug("FINAL ANSWER: ")
                for key, value in answer.items():
                    log.debug(f"{key}: {value}")

                # Store to Database
                dbh.insert_justification(
                    row[0],
                    answer["justification"],
                    answer["timestamp"],
                    answer["elapsed_time"],
                    answer["label"],
                    answer["model"],
                    answer["sources"],
                )
            else:
                raise Exception("Pipeline returned empty result")

        except Exception as e:
            log.error(f"Error during rag run: {e}")
    return


# Test tasks


@shared_task(bind=True, ignore_result=False)
def test_summarize_text(self, article_id, text):
    try:
        answer = None

        # # Keep only the actual text from the article's content
        # text = extract_text_from_html(text)

        # try invoking the groq llm
        api = groq_api()
        if api:
            answer = api.run(text)
            if answer is not None:
                result = {
                    "summarization": answer["response"],
                    "time": answer["elapsed_time"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }

        if (not api) or (answer is None):
            # invoke Gemini llm
            print("Trying to invoke gemini llm....")
            answer = google_llm(article_id, text)
            if answer:
                result = {
                    "summarization": answer["summarization"],
                    "time": answer["elapsed_time"],
                    "timestamp": answer["timestamp"],
                }
            else:
                # if everything fails, invoke the local model
                result = invoke_local_llm(text, article_id)

        print(f"Finished generating summary in: {result['time']} seconds")

        return result
    except Exception as e:
        print(f"An exception occurred: {e}")


@shared_task(bind=False, ignore_result=False)
def run_rag(article_id, claim, n):
    try:
        answer = run_pipeline(article_id, claim, n, None)
        if answer:
            print("FINAL ANSWER: ")
            print()
            for key, value in answer.items():
                print(f"{key}: {value}")
            return answer
        else:
            raise Exception("Pipeline returned empty result")

    except Exception as e:
        print(f"Error during rag run: {e}")


@shared_task(bind=False, ignore_result=False)
def run_batch_rag():
    try:
        testing()
    except Exception as e:
        print(f"Error during batch rag run: {e}")

    pass

        
@shared_task(bind=True, ignore_result=False)
def dummy_task(self):
    from check4facts.api import dbh
    for i in range(5):
        progress = {"taskId": self.request.id, "progress": i * 20, "status": "PROGRESS"}
        dbh.notify(task_channel_name(self.request.id), json.dumps(progress))
        time.sleep(10)

    # ✅ Final notify with status=SUCCESS
    final = {"taskId": self.request.id, "progress": 100, "status": "SUCCESS"}
    dbh.notify(task_channel_name(self.request.id), json.dumps(final))
