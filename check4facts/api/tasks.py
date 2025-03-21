import os
import time
import yaml
import numpy as np
from celery import result, shared_task
from check4facts.train import Trainer
from check4facts.config import DirConf
from check4facts.predict import Predictor
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.search import SearchEngine
from check4facts.scripts.features import FeaturesExtractor

# imports for text summarization
from check4facts.scripts.text_sum.local_llm import invoke_local_llm, google_llm
from check4facts.scripts.text_sum.groq_api import groq_api
from check4facts.scripts.text_sum.text_process import (
    extract_text_from_html,
)

# imports for rag
from check4facts.scripts.rag.pipeline import run_pipeline


@shared_task(bind=True, ignore_result=False)
def status_task(self, task_id):
    return result.AsyncResult(task_id)


@shared_task(bind=True, ignore_result=False)
def analyze_task(self, statement):
    from check4facts.api import dbh

    statement_id = statement.get("id")
    statement_text = statement.get("text")

    print(
        f'[Worker: {os.getpid()}] Started analyze procedure for statement id: "{statement_id}"'
    )

    path = os.path.join(DirConf.CONFIG_DIR, "search_config.yml")  # when using uwsgi.
    with open(path, "r") as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    statements = [statement_text]

    self.update_state(
        state="PROGRESS",
        meta={"current": 1, "total": 4, "type": f"ANALYZE_{statement_id}"},
    )
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(statements)[0]

    path = os.path.join(DirConf.CONFIG_DIR, "harvest_config.yml")  # while using uwsgi.
    with open(path, "r") as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [
        {"s_id": statement_id, "s_text": statement_text, "s_resources": search_results}
    ]

    self.update_state(
        state="PROGRESS",
        meta={"current": 2, "total": 4, "type": f"ANALYZE_{statement_id}"},
    )
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]

    if harvest_results.empty:
        print(
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

    self.update_state(
        state="PROGRESS",
        meta={"current": 3, "total": 4, "type": f"ANALYZE_{statement_id}"},
    )
    features_results = fe.run(statement_dicts)[0]

    if harvest_results.empty:
        predict_result = np.array([-1.0, -1.0])
    else:
        path = os.path.join(DirConf.CONFIG_DIR, "predict_config.yml")
        with open(path, "r") as f:
            predict_params = yaml.safe_load(f)
        p = Predictor(**predict_params)

        self.update_state(
            state="PROGRESS",
            meta={"current": 4, "total": 4, "type": f"ANALYZE_{statement_id}"},
        )
        predict_result = p.run([features_results]).loc[0, ["pred_0", "pred_1"]].values

    resource_records = harvest_results.to_dict("records")
    dbh.insert_statement_resources(statement_id, resource_records)
    print(
        f'[Worker: {os.getpid()}] Finished storing harvest results for statement id: "{statement_id}"'
    )
    dbh.insert_statement_features(statement_id, features_results, predict_result, None)
    print(
        f'[Worker: {os.getpid()}] Finished storing features results for statement id: "{statement_id}"'
    )


@shared_task(bind=True, ignore_result=False)
def train_task(self):
    from check4facts.api import dbh

    self.update_state(
        state="PROGRESS", meta={"current": 1, "total": 2, "type": "TRAIN"}
    )
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
    self.update_state(
        state="PROGRESS", meta={"current": 2, "total": 2, "type": "TRAIN"}
    )


@shared_task(bind=True, ignore_result=False)
def intial_train_task(self):
    from check4facts.api import dbh

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
    self.update_state(
        state="PROGRESS",
        meta={"current": 1, "total": (4 * total_count) + 1, "type": "INITIAL_TRAIN"},
    )

    # Execute all steps for each statement.
    for statement in statements:
        statement_id, text, true_label = statement[0], statement[1], statement[2]
        counter += 1

        print(f"Starting search for {statement_id}")

        self.update_state(
            state="PROGRESS",
            meta={
                "current": (4 * counter) + 1,
                "total": (4 * total_count) + 1,
                "type": "INITIAL_TRAIN",
            },
        )
        search_results = se.run([text])[0]

        print(f"Starting harvest for {statement_id}")
        articles = [
            {"s_id": statement_id, "s_text": text, "s_resources": search_results}
        ]
        self.update_state(
            state="PROGRESS",
            meta={
                "current": (4 * counter) + 2,
                "total": (4 * total_count) + 1,
                "type": "INITIAL_TRAIN",
            },
        )
        harvest_results = h.run(articles)[0]

        print(f"Saving Harvest Results to db  for {statement_id}")
        resource_records = harvest_results.to_dict("records")
        dbh.insert_statement_resources(statement_id, resource_records)

        print(f"Starting feature for {statement_id}")
        statement_dicts = [
            {"s_id": statement_id, "s_text": text, "s_resources": harvest_results}
        ]
        self.update_state(
            state="PROGRESS",
            meta={
                "current": (4 * counter) + 3,
                "total": (4 * total_count) + 1,
                "type": "INITIAL_TRAIN",
            },
        )
        features_results = fe.run(statement_dicts)[0]

        print(f"Saving Feature Results to db for {statement_id}")
        dbh.insert_statement_features(statement_id, features_results, None, true_label)
        self.update_state(
            state="PROGRESS",
            meta={
                "current": (4 * counter) + 4,
                "total": (4 * total_count) + 1,
                "type": "INITIAL_TRAIN",
            },
        )

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


# Tasks for text summarization


@shared_task(bind=True, ignore_result=False)
def summarize_text(self, article_id):
    from check4facts.api import dbh

    try:
        content = dbh.fetch_article_content(article_id)

        answer = None
        self.update_state(
            state="PROGRESS",
            meta={
                "current": 1,
                "total": 2,
                "type": "SUMMARIZE",
            },
        )

        # Keep only the actual text from the article's content
        content = extract_text_from_html(content)

        # invoke Gemini llm
        print("Trying to invoke gemini llm....")
        answer = google_llm(article_id, content)
        if answer:
            result = {
                "summarization": answer["summarization"],
                "time": answer["elapsed_time"],
                "article_id": article_id,
                "timestamp": answer["timestamp"],
            }
        else:
            # try invoking the groq llm
            print("Trying to invoke groq llm....")
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

        print(
            f"Finished generating summary in: {result['time']} seconds. Storing in database..."
        )

        self.update_state(
            state="PROGRESS",
            meta={
                "current": 2,
                "total": 2,
                "type": "SUMMARIZE",
            },
        )
        dbh.insert_summary(article_id, result["summarization"])
        dbh.disconnect()
    except Exception as e:
        print(f"Error generating summary for article with id {article_id}: {e}")


@shared_task(bind=True, ignore_result=False)
def batch_summarize_text(self):
    from check4facts.api import dbh

    try:

        # Fetch the id and content (extract_text_from_html function used)
        # from all published articles and null summary
        articles = dbh.fetch_articles_without_summary()

        print(f"Total articles fetched: {len(articles)}")
        for index, article in enumerate(articles):
            print(
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
            print("Trying to invoke gemini llm....")
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

            print(result["summarization"])
            dbh.insert_summary(article_id, result["summarization"])
            time.sleep(5)

        dbh.disconnect()

    except Exception as e:
        print(f"Error generating summary for published articles: {e}")


@shared_task(bind=True, ignore_result=False)
def justify_task(self, statement_id, n):
    from check4facts.api import dbh

    text = dbh.fetch_statement_text(statement_id)
    try:

        answer = run_pipeline(statement_id, text, n)
        if answer:
            print("FINAL ANSWER: ")
            print()
            for key, value in answer.items():
                print(f"{key}: {value}")

            # Store to Database
            dbh.insert_justification(
                statement_id,
                answer["justification"],
                answer["timestamp"],
                answer["llm_response_time"],
                answer["label"],
                answer["model"],
                answer["sources"],
            )
        else:
            raise Exception("Pipeline returned empty result")

    except Exception as e:
        print(f"Error during rag run: {e}")

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
        answer = run_pipeline(article_id, claim, n)
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
