import os
import time
import yaml
import numpy as np
from celery import result, shared_task
from check4facts.train import Trainer
from check4facts.config import DirConf
from check4facts.predict import Predictor
from check4facts.database import DBHandler
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.search import SearchEngine
from check4facts.scripts.features import FeaturesExtractor

#imports for text summarization
from check4facts.text_sum_scripts.llm_request import invoke_local_llm
from check4facts.text_sum_scripts.text_process import text_to_bulleted_list
from check4facts.text_sum_scripts.groq_api import groq_api


db_path = os.path.join(DirConf.CONFIG_DIR, 'db_config.yml')  # while using uwsgi
with open(db_path, 'r') as db_f:
    db_params = yaml.safe_load(db_f)
dbh = DBHandler(**db_params)


@shared_task(bind=True)
def status_task(self, task_id):
    return result.AsyncResult(task_id)


@shared_task(bind=True)
def analyze_task(self, statement):
    statement_id = statement.get('id')
    statement_text = statement.get('text')

    print(f'[Worker: {os.getpid()}] Started analyze procedure for statement id: "{statement_id}"')

    path = os.path.join(DirConf.CONFIG_DIR, 'search_config.yml')  # when using uwsgi.
    with open(path, 'r') as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    statements = [statement_text]

    self.update_state(state='PROGRESS',
                      meta={'current': 1, 'total': 4,
                            'type': f'{statement_id}'})
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(statements)[0]

    path = os.path.join(DirConf.CONFIG_DIR, 'harvest_config.yml')  # while using uwsgi.
    with open(path, 'r') as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [{
        's_id': statement_id,
        's_text': statement_text,
        's_resources': search_results}]

    self.update_state(state='PROGRESS',
                      meta={'current': 2, 'total': 4,
                            'type': f'{statement_id}'})
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]

    if harvest_results.empty:
        print(f'[Worker: {os.getpid()}] No resources found for statement id: "{statement_id}"')
    #     return

    path = os.path.join(DirConf.CONFIG_DIR, 'features_config.yml')  # while using uwsgi
    with open(path, 'r') as f:
        features_params = yaml.safe_load(f)
    fe = FeaturesExtractor(**features_params)
    statement_dicts = [{'s_id': statement_id, 's_text': statement_text,
                        's_resources': harvest_results}]

    self.update_state(state='PROGRESS',
                      meta={'current': 3, 'total': 4,
                            'type': f'{statement_id}'})
    features_results = fe.run(statement_dicts)[0]

    if harvest_results.empty:
        predict_result = np.array([-1.0, -1.0])
    else:
        path = os.path.join(DirConf.CONFIG_DIR, 'predict_config.yml')
        with open(path, 'r') as f:
            predict_params = yaml.safe_load(f)
        p = Predictor(**predict_params)

        self.update_state(state='PROGRESS',
                          meta={'current': 4, 'total': 4,
                                'type': f'{statement_id}'})
        predict_result = p.run([features_results]).loc[0, ['pred_0', 'pred_1']].values

    resource_records = harvest_results.to_dict('records')
    dbh.insert_statement_resources(statement_id, resource_records)
    print(f'[Worker: {os.getpid()}] Finished storing harvest results for statement id: "{statement_id}"')
    dbh.insert_statement_features(statement_id, features_results, predict_result, None)
    print(f'[Worker: {os.getpid()}] Finished storing features results for statement id: "{statement_id}"')


@shared_task(bind=True)
def train_task(self):
    self.update_state(
        state='PROGRESS',
        meta={
            'current': 1,
            'total': 2,
            'type': 'TRAIN'
        }
    )
    path = os.path.join(DirConf.CONFIG_DIR, 'train_config.yml')
    with open(path, 'r') as f:
        train_params = yaml.safe_load(f)
    t = Trainer(**train_params)

    features_records = dbh.fetch_statement_features(train_params['features'])
    features = np.vstack([np.hstack(f) for f in features_records])
    labels = dbh.fetch_statement_labels()
    t.run(features, labels)

    if not os.path.exists(DirConf.MODELS_DIR):
        os.mkdir(DirConf.MODELS_DIR)
    fname = t.best_model['clf'] + '_' + time.strftime(
        '%Y-%m-%d-%H:%M') + '.joblib'
    path = os.path.join(DirConf.MODELS_DIR, fname)
    t.save_best_model(path)
    self.update_state(
        state='PROGRESS',
        meta={
            'current': 2,
            'total': 2,
            'type': 'TRAIN'
        }
    )


@shared_task(bind=True)
def intial_train_task(self):
    # Initialize all python modules.
    path = os.path.join(DirConf.CONFIG_DIR, 'search_config.yml')  # when using uwsgi.
    with open(path, 'r') as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)

    path = os.path.join(DirConf.CONFIG_DIR, 'harvest_config.yml')  # while using uwsgi.
    with open(path, 'r') as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)

    path = os.path.join(DirConf.CONFIG_DIR, 'features_config.yml')  # while using uwsgi
    with open(path, 'r') as f:
        features_params = yaml.safe_load(f)
    fe = FeaturesExtractor(**features_params)

    # Get all statements from database.
    statements = dbh.fetch_statements()
    total_count = len(statements)
    counter = 0
    self.update_state(
        state='PROGRESS',
        meta={
            'current': 1,
            'total': (4 * total_count) + 1,
            'type': 'INITIAL_TRAIN'
        }
    )

    # Execute all steps for each statement.
    for statement in statements:
        statement_id, text, true_label = statement[0], statement[1], statement[2]
        counter += 1

        print(f'Starting search for {statement_id}')

        self.update_state(
            state='PROGRESS',
            meta={
                'current': (4 * counter) + 1,
                'total': (4 * total_count) + 1,
                'type': 'INITIAL_TRAIN'
            }
        )
        search_results = se.run([text])[0]

        print(f'Starting harvest for {statement_id}')
        articles = [{
            's_id': statement_id,
            's_text': text,
            's_resources': search_results
        }]
        self.update_state(
            state='PROGRESS',
            meta={
                'current': (4 * counter) + 2,
                'total': (4 * total_count) + 1,
                'type': 'INITIAL_TRAIN'
            }
        )
        harvest_results = h.run(articles)[0]

        print(f'Saving Harvest Results to db  for {statement_id}')
        resource_records = harvest_results.to_dict('records')
        dbh.insert_statement_resources(statement_id, resource_records)

        print(f'Starting feature for {statement_id}')
        statement_dicts = [{
            's_id': statement_id,
            's_text': text,
            's_resources': harvest_results
        }]
        self.update_state(
            state='PROGRESS',
            meta={
                'current': (4 * counter) + 3,
                'total': (4 * total_count) + 1,
                'type': 'INITIAL_TRAIN'
            }
        )
        features_results = fe.run(statement_dicts)[0]

        print(f'Saving Feature Results to db for {statement_id}')
        dbh.insert_statement_features(statement_id, features_results, None, true_label)
        self.update_state(
            state='PROGRESS',
            meta={
                'current': (4 * counter) + 4,
                'total': (4 * total_count) + 1,
                'type': 'INITIAL_TRAIN'
            }
        )

    print(f'Initiating model training.')
    path = os.path.join(DirConf.CONFIG_DIR, cmd_args.train_settings)
    with open(path, 'r') as f:
        train_params = yaml.safe_load(f)
    t = Trainer(**train_params)

    features_records = dbh.fetch_statement_features(
        train_params['features'])
    features = np.vstack([np.hstack(f) for f in features_records])
    labels = dbh.fetch_statement_labels()
    t.run(features, labels)

    if not os.path.exists(DirConf.MODELS_DIR):
        os.mkdir(DirConf.MODELS_DIR)
    fname = t.best_model['clf'] + '_' + time.strftime(
        '%Y-%m-%d-%H:%M') + '.joblib'
    path = os.path.join(DirConf.MODELS_DIR, fname)
    t.save_best_model(path)
    print(f'Successfully saved the best model.')


#Tasks for text summarization




@shared_task
def summarize_text(user_input, article_id):

    # Check if text is valid
    if len(user_input.split()) < 5:
        return {"error": "Please enter a valid text"}
    try:
        article_id = int(article_id)
    except ValueError:
        print("Error: article_id is not an integer")

    # Try invoking the groq_api to generate a summary
    api = groq_api()
    answer = api.run(user_input)

    # If the invoking fails, or the input is too large, call the local implementation
    if answer['response'] is None or len(user_input.split()) >= 1900:
        result = invoke_local_llm(user_input, article_id)
        return result
    else:
        return {"summarization": text_to_bulleted_list(answer['response']), "time": answer['elapsed_time'], 
                "article_id": article_id}



@shared_task
def celery_insert(task_id):
    dbh.connect()
    dbh.insert_summary(task_id)
    dbh.disconnect()
    

@shared_task
def celery_get_task_result(task_id):
    dbh.connect()
    result = dbh.get_successful_task_result(task_id)
    dbh.disconnect()
    return result
