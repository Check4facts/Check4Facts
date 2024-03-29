import os
import argparse
import time

import numpy as np
import pandas as pd
import yaml

from check4facts.scripts.search import SearchEngine
from check4facts.scripts.harvest import Harvester
from check4facts.scripts.features import FeaturesExtractor
from check4facts.train import Trainer
from check4facts.predict import Predictor
from check4facts.database import DBHandler
from check4facts.config import DirConf


class Interface:
    """
    This is the CLI of the C4F project. It is responsible
    for handling different types of actions based on given arguments.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Run various ops of the C4F project.')

        self.subparsers = self.parser.add_subparsers(
            help='sub-command help', dest='action')

        # create parser for "search_dev" command
        self.search_dev_parser = self.subparsers.add_parser(
            'search_dev', help='triggers relevant resource search (dev)')

        # create parser for "search" command
        self.search_parser = self.subparsers.add_parser(
            'search', help='triggers relevant resource search')

        # create parser for "harvest_dev" command
        self.harvest_dev_parser = self.subparsers.add_parser(
            'harvest_dev', help='triggers search results harvest (dev)')

        # create parser for "harvest" command
        self.harvest_parser = self.subparsers.add_parser(
            'harvest', help='triggers search results harvest')

        # create parser for "features_dev" command
        self.features_dev_parser = self.subparsers.add_parser(
            'features_dev', help='triggers features extraction (dev)')

        # create parser for "features" command
        self.features_parser = self.subparsers.add_parser(
            'features', help='triggers features extraction')

        # create parser for "predict_dev" command
        self.predict_dev_parser = self.subparsers.add_parser(
            'predict_dev', help='triggers model predictions (dev)')

        # create parser for "train_dev" command
        self.train_dev_parser = self.subparsers.add_parser(
            'train_dev', help='triggers model training (dev)')

        # create parser for "analyze_task_demo" command
        self.analyze_task_demo_parser = self.subparsers.add_parser(
            'analyze_task_demo', help='triggers a full analysis workflow')

        # create parser for "train_task_demo" command
        self.train_task_demo_parser = self.subparsers.add_parser(
            'train_task_demo', help='triggers model training workflow')

        # create parser for "initial_train" command
        self.initial_train_parser = self.subparsers.add_parser(
            'initial_train', help='triggers an initial training workflow')

    def run(self):
        # arguments for "search_dev" command
        self.search_dev_parser.add_argument(
            '--settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')

        # arguments for "search" command
        self.search_parser.add_argument(
            '--settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')

        # arguments for "harvest_dev" command
        self.harvest_dev_parser.add_argument(
            '--settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')

        # arguments for "harvest" command
        self.harvest_parser.add_argument(
            '--settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')

        # arguments for "features_dev" command
        self.features_dev_parser.add_argument(
            '--settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')

        # arguments for "features" command
        self.features_parser.add_argument(
            '--settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')

        # arguments for "predict_dev" command
        self.predict_dev_parser.add_argument(
            '--settings', type=str, default='predict_config.yml',
            help='name of YAML configuration file containing predict params')

        # arguments for "train_dev" command
        self.train_dev_parser.add_argument(
            '--settings', type=str, default='train_config.yml',
            help='name of YAML configuration file containing training params')

        # arguments for "analyze_task_demo" command
        self.analyze_task_demo_parser.add_argument(
            '--search_settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')
        self.analyze_task_demo_parser.add_argument(
            '--harvest_settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')
        self.analyze_task_demo_parser.add_argument(
            '--features_settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')
        self.analyze_task_demo_parser.add_argument(
            '--predict_settings', type=str, default='predict_config.yml',
            help='name of YAML configuration file containing predict params')
        self.analyze_task_demo_parser.add_argument(
            '--db_settings', type=str, default='db_config.yml',
            help='name of YAML configuration file containing database params')

        # arguments for "train_task_demo" command
        self.train_task_demo_parser.add_argument(
            '--train_settings', type=str, default='train_config.yml',
            help='name of YAML configuration file containing training params')
        self.train_task_demo_parser.add_argument(
            '--db_settings', type=str, default='db_config.yml',
            help='name of YAML configuration file containing database params')

        # arguments for "initial_train" command
        self.initial_train_parser.add_argument(
            '--search_settings', type=str, default='search_config.yml',
            help='name of YAML configuration file containing search params')
        self.initial_train_parser.add_argument(
            '--harvest_settings', type=str, default='harvest_config.yml',
            help='name of YAML configuration file containing harvest params')
        self.initial_train_parser.add_argument(
            '--features_settings', type=str, default='features_config.yml',
            help='name of YAML configuration file containing features params')
        self.initial_train_parser.add_argument(
            '--db_settings', type=str, default='db_config.yml',
            help='name of YAML configuration file containing database params')
        self.initial_train_parser.add_argument(
            '--train_settings', type=str, default='train_config.yml',
            help='name of YAML configuration file containing training params')

        cmd_args = self.parser.parse_args()

        if cmd_args.action == 'search_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            se = SearchEngine(**params)
            se.run_dev()

        elif cmd_args.action == 'search' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            se = SearchEngine(**params)
            statements_texts = ['Τι χρήματα παίρνουν οι αιτούντες άσυλο']
            results = se.run(statements_texts)

        elif cmd_args.action == 'harvest_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            h = Harvester(**params)
            h.run_dev()

        elif cmd_args.action == 'harvest' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            h = Harvester(**params)
            data = {'index': [0], 'link': ['https://www.liberal.gr/eidiseis/']}
            statements_dicts = [{
                's_id': 1, 's_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                's_resources': pd.DataFrame(data)}]
            results = h.run(statements_dicts)

        elif cmd_args.action == 'features_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            fe = FeaturesExtractor(**params)
            fe.run_dev()

        elif cmd_args.action == 'features' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            fe = FeaturesExtractor(**params)
            data = {
                'title': ['title'],
                'body': ['This a the body. It contains paragraphs.'],
                'sim_par': ['A quite similar paragraph.'],
                'sim_sent': ['A very similar sentence!']}
            statement_dicts = [{
                's_id': 1, 's_text': 'Τι χρήματα παίρνουν οι αιτούντες άσυλο',
                's_resources': pd.DataFrame(data)}]
            results = fe.run(statement_dicts)

        elif cmd_args.action == 'predict_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            p = Predictor(**params)
            p.run_dev()

        elif cmd_args.action == 'train_dev' and cmd_args.settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.settings)
            with open(path, 'r') as f:
                params = yaml.safe_load(f)
            t = Trainer(**params)
            t.run_dev()

        elif cmd_args.action == 'analyze_task_demo' \
                and cmd_args.search_settings and cmd_args.harvest_settings \
                and cmd_args.features_settings and cmd_args.db_settings:

            statement_ids = [4, 5, 7, 16]
            statement_texts = [
                'Μετανάστες στον Έβρο: «Μας έβγαλαν από τη φυλακή και μας έστειλαν στα σύνορα». ',
                'Έτοιμοι για «απόβαση» στη Λέσβο 150.000 μετανάστες και πρόσφυγες.',
                'Το 80-85% είναι πλέον οικονομικοί μετανάστες.',
                'Πέτσας: η Σύνοδος των χωρών του Νότου (Med 7) είχε προγραμματιστεί για τις 2 Ιουλίου στην Κορσική'
            ]

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.search_settings)
            with open(path, 'r') as f:
                search_params = yaml.safe_load(f)
            se = SearchEngine(**search_params)
            search_results = se.run(statement_texts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.harvest_settings)
            with open(path, 'r') as f:
                harvest_params = yaml.safe_load(f)
            h = Harvester(**harvest_params)
            statement_dicts = [{
                's_id': statement_ids[i],
                's_text': statement_texts[i],
                's_resources': search_results[i]}
                for i in range(len(statement_texts))]
            harvest_results = h.run(statement_dicts)

            # TODO manage harvest_results items referring to statements
            #  with no resources. We discard those statements but we have to
            #  keep items aligned. Prediction and db storing are not
            #  supporting those statements.

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.features_settings)
            with open(path, 'r') as f:
                features_params = yaml.safe_load(f)
            fe = FeaturesExtractor(**features_params)
            statement_dicts = [{
                's_id': statement_ids[i],
                's_text': statement_texts[i],
                's_resources': harvest_results[i]}
                for i in range(len(statement_texts))]
            features_results = fe.run(statement_dicts)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.predict_settings)
            with open(path, 'r') as f:
                predict_params = yaml.safe_load(f)
            p = Predictor(**predict_params)
            predict_results = p.run(features_results)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.db_settings)
            with open(path, 'r') as f:
                db_params = yaml.safe_load(f)
            dbh = DBHandler(**db_params)
            for s_id, s_resources, s_features, s_preds in \
                    zip(statement_ids, harvest_results,
                        features_results, predict_results):
                resource_records = s_resources.to_dict('records')
                dbh.insert_statement_resources(s_id, resource_records)
                dbh.insert_statement_features(s_id, s_features, s_preds)

        elif cmd_args.action == 'train_task_demo' \
                and cmd_args.train_settings and cmd_args.db_settings:
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.train_settings)
            with open(path, 'r') as f:
                train_params = yaml.safe_load(f)
            t = Trainer(**train_params)

            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.db_settings)
            with open(path, 'r') as f:
                db_params = yaml.safe_load(f)
            dbh = DBHandler(**db_params)

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

        elif cmd_args.action == 'initial_train' \
                and cmd_args.search_settings and cmd_args.harvest_settings \
                and cmd_args.features_settings and cmd_args.train_settings and cmd_args.db_settings:
            # Initialize all python modules.
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.db_settings)
            with open(path, 'r') as f:
                db_params = yaml.safe_load(f)
            dbh = DBHandler(**db_params)
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.search_settings)
            with open(path, 'r') as f:
                search_params = yaml.safe_load(f)
            se = SearchEngine(**search_params)
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.harvest_settings)
            with open(path, 'r') as f:
                harvest_params = yaml.safe_load(f)
            h = Harvester(**harvest_params)
            path = os.path.join(DirConf.CONFIG_DIR, cmd_args.features_settings)
            with open(path, 'r') as f:
                features_params = yaml.safe_load(f)
            fe = FeaturesExtractor(**features_params)

            # Get all statements from database.
            statements = dbh.fetch_statements()
            total_count = len(statements)
            print(f'Initiating training on {total_count} Statements from DB.')
            counter = 0
            for statement in statements:
                statement_id, text, true_label = statement[0], statement[1], statement[2]
                counter += 1

                print(f'Starting search for Statement: "{statement_id}"')
                search_results = se.run([text])[0]

                print(f'Starting harvest for Statement: "{statement_id}"')
                articles = [{
                    's_id': statement_id,
                    's_text': text,
                    's_resources': search_results
                }]
                harvest_results = h.run(articles)[0]

                print(f'Saving Harvest results of Statement: "{statement_id}"')
                resource_records = harvest_results.to_dict('records')
                dbh.insert_statement_resources(statement_id, resource_records)

                print(f'Starting feature for Statement: "{statement_id}"')
                statement_dicts = [{
                    's_id': statement_id,
                    's_text': text,
                    's_resources': harvest_results
                }]
                features_results = fe.run(statement_dicts)[0]

                print(f'Saving Feature results of Statement: "{statement_id}"')
                dbh.insert_statement_features(statement_id, features_results, None, true_label)

            print(f'Finished search, harvest and feature procedures for all {total_count} Statements.')
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


if __name__ == "__main__":
    interface = Interface()
    interface.run()
