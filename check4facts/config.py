import os


class DirConf:
    DATA_DIR = 'data'
    CSV_FILE = os.path.join(DATA_DIR, 'statements_test.csv')
    SEARCH_RESULTS_DIR = os.path.join(DATA_DIR, 'search_results')
    HARVEST_RESULTS_DIR = os.path.join(DATA_DIR, 'harvest_results')
    HARVEST_XML_DIR = os.path.join(HARVEST_RESULTS_DIR, 'xml_files')
    FEATURES_RESULTS_DIR = os.path.join(DATA_DIR, 'features_results')
    TRAINER_RESULTS_DIR = os.path.join(DATA_DIR, 'trainer_results')
    PREDICTOR_RESULTS_DIR = os.path.join(DATA_DIR, 'predictor_results')

    CONFIG_DIR = '../config'
    MODELS_DIR = 'models'
