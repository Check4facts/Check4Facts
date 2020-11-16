import os
import yaml

from flask import request
from flask_cors import CORS
from flask_marshmallow import Marshmallow

from check4facts.api.init import create_app
from check4facts.models.models import Statement
from check4facts.config import DirConf
from check4facts.scripts.search import SearchEngine
from check4facts.scripts.harvest import Harvester

app = create_app()
ma = Marshmallow(app)
CORS(app)


class ResourceSchema(ma.Schema):
    class Meta:
        fields = ("body", "fileFormat", "harvestDate", "harvestIteration", "id", "simParagraph", "simSentence", "title",
                  "url")


class StatementSourcesSchema(ma.Schema):
    class Meta:
        fields = ("id", "snippet", "title", "url")


class TopicSchema(ma.Schema):
    class Meta:
        fields = ("id", "name")


class StatementSchema(ma.Schema):
    class Meta:
        fields = ("author", "id", "main_article_text", "main_article_url", "registration_date", "statement_date", "text"
                  , "topic")

    topic = ma.Nested(TopicSchema)


statement_schema = StatementSchema(many=True)
resources_schema = ResourceSchema(many=True)


@app.route('/search_harvest', methods=['POST'])
def search_harvest():
    path = os.path.join('../../', DirConf.CONFIG_DIR, 'search_config.yml')
    with open(path, 'r') as f:
        search_params = yaml.safe_load(f)
    se = SearchEngine(**search_params)
    statement = request.json
    claims = [statement.get('text')]
    # Using first element only for the result cause only one statement is being checked.
    search_results = se.run(claims)[0]

    # Clean results from pdf/doc links.
    if 'fileFormat' not in search_results:
        search_results['fileFormat'] = 'html'
    search_results['fileFormat'] = search_results['fileFormat'].fillna('html')
    search_results = search_results[search_results['fileFormat'] == 'html']

    path = os.path.join('../../', DirConf.CONFIG_DIR, 'harvest_config.yml')
    with open(path, 'r') as f:
        harvest_params = yaml.safe_load(f)
    h = Harvester(**harvest_params)
    articles = [{
        'c_id': statement.get('id'),
        'c_text': statement.get('text'),
        'articles': search_results}]
    # Using first element only for the result cause only one statement is being checked.
    harvest_results = h.run(articles)[0]
    response_result = []
    for row in harvest_results.itertuples():
        response_result.append({
            'url': row.url,
            'title': row.title,
            'body': row.body,
            'simParagraph': row.sim_par,
            'simSentence': row.sim_sent,
        })

    # TODO save harvest results(Resources) to database.
    return resources_schema.jsonify(response_result)


# Successfully returns the list of all statements in database.
@app.route('/statement')
def fetch_statements():
    statements = Statement.query.all()
    result = statement_schema.dump(statements)
    return {"statements": result}


if __name__ == '__main__':
    app.run(debug=True)