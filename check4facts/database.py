import json
import select
import asyncio
import numpy as np
import psycopg2
from psycopg2.extensions import register_adapter, AsIs, ISOLATION_LEVEL_AUTOCOMMIT
from bs4 import BeautifulSoup
from check4facts.logging import get_logger

from check4facts.scripts.text_sum.text_process import extract_text_from_html

log = get_logger()


# Functions to adapt NumPy types
def adapt_numpy_scalar(numpy_scalar):
    """
    Convert NumPy scalar types to their native Python equivalents.
    """
    return AsIs(numpy_scalar.item())


def adapt_numpy_array(numpy_array):
    """
    Convert numpy.ndarray to a Python list for PostgreSQL compatibility.
    """
    return AsIs(list(numpy_array))


def add_numpy_adapters():
    """
    Register psycopg2 adapters for NumPy scalar types and arrays.
    """
    # Register scalar adapters
    scalar_types = [np.float32, np.float64, np.int32, np.int64, np.bool_]
    for scalar_type in scalar_types:
        register_adapter(scalar_type, adapt_numpy_scalar)

    # Register array adapter
    register_adapter(np.ndarray, adapt_numpy_array)


# Register the adapters
add_numpy_adapters()

def task_channel_name(task_id: str) -> str:
    return f"task_channel_{task_id.replace('-', '_')}"

def extract_task_id_from_channel(channel: str) -> str:
    prefix = "task_channel_"
    if channel.startswith(prefix):
        suffix = channel[len(prefix):]
        return suffix.replace("_", "-")
    raise ValueError(f"Invalid channel name: {channel}")


class DBHandler:

    def __init__(self, **kwargs):
        self.connection, self.cursor = None, None
        self.conn_params = kwargs

    def connect(self):
        try:

            self.connection = psycopg2.connect(**self.conn_params)
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

            self.cursor = self.connection.cursor()

            self.cursor.execute("SELECT version();")

            db_version = self.cursor.fetchone()
            log.info(f"Connected to: {db_version}")

        except Exception as e:
            log.error(f"Error: {e}")

    def disconnect(self):
        if self.connection:
            self.cursor.close()
            self.connection.close()
            log.info("Connection closed.")
        else:
            log.warning("Unable to close connection. Is the connection already closed?")

    def fetch_active_tasks_ids(self):
        if not self.connection:
            self.connect()

        response = []
        try:
            sql = """
                SELECT task_id
                FROM celery_taskmeta
                WHERE status = 'PROGRESS';
            """
            self.cursor.execute(sql)
            rows = self.cursor.fetchall()

            for row in rows:
                response.append(row[0])

        except Exception as e:
            log.error(f"Error fetching active celery tasks: {e}")
            self.connection.rollback()
            return []

        return response

    def insert_statement_resources(self, s_id, resource_records):
        conn = None
        sql1 = (
            "SELECT MAX(resource.harvest_iteration) FROM resource"
            " WHERE resource.statement_id = %s;"
        )
        sql2 = (
            "INSERT INTO resource (url, title, body, sim_paragraph,"
            " sim_sentence, file_format, harvest_date, harvest_iteration,"
            " statement_id) VALUES (%(url)s, %(title)s, %(body)s,"
            " %(sim_par)s, %(sim_sent)s, %(file_format)s,"
            " %(harvest_date)s, %(harvest_iteration)s, %(statement_id)s);"
        )
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql1, (s_id,))
            res = cur.fetchone()[0]
            h_iter = res + 1 if res else 1
            resource_records = [
                {**r, **{"statement_id": s_id, "harvest_iteration": h_iter}}
                for r in resource_records
            ]
            cur.executemany(sql2, resource_records)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()

    def insert_statement_features(self, s_id, features_record, s_preds, true_label):
        conn = None
        sql1 = (
            "SELECT MAX(feature_statement.harvest_iteration)"
            " FROM feature_statement"
            " WHERE feature_statement.statement_id = %s;"
        )
        sql2 = (
            "INSERT INTO feature_statement ("
            " s_embedding,"
            " s_fertile_terms,"
            " s_subjectivity,"
            " s_subjectivity_counts,"
            " s_sentiment,"
            " s_sentiment_counts,"
            " s_emotion_anger,"
            " s_emotion_disgust,"
            " s_emotion_fear,"
            " s_emotion_happiness,"
            " s_emotion_sadness,"
            " s_emotion_surprise,"
            " s_pg_polarity_counts,"
            " r_title_embedding,"
            " r_title_fertile_terms,"
            " r_title_similarity,"
            " r_title_subjectivity,"
            " r_title_subjectivity_counts,"
            " r_title_sentiment,"
            " r_title_sentiment_counts,"
            " r_title_emotion_anger,"
            " r_title_emotion_disgust,"
            " r_title_emotion_fear,"
            " r_title_emotion_happiness,"
            " r_title_emotion_sadness,"
            " r_title_emotion_surprise,"
            " r_title_pg_polarity_counts,"
            " r_body_embedding,"
            " r_body_similarity,"
            " r_body_subjectivity,"
            " r_body_subjectivity_counts,"
            " r_body_sentiment,"
            " r_body_sentiment_counts,"
            " r_body_emotion_anger,"
            " r_body_emotion_disgust,"
            " r_body_emotion_fear,"
            " r_body_emotion_happiness,"
            " r_body_emotion_sadness,"
            " r_body_emotion_surprise,"
            " r_body_pg_polarity_counts,"
            " r_sim_par_embedding,"
            " r_sim_par_fertile_terms,"
            " r_sim_par_similarity,"
            " r_sim_par_subjectivity,"
            " r_sim_par_subjectivity_counts,"
            " r_sim_par_sentiment,"
            " r_sim_par_sentiment_counts,"
            " r_sim_par_emotion_anger,"
            " r_sim_par_emotion_disgust,"
            " r_sim_par_emotion_fear,"
            " r_sim_par_emotion_happiness,"
            " r_sim_par_emotion_sadness,"
            " r_sim_par_emotion_surprise,"
            " r_sim_par_pg_polarity_counts,"
            " r_sim_sent_embedding,"
            " r_sim_sent_fertile_terms,"
            " r_sim_sent_similarity,"
            " r_sim_sent_subjectivity,"
            " r_sim_sent_subjectivity_counts,"
            " r_sim_sent_sentiment,"
            " r_sim_sent_sentiment_counts,"
            " r_sim_sent_emotion_anger,"
            " r_sim_sent_emotion_disgust,"
            " r_sim_sent_emotion_fear,"
            " r_sim_sent_emotion_happiness,"
            " r_sim_sent_emotion_sadness,"
            " r_sim_sent_emotion_surprise,"
            " r_sim_sent_pg_polarity_counts,"
            " true_label,"
            " predict_label,"
            " predict_proba,"
            " harvest_iteration,"
            " statement_id)"
            " VALUES ("
            " %(s_embedding)s,"
            " %(s_fertile_terms)s,"
            " %(s_subjectivity)s,"
            " %(s_subjectivity_counts)s,"
            " %(s_sentiment)s,"
            " %(s_sentiment_counts)s,"
            " %(s_emotion_anger)s,"
            " %(s_emotion_disgust)s,"
            " %(s_emotion_fear)s,"
            " %(s_emotion_happiness)s,"
            " %(s_emotion_sadness)s,"
            " %(s_emotion_surprise)s,"
            " %(s_pg_polarity_counts)s,"
            " %(r_title_embedding)s,"
            " %(r_title_fertile_terms)s,"
            " %(r_title_similarity)s,"
            " %(r_title_subjectivity)s,"
            " %(r_title_subjectivity_counts)s,"
            " %(r_title_sentiment)s,"
            " %(r_title_sentiment_counts)s,"
            " %(r_title_emotion_anger)s,"
            " %(r_title_emotion_disgust)s,"
            " %(r_title_emotion_fear)s,"
            " %(r_title_emotion_happiness)s,"
            " %(r_title_emotion_sadness)s,"
            " %(r_title_emotion_surprise)s,"
            " %(r_title_pg_polarity_counts)s,"
            " %(r_body_embedding)s,"
            " %(r_body_similarity)s,"
            " %(r_body_subjectivity)s,"
            " %(r_body_subjectivity_counts)s,"
            " %(r_body_sentiment)s,"
            " %(r_body_sentiment_counts)s,"
            " %(r_body_emotion_anger)s,"
            " %(r_body_emotion_disgust)s,"
            " %(r_body_emotion_fear)s,"
            " %(r_body_emotion_happiness)s,"
            " %(r_body_emotion_sadness)s,"
            " %(r_body_emotion_surprise)s,"
            " %(r_body_pg_polarity_counts)s,"
            " %(r_sim_par_embedding)s,"
            " %(r_sim_par_fertile_terms)s,"
            " %(r_sim_par_similarity)s,"
            " %(r_sim_par_subjectivity)s,"
            " %(r_sim_par_subjectivity_counts)s,"
            " %(r_sim_par_sentiment)s,"
            " %(r_sim_par_sentiment_counts)s,"
            " %(r_sim_par_emotion_anger)s,"
            " %(r_sim_par_emotion_disgust)s,"
            " %(r_sim_par_emotion_fear)s,"
            " %(r_sim_par_emotion_happiness)s,"
            " %(r_sim_par_emotion_sadness)s,"
            " %(r_sim_par_emotion_surprise)s,"
            " %(r_sim_par_pg_polarity_counts)s,"
            " %(r_sim_sent_embedding)s,"
            " %(r_sim_sent_fertile_terms)s,"
            " %(r_sim_sent_similarity)s,"
            " %(r_sim_sent_subjectivity)s,"
            " %(r_sim_sent_subjectivity_counts)s,"
            " %(r_sim_sent_sentiment)s,"
            " %(r_sim_sent_sentiment_counts)s,"
            " %(r_sim_sent_emotion_anger)s,"
            " %(r_sim_sent_emotion_disgust)s,"
            " %(r_sim_sent_emotion_fear)s,"
            " %(r_sim_sent_emotion_happiness)s,"
            " %(r_sim_sent_emotion_sadness)s,"
            " %(r_sim_sent_emotion_surprise)s,"
            " %(r_sim_sent_pg_polarity_counts)s,"
            " %(true_label)s,"
            " %(predict_label)s,"
            " %(predict_proba)s,"
            " %(harvest_iteration)s,"
            " %(statement_id)s);"
        )
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql1, (s_id,))
            res = cur.fetchone()[0]
            h_iter = res + 1 if res else 1
            features_record["true_label"] = (
                true_label if true_label is not None else None
            )
            if s_preds is None:
                features_record["predict_label"] = None
                features_record["predict_proba"] = None
            elif np.array_equal(s_preds, np.array([-1.0, -1.0])):
                features_record["predict_label"] = None
                features_record["predict_proba"] = -1.0
            else:
                features_record["predict_label"] = (
                    True if np.argmax(s_preds) == 1 else False
                )
                features_record["predict_proba"] = np.max(s_preds)
            features_record["harvest_iteration"] = h_iter
            features_record["statement_id"] = s_id
            # Convert NumPy arrays to Python lists
            features_record = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in features_record.items()
            }

            cur.execute(sql2, features_record)
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()

    def fetch_statement_features(self, features):
        conn, res = None, None
        sql = "SELECT {} FROM feature_statement;".format(", ".join(features))
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql)
            res = cur.fetchall()
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()
            return res

    def fetch_statement_labels(self):
        conn, res = None, None
        sql = "SELECT true_label FROM feature_statement;"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql)
            res = [r[0] for r in cur.fetchall()]
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()
            return res

    def count_statements(self):
        conn, res = None, None
        sql = "SELECT COUNT(*) FROM statement;"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql)
            res = cur.fetchone()[0]
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()
            return res

    # TODO later make it to fetch batch of statements.
    def fetch_statements(self):
        conn, res = None, None
        # Change the requested columns according to the ML model.
        sql = "SELECT id, text, fact_checker_label FROM statement;"
        try:
            conn = psycopg2.connect(**self.conn_params)
            cur = conn.cursor()
            cur.execute(sql)
            res = cur.fetchall()
            conn.commit()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            log.error(error)
        finally:
            if conn is not None:
                conn.close()
            return res

    def fetch_single_statement(self, statement_id):
        if not self.connection:
            self.connect()

        try:
            print(f"Fetching statement with id: {statement_id}")
            sql = f"""
                SELECT s.text
                FROM statement s
                WHERE s.id = {statement_id}
            """
            self.cursor.execute(sql)
            result = self.cursor.fetchone()[0]
            print(f"Statement text: {result}")
            return result

        except Exception as e:
            print(f"Error fetching text from statement: {e}")
            self.connection.rollback()

    def fetch_article_content(self, article_id):
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            sql = f"""
                SELECT a.content
                FROM article a
                WHERE a.id = {article_id};
            """
            self.cursor.execute(sql)
            result = self.cursor.fetchone()[0]
            return result

        except Exception as e:
            log.error(f"Error fetching content from article: {e}")
            self.connection.rollback()

    def fetch_articles_without_summary(self):
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            sql = """
                SELECT id, content
                FROM article
                WHERE summary IS NULL AND published = TRUE;
            """
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            results = list(
                map(lambda item: (item[0], extract_text_from_html(item[1])), results)
            )
            return results

        except Exception as e:
            log.error(f"Error fetching articles without summary: {e}")
            self.connection.rollback()
            return []

    def fetch_sources_from_articles_content(self, page_size=50):
        # Method to fetch sources from articles content mapped by statement_id
        if not self.connection or self.connection.closed:
            self.connect()

        last_id = 0

        fact_checker_sources = {}
        try:
            while True:
                query = f"""
                    SELECT statement_id, content
                    FROM article
                    WHERE statement_id > {last_id}
                    ORDER BY statement_id
                    LIMIT {page_size};
                """
                self.cursor.execute(query)
                results = self.cursor.fetchall()

                if not results:
                    break
                last_id = int(results[-1][0])
                for index, row in enumerate(results):
                    statement_id = row[0]
                    content = row[1]

                    fact_checker_sources[statement_id] = []

                    soup = BeautifulSoup(content, "html.parser")
                    sources_element = soup.find(
                        lambda tag: tag.name and "Πηγές" in tag.get_text()
                    )
                    if sources_element:
                        elements_after_sources = sources_element.find_all_next()
                        for elem in elements_after_sources:
                            if elem.name == "a":
                                fact_checker_sources[statement_id].append(elem["href"])

            # filter out empty lists
            fact_checker_sources = {k: v for k, v in fact_checker_sources.items() if v}
            return fact_checker_sources
        except Exception as e:
            log.error(f"Error fetching page of articles contents: {e}")
            self.connection.rollback()
            return {}

    def fetch_sources_from_articles_content_category_wise(self, id, page_size=50):
        # Method to fetch sources from articles content mapped by statement_id
        if not self.connection or self.connection.closed:
            self.connect()

        last_id = 0

        fact_checker_sources = {}
        try:
            while True:
                query = f"""
                    SELECT statement_id, content
                    FROM article
                    WHERE statement_id > {last_id} AND category_id = {id}
                    ORDER BY statement_id
                    LIMIT {page_size};
                """
                self.cursor.execute(query)
                results = self.cursor.fetchall()

                if not results:
                    break
                last_id = int(results[-1][0])
                for index, row in enumerate(results):
                    statement_id = row[0]
                    content = row[1]

                    fact_checker_sources[statement_id] = []

                    soup = BeautifulSoup(content, "html.parser")
                    sources_element = soup.find(
                        lambda tag: tag.name and "Πηγές" in tag.get_text()
                    )
                    if sources_element:
                        elements_after_sources = sources_element.find_all_next()
                        for elem in elements_after_sources:
                            if elem.name == "a":
                                fact_checker_sources[statement_id].append(elem["href"])

            # filter out empty lists
            fact_checker_sources = {k: v for k, v in fact_checker_sources.items() if v}
            return fact_checker_sources
        except Exception as e:
            print(f"Error fetching page of articles contents: {e}")
            self.connection.rollback()
            return {}

    def fetch_statement_text(self, statement_id):
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            log.debug(f"Fetching text from stament with id: {statement_id}")
            sql = f"""
                SELECT s.text
                FROM statement s
                WHERE s.id = {statement_id}
            """
            self.cursor.execute(sql)
            result = self.cursor.fetchone()[0]
            log.debug(f"Statement text: {result}")
            return result

        except Exception as e:
            log.error(f"Error fetching text from statement: {e}")
            self.connection.rollback()

    def fetch_all_statement_texts(self):
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            log.debug(f"Fetching all texts from statement table...")
            sql = """
                SELECT s.id, s.text
                FROM statement s
                WHERE s.fact_checker_accuracy IS NOT NULL;
            """
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            result = [list((item[0], item[1])) for item in result]
            log.debug(f"Fetched {len(result)} statement texts from database!")
            return result

        except Exception as e:
            log.error(f"Error fetching text from statement: {e}")
            self.connection.rollback()

    def insert_justification(
        self, statement_id, text, timestamp, elapsed_time, label, model, sources
    ):
        if not self.connection or self.connection.closed:
            self.connect()

        try:
            log.info(
                f"Inserting new justification with text: {text} for stament with id: {statement_id}"
            )
            sql = """
                INSERT INTO justification (statement_id, text, timestamp, elapsed_time, label, model, sources)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            sources_as_list = list(sources)
            self.cursor.execute(
                sql,
                (statement_id, text, timestamp, elapsed_time, label, model, sources_as_list),
            )
            self.connection.commit()
            log.info(
                f"Justification for statement id: {statement_id} inserted successfully."
            )
        except Exception as e:
            log.error(f"Error inserting justification for statement_id {statement_id}: {e}")
            self.connection.rollback()

    # added extra functions for text summarization handling

    def insert_summary(self, article_id, article_summary):
        if not self.connection or self.connection.closed:
            self.connect()

        try:

            self.cursor.execute(
                """
            SELECT summary FROM article WHERE id = %s FOR UPDATE;""",
                (article_id,),
            )
            row = self.cursor.fetchone()
            if row and row[0]:
                log.debug(
                    "Summary already exists in the for this article_id. Deleting previous registrations..."
                )
                self.remove_summary_by_article_id(article_id)

            log.debug("Inserting summary....")
            query = """
            UPDATE article SET summary = %s WHERE id = %s;
            """
            self.cursor.execute(query, (article_summary, article_id))
            self.connection.commit()
            log.info(f"Summary with article id: {article_id} inserted successfully.")

        except Exception as e:
            log.error(f"Error inserting summary: {e}")
            self.connection.rollback()

    def remove_summary_by_article_id(self, article_id):
        if not self.connection or self.connection.closed:
            self.connect()

        try:

            self.cursor.execute(
                """SELECT summary FROM article WHERE id = %s FOR UPDATE;""",
                (article_id,),
            )

            row = self.cursor.fetchone()
            if row:
                query = """
                UPDATE article SET summary = NULL where id = %s;
            """
                self.cursor.execute(query, (article_id,))
                self.connection.commit()
                log.debug(f"Summary with article id: {article_id} deleted successfully.")
            else:
                log.warning(
                    f"Cannot delete summary for article id: {article_id}. Summary doesn't exist"
                )
                self.connection.rollback()
        except Exception as e:
            log.error(f"Error deleting row: {e}")
            self.connection.rollback()

    def fetch_blacklist(self):
        if not self.connection or self.connection.closed:
            self.connect()
        try:
            self.cursor.execute(
                """SELECT url FROM justification_source WHERE black_listed is TRUE""",
            )
            result = self.cursor.fetchall()
            return [row[0] for row in result]

        except Exception as e:
            print(f"Error retrieving result: {e}")
            self.connection.rollback()
            return []

    def fetch_whitelist(self):
        pass

    def fetch_ground_truth_label(self, statement_id):
        if not self.connection or self.connection.closed:
            self.connect()
        try:
            self.cursor.execute(
                """SELECT fact_checker_accuracy FROM statement WHERE id = %s""",
                (statement_id,),
            )
            result = self.cursor.fetchone()[0]

            return int(result)
        except Exception as e:
            print(f"Error retrieving result: {e}")
            self.connection.rollback()
            return None

    def fetch_statements_urls(self):
        from urllib.parse import urlparse

        if not self.connection or self.connection.closed:
            self.connect()

        try:
            sql = """
                SELECT url
                FROM statement_source;
            """
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            domains = {urlparse(row[0]).netloc for row in results if row[0]}
            return sorted(domains)

        except Exception as e:
            print(f"Error fetching articles without summary: {e}")
            self.connection.rollback()
            return []
