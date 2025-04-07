import psycopg2
import pandas as pd
from bs4 import BeautifulSoup


def fetch_sources_from_articles_content(page_size=50):
    # Database connection parameters
    DB_PARAMS = {
        "dbname": "check4facts_2",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432",
    }

    fact_checker_sources = {}
    last_id = 0

    try:
        # Connect to the database
        connection = psycopg2.connect(**DB_PARAMS)
        cursor = connection.cursor()

        while True:
            query = f"""
                SELECT statement_id, content
                FROM article
                WHERE statement_id > {last_id}
                ORDER BY statement_id
                LIMIT {page_size};
            """
            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                break

            last_id = int(results[-1][0])

            for statement_id, content in results:
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

        # Filter out empty lists
        fact_checker_sources = {k: v for k, v in fact_checker_sources.items() if v}

        return fact_checker_sources

    except Exception as e:
        print(f"Error fetching page of articles contents: {e}")
        if connection:
            connection.rollback()
        return {}

    finally:
        if connection:
            cursor.close()
            connection.close()


# Run the function and save the results to a DataFrame
if __name__ == "__main__":
    sources = fetch_sources_from_articles_content()

    # Convert dictionary to DataFrame
    df = pd.DataFrame(
        [(k, url) for k, urls in sources.items() for url in urls],
        columns=["statement_id", "urls"],
    )

    # Save to CSV
    df.to_csv("sources.csv", index=False)

    print(df)
