import pandas as pd
from check4facts.scripts.rag.pipeline import run_pipeline
from check4facts.scripts.rag.harvester import *
from check4facts.scripts.web_crawler_rag.crawl4ai import crawl4ai
from check4facts.scripts.web_crawler_rag.url_generation import url_generation


def testing_new():
    from check4facts.api import dbh
    import os
    import time
    import numpy as np
    import pandas as pd
    import re
    from urllib.parse import urlparse

    ids_to_process = [
        40,
        53,
        72,
        79,
        93,
        99,
        112,
        128,
        129,
        132,
        154,
        155,
        157,
        158,
        160,
        163,
        169,
        170,
        173,
        175,
        176,
        178,
        223,
        247,
        248,
        250,
    ]

    results_path = "data/rag_results.csv"

    # Load already processed IDs if file exists
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        processed_ids = set(existing_df["id"])
        results = existing_df.to_dict(orient="records")
    else:
        processed_ids = set()
        results = []

    statements = dbh.fetch_all_statement_texts()

    for id, statement in statements:
        if int(id) not in ids_to_process:
            continue
        if id in processed_ids:
            continue  # Skip already processed claims

        print(f"Processing ID: {id}")
        print(statement)
        print("----")

        try:
            crawler = crawl4ai(
                claim=statement, web_sources=1, article_id=id, provided_urls=None
            )
            answer = crawler.run_crawler()

            if answer:
                print("FINAL ANSWER:")
                print()
                for key, value in answer.items():
                    print(f"{key}: {value}")

                answer["id"] = id
                answer["claim"] = statement

                results.append(answer)

                # Save after each result
                df = pd.DataFrame(results)
                df.to_csv(results_path, index=False)
                print(f"Progress saved to {results_path}")

            else:
                raise Exception("Pipeline returned empty result")

        except Exception as e:
            print(f"Error during new rag run for ID {id}: {e}")

    print("Finished processing.")
