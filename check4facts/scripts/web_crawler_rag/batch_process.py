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

    blacklisted_ids = [129]
    results_path = "data/rag_results_unverified_ids.csv"
    unverified_ids = [
        7,
        24,
        77,
        51,
        63,
        171,
        247,
        127,
        164,
        259,
        130,
        252,
        104,
        57,
        166,
        249,
        167,
        174,
        260,
        67,
        88,
        131,
        45,
        89,
        75,
        133,
        90,
        81,
        177,
        83,
        135,
        97,
        91,
        136,
        86,
        137,
        34,
        189,
        235,
    ]

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
        time.sleep(10)
        if id in processed_ids:
            print(f"ID:{id} is already processed. Skipping....")
            continue  # Skip already processed claims
        if id in blacklisted_ids:
            print(f"ID:{id} is blacklisted. Skipping....")
            continue
        if id not in unverified_ids:
            print(f"ID:{id} is not in the unverified list. Skipping....")
            continue

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
                answer["ground_truth"] = dbh.fetch_ground_truth_label(id)

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
