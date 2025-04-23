import pandas as pd
from check4facts.scripts.rag.pipeline import run_pipeline
from check4facts.scripts.rag.harvester import *


def testing():
    from check4facts.api import dbh
    import os

    index = 0
    results_list = []
    df = pd.read_csv("data/sources.csv")
    df2 = pd.read_csv("result_temp_2.csv")

    grouped_urls = df.groupby("statement_id")["urls"].apply(list)
    if not os.path.exists("result_temp.csv"):
        pd.DataFrame(columns=["statement_id", "claim", "urls", "result"]).to_csv(
            "result_temp.csv", index=False
        )

    for statement_id, urls in grouped_urls.items():

        if statement_id in df2.index:
            print("Already processed")
            continue

        for url in urls:
            if "https://www.icao.int/environmental-protection" in url:
                urls.remove(url)

        # fetch statement claim from statement table
        claim = dbh.fetch_single_statement(statement_id)
        print(
            f"""
            statement_id: {statement_id}, 
            claim: {claim}, 
            urls: {urls}"""
        )
        try:
            llm_instance = run_pipeline(
                article_id=9999,
                claim=claim,
                num_of_web_sources=len(urls),
                provided_urls=urls,
            )

            llm = None
            label = None

            for key, value in llm_instance.items():
                print(f"{key}: {value}")
                if key == "label":
                    label = value
                    print(label)
                if key == "model":
                    llm = value
                    print(llm)

            result = {
                "statement_id": statement_id,
                "claim": claim,
                "urls": urls,
                "label": label,
                "llm": llm,
            }
        except Exception as e:
            print(e)
            result = {
                "statement_id": statement_id,
                "claim": claim,
                "urls": urls,
                "label": None,
                "llm": None,
            }

        print(result)
        results_list.append(result)
        pd.DataFrame([result]).to_csv(
            "result_temp.csv", mode="a", header=False, index=False
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.to_csv("results.csv", index=False)
    print("Finished!")
    return results_df
