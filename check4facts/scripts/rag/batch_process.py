import pandas as pd
from check4facts.scripts.rag.pipeline import run_pipeline
from check4facts.scripts.rag.harvester import *


def testing():
    from check4facts.api import dbh
    import os

    index = 0
    results_list = []
    df = pd.read_csv("./data/updated_sources.csv")

    grouped_urls = df.groupby("statement_id")["urls"].apply(list)
    if not os.path.exists("result_temp.csv"):
        pd.DataFrame(columns=["statement_id", "claim", "urls", "result"]).to_csv(
            "result_temp.csv", index=False
        )

    for statement_id, urls in grouped_urls.items():

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
                provided_urls=None,
            )
            retrieved_knowledge = None
            justification = None
            llm = None
            label = None
            urls = None

            for key, value in llm_instance.items():
                if key == "label":
                    label = value
                if key == "model":
                    llm = value
                if key == "external_sources":
                    retrieved_knowledge = value
                if key == "justification":
                    justification = value
                if key == "sources":
                    urls = value

            result = {
                "statement_id": statement_id,
                "claim": claim,
                "urls": urls,
                "label": label,
                "true_label": dbh.fetch_ground_truth_label(statement_id),
                "llm": llm,
                "retrieved_knowledge": retrieved_knowledge,
                "justification": justification,
            }
        except Exception as e:
            print(e)
            result = {
                "statement_id": statement_id,
                "claim": claim,
                "urls": urls,
                "label": None,
                "true_label": dbh.fetch_ground_truth_label(statement_id),
                "llm": None,
                "retrieved_knowledge": None,
                "justification": e,
            }

        print(result)
        results_list.append(result)
        pd.DataFrame([result]).to_csv(
            "result_temp.csv", mode="a", header=False, index=False
        )

    results_df = pd.DataFrame(results_list)
    results_df = results_df.to_csv("./data/results_supervised_0_1.csv", index=False)
    print("Finished!")
    return results_df
