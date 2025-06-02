import pandas as pd
from check4facts.scripts.rag.pipeline import run_pipeline
from check4facts.scripts.rag.harvester import *
from check4facts.scripts.web_crawler_rag.crawl4ai import crawl4ai
from check4facts.scripts.web_crawler_rag.url_generation import url_generation


def testing_new():
    from check4facts.api import dbh
    import os
    from urllib.parse import urlparse

    # whitelist = dbh.fetch_statements_urls()
    # print(whitelist)

    # whitelist_pandemic = dbh.fetch_sources_from_articles_content_category_wise(id=1005)
    # whitelist_crime = dbh.fetch_sources_from_articles_content_category_wise(id=3)
    # whitelist_climate = dbh.fetch_sources_from_articles_content_category_wise(id=1004)
    # whitelist_immigration = dbh.fetch_sources_from_articles_content_category_wise(id=2)
    # whitelist_digital_transition = (
    #     dbh.fetch_sources_from_articles_content_category_wise(id=1013)
    # )

    # pandemic_domains = list(
    #     {urlparse(url).netloc for urls in whitelist_pandemic.values() for url in urls}
    # )
    # crime_domains = list(
    #     {urlparse(url).netloc for urls in whitelist_crime.values() for url in urls}
    # )
    # climate_domains = list(
    #     {urlparse(url).netloc for urls in whitelist_climate.values() for url in urls}
    # )
    # immigration_domains = list(
    #     {
    #         urlparse(url).netloc
    #         for urls in whitelist_immigration.values()
    #         for url in urls
    #     }
    # )
    # digital_transition_domains = list(
    #     {
    #         urlparse(url).netloc
    #         for urls in whitelist_digital_transition.values()
    #         for url in urls
    #     }
    # )

    claim = "Ο Βελόπουλος ισχυρίστηκε ότι τα ΜΜΕ δεν καλύπτουν τους κινδύνους της Τεχνητής Νοημοσύνης."

    domains = url_generation(claim=claim)
    domains.return_whitelist()
    # print("Pandemic: ")
    # print(pandemic_domains)
    # print()
    # print("Crime: ")
    # print(crime_domains)
    # print()
    # print("Climate: ")
    # print(climate_domains)
    # print()
    # print("Immigration: ")
    # print(immigration_domains)
    # print()

    # index = 0
    # results_list = []
    # df = pd.read_csv("./data/updated_sources.csv")

    # grouped_urls = df.groupby("statement_id")["urls"].apply(list)
    # if not os.path.exists("result_temp.csv"):
    #     pd.DataFrame(columns=["statement_id", "claim", "urls", "result"]).to_csv(
    #         "result_temp.csv", index=False
    #     )

    # for statement_id, urls in grouped_urls.items():

    #     claim = dbh.fetch_single_statement(statement_id)
    #     print(
    #         f"""
    #         statement_id: {statement_id},
    #         claim: {claim},
    #         urls: {urls}"""
    #     )

    #     try:
    #         crawler = crawl4ai(claim, len(urls), article_id=9999, provided_urls=urls)
    #         llm_instance = crawler.run_crawler()

    #         # llm_instance = run_crawler(
    #         #     article_id=9999,
    #         #     claim=claim,
    #         #     num_of_web_sources=len(urls),
    #         #     provided_urls=urls,
    #         # )
    #         retrieved_knowledge = None
    #         justification = None
    #         llm = None
    #         label = None
    #         urls = None

    #         for key, value in llm_instance.items():
    #             if key == "label":
    #                 label = value
    #             if key == "model":
    #                 llm = value
    #             if key == "external_sources":
    #                 retrieved_knowledge = value
    #             if key == "justification":
    #                 justification = value
    #             if key == "sources":
    #                 urls = value

    #         result = {
    #             "statement_id": statement_id,
    #             "claim": claim,
    #             "urls": urls,
    #             "label": label,
    #             "true_label": dbh.fetch_ground_truth_label(statement_id),
    #             "llm": llm,
    #             "retrieved_knowledge": retrieved_knowledge,
    #             "justification": justification,
    #         }
    #     except Exception as e:
    #         print(f"Error during new rag batch process {e}")
    #         result = {
    #             "statement_id": statement_id,
    #             "claim": claim,
    #             "urls": urls,
    #             "label": None,
    #             "true_label": dbh.fetch_ground_truth_label(statement_id),
    #             "llm": None,
    #             "retrieved_knowledge": None,
    #             "justification": e,
    #         }
    #         break

    #     print(result)
    #     results_list.append(result)
    #     pd.DataFrame([result]).to_csv(
    #         "result_temp.csv", mode="a", header=False, index=False
    #     )

    # results_df = pd.DataFrame(results_list)
    # results_df = results_df.to_csv("./data/new_results_supervised_0_6.csv", index=False)
    # print("Finished!")
    # return results_df
