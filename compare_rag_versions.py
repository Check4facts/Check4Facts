import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix


db_url = "postgresql://check4facts@localhost:5432/check4facts"  # Update with your credentials
engine = create_engine(db_url)

try:
    with engine.connect() as connection:
        print("Database connection successful.")
except Exception as e:
    print(f"Database connection failed: {e}")

with engine.connect() as connection:
    # Load statement_rag.csv into a DataFrame
    statements_rag_df = pd.read_csv("data/statements_rag.csv")
    statements_rag_df.rename(columns={"label": "rag_v2_label"}, inplace=True)
    statements_rag_df["rag_v2_label"] = statements_rag_df["rag_v2_label"].astype(int)
    # print(statements_rag_df.dtypes)
    # print(statements_rag_df.head())
    # Convert statement IDs to a list
    statement_ids = tuple(statements_rag_df["statement_id"].tolist())
    # Fetch fact_checker_accuracy for statements in statements_rag_df using pandas
    query = f"""
        SELECT s.id, s.fact_checker_accuracy, j.label as rag_v1_label
        FROM statement s
        INNER JOIN justification j ON s.id = j.statement_id
        WHERE s.id IN {statement_ids} AND j.timestamp = (
            SELECT MAX(j2.timestamp)
            FROM justification j2
            WHERE j2.statement_id = j.statement_id
        )
        ORDER BY s.id;
    """
    result_df = pd.read_sql_query(query, connection)
    # print(result_df.head())
    # print(result_df.dtypes)
    # No justification from rag_v1 due to no sources found to harvest for statements 235,255
    print(
        f"Justification rag_v1 for {len(result_df)} statements\n",
        f"Justification rag_v2 for {len(statement_ids)} statements\n",
        len(result_df) == len(statement_ids),
    )

    fact_rag1_rag2_labels_df = pd.merge(
        statements_rag_df,
        result_df,
        left_on="statement_id",
        right_on="id",
        how="inner",
    )
    fact_rag1_rag2_labels_df["rag_v1_label"] = fact_rag1_rag2_labels_df[
        "rag_v1_label"
    ].astype(int)
    print(fact_rag1_rag2_labels_df.head())
    print(fact_rag1_rag2_labels_df.dtypes)
    print(
        f"Justification with both models for {len(fact_rag1_rag2_labels_df)} statements"
    )

    custom_labels = [
        "Unverifiable",
        "Inaccurate",
        "Relatively Inaccurate",
        "Relatively Accurate",
        "Accurate",
    ]

    # Create directory if it doesn't exist to store the plots
    output_dir = "data/rag_v1_vs_rag_v2_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plotting Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_rag_v1 = confusion_matrix(
        fact_rag1_rag2_labels_df["fact_checker_accuracy"],
        fact_rag1_rag2_labels_df["rag_v1_label"],
    )
    cm_rag_v2 = confusion_matrix(
        fact_rag1_rag2_labels_df["fact_checker_accuracy"],
        fact_rag1_rag2_labels_df["rag_v2_label"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    sns.heatmap(cm_rag_v1, annot=True, cmap="Blues", fmt="d", ax=axes[0])
    axes[0].set_title("RAG Version 1 vs Fact Checker")
    axes[0].set_xlabel("RAG Version 1 Predictions")
    axes[0].set_ylabel("Fact Checker Accuracy")

    sns.heatmap(cm_rag_v2, annot=True, cmap="Greens", fmt="d", ax=axes[1])
    axes[1].set_title("RAG Version 2 vs Fact Checker")
    axes[1].set_xlabel("RAG Version 2 Predictions")
    axes[1].set_ylabel("Fact Checker Accuracy")

    # Add a legend for custom labels
    legend_elements = [
        plt.Line2D(
            [0], [0], color="w", marker="o", markersize=10, label=f"{i}: {label}"
        )
        for i, label in enumerate(custom_labels)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, title="Legend")

    plt.savefig(os.path.join(output_dir, "cm_rag_v1_vs_rag_v2.png"))
    print(
        f"Confusion matrix saved as {os.path.join(output_dir, 'cm_rag_v1_vs_rag_v2.png')}"
    )

    # Plotting KDE plot
    plt.figure(figsize=(7, 5))
    sns.kdeplot(
        fact_rag1_rag2_labels_df["rag_v1_label"],
        label="RAG Version 1",
        fill=True,
        alpha=0.5,
    )
    sns.kdeplot(
        fact_rag1_rag2_labels_df["rag_v2_label"],
        label="RAG Version 2",
        fill=True,
        alpha=0.5,
    )
    sns.kdeplot(
        fact_rag1_rag2_labels_df["fact_checker_accuracy"],
        label="Fact Checker",
        fill=True,
        alpha=0.5,
        linestyle="dashed",
    )
    print("RAG V1 unique labels:", fact_rag1_rag2_labels_df["rag_v1_label"].unique())
    print("RAG V2 unique labels:", fact_rag1_rag2_labels_df["rag_v2_label"].unique())
    print("Fact checker unique labels:", fact_rag1_rag2_labels_df["fact_checker_accuracy"].unique())


    plt.xlabel("Accuracy Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of Accuracy Scores (with Fact Checker)")

    plt.savefig(os.path.join(output_dir, "kde_rag_v1_vs_rag_v2.png"))
    print(f"KDE plot saved as {os.path.join(output_dir, 'kde_rag_v1_vs_rag_v2.png')}")
    
    # Set the style
    sns.set(style="whitegrid")

    # Set up the plot
    plt.figure(figsize=(8, 5))

    # Plot histograms
    bin_edges = [i - 0.5 for i in range(6)]  # [-0.5, 0.5, 1.5, ..., 4.5] for integer bins

    sns.histplot(fact_rag1_rag2_labels_df["rag_v1_label"], bins=bin_edges, stat="density", label="RAG V1",
                color="skyblue", kde=True, element="step", linewidth=2)

    sns.histplot(fact_rag1_rag2_labels_df["rag_v2_label"], bins=bin_edges, stat="density", label="RAG V2",
                color="lightgreen", kde=True, element="step", linewidth=2)

    sns.histplot(fact_rag1_rag2_labels_df["fact_checker_accuracy"], bins=bin_edges, stat="density", label="Fact Checker",
                color="gray", kde=True, element="step", linewidth=2, linestyle="dashed")

    # Set ticks to match integer classes
    plt.xticks(range(5))
    plt.xlim(-0.5, 4.5)
    plt.xlabel("Accuracy Label")
    plt.ylabel("Density")
    plt.title("Discrete Accuracy Distribution (with KDE overlay)")
    plt.legend()

    # Save to file
    plt.savefig(os.path.join(output_dir, "hist_rag_v1_vs_rag_v2.png"))
    print(f"KDE/Histogram plot saved as {os.path.join(output_dir, 'hist_rag_v1_vs_rag_v2.png')}")

    # Plotting Scatter plot with Jittering
    plt.figure(figsize=(6, 5))

    fact_rag1_rag2_labels_df["Jitter"] = np.random.normal(
        0, 0.1, len(fact_rag1_rag2_labels_df)
    )  # Small noise to avoid overlapping points

    plt.scatter(
        fact_rag1_rag2_labels_df["rag_v1_label"] + fact_rag1_rag2_labels_df["Jitter"],
        fact_rag1_rag2_labels_df["fact_checker_accuracy"]
        + fact_rag1_rag2_labels_df["Jitter"],
        alpha=0.5,
        label="RAG Version 1",
        color="blue",
    )

    plt.scatter(
        fact_rag1_rag2_labels_df["rag_v2_label"] + fact_rag1_rag2_labels_df["Jitter"],
        fact_rag1_rag2_labels_df["fact_checker_accuracy"]
        + fact_rag1_rag2_labels_df["Jitter"],
        alpha=0.5,
        label="RAG Version 2",
        color="green",
    )

    plt.plot([0, 4], [0, 4], "r--", label="Perfect Agreement")

    plt.xlabel("Predicted Score")
    plt.ylabel("Fact Checker Score")
    plt.legend()
    plt.title("Comparison of Accuracy Scores vs Fact Checker")

    plt.savefig(os.path.join(output_dir, "scatter_plot_rag_v1_vs_rag_v2.png"))
    print(
        f"Scatter plot saved as {os.path.join(output_dir, 'scatter_plot_rag_v1_vs_rag_v2.png')}"
    )

    # Plotting Bar Chart of Mean Differences
    plt.figure(figsize=(6, 5))
    fact_rag1_rag2_labels_df["RAG_v1_Difference"] = (
        fact_rag1_rag2_labels_df["rag_v1_label"]
        - fact_rag1_rag2_labels_df["fact_checker_accuracy"]
    )
    fact_rag1_rag2_labels_df["RAG_v2_Difference"] = (
        fact_rag1_rag2_labels_df["rag_v2_label"]
        - fact_rag1_rag2_labels_df["fact_checker_accuracy"]
    )

    mean_diff_old = [
        fact_rag1_rag2_labels_df[
            fact_rag1_rag2_labels_df["fact_checker_accuracy"] == i
        ]["RAG_v1_Difference"].mean()
        for i in range(5)
    ]
    mean_diff_rag = [
        fact_rag1_rag2_labels_df[
            fact_rag1_rag2_labels_df["fact_checker_accuracy"] == i
        ]["RAG_v2_Difference"].mean()
        for i in range(5)
    ]

    bar_width = 0.4
    x_labels = range(5)

    plt.bar(
        x_labels,
        mean_diff_old,
        width=bar_width,
        label="RAG Version 1",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        [x + bar_width for x in x_labels],
        mean_diff_rag,
        width=bar_width,
        label="RAG Version 2",
        color="green",
        alpha=0.7,
    )

    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fact Checker Score")
    plt.ylabel("Mean Difference (Model - Fact Checker)")
    plt.legend()
    plt.title("Mean Difference in Accuracy Scores vs Fact Checker")

    plt.savefig(os.path.join(output_dir, "bar_chart_rag_v1_vs_rag_v2.png"))
    print(
        f"Bar chart saved as {os.path.join(output_dir, 'bar_chart_rag_v1_vs_rag_v2.png')}"
    )

    # Plotting Bland-Altman Plot
    plt.figure(figsize=(6, 5))

    fact_rag1_rag2_labels_df["RAG_v1_Mean"] = (
        fact_rag1_rag2_labels_df["rag_v1_label"]
        + fact_rag1_rag2_labels_df["fact_checker_accuracy"]
    ) / 2
    fact_rag1_rag2_labels_df["RAG_v1_Diff"] = (
        fact_rag1_rag2_labels_df["rag_v1_label"]
        - fact_rag1_rag2_labels_df["fact_checker_accuracy"]
    )

    fact_rag1_rag2_labels_df["RAG_v2_Mean"] = (
        fact_rag1_rag2_labels_df["rag_v2_label"]
        + fact_rag1_rag2_labels_df["fact_checker_accuracy"]
    ) / 2
    fact_rag1_rag2_labels_df["RAG_v2_Diff"] = (
        fact_rag1_rag2_labels_df["rag_v2_label"]
        - fact_rag1_rag2_labels_df["fact_checker_accuracy"]
    )

    plt.scatter(
        fact_rag1_rag2_labels_df["RAG_v1_Mean"],
        fact_rag1_rag2_labels_df["RAG_v1_Diff"],
        alpha=0.5,
        label="RAG Version 1",
        color="blue",
    )
    plt.scatter(
        fact_rag1_rag2_labels_df["RAG_v2_Mean"],
        fact_rag1_rag2_labels_df["RAG_v2_Diff"],
        alpha=0.5,
        label="RAG Version 2",
        color="green",
    )

    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Mean Accuracy Score (Model & Fact Checker)")
    plt.ylabel("Difference (Model - Fact Checker)")
    plt.legend()
    plt.title("Bland-Altman Plot vs Fact Checker")

    plt.savefig(os.path.join(output_dir, "bland_altman_plot_rag_v1_vs_rag_v2.png"))
    print(
        f"Bland-Altman plot saved as {os.path.join(output_dir, 'bland_altman_plot_rag_v1_vs_rag_v2.png')}"
    )
