"""
PySpark ETL step.

Reads the raw e-commerce CSVs (products, reviews, inventory, support_tickets),
joins and aggregates them with Spark DataFrames, and writes one compact
"knowledge base" file: data/processed/knowledge_base.jsonl

Each line in that file is one retrievable text chunk about one product
(doc_type is one of: description, reviews, inventory, support). The RAG
agent later embeds and searches over these chunks.

Run with:
    python src/etl_pyspark.py
"""

import json
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "processed", "knowledge_base.jsonl")


def main():
    spark = SparkSession.builder.appName("ecommerce-qa-etl").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")  # quiet Spark's log noise for a cleaner course demo

    products = spark.read.csv(os.path.join(RAW_DIR, "products.csv"), header=True, inferSchema=True)
    reviews = spark.read.csv(os.path.join(RAW_DIR, "reviews.csv"), header=True, inferSchema=True)
    inventory = spark.read.csv(os.path.join(RAW_DIR, "inventory.csv"), header=True, inferSchema=True)
    tickets = spark.read.csv(os.path.join(RAW_DIR, "support_tickets.csv"), header=True, inferSchema=True)

    # Average rating + review count per product, plus a simple recommendation
    # label derived with Spark's when/otherwise (a real transformation, not
    # just a passthrough).
    review_stats = (
        reviews.groupBy("product_id")
        .agg(
            F.round(F.avg("rating"), 2).alias("avg_rating"),
            F.count("*").alias("review_count"),
        )
        .withColumn(
            "sentiment_label",
            F.when(F.col("avg_rating") >= 4.0, "Recommended")
            .when(F.col("avg_rating") >= 3.0, "Mixed")
            .otherwise("Not recommended"),
        )
    )

    # Ticket count + list of issue summaries per product.
    ticket_stats = tickets.groupBy("product_id").agg(
        F.count("*").alias("ticket_count"),
        F.collect_list("issue_summary").alias("issues"),
    )

    # Left-join everything onto the product catalog: not every product has
    # reviews or tickets, so missing sides should stay null rather than
    # dropping the product.
    joined = (
        products.join(review_stats, on="product_id", how="left")
        .join(inventory, on="product_id", how="left")
        .join(ticket_stats, on="product_id", how="left")
    )

    # The joined result is tiny (one row per product). At this scale it's
    # simplest and most beginner-friendly to collect it to the driver and
    # write one clean JSON file in plain Python, instead of dealing with
    # Spark's multi-part distributed file output.
    joined_rows = joined.collect()

    # A few representative review quotes per product are easiest to grab
    # straight from the raw reviews table after collecting.
    review_rows = reviews.select("product_id", "rating", "review_text").collect()
    quotes_by_product = {}
    for r in review_rows:
        quotes_by_product.setdefault(r["product_id"], []).append((r["rating"], r["review_text"]))

    chunks = []
    for row in joined_rows:
        pid = row["product_id"]
        name = row["product_name"]

        # description chunk
        description_text = f"{name} ({row['category']}, ${row['price']:.2f}). {row['description']}"
        chunks.append({"product_id": pid, "product_name": name, "doc_type": "description", "text": description_text})

        # reviews chunk
        avg_rating = row["avg_rating"]
        if avg_rating is not None:
            quotes = [f'"{text}" ({rating}/5)' for rating, text in quotes_by_product.get(pid, [])[:3]]
            reviews_text = (
                f"{name} has an average rating of {avg_rating}/5 from {row['review_count']} reviews. "
                f"Overall customer sentiment: {row['sentiment_label']}. "
                f"Example reviews: " + " | ".join(quotes)
            )
        else:
            reviews_text = f"{name} has no customer reviews yet."
        chunks.append({"product_id": pid, "product_name": name, "doc_type": "reviews", "text": reviews_text})

        # inventory chunk
        stock_qty = row["stock_quantity"]
        if stock_qty is None:
            inventory_text = f"No inventory record found for {name}."
        elif stock_qty > 0:
            inventory_text = f"{name} is in stock: {stock_qty} units available at {row['warehouse_location']}."
        else:
            inventory_text = (
                f"{name} is currently out of stock at {row['warehouse_location']}. "
                f"Expected restock date: {row['restock_date']}."
            )
        chunks.append({"product_id": pid, "product_name": name, "doc_type": "inventory", "text": inventory_text})

        # support chunk
        ticket_count = row["ticket_count"]
        if not ticket_count:
            support_text = f"No reported issues or support tickets for {name}."
        else:
            issues_list = "; ".join(row["issues"])
            support_text = f"{name} has {ticket_count} reported support ticket(s). Common issues: {issues_list}."
        chunks.append({"product_id": pid, "product_name": name, "doc_type": "support", "text": support_text})

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {len(chunks)} chunks for {len(joined_rows)} products to {OUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
