import marimo

__generated_with = "0.11.14"
app = marimo.App(
    width="full",
    app_title="Sales Analysis",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# <font color="orange">Sales Analysis 📊</font>""").center()
    return


@app.cell
def _(mo):
    mo.md(
        """/// details | **SOME HEADS UP BEFORE WE GO AHEAD!**_(Click to view)_

    ### *Analysis outcomes* :
    >1. Find the characteristics of the most successful merchants<br>
    >2. What are the top 2 shipping carriers? Why should or shouldn't we try to use those two for all orders? 
    >3. Choose the top two print providers to give discount and end contract with the worst two. Explain with reason.
    ///

    """
    ).callout("info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Business model 🧭""").center()
    return


@app.cell
def _(mo):
    mo.image(r'public/Picture.jpg').center()
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "## Shipping Carriers:": """
                 The model relies on two shipping carriers to handle logistics and delivery. These carriers are essential for transporting products from the business to the end customers, ensuring timely and efficient distribution.
                """,
            "## End Customers:": """
                The ultimate recipients of the products are the end customers. They can purchase products through various sales channels, including Shop 1 and Shop 2. This indicates that the business leverages multiple retail outlets or online platforms to reach a broader audience.
                """,
            "## Print Providers:": f"""
                The inclusion of print providers suggests that the business offers customized products or packaging. These providers are responsible for adding personalized elements, which could be a key differentiator in the market.
                """,
            "## Line Items:": f"""
               These represent individual products or services within an order. The presence of multiple line items indicates that customers can order several products simultaneously, enhancing the convenience and variety offered by the business.""",
            "## Orders:": """This section lists individual transactions or batches of products being processed. The sequence of orders (Order 1, Order 2, etc.) highlights the business's capability to handle multiple transactions efficiently.""",
        }
    )
    return


@app.cell(hide_code=True)
def _():
    # Polars for data manipulation

    import polars as pl

    # datetime for date & time operations

    import datetime as dt


    from matplotlib import pyplot as plt

    # missingno to visualize missing data

    import missingno as msno

    # duckdb for sql operations

    import duckdb as db

    # Marimo for notebook

    import marimo as mo

    # plotly for visualization

    import plotly.express as px

    import plotly.graph_objects as go

    import numpy as np
    return db, dt, go, mo, msno, np, pl, plt, px


@app.cell(hide_code=True)
def _(pl):
    # Loading line_items csv

    lineItems = pl.read_csv(
        r"https://raw.githubusercontent.com/fadirasak/fadirasak.github.io/refs/heads/main/apps/public/line_items.csv", try_parse_dates=True
    )

    # Loading orders csv

    orders = pl.read_parquet(
        r"public/orders2.parquet"
    )
    return lineItems, orders


@app.cell(hide_code=True)
def _(lineItems, msno, orders, pl, plt):
    # visualizing the missing data

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    msno.matrix(lineItems.to_pandas(), ax=axs[0],sparkline=False,fontsize=7)
    axs[0].set_title('Missing Values Matrix - lineItems')
    msno.matrix(orders.to_pandas(), ax=axs[1],sparkline=False,fontsize=7)
    axs[1].set_title('Missing Values Matrix - orders')
    plt.tight_layout()
    plt.show()

    print(f'''number of null values in lineItems: {lineItems.select(pl.all().is_null().sum())}

    number of null values in orders: {orders.select(pl.all().is_null().sum())}''')
    return axs, fig


@app.cell(hide_code=True)
def _(orders, pl):
    orders_1 = orders.filter(pl.col("TOTAL_COST") != 0)
    return (orders_1,)


@app.cell(hide_code=True)
def _(lineItems, orders_1):
    lineItems_1 = lineItems.unique()
    orders_2 = orders_1.unique()
    return lineItems_1, orders_2


@app.cell(hide_code=True)
def _(lineItems_1, orders_2, pl):
    lineItems_2 = lineItems_1.with_columns(
        pl.col("REPRINT_FLAG")
        .fill_null("false")
        .replace({"true": True, "false": False}, default=None),
        pl.col("PRODUCT_BRAND", "PRODUCT_TYPE").cast(pl.Categorical),
    )
    orders_3 = orders_2.with_columns(
        pl.col("ADDRESS_TO_COUNTRY", "SUB_PLAN").cast(pl.Categorical)
    )
    return lineItems_2, orders_3


@app.cell(hide_code=True)
def _(orders_3, pl):
    orders_4 = orders_3.with_columns(
        pl.col("SHIPMENT_CARRIER")
        .str.replace_all("Usps[^\\s]*$", "USPS")
        .str.replace_all("Ups[^\\s]*$", "UPS")
        .str.replace_all("Ups Mail Innovations.*", "UPS Mail Innovations")
        .str.replace_all("Dhl[^\\s]*$", "DHL")
        .str.replace_all("Dhl Global.*", "DHL Global")
        .str.replace_all("Dhl Germany.*", "DHL Germany")
        .str.replace_all("Dhl Express.*", "DHL Express")
        .str.replace_all("Asendia.*", "Asendia")
        .str.replace_all("Canada.*", "Canada Post")
        .str.replace_all("Globegistics.*", "Globegistics")
        .str.replace_all("Deutsche.*", "Deutsche Post")
        .str.replace_all("Fedex.*", "Fedex")
        .str.replace_all("Dpd.*", "DPD")
        .str.replace_all("Couriersplease.*", "CouriersPlease")
        .replace({"Unknown": None})
        .cast(pl.Categorical)
    )
    return (orders_4,)


@app.cell(hide_code=True)
def _(orders_4):
    orders_5 = orders_4.drop('ADDRESS_TO_REGION')
    return (orders_5,)


@app.cell(hide_code=True)
def _(orders_5, pl):
    orders_6 = orders_5.with_columns(
        pl.col("FULFILLED_DT")
        .dt.to_string("%Y-%m-%d %H:%M:%S")
        .replace({"2012-12-06 00:00:00": "2021-12-06 00:00:00"})
        .str.to_datetime("%Y-%m-%d %H:%M:%S")
    )
    return (orders_6,)


@app.cell(hide_code=True)
def _(lineItems_2, orders_6):
    lineItemOrders = lineItems_2.join(other=orders_6, on='ORDER_ID', how='inner')
    return (lineItemOrders,)


@app.cell(hide_code=True)
def _(lineItemOrders, pl):
    lineItemOrders_1 = lineItemOrders.with_columns(
        pl.when(pl.col("SHOP_ID") == 3592957)
        .then(pl.col("PRINT_PROVIDER_ID").fill_null(29))
        .otherwise(pl.col("PRINT_PROVIDER_ID"))
        .alias("PRINT_PROVIDER_ID"),
        pl.when(pl.col("SHOP_ID") == 3592957)
        .then(pl.col("SHIPMENT_CARRIER").fill_null("USPS"))
        .otherwise(pl.col("SHIPMENT_CARRIER"))
        .alias("SHIPMENT_CARRIER"),
    )
    return (lineItemOrders_1,)


@app.cell(hide_code=True)
def _(lineItemOrders_1, pl):
    lineItemOrders_2 = (
        lineItemOrders_1.with_columns(
            pl.when(
                (pl.col("FULFILLED_DT") < pl.col("ORDER_DT")).or_(
                    pl.col("FULFILLED_DT").is_null()
                )
            )
            .then(None)
            .otherwise("FULFILLED_DT")
            .alias("FULFILLED_DT")
        )
        .with_columns(
            pl.when(pl.col("FULFILLED_DT") >= pl.col("ORDER_DT"))
            .then(pl.col("FULFILLED_DT") - pl.col("ORDER_DT"))
            .otherwise(None)
            .alias("duration")
        )
        .with_columns(
            pl.col("duration")
            .mean()
            .over("PRINT_PROVIDER_ID")
            .alias("avg_duration")
        )
        .with_columns(
            (pl.col("ORDER_DT") + pl.col("avg_duration")).alias("avg_duration2")
        )
        .with_columns(pl.col("FULFILLED_DT").fill_null(pl.col("avg_duration2")))
        .with_columns(
            pl.col("duration").fill_null(
                pl.col("FULFILLED_DT") - pl.col("ORDER_DT")
            )
        )
        .drop(["avg_duration", "avg_duration2"])
    )
    return (lineItemOrders_2,)


@app.cell
def _(lineItemOrders_2, mo):
    mo.vstack(
        [mo.md("""##LineItem Dataframe _(Data used for this analysis)_""").center(), mo.ui.dataframe(lineItemOrders_2)]
    ) 
    return


@app.cell
def _(mo):
    mo.md(r"""## Which are the most successful merchants? """).callout('warn')
    return


@app.cell
def _(lineItemOrders_2, pl):
    Top5Merchants = lineItemOrders_2.group_by("MERCHANT_ID").agg(
        pl.sum("TOTAL_COST").alias("Total Sales"),
        pl.col("ORDER_ID").n_unique().alias("No. of orders"),
        pl.col("ADDRESS_TO_COUNTRY").n_unique().alias("No. of countries"),
    ).join(
        lineItemOrders_2.group_by(["MERCHANT_ID", "ORDER_ID"])
        .agg(
            pl.mean("TOTAL_COST").alias("Avg Sales"),
            pl.mean("QUANTITY").alias("Avg Quantity"),
        )
        .group_by("MERCHANT_ID")
        .agg(pl.mean("Avg Sales"), pl.mean("Avg Quantity").round()),
        on="MERCHANT_ID",
        how="inner",
    ).filter(
        pl.col("Total Sales").ge(pl.col("Total Sales").mean()),
        pl.col("No. of orders").ge(pl.col("No. of orders").mean()),
        pl.col("No. of countries").ge(pl.col("No. of countries").mean()),
        pl.col("Avg Sales").ge(pl.col("Avg Sales").mean()),
    ).sort(
        by=["Total Sales", "No. of orders", "No. of countries", "Avg Sales"],
        descending=True,
    ).cast({"MERCHANT_ID":pl.String}).head(5)
    return (Top5Merchants,)


@app.cell
def _(Top5Merchants, go, mo):
    def create_figure():
        figs = go.Figure()
    
        # Add Total Sales bars
        figs.add_trace(
            go.Bar(
                x=Top5Merchants["MERCHANT_ID"].to_list(),
                y=Top5Merchants["Total Sales"].to_list(),
                name="Total Sales",
                marker=dict(color="blue"),
            )
        )

        # Add No. of orders bars
        figs.add_trace(
            go.Bar(
                x=Top5Merchants["MERCHANT_ID"].to_list(),
                y=Top5Merchants["No. of orders"].to_list(),
                name="No. of orders",
                yaxis="y2",  # Use secondary y-axis for smaller metric
                marker=dict(color="green"),
            )
        )

        # Add No. of countries bars
        figs.add_trace(
            go.Bar(
                x=Top5Merchants["MERCHANT_ID"].to_list(),
                y=Top5Merchants["No. of countries"].to_list(),
                name="No. of countries",
                yaxis="y2",  # Use secondary y-axis for smaller metric
                marker=dict(color="orange"),
            )
        )

        # Add Avg Sales bars
        figs.add_trace(
            go.Bar(
                x=Top5Merchants["MERCHANT_ID"].to_list(),
                y=Top5Merchants["Avg Sales"].to_list(),
                name="Avg Sales",
                yaxis="y2",  # Use secondary y-axis for smaller metric
                marker=dict(color="red"),
            )
        )

        # Update layout for dual-axis
        figs.update_layout(
            title="Top 5 Merchants by Various Metrics",
            barmode="group",
            xaxis=dict(title="MERCHANT_ID"),
            yaxis=dict(
                title="Total Sales",
                tickfont=dict(color="white"),
            ),
            yaxis2=dict(
                title="Other Metrics",
                tickfont=dict(color="white"),
                overlaying="y",
                side="right",
            ),
            legend=dict(x=0.85, y=1),
        )
    
        return mo.ui.plotly(figs)

    plot1 = create_figure()

    return create_figure, plot1


@app.cell
def _(Top5Merchants, mo, pl, plot1):
    mo.hstack(
        [
            plot1,
            Top5Merchants
            if pl.DataFrame(plot1.value).is_empty()
            else pl.DataFrame(plot1.value),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What characteristics do the most successful merchants share?

        1. Number of Transactions 💰 – A higher number of transactions usually indicates strong demand and customer engagement.
        2. Total Sales (Revenue) 🏆 – The ultimate measure of success, as revenue directly reflects business performance.
        3. Number of Orders 📦 – A higher number of transactions usually indicates strong demand and customer engagement.
        4. Number of Countries 🌍 – Expansion across multiple regions is a sign of a strong global presence and scalability.
        5. Average Sales per Order 💰 – Higher average sales mean customers are spending more per transaction, boosting profitability.
        """
    ).callout(kind='success')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""## What are the top two shipping carriers? Why should or shouldn’t we try to use those two for all orders?"""
    ).callout("warn")
    return


@app.cell
def _(mo, px, top2ShippingCarriers):
    plot2 = mo.ui.plotly(px.scatter(
                top2ShippingCarriers,
                x="avg days",
                y="count",
                size="avg shipping cost",
                color="SHIPMENT_CARRIER",
                hover_name="SHIPMENT_CARRIER",
            ).update_layout(
                title="Top 5 shipping carriers",
                xaxis_title="Average Days",
                yaxis_title="Shipments",
            ))
    return (plot2,)


@app.cell
def _(mo, pl, plot2, top2Bottom2):
    mo.hstack(
        [
            plot2,
            top2Bottom2
            if pl.DataFrame(plot2.value).is_empty()
            else pl.DataFrame(plot2.value),
        ]
    )
    return


@app.cell
def _(lineItemOrders_2, pl):
    top2ShippingCarriers = (
        lineItemOrders_2.select(pl.col("SHIPMENT_CARRIER").value_counts())
        .unnest("SHIPMENT_CARRIER")
        .sort("count", descending=True)
        .head(15)
        .join(
            lineItemOrders_2.select(
                pl.col("SHIPMENT_CARRIER"),
                (pl.col("SHIPMENT_DELIVERED_AT") - pl.col("FULFILLED_DT"))
                .alias("duration")
                .dt.total_days(),
            )
            .group_by("SHIPMENT_CARRIER")
            .agg(pl.mean("duration").alias("avg days"))
            .sort("avg days")
            .head(15),
            on="SHIPMENT_CARRIER",
            how="inner",
        )
        .join(
            lineItemOrders_2.group_by("SHIPMENT_CARRIER")
            .agg(pl.mean("TOTAL_SHIPPING").alias("avg shipping cost"))
            .sort("avg shipping cost", descending=False)
            .head(15),
            on="SHIPMENT_CARRIER",
            how="inner",
        )
        .sort(by=["count", "avg days", "avg shipping cost"], descending=True)
        .limit(5)
    )
    return (top2ShippingCarriers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The top 2 Shipping Carriers are USPS and UPS Mail Innovations:

        - They have a high count of orders
        - Their avg shipment time is low
        - Shipment cost is also low
        """
    ).callout('success')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Print Providers control the print quality, stock of items, and production time (the time from ordered to fulfilled). We want to provide a discount to the two best Print Providers and end our contracts with the worst two. Which do you choose and why?""").callout('warn')
    return


@app.cell
def _(lineItemOrders_2, pl):
    top2Bottom2 = lineItemOrders_2.group_by("PRINT_PROVIDER_ID").agg(
        pl.mean("duration").alias("Avg duration"),
        pl.col("REPRINT_FLAG")
        .map_elements(lambda flags: flags.sum() / len(flags) * 100)
        .alias("reprint %"),
        pl.col("ORDER_ID").n_unique().alias("count of orders"),
    ).sort(
        "Avg duration",
        "reprint %",
        "count of orders",
        descending=[False, False, True],
    ).filter(pl.col("Avg duration").is_not_null()).filter(
        pl.col("count of orders") > 1000
    ).head(2).vstack(
        lineItemOrders_2.group_by("PRINT_PROVIDER_ID")
        .agg(
            pl.mean("duration").alias("Avg duration"),
            pl.col("REPRINT_FLAG")
            .map_elements(lambda flags: flags.sum() / len(flags) * 100)
            .alias("reprint %"),
            pl.col("ORDER_ID").n_unique().alias("count of orders"),
        )
        .sort(
            "Avg duration",
            "reprint %",
            "count of orders",
            descending=[False, False, True],
        )
        .filter(pl.col("Avg duration").is_not_null())
        .tail(2),
    ).with_columns(
        pl.col('Avg duration').dt.total_days().alias('Avg duration in days') , pl.col('PRINT_PROVIDER_ID').cast(pl.String).cast(pl.Categorical),
        pl.col('count of orders').cast(pl.Int32)
    ).drop('Avg duration')
    return (top2Bottom2,)


@app.cell
def _(mo, px, top2Bottom2):
    mo.hstack(
        [
            px.scatter_3d(
                top2Bottom2,
                x="reprint %",
                y="count of orders",
                z="Avg duration in days",
                color="PRINT_PROVIDER_ID",
                hover_name="PRINT_PROVIDER_ID",
            ).update_layout(
                title="Top 2 and Bottom 2 Print Providers",
                xaxis_title="Reprint %",
                yaxis_title="Count of Prints",
            ),
            top2Bottom2,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## We provide discounts to 💸: 

        - Print providers 29 and 27 as they take huge number of orders for print with below 50 % reprint ratio all while delivering in 2 days

        ## We end our contracts with 🚨: 
        - Print providers 92 and 103 as they have the least number of orders out of all print provider plus they take very long to deliver (47 and 202 days respectively)
        """
    ).callout('success')
    return


if __name__ == "__main__":
    app.run()
