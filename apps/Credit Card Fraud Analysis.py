import marimo

__generated_with = "0.11.21"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    with mo.status.spinner(title="Loading Libraries..⏳", remove_on_exit=True):
        import polars as pl
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import (
            RandomForestClassifier,
            GradientBoostingClassifier,
            VotingClassifier,
        )

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_curve,
            f1_score,
        )
        from imblearn.over_sampling import SMOTE
        import matplotlib.pyplot as plt
        from scipy.optimize import minimize_scalar
    return (
        GradientBoostingClassifier,
        GridSearchCV,
        LabelEncoder,
        RandomForestClassifier,
        SMOTE,
        StandardScaler,
        VotingClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        go,
        minimize_scalar,
        np,
        pd,
        pl,
        plt,
        precision_recall_curve,
        px,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # CREDIT CARD APPLICATIONS 💳
        <p>Commercial banks receive <em>a lot</em> of applications for credit cards. Many of them get rejected for many reasons, like high loan balances, low income levels, or too many inquiries on an individual's credit report, for example. Manually analyzing these applications is mundane, error-prone, and time-consuming (and time is money!). Luckily, this task can be automated with the power of machine learning and pretty much every commercial bank does so nowadays. In this notebook, we will build an automatic credit card approval predictor using machine learning techniques, just like the real banks do.</p>

        This project aims to detect fraudulent credit card transactions using machine learning. 
        We generate a synthetic dataset with transaction details, perform exploratory data analysis (EDA), 
        and train an **ensemble model combining Random Forest & Gradient Boosting** with **hyperparameter tuning** 
        to classify transactions as either legitimate or fraudulent. 
        We also experiment with **dynamic probability thresholds, cost-sensitive learning, and time-based features** 
        to balance recall across both classes.
        <p><img src="https://www.adcb.com/en/Images/navigating-your-cc-applications-1250x560_tcm41-532978.jpg" alt="Credit card being held in hand"></p>
        """
    )
    return


@app.cell
def _(np, pd, pl):
    def generate_data(n=5000, fraud_ratio=0.2):
        np.random.seed(42)

        transaction_ids = np.arange(1, n + 1)
        amounts = np.round(np.random.exponential(scale=50, size=n), 2)

        # Increase fraud cases to 20%
        fraud_labels = np.random.choice(
            [0, 1], size=n, p=[1 - fraud_ratio, fraud_ratio]
        )

        transaction_types = np.random.choice(
            ["Online", "POS", "ATM"], size=n, p=[0.5, 0.4, 0.1]
        )
        locations = np.random.choice(
            ["USA", "UK", "India", "UAE", "Germany"], size=n
        )
        timestamps = pd.date_range(start="2024-01-01", periods=n, freq="T").astype(
            str
        )
        # Convert timestamps to datetime before extracting the hour
        timestamps = pd.to_datetime(timestamps)
        hours_of_day = timestamps.hour

        return pl.DataFrame(
            {
                "TransactionID": transaction_ids,
                "Amount": amounts,
                "Fraud": fraud_labels,
                "TransactionType": transaction_types,
                "Location": locations,
                "Timestamp": timestamps.astype(str),
                "HourOfDay": hours_of_day,
            }
        )
    return (generate_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## **Step 1: Generate Synthetic Data**
        We create a synthetic dataset of 5000 credit card transactions where:

        - `fraud_ratio=0.2` ensures 20% fraud
        - Each transaction gets:
        - a unique `TransactionID`
        - an `Amount` (from an exponential distribution)
        - a `TransactionType` (Online/POS/ATM)
        - a `Location` (USA, UK, India, UAE, Germany)
        - a `Timestamp` (one-minute increments)
        - `HourOfDay` (extracted from timestamp for time-based fraud insights)
        """
    )
    return


@app.cell
def _(generate_data, mo):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(
                        "**Creating a Numpy UDF to generate this Synthetic data 👉🏼**"
                    ),
                    mo.md("""
    ```python
    def generate_data(n=5000, fraud_ratio=0.2):
        np.random.seed(42)

        transaction_ids = np.arange(1, n + 1)
        amounts = np.round(np.random.exponential(scale=50, size=n), 2)

        # Increase fraud cases to 20%
        fraud_labels = np.random.choice([0, 1], size=n, p=[1 - fraud_ratio, fraud_ratio])

        transaction_types = np.random.choice(["Online", "POS", "ATM"], size=n, p=[0.5, 0.4, 0.1])
        locations = np.random.choice(["USA", "UK", "India", "UAE", "Germany"], size=n)
        timestamps = pd.date_range(start="2024-01-01", periods=n, freq="T").astype(str)
        # Convert timestamps to datetime before extracting the hour
        timestamps = pd.to_datetime(timestamps)
        hours_of_day = timestamps.hour

        return pl.DataFrame({
            "TransactionID": transaction_ids,
            "Amount": amounts,
            "Fraud": fraud_labels,
            "TransactionType": transaction_types,
            "Location": locations,
            "Timestamp": timestamps.astype(str),
            "HourOfDay": hours_of_day
        })
    ```




    """),
                ]
            ),
            mo.vstack(
                [
                    mo.ui.button(
                        label='<div data-tooltip="Feel free to take a look and play with the data below!">This is the generated data</div>'
                    ).center(),
                    mo.ui.dataframe(generate_data()),
                ]
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## **Step 2: Data Preprocessing & Feature Engineering**
        - Encode categorical columns with `LabelEncoder`.
        - Create engineered features:
        - `Amount_per_Type` = Amount / (TransactionTypeCode+1)
        - `Amount_per_Location` = Amount / (LocationCode+1)
        - Drop original categorical columns.
        """
    )
    return


@app.cell
def _(LabelEncoder, generate_data, mo, pl):
    df = generate_data()


    le_transaction = LabelEncoder()
    df = df.with_columns(
        pl.Series(
            "TransactionTypeCode",
            le_transaction.fit_transform(df["TransactionType"].to_list()),
        ).cast(pl.Int32),
    )

    le_location = LabelEncoder()
    df = df.with_columns(
        pl.Series(
            "LocationCode", le_location.fit_transform(df["Location"].to_list())
        ).cast(pl.Int32),
    )

    # Drop original categorical columns
    df = df.drop(["TransactionType", "Location"])

    df = df.with_columns(
        (df["Amount"] / (df["TransactionTypeCode"] + 1)).alias("Amount_per_Type"),
        (df["Amount"] / (df["LocationCode"] + 1)).alias("Amount_per_Location"),
    )

    mo.show_code()
    return df, le_location, le_transaction


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## **Step 3: Exploratory Data Analysis (EDA)**
        We use Plotly to visualize:

        - Fraud distribution (Pie chart)
        - Transaction amounts (Histogram)
        """
    )
    return


@app.cell
def _(df, mo, pl, px):
    fraud_counts = (
        df.group_by("Fraud")
        .agg(pl.col("TransactionID").count().alias("count"))
        .with_columns(
            pl.when(pl.col("Fraud") == 1)
            .then(True)
            .otherwise(False)
            .alias("Fraud")
        )
    )
    px.pie(
        fraud_counts.to_pandas(),
        names="Fraud",
        values="count",
        title="Fraud vs. Legitimate Transactions",
        template="simple_white",
    )

    mo.show_code()
    return (fraud_counts,)


@app.cell
def _(fraud_counts, px):
    px.pie(
        fraud_counts.to_pandas(),
        names="Fraud",
        values="count",
        title="Fraud vs. Legitimate Transactions",
        template="none",
    )
    return


@app.cell
def _(df, mo, pl, px):
    px.histogram(
        df.with_columns(
            pl.when(pl.col("Fraud") == 1)
            .then(True)
            .otherwise(False)
            .alias("Fraud")
        ).to_pandas(),
        x="Amount",
        color="Fraud",
        nbins=50,
        title="Transaction Amount Distribution",
        barmode="overlay",
        template="none",
    )

    mo.show_code()
    return


@app.cell
def _(df, pl, px):
    px.histogram(
        df.with_columns(
            pl.when(pl.col("Fraud") == 1)
            .then(True)
            .otherwise(False)
            .alias("Fraud")
        ).to_pandas(),
        x="Amount",
        color="Fraud",
        nbins=50,
        title="Transaction Amount Distribution",
        barmode="overlay",
        template="none",
    )
    return


@app.cell
def _(df, mo, pl, px):
    fraud_by_type = (
        df.group_by(["TransactionTypeCode", "Fraud"])
        .agg(pl.col("TransactionID").count().alias("count"))
        .pivot(values="count", index="TransactionTypeCode", columns="Fraud")
        .fill_null(0)
    )
    fraud_by_type = fraud_by_type.with_columns(
        (pl.col("1") / (pl.col("0") + pl.col("1")) * 100).alias("FraudRate")
    )
    fig3 = px.bar(
        fraud_by_type.to_pandas(),
        x="TransactionTypeCode",
        y="FraudRate",
        title="Fraud Rate by Transaction Type",
        template="none",
    )

    mo.show_code()
    return fig3, fraud_by_type


@app.cell
def _(fraud_by_type, px):
    px.bar(
        fraud_by_type.to_pandas(),
        x="TransactionTypeCode",
        y="FraudRate",
        title="Fraud Rate by Transaction Type",
        template="none",
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## **Step 4: Model Training with Cost-Sensitive Learning & Threshold Tuning**
        1. We use **SMOTE** to oversample fraud cases.
        2. Train an **Ensemble** (Random Forest + Gradient Boosting) with:
           - **Increased class_weight** for fraud (0:1, 1:4)
           - `max_depth` and `min_samples_split` tuned
        3. Optimize threshold dynamically to maximize macro F1-score
        4. Lower threshold further to increase fraud recall
        """
    )
    return


@app.cell
def _(df, mo):
    X = df.select(
        [
            "Amount",
            "TransactionTypeCode",
            "LocationCode",
            "Amount_per_Type",
            "Amount_per_Location",
            "HourOfDay",
        ]
    ).to_numpy()

    y = df["Fraud"].to_numpy()

    mo.show_code()
    return X, y


@app.cell
def _(SMOTE, X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, smote, y_resampled


@app.cell
def _(StandardScaler, X_resampled, mo, train_test_split, y_resampled):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mo.show_code()
    return X_test, X_train, scaler, y_test, y_train


@app.cell
def _(
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    X_train,
    mo,
    y_train,
):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight={0: 1, 1: 4},  # Bumped up fraud weight
        random_state=42,
    )


    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    )

    # Create ensemble model
    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")
    ensemble.fit(X_train, y_train)


    mo.show_code()
    return ensemble, gb, rf


@app.cell(disabled=True)
def _(mo):
    thresholds = {"0.35": 0.35, "0.40": 0.40, "0.45": 0.45, "0.50": 0.50}
    dropdown = mo.ui.dropdown(
        options=thresholds, label="**Choose Threshold**", value="0.35"
    )
    return dropdown, thresholds


@app.cell(disabled=True)
def _(dropdown):
    dropdown
    return


@app.cell
def _(X_test, ensemble, f1_score, minimize_scalar, mo, y_test):
    # Get prediction probabilities
    y_probs = ensemble.predict_proba(X_test)[:, 1]


    # Define the function to optimize threshold dynamically
    def optimize_threshold(threshold):
        y_pred = (y_probs > threshold).astype(int)
        return -f1_score(
            y_test, y_pred, average="macro"
        )  # Optimize for balanced F1-score


    # Optimize threshold dynamically
    best_threshold = minimize_scalar(
        optimize_threshold, bounds=(0.2, 0.6), method="bounded"
    ).x

    # Lower the threshold slightly to improve fraud recall
    if best_threshold > 0.3:
        best_threshold -= 0.1
    print(f"Adjusted Threshold: {best_threshold:.3f}")


    mo.show_code()
    return best_threshold, optimize_threshold, y_probs


@app.cell
def _(mo):
    mo.md(
        r"""
        ### **Step 5: Model Evaluation**
        We measure model accuracy, generate a classification report, and visualize the confusion matrix.
        """
    )
    return


@app.cell
def _(best_threshold, classification_report, y_probs, y_test):
    # Evaluate model at best threshold
    y_pred_optimal = (y_probs > best_threshold).astype(int)
    print(classification_report(y_test, y_pred_optimal))
    return (y_pred_optimal,)


@app.cell
def _(conf_matrix_df, mo, report_df):
    mo.hstack(
        [
            mo.vstack([mo.md("### Classification Report").center(), report_df]),
            mo.vstack([mo.md("### Confusion Matrix").center(), conf_matrix_df]),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Classification Report :""")
    return


@app.cell
def _(classification_report, pd, y_pred_optimal, y_test):
    report_dict = classification_report(y_test, y_pred_optimal, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df, report_dict


@app.cell
def _(confusion_matrix, pd, y_pred_optimal, y_test):
    conf_matrix = confusion_matrix(y_test, y_pred_optimal)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Legit", "Actual Fraud"],
        columns=["Predicted Legit", "Predicted Fraud"],
    )
    return conf_matrix, conf_matrix_df


@app.cell
def _(conf_matrix, mo, px):
    # Plot confusion matrix

    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legit", "Fraud"],
        y=["Legit", "Fraud"],
        title="Confusion Matrix",
    )

    mo.show_code()
    return (fig,)


@app.cell
def _(conf_matrix, px):
    px.imshow(
        conf_matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Legit", "Fraud"],
        y=["Legit", "Fraud"],
        title="Confusion Matrix",
        template="none",
    )
    return


@app.cell
def _(mo, plt, precision_recall_curve, y_probs, y_test):
    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label="Random Forest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    mo.show_code()
    return precision, recall


@app.cell
def _(plt, precision, precision_recall_curve, recall, y_probs, y_test):
    _precision, _recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label="Ensemble Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    return


if __name__ == "__main__":
    app.run()
