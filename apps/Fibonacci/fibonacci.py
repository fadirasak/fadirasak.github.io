import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        rf"""
        # Fibonacci Calculator
        """
    ).center()
    return


@app.cell
def _(mo):
    mo.image(r"""/Users/fadirasak/Desktop/Data science/projects/Fibonacci/Screen-Shot-2021-05-26-at-6.04.09-PM-1024x661.png""").center()
    return


@app.cell
def _(mo, n):
    mo.md(fr''' Use the slider below to calculate the first {n.value} numbers in the Fibonacci sequence. ''')
    return


@app.cell
def _(mo):
    # Create an interactive slider
    n = mo.ui.slider(1, 100, value=50, label="Number of Fibonacci numbers")
    n
    return (n,)


@app.cell
def _(fibonacci, mo, n):
    fib = fibonacci(n.value)
    mo.md(", ".join([str(f) for f in fib]))
    return (fib,)


@app.cell
def _():
    # Generate Fibonacci sequence
    def fibonacci(n):
        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i - 1] + sequence[i - 2])
        return sequence
    return (fibonacci,)


@app.cell
def _():
    import numpy as np
    import marimo as mo
    return mo, np


if __name__ == "__main__":
    app.run()
