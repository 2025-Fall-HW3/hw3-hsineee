"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        n_assets = len(assets)

        # 一開始先等權
        current_weights = pd.Series(1.0 / n_assets, index=assets)

        # 動能參數：長期約 12 個月、短期約 1 個月
        long_lb = 252   # 約一年交易日
        short_lb = 21   # 約一個月交易日

        dates = self.price.index

        for i, date in enumerate(dates):
            # 每個月第一個交易日，且有足夠歷史資料 => 重新算權重
            if (
                i > long_lb
                and i > 0
                and dates[i].month != dates[i - 1].month
            ):
                # 長期動能 (12 個月)
                price_now = self.price[assets].iloc[i - 1]
                price_long_past = self.price[assets].iloc[i - long_lb]
                long_mom = price_now / price_long_past - 1

                # 短期動能 (1 個月)
                price_short_past = self.price[assets].iloc[i - short_lb]
                short_mom = price_now / price_short_past - 1

                # 組合動能分數
                score = 0.7 * long_mom + 0.3 * short_mom
                score = score.replace([np.inf, -np.inf], np.nan)

                if not score.isna().all():
                    # NaN 當成很差的分數，避免被選到
                    score = score.fillna(-1e9)

                    # 挑出前 3 名 sector
                    top_assets = score.nlargest(3).index

                    new_weights = pd.Series(0.0, index=assets)
                    new_weights.loc[top_assets] = 1.0 / len(top_assets)

                    current_weights = new_weights

            self.portfolio_weights.loc[date, assets] = current_weights.values

        # SPY 權重固定為 0
        self.portfolio_weights[self.exclude] = 0.0

        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
