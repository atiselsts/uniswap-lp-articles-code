# Code for the Uniswap LP Articles

This repository contains Python code for the artice series ["Liquidity Provider Strategies for Uniswap v3"](https://atise.medium.com/liquidity-provider-strategies-for-uniswap-v3-table-of-contents-64725c6c0b10).

The code is used to produce the graphs in the published articles.

For visualization, it uses Matplotlib in combination with the [ING theme](https://pypi.org/project/ing-theme-matplotlib/).

# Contents

* v2_math.py - math for full-range positions
* v3_math.py - math for concentrated liquidity positions
* plot_article_1.py - plots for the article "An Introduction to Uniswap LP Strategies and Hedging"
* plot_article_2.py - plots for the article "Dynamic Hedging"
* plot_article_3.py - plots for the article "Loss Versus Rebalancing (LVR)"
* plot_article_4.py - plots for the article "Power Perpetuals"
* plot_article_6.py - plots for the article "Liquidity Relocation (Rebalancing)"
* fit_power_perpetual_coefficients.py - code from the power perpetual article exported to a standalone module

# Simulations

The current implementation of price path simulations use significant RAM
and require several minutes to complete. They could be improved by using the JIT
decorator, or by rewriting the code to a faster programming language.