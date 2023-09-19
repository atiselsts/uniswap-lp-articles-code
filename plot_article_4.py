#!/usr/bin/env python

#
# This plots the figures for the article on power perpetuals.
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
import v2_math
import v3_math


# Constants for the LP positions

INITIAL_PRICE = 100

# set a wider range to get more interesting results than when using traditional +-5%
# the v3 position is asymmetric, so x and y initial values are not going to be the same
PRICE_B = INITIAL_PRICE * 2
PRICE_A = INITIAL_PRICE / 1.5

INITIAL_VALUE = 2 * INITIAL_PRICE

INITIAL_X = INITIAL_VALUE / INITIAL_PRICE / 2
INITIAL_Y = INITIAL_VALUE / 2  # this is not correct for the v3 position, and is recomputed in the code

FEE_VALUE = 10.0


# Constants for power perps

MAX_COEFFICIENTS = 5


# Constants for plotting

STEP = 0.01 * INITIAL_PRICE

YLIM_MIN = 0
YLIM_MAX = 600

PROFIT_YLIM_MIN = -300
PROFIT_YLIM_MAX = 80

#
# This uses hedging formula from the article "Spanning with Power Perpetuals" by Joseph Clark,
# where the AMM is approximated using the Taylor expansion around r=0.
#
# This assumes zero funding rate! (as time is not given as a parameter to this function)
#
def power_perp_v2(price, order, include_first_order=True):
    r = price / INITIAL_PRICE - 1.0 # r is the price return
    perp_return = 0
    if order >= 1 and include_first_order:
        perp_return += -0.5 * r
    if order >= 2:
        perp_return += 0.125 * (r ** 2)
    if order >= 3:
        perp_return += -3 / 48 * (r ** 3)
    if order >= 4:
        perp_return += 15 / 384 * (r ** 4)
    perp_value = INITIAL_VALUE * (perp_return + 1.0)
    #    print("power perp value at price", order, include_first_order, price, perp_value)
    return perp_value

#
# This uses numerically found coefficient array for a narrow-range v3 positions.
#
def power_perp_v3(price, coefficients, order, include_first_order=True):
    # assume that when the price crosses the LP range boundaries, the owner sells the power perp
    price = min(price, PRICE_B)
    price = max(price, PRICE_A)

    r = price / INITIAL_PRICE - 1.0 # r is the price return
    perp_return = 0
    if order >= 1 and include_first_order:
        perp_return += coefficients[1] * r
    if order >= 2:
        perp_return += coefficients[2] * (r ** 2)
    if order >= 3:
        perp_return += coefficients[3] * (r ** 3)
    if order >= 4:
        perp_return += coefficients[4] * (r ** 4)

    perp_value = INITIAL_VALUE * (perp_return + 1.0)
    #    print("power perp value at price", order, include_first_order, price, perp_value)
    return perp_value

#
# This shows approximations with power perps of a full-range LP position (v2 style)
#
def plot_power_perps(L, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [v2_math.position_value_from_liquidity(L, price) for price in x]
    y_1 = [2 * INITIAL_VALUE - power_perp_v2(price, order=1) for price in x]
    y_2 = [2 * INITIAL_VALUE - power_perp_v2(price, order=2) for price in x]
    y_3 = [2 * INITIAL_VALUE - power_perp_v2(price, order=3) for price in x]
    y_4 = [2 * INITIAL_VALUE - power_perp_v2(price, order=4) for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="green", label="LP")
    pl.plot(x, y_1, linewidth=2, color="#666666", label="First order")
    pl.plot(x, y_2, linewidth=2, color="grey", label="Second order")
    pl.plot(x, y_3, linewidth=2, color="lightgrey", label="Third order")
    pl.plot(x, y_4, linewidth=2, color="white", label="Fourth order")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    pl.xlim(0, mx + 0.1)
    pl.legend()

    pl.savefig("article_4_value_power_perps.png", bbox_inches='tight')
    pl.close()


#
# This shows approximations with power perps of a full-range LP position (v2 style)
#
def plot_hedging_power_perps(L, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_1 = [v2_math.position_value_from_liquidity(L, price) + power_perp_v2(price, order=1) - 2 * INITIAL_VALUE for price in x]
    y_2 = [v2_math.position_value_from_liquidity(L, price) + power_perp_v2(price, order=2) - 2 * INITIAL_VALUE for price in x]
    y_3 = [v2_math.position_value_from_liquidity(L, price) + power_perp_v2(price, order=3) - 2 * INITIAL_VALUE for price in x]
    y_4 = [v2_math.position_value_from_liquidity(L, price) + power_perp_v2(price, order=4) - 2 * INITIAL_VALUE for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_1, linewidth=2, color="#666666", label="First order")
    pl.plot(x, y_2, linewidth=2, color="grey", label="Second order")
    pl.plot(x, y_3, linewidth=2, color="lightgrey", label="Third order")
    pl.plot(x, y_4, linewidth=2, color="white", label="Fourth order")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Hedge error, $")

    pl.xlim(0, mx + 0.1)
    pl.legend()

    pl.savefig("article_4_hedge_error_power_perps.png", bbox_inches='tight')
    pl.close()


def power_hedged_v2(L, price, order, include_first_order):
    return v2_math.position_value_from_liquidity(L, price) \
        + power_perp_v2(price, order, include_first_order) \
        - INITIAL_VALUE

#
# Instead of using the generic full-range hedging,
# a custom power-perp formula is derived and applied.
#
def power_hedged_v3(L, price, price_a, price_b, coefficients, order, include_first_order):
    return v3_math.position_value_from_liquidity(L, price, price_a, price_b) \
        + power_perp_v3(price, coefficients, order, include_first_order) \
        - INITIAL_VALUE


#
# This finds positive and negative edges in a data series
#
def find_profit_range(series):
    in_profit = [1 if v > 0 and not (u > 0) else 0 for u, v in zip(series, series[1:])]
    out_of_profit = [1 if u > 0 and not (v > 0) else 0 for u, v in zip(series, series[1:])]

    # From the docs: "In case of multiple occurrences of the maximum values,
    # the indices corresponding to the first occurrence are returned."
    start = np.argmax(in_profit)
    end = np.argmax(out_of_profit)

    return (start, end)

#
# This shows full-range LP position (v2 style) hedged with power perps
#
def plot_lp_fullrange_power_hedged(L, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [power_hedged_v2(L, price, 2, False) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y_asset = [2 * price for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="black")
    pl.plot(x, y_hodl, linewidth=2, color="red")
    pl.plot(x, y_asset, linewidth=2, color="red")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    x1 = np.arange(mn, INITIAL_PRICE, STEP)
    x2 = np.arange(INITIAL_PRICE, mx, STEP)

    ax.fill_between(x1, 0, y_lp[:len(x1)], color="orange")
    ax.fill_between(x2, 0, y_lp[len(x1):], color="darkgreen")
   
    pl.text(130, 80, "LP position", weight='bold')
    pl.text(230, 270, "50:50 HODL", weight='bold')
    pl.text(170, 500, "100% asset", weight='bold')

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig("article_4_value_lp_fullrange_power_hedged.png", bbox_inches='tight')
    pl.close()


#
# This shows full-range LP position (v2 style) hedged with power perps vs 50:50 HODL
#
def plot_lp_fullrange_power_hedged_vs_hodl(L, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + power_hedged_v2(L, price, 2, False) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y = [u - v for u, v in zip(y_lp, y_hodl)]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y, linewidth=2, color="black")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Profit, $")

    start, _ = find_profit_range(y)
    x1 = x[0:start]; y1 = y[0:start]
    x2 = x[start:]; y2 = y[start:]
    ax.fill_between(x1, 0, y1, color="red")
    ax.fill_between(x2, 0, y2, color="darkgreen")

    pl.ylim(PROFIT_YLIM_MIN, PROFIT_YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig("article_4_profit_lp_fullrange_power_hedged_vs_hodl.png", bbox_inches='tight')
    pl.close()


#
# This shows narrow-range LP position hedged with power perps
# (not finished!)
#
def plot_lp_narrow_power_hedged(L, price_a, price_b, coefficients, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [power_hedged_v3(L, price, price_a, price_b, coefficients, 2, False) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y_asset = [2 * price for price in x]

    x1 = np.arange(mn, INITIAL_PRICE, STEP)
    x2 = np.arange(INITIAL_PRICE, mx, STEP)

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="black")
    pl.plot(x, y_hodl, linewidth=2, color="red")
    pl.plot(x, y_asset, linewidth=2, color="red")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    ax.fill_between(x1, 0, y_lp[:len(x1)], color="orange")
    ax.fill_between(x2, 0, y_lp[len(x1):], color="darkgreen")
    
    pl.text(130, 80, "LP position", weight='bold')
    pl.text(230, 410, "50:50 HODL", weight='bold')
    pl.text(170, 500, "100% asset", weight='bold')

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig("article_4_value_lp_narrow_power_hedged.png", bbox_inches='tight')
    pl.close()


#
# This shows narrow-range LP position hedged with power perps, relative to 50:50 HODL
# (not finished!)
#
def plot_lp_narrow_power_hedged_vs_hodl(L, price_a, price_b, coefficients, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + power_hedged_v3(L, price, price_a, price_b, coefficients, 2, False) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y = [u - v for u, v in zip(y_lp, y_hodl)]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y, linewidth=2, color="black")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Profit, $")

    start, end = find_profit_range(y)
    x1 = x[0:start]; y1 = y[0:start]
    x2 = x[start:end]; y2 = y[start:end]
    x3 = x[end:]; y3 = y[end:]
    ax.fill_between(x1, 0, y1, color="red")
    ax.fill_between(x2, 0, y2, color="darkgreen")
    ax.fill_between(x3, 0, y3, color="red")
    
    pl.ylim(PROFIT_YLIM_MIN, PROFIT_YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig("article_4_profit_lp_narrow_power_hedged_vs_hodl.png", bbox_inches='tight')
    pl.close()


def fit_power_hedging_coefficients(liquidity, price_a, price_b):
    # the max value is bounded by the amount of tokens `y` at price `price_b`
    v_max = liquidity * (price_b ** 0.5 - price_a ** 0.5)

    # check if the formula is correct
    step = (price_b - price_a) / 1000
    data_f = []
    data_a = []
    x = np.arange(price_a, price_b + step, step)
    for price in x:
        v_formula = v3_math.position_value_from_max_value(v_max, price, price_a, price_b)
        v_amounts = v3_math.position_value_from_liquidity(liquidity, price, price_a, price_b)
        data_f.append(v_formula)
        data_a.append(v_amounts)
        error = (v_formula - v_amounts) / v_amounts
        #print(price, v_formula, v_amounts, error)
        assert abs(error) < 0.001
    print("value formula checked")

    # convert from f(x) to f(r) before doing the fit
    r = [price / INITIAL_PRICE - 1.0 for price in x]
    data_transformed = [u / INITIAL_VALUE - 1.0 for u in data_f]

    # do a polynomial fit, with a high-order polynomial
    polyfit_many = np.polyfit(r, data_transformed, 20)
    coefficients = [0] * MAX_COEFFICIENTS
    coefficients[0] = 0 # always zero for the 0-th order hedge
    coefficients[1] = -polyfit_many[-2] # 1-st order power perp
    coefficients[2] = -polyfit_many[-3] # 2-nd order power perp
    coefficients[3] = -polyfit_many[-4] # 3-rd order power perp
    coefficients[4] = -polyfit_many[-5] # 4-th order power perp

    #
    # Example n-th order reconstructions:
    #
    # reconstruction1 = [2 * INITIAL_VALUE - power_perp_v3(price, coefficients, 1) for price in x]
    # reconstruction2 = [2 * INITIAL_VALUE - power_perp_v3(price, coefficients, 2) for price in x]
    # reconstruction3 = [2 * INITIAL_VALUE - power_perp_v3(price, coefficients, 3) for price in x]

    return coefficients

    
def main():
    mpl_style(True)

    price_a = PRICE_A
    price_b = PRICE_B

    initial_x = INITIAL_X
    initial_y = INITIAL_Y

    print(f"initial_x={initial_x:.2f} initial_y={initial_y:.2f}")
    L_v2 = v2_math.get_liquidity(initial_x, initial_y)
    print(f"L_v2={L_v2:.2f}")

    value_v2 = v2_math.position_value_from_liquidity(L_v2, INITIAL_PRICE)
    print(f"initial_value_v2={value_v2:.2f}")

    # normalizes the liquidity across the price range
    L_v3 = v3_math.get_liquidity(initial_x, initial_y,
                                 INITIAL_PRICE ** 0.5,
                                 price_a ** 0.5, price_b ** 0.5)
    print(f"L_v3={L_v3:.2f}")

    value_v3 = v3_math.position_value_from_liquidity(L_v3, INITIAL_PRICE, price_a, price_b)
    print(f"initial_value_v3={value_v3:.2f}")

    # scale the liquidity to get the right value
    L_v3 *= INITIAL_VALUE / value_v3
    value_v3 = v3_math.position_value_from_liquidity(L_v3, INITIAL_PRICE, price_a, price_b)
    print(f"now initial_value_v3={value_v3:.2f}")

    coefficients = fit_power_hedging_coefficients(L_v3, price_a, price_b)
    print("coefficients=", coefficients)
    print("")

    # min price
    mn = 0.01 * INITIAL_PRICE # don't use zero as the price
    # max price
    mx = 3.0 * INITIAL_PRICE

    plot_power_perps(L_v2, mn, mx)
    plot_hedging_power_perps(L_v2, mn, mx)

    plot_lp_narrow_power_hedged(L_v3, price_a, price_b, coefficients, mn, mx)
    plot_lp_fullrange_power_hedged(L_v2, mn, mx)

    plot_lp_narrow_power_hedged_vs_hodl(L_v3, price_a, price_b, coefficients, mn, mx)
    plot_lp_fullrange_power_hedged_vs_hodl(L_v2, mn, mx)



if __name__ == '__main__':
    main()
    print("all done!")
