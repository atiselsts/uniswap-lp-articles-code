#!/usr/bin/env python

#
# This plots the figures for the introduction article on Uniswap LP hedging.
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
import v2_math
import v3_math

# Constants for the LP positions

INITIAL_PRICE = 100

# select the value such that at 50:50 HODL we have 1.0 of the volatile asset X
INITIAL_VALUE = 2 * INITIAL_PRICE

INITIAL_X = INITIAL_VALUE / INITIAL_PRICE / 2
INITIAL_Y = INITIAL_VALUE / 2

FEE_YIELD = 0.05 # assume, 5% constant fee yield, no matter what the volatility & price
FEE_VALUE = INITIAL_VALUE * FEE_YIELD


# Constants for plotting

STEP = 0.01 * INITIAL_PRICE

YLIM_MIN = 0
YLIM_MAX = 600

PROFIT_YLIM_MIN = -300
PROFIT_YLIM_MAX = 80


#
# Compute the delta of a full-range LP position.
#   Delta(P) = Value'(P),
# where the value functions of a position is:
#   Value(P_t) = Value_0 * alpha_t,
# where `alpha` is the price change:
#   alpha_t := P_t / P_0
#
def delta(price_t):
    alpha_t = price_t / INITIAL_PRICE
    return INITIAL_VALUE / (2 * (alpha_t ** 0.5))


#
# Compute the gamma of a full-range LP position.
#   Gamma(P) = Delta'(P) = Value''(P)
#
def gamma(price_t):
    alpha_t = price_t / INITIAL_PRICE
    return -INITIAL_VALUE / (4 * (alpha_t ** 1.5))


#
# This computes the value of v2 position where some part of the volatile asset has been borrowed.
#
def borrowed_v2_lp_value(L, price, part_borrowed):
    assert 0 <= part_borrowed <= 1.0

    x = v2_math.calculate_x(L, price)
    y = v2_math.calculate_y(L, price)

    initial_x = INITIAL_X
    initial_y = INITIAL_Y

    borrowed_x = initial_x * part_borrowed
    # assume 100% loan-to-value ratio (this is not realistic!)
    collateral_y = initial_y * part_borrowed

    actual_x = x - borrowed_x # this can be negative, it's fine - does not change the math
    actual_y = y + collateral_y

    return v2_math.position_value(actual_x, actual_y, price)


#
# This computes the value of v2 position where some part of the volatile asset has been borrowed.
#
def borrowed_v3_lp_value(L, price, price_a, price_b, part_borrowed):
    assert 0 <= part_borrowed <= 1.0

    sp = price ** 0.5
    sa = price_a ** 0.5
    sb = price_b ** 0.5
    x = v3_math.calculate_x(L, sp, sa, sb)
    y = v3_math.calculate_y(L, sp, sa, sb)

    initial_x = INITIAL_X
    initial_y = INITIAL_Y

    borrowed_x = initial_x * part_borrowed
    # assume 100% loan-to-value ratio (this is not realistic!)
    collateral_y = initial_y * part_borrowed

    actual_x = x - borrowed_x # this can be negative, it's fine - does not change the math
    actual_y = y + collateral_y

    return v3_math.position_value(actual_x, actual_y, price)


#
# This finds positive and negative edges in a data series, useful for plotting
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
# This shows a narrow-range LP position
#
def plot_lp_narrow(L, price_a, price_b, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [v3_math.position_value_from_liquidity(L, price, price_a, price_b) for price in x]
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

    pl.savefig("article_1_value_lp_narrow.png", bbox_inches='tight')
    pl.close()

#
# This shows a narrow-range LP position zoomed in
#
def plot_lp_narrow_zoomed(L, price_a, price_b):
    step = 0.001 * INITIAL_PRICE
    mn = INITIAL_PRICE * 0.9
    mx = INITIAL_PRICE * 1.1
    x = np.arange(mn, mx, step)
    y_lp = [v3_math.position_value_from_liquidity(L, price, price_a, price_b) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y_asset = [2 * price for price in x]

    x1 = np.arange(mn, INITIAL_PRICE, step)
    x2 = np.arange(INITIAL_PRICE, mx, step)

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="black")
    pl.plot(x, y_hodl, linewidth=2, color="red")
    pl.plot(x, y_asset, linewidth=2, color="red")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    ax.fill_between(x1, 0, y_lp[:len(x1)], color="orange")
    y_rest = y_lp[len(x1):]
    ax.fill_between(x2[-len(y_rest):], 0, y_rest, color="darkgreen")
    
    pl.text(103, 195, "LP position", weight='bold')
    pl.text(105, 204, "50:50 HODL", weight='bold')
    pl.text(98, 207, "100% asset", weight='bold')

    pl.ylim(185, 215)
    pl.xlim(mn - 0.1, mx + 0.1)

    pl.savefig("article_1_value_lp_narrow_zoomed.png", bbox_inches='tight')
    pl.close()


#
# This shows a narrow-range LP position relative to 50:50 HODL
#
def plot_lp_narrow_vs_hodl(L, price_a, price_b, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + v3_math.position_value_from_liquidity(L, price, price_a, price_b) for price in x]
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

    pl.savefig("article_1_profit_lp_narrow_vs_hodl.png", bbox_inches='tight')
    pl.close()

    
#
# This shows a narrow-range LP position relative to 50:50 HODL
#
def plot_lp_narrow_vs_flat(L, price_a, price_b, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + v3_math.position_value_from_liquidity(L, price, price_a, price_b) for price in x]
    y_flat = [INITIAL_VALUE for price in x]
    y = [u - v for u, v in zip(y_lp, y_flat)]

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

    pl.savefig("article_1_profit_lp_narrow_vs_flat.png", bbox_inches='tight')
    pl.close()


#
# This shows a full-range LP position (v2 style)
#
def plot_lp_fullrange(L, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [v2_math.position_value_from_liquidity(L, price) for price in x]
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

    pl.savefig("article_1_value_lp_fullrange.png", bbox_inches='tight')
    pl.close()


#
# This shows a full-range LP position (v2 style) vs 50:50 HODL
#
def plot_lp_fullrange_vs_hodl(L, mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + v2_math.position_value_from_liquidity(L, price) for price in x]
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

    pl.savefig("article_1_profit_lp_fullrange_vs_hodl.png", bbox_inches='tight')
    pl.close()


#
# This shows a narrow-range LP position with borrowed assets
#
def plot_lp_narrow_borrowed(L, price_a, price_b, mn, mx, part_borrowed):
    x = np.arange(mn, mx, STEP)
    y_lp = [borrowed_v3_lp_value(L, price, price_a, price_b, part_borrowed) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y_asset = [2 * price for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="black")
    pl.plot(x, y_hodl, linewidth=2, color="red")
    pl.plot(x, y_asset, linewidth=2, color="red")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    ax.fill_between(x, 0, y_lp, color="orange")
    
    pl.text(130, 80, "LP position", weight='bold')
    pl.text(230, 410, "50:50 HODL", weight='bold')
    pl.text(170, 500, "100% asset", weight='bold')

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig(f"article_1_value_lp_narrow_borrowed_{part_borrowed:.1f}.png", bbox_inches='tight')
    pl.close()


#
# This shows a narrow-range LP position with borrowed assets relative to 50:50 HODL
#
def plot_lp_narrow_borrowed_vs_hodl(L, price_a, price_b, mn, mx, part_borrowed):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + borrowed_v3_lp_value(L, price, price_a, price_b, part_borrowed) for price in x]
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

    pl.savefig(f"article_1_profit_lp_narrow_borrowed_{part_borrowed:.1f}_vs_hodl.png", bbox_inches='tight')
    pl.close()

#
# This shows a narrow-range LP position with borrowed assets relative to 50:50 HODL
#
def plot_lp_narrow_borrowed_vs_flat(L, price_a, price_b, mn, mx, part_borrowed):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + borrowed_v3_lp_value(L, price, price_a, price_b, part_borrowed) for price in x]
    y_flat = [INITIAL_VALUE for price in x]
    y = [u - v for u, v in zip(y_lp, y_flat)]

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

    pl.savefig(f"article_1_profit_lp_narrow_borrowed_{part_borrowed:.1f}_vs_flat.png", bbox_inches='tight')
    pl.close()


#
# This shows a full-range LP position (v2 style) using borrowed assets
#
def plot_lp_fullrange_borrowed(L, mn, mx, part_borrowed):
    x = np.arange(mn, mx, STEP)
    y_lp = [borrowed_v2_lp_value(L, price, part_borrowed) for price in x]
    y_hodl = [INITIAL_VALUE / 2 + price for price in x]
    y_asset = [2 * price for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="black")
    pl.plot(x, y_hodl, linewidth=2, color="red")
    pl.plot(x, y_asset, linewidth=2, color="red")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    if part_borrowed == 1.0:
        ax.fill_between(x, 0, y_lp, color="orange")
    else:
        x1 = np.arange(mn, INITIAL_PRICE, STEP)
        x2 = np.arange(INITIAL_PRICE, mx, STEP)
        ax.fill_between(x1, 0, y_lp[:len(x1)], color="orange")
        ax.fill_between(x2, 0, y_lp[len(x1):], color="darkgreen")
    
    pl.text(130, 80, "LP position", weight='bold')
    pl.text(230, 410, "50:50 HODL", weight='bold')
    pl.text(170, 500, "100% asset", weight='bold')

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig(f"article_1_value_lp_fullrange_borrowed_{part_borrowed:.1f}.png", bbox_inches='tight')
    pl.close()


#
# This shows a full-range LP position (v2 style) with borrowed asset vs 50:50 HODL
#
def plot_lp_fullrange_borrowed_vs_hodl(L, mn, mx, part_borrowed):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + borrowed_v2_lp_value(L, price, part_borrowed) for price in x]
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

    pl.savefig(f"article_1_profit_lp_fullrange_borrowed_{part_borrowed:.1f}_vs_hodl.png", bbox_inches='tight')
    pl.close()


#
# This shows a full-range LP position (v2 style) with borrowed asset vs flat
#
def plot_lp_fullrange_borrowed_vs_flat(L, mn, mx, part_borrowed):
    x = np.arange(mn, mx, STEP)
    y_lp = [FEE_VALUE + borrowed_v2_lp_value(L, price, part_borrowed) for price in x]
    y_flat = [INITIAL_VALUE for price in x]
    y = [u - v for u, v in zip(y_lp, y_flat)]

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

    pl.savefig(f"article_1_profit_lp_fullrange_borrowed_{part_borrowed:.1f}_vs_flat.png", bbox_inches='tight')
    pl.close()

#
# This shows 100% HODL of the volatile asset
#
def plot_asset(mn, mx):
    x = np.arange(mn, mx, STEP)
    y = [2 * price for price in x]

    x1 = np.arange(mn, INITIAL_PRICE, STEP)
    x2 = np.arange(INITIAL_PRICE, mx, STEP)

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y, linewidth=2, color="black")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    ax.fill_between(x1, 0, y[:len(x1)], color="orange")
    ax.fill_between(x2, 0, y[len(x1):], color="darkgreen")

    pl.text(130, 80, "100% HODL", weight='bold')

    pl.savefig("article_1_value_asset.png", bbox_inches='tight')
    pl.close()


#
# This shows 50:50 HODL
#
def plot_hodl(mn, mx):
    x = np.arange(mn, mx, STEP)
    y = [INITIAL_VALUE / 2 + price for price in x]

    x1 = np.arange(mn, INITIAL_PRICE, STEP)
    x2 = np.arange(INITIAL_PRICE, mx, STEP)

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y, linewidth=2, color="black")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    ax.fill_between(x1, 0, y[:len(x1)], color="orange")
    ax.fill_between(x2, 0, y[len(x1):], color="darkgreen")

    pl.text(130, 80, "50:50 HODL", weight='bold')

    pl.savefig("article_1_value_hodl.png", bbox_inches='tight')
    pl.close()


#
# This plots the delta of full-range LP position (v2 style)
#
def plot_delta(mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [delta(price) for price in x]
    y_hodl = [INITIAL_VALUE / 2 for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="white", label="$\Delta_{LP}$")
    pl.plot(x, y_hodl, linewidth=2, color="green", label="$\Delta_{HODL}$")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("$\Delta$")

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)
    pl.legend()

    pl.savefig("article_1_delta_fullrange.png", bbox_inches='tight')
    pl.close()


#
# This plots the gamma of full-range LP position (v2 style)
#
def plot_gamma(mn, mx):
    x = np.arange(mn, mx, STEP)
    y_lp = [gamma(price) for price in x]
    y_hodl = [0.0 for price in x]

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="white", label="$\Gamma_{LP}$")
    pl.plot(x, y_hodl, linewidth=2, color="green", label="$\Gamma_{HODL}$")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("$\Gamma_{LP}$")

    pl.ylim(PROFIT_YLIM_MIN, PROFIT_YLIM_MAX)
    pl.xlim(0, mx + 0.1)
    pl.legend()

    pl.savefig("article_1_gamma_fullrange.png", bbox_inches='tight')
    pl.close()


def main():
    mpl_style(True)

    # set price range to +5% above the current price, and symmetrical range below
    price_b = INITIAL_PRICE * 1.05
    price_a = INITIAL_PRICE / 1.05

    initial_x = INITIAL_X
    initial_y = INITIAL_Y

    print(f"initial_x={initial_x:.2f} initial_y={initial_y:.2f}")
    L_v2 = v2_math.get_liquidity(initial_x, initial_y)
    print(f"L_v2={L_v2:.2f}")

    # normalizes the liquidity across the price range
    L_v3 = v3_math.get_liquidity(initial_x, initial_y,
                                 INITIAL_PRICE ** 0.5,
                                 price_a ** 0.5, price_b ** 0.5)
    print(f"L_v3={L_v3:.2f}")

    value_v2 = v2_math.position_value_from_liquidity(L_v2, INITIAL_PRICE)
    print(f"initial_value_v2={value_v2:.2f}")

    value_v3 = v3_math.position_value_from_liquidity(L_v3, INITIAL_PRICE, price_a, price_b)
    print(f"initial_value_v3={value_v3:.2f}")

    # min price
    mn = 0.01 * INITIAL_PRICE # don't use zero as the price
    # max price
    mx = 3.0 * INITIAL_PRICE

    plot_asset(mn, mx)
    plot_hodl(mn, mx)

    plot_delta(mn, mx)
    plot_gamma(mn, mx)

    plot_lp_narrow_zoomed(L_v3, price_a, price_b)

    plot_lp_narrow(L_v3, price_a, price_b, mn, mx)
    plot_lp_fullrange(L_v2, mn, mx)

    plot_lp_narrow_borrowed(L_v3, price_a, price_b, mn, mx, 0.5)
    plot_lp_fullrange_borrowed(L_v2, mn, mx, 0.5)

    plot_lp_narrow_borrowed(L_v3, price_a, price_b, mn, mx, 1.0)
    plot_lp_fullrange_borrowed(L_v2, mn, mx, 1.0)

    plot_lp_narrow_vs_hodl(L_v3, price_a, price_b, mn, mx)
    plot_lp_narrow_vs_flat(L_v3, price_a, price_b, mn, mx)
    plot_lp_fullrange_vs_hodl(L_v2, mn, mx)

    plot_lp_narrow_borrowed_vs_hodl(L_v3, price_a, price_b, mn, mx, 0.5)
    plot_lp_narrow_borrowed_vs_flat(L_v3, price_a, price_b, mn, mx, 0.5)
    plot_lp_fullrange_borrowed_vs_hodl(L_v2, mn, mx, 0.5)
    plot_lp_fullrange_borrowed_vs_flat(L_v2, mn, mx, 0.5)

    plot_lp_narrow_borrowed_vs_hodl(L_v3, price_a, price_b, mn, mx, 1.0)
    plot_lp_narrow_borrowed_vs_flat(L_v3, price_a, price_b, mn, mx, 1.0)
    plot_lp_fullrange_borrowed_vs_hodl(L_v2, mn, mx, 1.0)
    plot_lp_fullrange_borrowed_vs_flat(L_v2, mn, mx, 1.0)



if __name__ == '__main__':
    main()
    print("all done!")
