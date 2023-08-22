#!/usr/bin/env python

#
# This plots the figures for the article on continuous hedging.
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
import v2_math
import v3_math
from math import sqrt


# Constants for the LP positions

INITIAL_PRICE = 100

# select the value such that at 50:50 HODL we have 1.0 of the volatile asset X
INITIAL_VALUE = 2 * INITIAL_PRICE

INITIAL_X = INITIAL_VALUE / INITIAL_PRICE / 2
INITIAL_Y = INITIAL_VALUE / 2


# Constants for price simulations

# similar to the 1-day volatility for ETH-USD
SIGMA = 0.05
# assume 12 second blocks as in the mainnet
BLOCKS_PER_DAY = 86400 // 12

NUM_DAYS = 365

# assume 0.3% swap fee
SWAP_FEE = 0.3 / 100

NUM_SIMULATIONS = 100


# Constants for plotting

pl.rcParams["savefig.dpi"] = 200

############################################################

#
# Use geometrical Brownian motion to simulate price evolution.
#
def get_price_path(sigma_per_day):
    np.random.seed(123) # make it repeatable
    mu = 0.0   # assume delta neutral behavior
    T = NUM_DAYS
    n = T * BLOCKS_PER_DAY
    # calc each time step
    dt = T/n
    # simulation using numpy arrays
    St = np.exp(
        (mu - sigma_per_day ** 2 / 2) * dt
        + sigma_per_day * np.random.normal(0, np.sqrt(dt), size=(NUM_SIMULATIONS, n-1)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(NUM_SIMULATIONS), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
    St = INITIAL_PRICE * St.cumprod(axis=0)
    return St

############################################################

def get_hedging_costs_v2(step):
    step += 1.0

    initial_x = INITIAL_X
    initial_y = INITIAL_Y
    initial_capital = 5 * INITIAL_Y  # use 4 parts of assets for lending, 1 straight to the pool
    x_borrowed = initial_x

    L = v2_math.get_liquidity(initial_x, initial_y)
    V0 = v2_math.position_value_from_liquidity(L, INITIAL_PRICE)

    hedging_costs = []
    num_tx = 0
    all_prices = get_price_path(SIGMA)

    final_prices = all_prices[-1,:]
    returns = final_prices / INITIAL_PRICE
    year_sigma = SIGMA * sqrt(NUM_DAYS)
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]
        p_low = prices[0] / step
        p_high = prices[0] * step
        total_fees = 0
        tx = 0
        for price in prices:
            if not (p_low <= price <= p_high):
                p_low = price / step
                p_high = price * step

                x_in_pos = v2_math.calculate_x(L, price)
                delta_x = x_borrowed - x_in_pos
                delta_y = delta_x * price

                x_borrowed = x_in_pos # repay / add some ETH
                swap_fee = abs(delta_y) * SWAP_FEE
                total_fees += swap_fee # assume zero transaction fees
                num_tx += 1
        hedging_costs.append(total_fees)

    mean_hedging_costs = np.mean(hedging_costs)
    mean_hedging_costs /= initial_capital
    print(f"step={step} mean costs={100 * mean_hedging_costs:.2f}%, per day ={100 * mean_hedging_costs / NUM_DAYS:.2f}%")
    print(f" average number of transactions: {num_tx / NUM_SIMULATIONS:.1f}")
    return mean_hedging_costs, num_tx / NUM_SIMULATIONS

############################################################

def get_hedging_costs_v3(step):
    step += 1.0

    # set price range to +50% above the current price, and symmetrical range below
    price_b = INITIAL_PRICE * 1.5
    price_a = INITIAL_PRICE / 1.5

    sa = sqrt(price_a)
    sb = sqrt(price_b)

    initial_x = INITIAL_X
    initial_y = INITIAL_Y
    initial_capital = 5 * INITIAL_Y  # use 4 parts of assets for lending, 1 straight to the pool
    x_borrowed = initial_x

    L = v3_math.get_liquidity(initial_x, initial_y, sqrt(INITIAL_PRICE), sa, sb)
    V0 = v3_math.position_value_from_liquidity(L, INITIAL_PRICE, price_a, price_b)

    hedging_costs = []
    num_tx = 0
    all_prices = get_price_path(SIGMA)

    final_prices = all_prices[-1,:]
    returns = final_prices / INITIAL_PRICE
    year_sigma = SIGMA * sqrt(NUM_DAYS)
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]
        
        p_low = prices[0] / step
        p_high = prices[0] * step
        total_fees = 0
        tx = 0
        for price in prices:
            if not (p_low <= price <= p_high):
                p_low = price / step
                p_high = price * step

                x_in_pos = v3_math.calculate_x(L, sqrt(price), sa, sb)
                delta_x = x_borrowed - x_in_pos
                delta_y = delta_x * price

                x_borrowed = x_in_pos # repay / add some ETH
                swap_fee = abs(delta_y) * SWAP_FEE
                total_fees += swap_fee # assume zero transaction fees
                num_tx += 1
        hedging_costs.append(total_fees)

    mean_hedging_costs = np.mean(hedging_costs)
    mean_hedging_costs /= initial_capital
    print(f"step={step} mean costs={100 * mean_hedging_costs:.2f}%, per day ={100 * mean_hedging_costs / NUM_DAYS:.2f}%")
    print(f" average number of transactions: {num_tx / NUM_SIMULATIONS:.1f}")
    return mean_hedging_costs, num_tx / NUM_SIMULATIONS

############################################################

def plot_hedging_costs_v2():
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))

    costs_percent = []
    numtx = []
    steps = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    for step in steps:
        c, ntx = get_hedging_costs_v2(step)
        costs_percent.append(c * 100)
        numtx.append(ntx)

    x = [u * 100 for u in steps]
    pl.plot(x, costs_percent, linewidth=2, marker="D")

    pl.xlabel("Step size, %")
    pl.ylabel("Yearly hedging costs, %")

    pl.savefig("article_2_hedging_costs_v2.png", bbox_inches='tight')
    pl.close()


def plot_hedging_costs_v3():
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))

    costs_percent = []
    numtx = []

    steps = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    for step in steps:
        c, ntx = get_hedging_costs_v3(step)
        costs_percent.append(c * 100)
        numtx.append(ntx)

    x = [u * 100 for u in steps]
    pl.plot(x, costs_percent, linewidth=2, marker="D")

    pl.xlabel("Step size, %")
    pl.ylabel("Yearly hedging costs, %")

    pl.savefig("article_2_hedging_costs_v3.png", bbox_inches='tight')
    pl.close()

############################################################

def is_liquidated(y_lent, x_borrowed, price):
    MAX_LTV = 1.0  # note: unrealistically high!
    current_ltv = x_borrowed * price / y_lent
    #print(price, "ltv=", current_ltv)
    return current_ltv >= MAX_LTV


#
# The scenario to simulate:
#  - ETH price is $100
#  - have 500 USDC initial capital
#  - buy ETH for 100 * (1 - borrow_ratio) USDC
#  - put 400 USDC in lending protocol
#  - borrow  1 * borrow_ratio ETH, worth 100 * borrow_ratio USDC
#  - put 100 USDC and 1 ETH in the pool
#
def rebalanced_value(L, p_min, p_max, step, borrow_ratio):
    print(p_min, p_max)
    prices = [INITIAL_PRICE]
    values = [5 * INITIAL_Y]

    step += 1.0
    print("step=", step)

    if borrow_ratio < 0:
        is_dynamic = True
        borrow_ratio += 1
        borrow_ratio0 = borrow_ratio
        adjustable_borrow_ratio = 1.0 - borrow_ratio
    else:
        is_dynamic = False

    # price increasing run
    price = INITIAL_PRICE
    if is_dynamic:
        y_lent = INITIAL_Y * 4
        x_borrowed = INITIAL_X
    else:
        y_lent = INITIAL_Y * 4 - INITIAL_X * (1 - borrow_ratio) * price
        x_borrowed = INITIAL_X * borrow_ratio

    while price < p_max and L > 0:
        price *= step

        if is_liquidated(y_lent, x_borrowed, price):
            print("liquidated at ", price)
            break

        # evaluation step: what is the value at the new price?
        v_lp = v2_math.position_value_from_liquidity(L, price)
        v_hedge = v2_math.position_value(-x_borrowed, y_lent, price)
        v = v_lp + v_hedge

        print(f"L={L:.0f} price={price:.0f} x_borrowed={x_borrowed} y_lent={y_lent} v_lp={v_lp:.0f} v={v:.0f}")
        prices.append(price)
        values.append(v)

        if is_dynamic:
            effective_price = min(price, p_max)
            remaining = (p_max - effective_price) / (p_max - INITIAL_PRICE)
            borrow_ratio = borrow_ratio0 + adjustable_borrow_ratio * remaining
            print(" ", borrow_ratio)

        x_in_pos = v2_math.calculate_x(L, price)
        delta_x = x_borrowed - x_in_pos * borrow_ratio
        print("  x_borrowed=", x_borrowed, "x_in_pos=", x_in_pos, "delta_x=", delta_x)

        delta_y = delta_x * price
        y_lent -= delta_y     # remove USDC collateral
        x_borrowed = x_in_pos * borrow_ratio # repay some ETH

    print("")

    # price decreasing run
    price = INITIAL_PRICE
    if is_dynamic:
        borrow_ratio = 1.0
    y_lent = INITIAL_Y * 4 - INITIAL_X * (1 - borrow_ratio) * price
    x_borrowed = INITIAL_X * borrow_ratio

    while price > p_min and L > 0:
        price /= step

        # liquidation is not possible

        # evaluation step: what is the value at the new price?
        v_lp = v2_math.position_value_from_liquidity(L, price)
        v_hedge = v2_math.position_value(-x_borrowed, y_lent, price)
        v = v_lp + v_hedge

        print(f"L={L:.0f} price={price:.0f} x_borrowed={x_borrowed} y_lent={y_lent} v_lp={v_lp:.0f} v={v:.0f}")
        prices = [price] + prices
        values = [v] + values

        if is_dynamic:
            effective_price = max(price, p_min)
            remaining = (effective_price - p_min) / (INITIAL_PRICE - p_min)
            borrow_ratio = 1.0 + borrow_ratio0 + adjustable_borrow_ratio * (1 - remaining)
            print(" ", borrow_ratio)

        x_in_pos = v2_math.calculate_x(L, price)
        delta_x = x_borrowed - x_in_pos * borrow_ratio

        delta_y = delta_x * price
        x_borrowed = x_in_pos * borrow_ratio # borrow more ETH
        y_lent -= delta_y     # add more USDC collateral


    return prices, values


############################################################

#
# This shows value of LP position (v2 style) hedged with rebalancing hedges
#
def plot_portfolio_value(L, mn, mx, step_sizes, borrow_ratio, filename):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))

    for step in step_sizes:
        x, y = rebalanced_value(L, mn, mx, step, borrow_ratio)
        pl.plot(x, y, linewidth=2, label=f"Step={step*100:.0f}%") #, color="black")

    if False:
        # optional: plot how the HODL looks like
        y = [4 * INITIAL_Y + INITIAL_X * price for price in x]
        pl.plot(x, y, linewidth=2, label=f"HODL 4:1")

    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Total portfolio value, $")

    pl.legend()

    pl.savefig("article_2_" + filename, bbox_inches='tight')
    pl.close()

    
#
# This shows value of LP position (v2 style) partially hedged with rebalancing hedges
#
def plot_partial_hedged_portfolio_value(L, mn, mx, step, borrow_ratios, filename):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))

    for borrow_ratio in borrow_ratios:
        x, y = rebalanced_value(L, mn, mx, step, borrow_ratio)
        if borrow_ratio < 0:
            label = f"Dynamic borrow ratio"
        else:
            label = f"Borrow ratio={borrow_ratio*100:.0f}%"
        pl.plot(x, y, linewidth=2, label=label)

    y = [4 * INITIAL_Y + INITIAL_X * price for price in x]
    pl.plot(x, y, linewidth=2, label=f"HODL 4:1", color="white")

    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Total portfolio value, $")

    pl.legend()

    pl.savefig("article_2_" + filename, bbox_inches='tight')
    pl.close()

############################################################

def plot_value_functions(L, mn, mx, filename):
    STEP = 0.01 * INITIAL_PRICE
    YLIM_MIN = 0
    YLIM_MAX = 1000

    x = np.arange(mn, mx, STEP)
    y_lp = [v2_math.position_value_from_liquidity(L, price) for price in x]
    y_hodl = [(INITIAL_VALUE / 2 + price) / 2 for price in x]
    y_asset = [price for price in x]

    x1 = np.arange(mn, INITIAL_PRICE, STEP)
    x2 = np.arange(INITIAL_PRICE, mx, STEP)

    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3))
    pl.plot(x, y_lp, linewidth=2, color="orange")
    pl.plot(x, y_hodl, linewidth=2, color="darkgreen")
    pl.plot(x, y_asset, linewidth=2, color="green")
    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Value, $")

    pl.text(350, 550, "$y=x$ [100% asset]", weight='bold')
    pl.text(450, 325, "$y=x/2 + const$ [50:50 HODL]", weight='bold')
    pl.text(500, 170, "$y=sqrt(x)$ [LP position]", weight='bold')

    pl.ylim(YLIM_MIN, YLIM_MAX)
    pl.xlim(0, mx + 0.1)

    pl.savefig("article_2_" + filename, bbox_inches='tight')
    pl.close()
    

############################################################

def main():
    mpl_style(True)

    L_v2 = v2_math.get_liquidity(INITIAL_X, INITIAL_Y)
    print(f"L_v2={L_v2:.2f}")

    value_v2 = v2_math.position_value_from_liquidity(L_v2, INITIAL_PRICE)
    print(f"initial_value_v2={value_v2:.2f}")

    # min price
    mn = INITIAL_PRICE / 10
    # max price
    mx = 10 * INITIAL_PRICE

    plot_value_functions(L_v2, mn, mx, "value_functions.png")

    plot_portfolio_value(L_v2, mn, mx, [0.1, 0.5, 1, 2], 1.0,
                         "rebalancing_value_lp_fullrange.png")

    # min price
    mn = INITIAL_PRICE / 4
    # max price
    mx = 4 * INITIAL_PRICE
    plot_portfolio_value(L_v2, mn, mx, [0.01, 0.05, 0.1], 1.0,
                         "rebalancing_value_lp_fullrange_fine.png")

    plot_partial_hedged_portfolio_value(L_v2, mn, mx, 0.1, [1.0, 0.5, -1],
                                        "rebalancing_value_lp_fullrange_parthedged.png")


    plot_hedging_costs_v2()
    plot_hedging_costs_v3()


if __name__ == '__main__':
    main()
    print("all done!")
