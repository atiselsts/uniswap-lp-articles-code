#!/usr/bin/env python

#
# This plots the figures for the article on LVR.
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
import v2_math
import v3_math
from math import sqrt


# Constants for the LP positions

INITIAL_PRICE = 1000

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

NUM_SIMULATIONS = 10000


# Constants for plotting
pl.rcParams["savefig.dpi"] = 200


############################################################

#
# Use geometrical Brownian motion to simulate price evolution.
#
def get_price_path(sigma_per_day, blocks_per_day=BLOCKS_PER_DAY, M=NUM_SIMULATIONS):
    np.random.seed(123) # make it repeatable
    mu = 0.0   # assume delta neutral behavior
    T = NUM_DAYS
    n = T * blocks_per_day
    # calc each time step
    dt = T/n
    # simulation using numpy arrays
    St = np.exp(
        (mu - sigma_per_day ** 2 / 2) * dt
        + sigma_per_day * np.random.normal(0, np.sqrt(dt), size=(M, n-1)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(M), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0). 
    St = INITIAL_PRICE * St.cumprod(axis=0)
    return St

############################################################

# returns swap amounts from the pool's perspective (changes in the pool's composition)
def get_swap_amounts(L, reserve_x, reserve_y, target_price, fee_tier):
    sqrt_tp = sqrt(target_price)
    x = L / sqrt_tp - reserve_x
    y = L * sqrt_tp - reserve_y
    if x > 0:
        # arber sells X, LP buys X
        x_with_fee = x / (1 - fee_tier)
        y_with_fee = y
        swap_fee = (x_with_fee - x) * target_price
    else:
        # arber buys X, LP sells X
        x_with_fee = x
        y_with_fee = y / (1 - fee_tier)
        swap_fee = y_with_fee - y

    return x, x_with_fee, y, y_with_fee, swap_fee

############################################################

# this does some quick checks that the math is correct
def test_amounts():
    L = v2_math.get_liquidity(INITIAL_X, INITIAL_Y)
    price = INITIAL_PRICE
    reserve_x = v2_math.calculate_x(L, price)
    reserve_y = v2_math.calculate_y(L, price)

    fee_tier = SWAP_FEE

    for c in [1.0001, 1.0002, 1.01, 1.1, 1.2]:
        x, x_with_fee, y, y_with_fee, swap_fee = get_swap_amounts(L, reserve_x, reserve_y, price / c, fee_tier)
        #print(price / c, "to swap:", x, y)

        verify_y = v2_math.sell_x(reserve_x, reserve_y, x_with_fee, fee_tier)
        error = (verify_y + y) / verify_y
        assert error < 1e-12
        #print("  error=", error * 100, "%")

        new_price = price / c
        arb_pnl = -(x * new_price + y) - swap_fee
        #print("arb_pnl=", arb_pnl)
        if c == 1.2:
            assert arb_pnl > 0
        elif c == 1.0001:
            assert arb_pnl < 0

    for c in [1.0001, 1.0002, 1.01, 1.1, 1.2]:
        x, x_with_fee, y, y_with_fee, swap_fee = get_swap_amounts(L, reserve_x, reserve_y, price * c, fee_tier)

        verify_x = v2_math.buy_x(reserve_x, reserve_y, y_with_fee, fee_tier)
        error = (verify_x + x) / verify_x
        assert error < 1e-12

        new_price = price * c
        arb_pnl = -(x * new_price + y) - swap_fee
        if c == 1.2:
            assert arb_pnl > 0
        elif c == 1.0001:
            assert arb_pnl < 0


############################################################

def compute_lvr(all_prices, swap_tx_cost):
    print(f"compute_lvr, swap_tx_cost={swap_tx_cost}")
    fee_multiplier = 1 / (1 - SWAP_FEE)

    all_lvr = []
    all_fees = []

    L0 = v2_math.get_liquidity(INITIAL_X, INITIAL_Y)

    if len(all_prices.shape) > 2:
        # take the first elements from the second dimension
        all_prices = all_prices[:,0,:]

    for sim in range(all_prices.shape[1]):
        prices = all_prices[:,sim]
        sqrt_prices = np.sqrt(prices)

        reserve_x = INITIAL_X
        reserve_y = INITIAL_Y

        # compute lvr and fees
        lvr = 0
        fees = 0
        num_tx = 0

        for cex_price, cex_sqrt_price in zip(prices, sqrt_prices):
            x = L0 / cex_sqrt_price - reserve_x
            y = L0 * cex_sqrt_price - reserve_y
            if x > 0:
                # arber sells X, LP buys X
                x_with_fee = x * fee_multiplier
                swap_fee = (x_with_fee - x) * cex_price
            else:
                # arber buys X, LP sells X
                y_with_fee = y * fee_multiplier
                swap_fee = y_with_fee - y

            # assume fixed gas fees
            arb_gain = -(x * cex_price + y) - swap_fee - swap_tx_cost
            if arb_gain > 0:
                lvr += -(x * cex_price + y) # account without swap fees and tx fees
                reserve_x += x
                reserve_y += y
                fees += swap_fee
                num_tx += 1

        # normalize by dividing with the initial value of the capital rather than the final value
        lvr /= INITIAL_VALUE
        fees /= INITIAL_VALUE
        all_lvr.append(lvr)
        all_fees.append(fees)

    return np.mean(all_lvr), np.mean(all_fees)


def plot_lvr_and_tx_cost():
    # assume pool with $1 million liquidity
    # if swap transaction costs:
    # * $100 -> 1 bps fee per tranasaction
    # * $10 -> 0.1 bps fee per tranasaction
    tx_cost_bps = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02])

    swap_tx_cost = INITIAL_VALUE * tx_cost_bps / (100 * 100)
    # multiply by 500 because the assumption is $1M liquidity instead of $2000 as in this model
    swap_tx_cost *= 500

    print(f"tx_cost_bps={tx_cost_bps} tx_cost_$={swap_tx_cost}")

    fig, ax = pl.subplots()
    fig.set_size_inches((4, 3))

    # reduce the number of simulations, since we iterate over each block
    num_simulations = 100

    all_prices = get_price_path(SIGMA, blocks_per_day=BLOCKS_PER_DAY, M=num_simulations)
    final_prices = all_prices[-1,:]
    returns = final_prices / INITIAL_PRICE
    year_sigma = SIGMA * sqrt(NUM_DAYS) # convert from daily to yearly volatility
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    lvr_and_fees = [compute_lvr(all_prices, cost) for cost in swap_tx_cost]

    pl.plot(swap_tx_cost, [100*u[0] for u in lvr_and_fees], label="Losses to LVR", marker="v")
    pl.plot(swap_tx_cost, [100*u[1] for u in lvr_and_fees], label="Gains from arb fees", marker="o", color="green")

    pl.xlabel("Swap tx cost, $")
    pl.ylabel("APR, %")
    pl.legend()
    pl.ylim(ymin=0)

    pl.savefig("article_3_lvr_and_tx_cost.png", bbox_inches='tight')
    pl.close()



def plot_lvr_and_block_time():
    tx_cost_bps = 0.01
    swap_tx_cost = INITIAL_VALUE * tx_cost_bps / (100 * 100)
    # multiply by 500 because the assumption is $1M liquidity instead of $2000 as in this model
    swap_tx_cost *= 500

    fig, ax = pl.subplots()
    fig.set_size_inches((4, 3))


    # reduce the number of simulations, sicne we iterate over each block
    num_simulations = 50

    num_blocks = 86400 // 12

    all_prices = get_price_path(SIGMA, blocks_per_day=num_blocks, M=num_simulations)
    print(all_prices.shape)
    final_prices = all_prices[-1,:]
    print(final_prices)
    returns = final_prices / INITIAL_PRICE
    year_sigma = SIGMA * sqrt(NUM_DAYS) # convert from daily to yearly volatility
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    block_div = [1, 3, 10, 30, 100]
    lvr_and_fees = []
    for div in block_div:
        if div > 1:
            all_prices = all_prices.reshape((num_blocks * NUM_DAYS) // div, div, num_simulations)

            final_prices = all_prices[-1,:][-1,:]
            returns = final_prices / INITIAL_PRICE
            year_sigma = SIGMA * sqrt(NUM_DAYS) # convert from daily to yearly volatility
            print(f"  div={div} sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

        lvr, fees1 = compute_lvr(all_prices, 0.0)
        _, fees2   = compute_lvr(all_prices, swap_tx_cost)
        lvr_and_fees.append((lvr, fees1, fees2))

    block_time_sec = [12 * d for d in block_div]
    pl.plot(block_time_sec, [100*u[0] for u in lvr_and_fees], label="Losses to LVR", marker="v")
    pl.plot(block_time_sec, [100*u[1] for u in lvr_and_fees], label="Gains from arb fees, tx cost=$0", marker="o", color="green")
    pl.plot(block_time_sec, [100*u[2] for u in lvr_and_fees], label="Gains from arb fees, tx cost=$0.1", marker="o", color="lightgreen")

    pl.xlabel("Block time, seconds")
    pl.xscale("log")
    pl.ylabel("APR, %")
    pl.legend()
    pl.ylim(ymin=0)

    pl.savefig("article_3_lvr_and_block_time.png", bbox_inches='tight')
    pl.close()


############################################################

def compute_expected_divloss(sigma):
    initial_x = INITIAL_X
    initial_y = INITIAL_Y

    year_sigma = sigma
    sigma /= sqrt(NUM_DAYS) # convert from yearly to daily volatility

    all_divloss = []
    all_final_values = []

    all_prices = get_price_path(sigma, blocks_per_day=1)

    final_prices = all_prices[-1,:]
    returns = final_prices / INITIAL_PRICE
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        L = v2_math.get_liquidity(initial_x, initial_y)
        V0 = v2_math.position_value_from_liquidity(L, INITIAL_PRICE)

        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y
        Vfinal = v2_math.position_value_from_liquidity(L, prices[-1])
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)
        all_final_values.append(Vfinal)

    return np.mean(all_divloss)


def plot_divloss_from_sigma():
    fig, ax = pl.subplots()
    fig.set_size_inches((8, 5))

    fee_apr = np.arange(0, 30, 5)
    sigmas = np.arange(0.1, 1.5, 0.1)
    day_sigmas = [sigma / sqrt(NUM_DAYS) for sigma in sigmas]
    divloss = [compute_expected_divloss(sigma) * 100 for sigma in sigmas]
    model_divloss = [-100 * (sigma ** 2) / 8 for sigma in sigmas]

    for fee in fee_apr:
        pl.plot(sigmas, [dl + fee for dl in divloss], marker="o", label=f"Fee APR={fee:.0f}%")

    pl.plot(sigmas, model_divloss, linestyle="--", marker="None", label="Divergence loss (analytical model)")

    pl.xlabel("Annualized $\sigma$")
    pl.ylabel("Profit and loss APR, %")
    pl.legend()

    pl.savefig("article_3_divloss_vs_fees.png", bbox_inches='tight')
    pl.close()


############################################################
    
def main():
    mpl_style(True)

    # sanity check the math
    test_amounts()

    # check what % of LVR goes to the LP as fees, as a function of Tx cost
    plot_lvr_and_tx_cost()

    # check what % of LVR goes to the LP as fees, as a function of block time
    plot_lvr_and_block_time()

    # plot the expectd DL == LVR depending on volatility
    plot_divloss_from_sigma()


if __name__ == '__main__':
    main()
    print("all done!")
