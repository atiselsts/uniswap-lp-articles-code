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
#NUM_DAYS = 2

# assume 0.3% and 0.05% swap fees
SWAP_FEE_03 = 0.3 / 100
SWAP_FEE_005 = 0.05 / 100

NUM_SIMULATIONS = 10000


# Constants for plotting
pl.rcParams["savefig.dpi"] = 200


############################################################

#
# Use geometrical Brownian motion to simulate price evolution.
#
def get_price_path(sigma_per_day, blocks_per_day=BLOCKS_PER_DAY, M=NUM_SIMULATIONS, num_days=NUM_DAYS):
    np.random.seed(123) # make it repeatable
    mu = 0.0   # assume delta neutral behavior
    T = num_days
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

    fee_tier = SWAP_FEE_03

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

def estimate_lvr(prices, swap_tx_cost, fee_tier):
    fee_factor_down = 1.0 - fee_tier
    fee_factor_up = 1.0 + fee_tier

    # assume $1 million of liquidity in the pool (the larger, the better for all parties)
    reserve_x = 500
    reserve_y = 500_000
    pool_value0 = reserve_x * INITIAL_PRICE + reserve_y
    L = v2_math.get_liquidity(reserve_x, reserve_y)

    # compute lvr and fees
    lvr = 0
    collected_fees = 0
    num_tx = 0

    for cex_price in prices:
        pool_price = reserve_y / reserve_y
        if cex_price > pool_price:
            to_price = cex_price * fee_factor_down
            if to_price < pool_price:
                continue
        else:
            to_price = cex_price * fee_factor_up
            if to_price > pool_price:
                continue

        to_sqrt_price = sqrt(to_price)
        delta_x = L / to_sqrt_price - reserve_x
        delta_y = L * to_sqrt_price - reserve_y
        if delta_x > 0:
            # arber sells X, LP buys X
            swap_fee = fee_tier * delta_x * cex_price
        else:
            # arber buys X, LP sells X
            swap_fee = fee_tier * delta_y

        # assume fixed gas fees
        lp_loss_vs_cex = -(delta_x * cex_price + delta_y)

        arb_gain = lp_loss_vs_cex - swap_fee - swap_tx_cost
        if arb_gain > 0:
            lvr += lp_loss_vs_cex # account without swap fees and tx fees
            reserve_x += delta_x
            reserve_y += delta_y
            collected_fees += swap_fee
            num_tx += 1

    # normalize by dividing with the initial value of the capital rather than the final value
    lvr /= pool_value0
    collected_fees /= pool_value0
    return lvr, collected_fees, num_tx


############################################################

def compute_lvr(all_prices, swap_tx_cost, fee_tier):
    print(f"compute_lvr, swap_tx_cost={swap_tx_cost}, fee_tier={100*fee_tier:.2}%")
    #fee_multiplier = 1 / (1 - swap_fee)

    all_lvr = []
    all_fees = []
    all_tx_per_block = []

    # assume $1 million of liquidity in the pool (the larger, the better for all parties)
#    pool_x0 = 500
#    pool_y0 = 500_000
#    pool_value0 = pool_x0 * INITIAL_PRICE + pool_y0

#    L = v2_math.get_liquidity(pool_x0, pool_y0)

    if len(all_prices.shape) > 2:
        # take the first elements from the second dimension
        all_prices = all_prices[:,0,:]


    for sim in range(all_prices.shape[1]):
        prices = all_prices[:,sim]

        lvr, collected_fees, num_tx = estimate_lvr(prices, swap_tx_cost, fee_tier)
        all_lvr.append(lvr)
        all_fees.append(collected_fees)
        all_tx_per_block.append(num_tx / len(prices))

    return np.mean(all_lvr), np.mean(all_fees), np.mean(all_tx_per_block)


def plot_lvr_and_tx_cost():
    # assume pool with $1 million liquidity
    # if swap transaction costs:
    # * $100 -> 1 bps fee per tranasaction
    # * $10 -> 0.1 bps fee per tranasaction
    # * $1 -> 0.01 bps fee per tranasaction
    # * $0.1 -> 0.001 bps fee per tranasaction
    #tx_cost_bps = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02])

    #swap_tx_cost = INITIAL_VALUE * tx_cost_bps / (100 * 100)
    # multiply by 500 because the assumption is $1M liquidity instead of $2000 as in this model
    #swap_tx_cost *= 500

    swap_tx_cost_dollars = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])

    fig, ax = pl.subplots()
    fig.set_size_inches((4, 3))

    # reduce the number of simulations, since we iterate over each block
    num_simulations = 50

    all_prices = get_price_path(SIGMA, blocks_per_day=BLOCKS_PER_DAY, M=num_simulations)
    final_prices = all_prices[-1,:]
    returns = final_prices / INITIAL_PRICE
    year_sigma = SIGMA * sqrt(NUM_DAYS) # convert from daily to yearly volatility
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    lvr_and_fees_03 = [compute_lvr(all_prices, cost, SWAP_FEE_03) for cost in swap_tx_cost_dollars]
    lvr_and_fees_005 = [compute_lvr(all_prices, cost, SWAP_FEE_005) for cost in swap_tx_cost_dollars]

    x = swap_tx_cost_dollars
    pl.plot(x, [100*u[0] for u in lvr_and_fees_03], label="Losses to LVR", marker="v")
    pl.plot(x, [100*u[1] for u in lvr_and_fees_03], label="Gains from arb fees, 0.3% pool", marker="o", color="green")
    pl.plot(x, [100*u[1] for u in lvr_and_fees_005], label="Gains from arb fees, 0.05% pool", marker="o", color="lightgreen")

    pl.xlabel("Swap tx cost, $")
    pl.ylabel("APR, %")
    pl.legend()
    pl.ylim(ymin=0)

    pl.savefig("article_3_lvr_and_tx_cost.png", bbox_inches='tight')
    pl.close()



def plot_lvr_and_block_time():
    #tx_cost_bps = 0.01
    #swap_tx_cost = INITIAL_VALUE * tx_cost_bps / (100 * 100)
    # multiply by 500 because the assumption is $1M liquidity instead of $2000 as in this model
    #swap_tx_cost *= 500

    # assume not so cheap transactions
    swap_tx_cost_dollars = 5

    fig, ax = pl.subplots()
    fig.set_size_inches((4, 3))

    # reduce the number of simulations, since we iterate over each block
    num_simulations = 10 # very small, but the sigma and mean are still quite accurate

    base_blocktime_sec = 1
    blocks_per_day = 86400 // base_blocktime_sec

    all_prices = get_price_path(SIGMA, blocks_per_day=blocks_per_day, M=num_simulations)
    final_prices = all_prices[-1,:]

    returns = final_prices / INITIAL_PRICE
    year_sigma = SIGMA * sqrt(NUM_DAYS) # convert from daily to yearly volatility
    print(f"sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    block_time_multipliers = [1, 2, 4, 8, 12]
    lvr_and_fees = []
    for multiplier in block_time_multipliers:
        if multiplier > 1:
            all_prices = all_prices.reshape((blocks_per_day * NUM_DAYS) // multiplier, multiplier, num_simulations)

            final_prices = all_prices[-1,:][-1,:]
            returns = final_prices / INITIAL_PRICE
            year_sigma = SIGMA * sqrt(NUM_DAYS) # convert from daily to yearly volatility
            print(f"  m={multiplier} sigma={year_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

        lvr, fees1, _ = compute_lvr(all_prices, 0.0, SWAP_FEE_03)
        lvr, fees2, _   = compute_lvr(all_prices, swap_tx_cost_dollars, SWAP_FEE_03)
        lvr_and_fees.append((lvr, fees1, fees2))

    block_time_sec = [u * base_blocktime_sec for u in block_time_multipliers]
    pl.plot(block_time_sec, [100*u[0] for u in lvr_and_fees], label="Losses to LVR", marker="v")
    pl.plot(block_time_sec, [100*u[1] for u in lvr_and_fees], label="Gains from arb fees, tx cost=$0", marker="o", color="green")
    pl.plot(block_time_sec, [100*u[2] for u in lvr_and_fees], label=f"Gains from arb fees, tx cost=${swap_tx_cost_dollars}", marker="o", color="lightgreen")

    pl.xlabel("Block time, seconds")
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

#
# This aims to replicate Table 1 from the paper "LVR-with-fees", i.e.
# "Automated Market Making and Arbitrage Profits in the Presence of Fees".
#
def simulate_prob_trade_per_block():
    # don't simulate the 50 msec case, it requires too much blocks & not practical
    base_blocktime_sec = 2
    # corresponds to 2 sec, 12 sec, 2 min, 12 min
    block_time_multipliers = [1, 6, 60, 300]
    # as in the paper
    swap_fee_bps = [1, 5, 10, 30, 100]

    # WARNING: assume zero-cost swap Tx (probably as in the paper!)
    tx_cost = 0.0
    #tx_cost = 1.0

    blocks_per_day = 86400 // base_blocktime_sec

    num_days = 10
    num_simulations = 100
    all_prices = get_price_path(SIGMA, blocks_per_day=blocks_per_day, M=num_simulations, num_days=num_days)
    final_prices = all_prices[-1,:]

    returns = final_prices / INITIAL_PRICE
    period_sigma = SIGMA * sqrt(num_days) # convert from daily to period volatility
    print(f"sigma={period_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

    all_prob_per_block = {}
    for multiplier in block_time_multipliers:
        if multiplier > 1:
            all_prices = all_prices.reshape((blocks_per_day * num_days) // multiplier, multiplier, num_simulations)

            final_prices = all_prices[-1,:][-1,:]
            returns = final_prices / INITIAL_PRICE
            print(f"  m={multiplier} sigma={period_sigma:.2f} mean={np.mean(final_prices):.4f} std={np.std(np.log(returns)):.4f}")

        sim_result = [compute_lvr(all_prices, tx_cost, bps / 10_000) for bps in swap_fee_bps]
        all_prob_per_block[multiplier] = [u[2] for u in sim_result]


    for multiplier in block_time_multipliers[::-1]:
        print(f"block time {multiplier * base_blocktime_sec: 5d} sec, arb prob %:", end="")
        for i in range(len(swap_fee_bps)):
            #print(f"fee={swap_fee_bps[i]} bps prob={all_prob_per_block[multiplier][i]:.2f} ", end="")
            print(f"{100*all_prob_per_block[multiplier][i]: 3.1f} ", end="")
        print("")

############################################################

def simulate_poisson_block_times():
    base_blocktime_sec = 2
    block_time_multipliers = [1, 6, 60, 300]

#    lam = block_time_multipliers[0]

    num_days = 2

    num_simulations = 100

    blocks_per_day = 86400 // base_blocktime_sec
    blocks_per_day //= block_time_multipliers[1] # XXX
    block_time_distr = np.random.exponential(scale=1.0, size=num_days * blocks_per_day)

    #count, bins, ignored = pl.hist(distr, 100, density=True)
    #pl.show()

    sigma = SIGMA * sqrt(num_days) # convert from daily to period volatility
    all_prices = get_price_path(sigma, blocks_per_day=blocks_per_day, M=num_simulations, num_days=num_days)

    all_lvr = []
    all_fees = []
    all_tx_per_block = []

    swap_tx_cost = 0.0

    # as in the paper
    swap_fee_bps = [1, 5, 10, 30, 100]

    for bps in swap_fee_bps:
        for sim in range(all_prices.shape[1]):
            prices = all_prices[:,sim]
#            print(prices)

            adj_prices = [prices[0]]
            price = prices[0]
            t = 0
            for i in range(1, len(prices)):
                f = prices[i] / prices[i-1]
                t += block_time_distr[i]
                # correct the factor in terms of the time passed between the blocks
                s_factor = sqrt(block_time_distr[i])
                new_f = 1.0 + s_factor * (f - 1)
#                if s_factor < 0.5 or s_factor > 2:
#                    print(f, new_f, s_factor)
                price *= new_f
                adj_prices.append(price)
#            print("total t=", t, "num blocks=", len(prices))

            fee_tier = bps / 10_000
#            print(adj_prices[:5], adj_prices[-5:])
            lvr, collected_fees, num_tx = estimate_lvr(adj_prices, swap_tx_cost, fee_tier)
            all_lvr.append(lvr)
            all_fees.append(collected_fees)
            all_tx_per_block.append(num_tx / len(adj_prices))

        tx_per_block = np.mean(all_tx_per_block)
        print(f"block_time={86400 // blocks_per_day} bps={bps} tx_per_block={100*tx_per_block:.2f}%")


############################################################
    
def main():
    mpl_style(True)

    # sanity check the math
    test_amounts()

    # try to match LVR paper results
#    simulate_prob_trade_per_block()

    simulate_poisson_block_times()

    # check what % of LVR goes to the LP as fees, as a function of Tx cost
#    plot_lvr_and_tx_cost()

    # check what % of LVR goes to the LP as fees, as a function of block time
#    plot_lvr_and_block_time()

    # plot the expectd DL == LVR depending on volatility
#    plot_divloss_from_sigma()


if __name__ == '__main__':
    main()
    print("all done!")
