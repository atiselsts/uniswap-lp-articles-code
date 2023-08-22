#!/usr/bin/env python

#
# This plots the figures for the article on liquidity relocation.
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
import v2_math
import v3_math
from math import sqrt

# Constants for the LP positions

INITIAL_PRICE = 100

RANGE_FACTOR = 1.1

# select the value such that at 50:50 HODL we have 1.0 of the volatile asset X
INITIAL_VALUE = 2 * INITIAL_PRICE

INITIAL_X = INITIAL_VALUE / INITIAL_PRICE / 2
INITIAL_Y = INITIAL_VALUE / 2


# Constants for the Gaussian LP

# should be an odd number; the more positions, the better the gaussian is simulated
NUM_GAUSSIAN_POSITIONS = 7


# Constants for simulations

# similar to the 1-day volatility for ETH-USD
SIGMA = 0.05

# set to 0.0 to get a martingale
ZERO_MU = +0.000

# set to nonzero to simulate directional price movements
POSITIVE_MU = +0.003

# assume 12 second blocks as in the mainnet
BLOCKS_PER_DAY = 86400 // 12

NUM_DAYS = 365

# assume 0.3% swap fee
SWAP_FEE = 0.3 / 100

NUM_SIMULATIONS = 10000


############################################################

#
# Use geometrical Brownian motion to simulate price evolution.
#
def get_price_path(sigma_per_day, mu, blocks_per_day=BLOCKS_PER_DAY, M=NUM_SIMULATIONS):
    np.random.seed(123) # make it repeatable
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

def evaluate_static(all_prices, range_factor):
    all_divloss = []
    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        price_a = INITIAL_PRICE / range_factor
        price_b = INITIAL_PRICE * range_factor

        L = v3_math.get_liquidity(INITIAL_X, INITIAL_Y,
                                  sqrt(INITIAL_PRICE),
                                  sqrt(price_a), sqrt(price_b))

        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y
        Vfinal = v3_math.position_value_from_liquidity(L, prices[-1], price_a, price_b)
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def evaluate_full_rebalancing(all_prices, range_factor):
    all_divloss = []
    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        price_a = INITIAL_PRICE / range_factor
        price_b = INITIAL_PRICE * range_factor

        L = v3_math.get_liquidity(INITIAL_X, INITIAL_Y,
                                  sqrt(INITIAL_PRICE),
                                  sqrt(price_a), sqrt(price_b))
        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y

        for p in prices:
            if not (price_a <= p <= price_b):
                sp = sqrt(p)
                sa = sqrt(price_a)
                sb = sqrt(price_b)

                # amounts before rebalancing
                x = v3_math.calculate_x(L, sp, sa, sb)
                y = v3_math.calculate_y(L, sp, sa, sb)
                v_old = v3_math.position_value(x, y, p)

                # amounts after rebalancing
                y = v_old / 2
                x = y / p

                # sanity check
                v_new = x * p + y
                assert v_new - 1e-8 < v_old < v_new + 1e-8

                price_a = p / range_factor
                price_b = p * range_factor
                sa = sqrt(price_a)
                sb = sqrt(price_b)
                L = v3_math.get_liquidity(x, y, sp, sa, sb)

                # sanity check
                v_new = v3_math.position_value_from_liquidity(L, p, price_a, price_b)
                assert v_new - 1e-8 < v_old < v_new + 1e-8


        Vfinal = v3_math.position_value_from_liquidity(L, prices[-1], price_a, price_b)
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def evaluate_two_sided(all_prices, range_factor, do_rebalance):
    all_divloss = []
    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        price_low_a = INITIAL_PRICE / (range_factor ** 2)
        price_low_b = INITIAL_PRICE / range_factor

        price_high_a = INITIAL_PRICE * range_factor
        price_high_b = INITIAL_PRICE * (range_factor ** 2)

        L_low = v3_math.get_liquidity(0, INITIAL_Y,
                                  sqrt(INITIAL_PRICE),
                                  sqrt(price_low_a), sqrt(price_low_b))
        # since the ranges are symmetric, we expect L_low == L_high
        L_high = L_low
        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y

        if do_rebalance:
          for p in prices:
            if not (price_low_a <= p <= price_high_b):
                sp = sqrt(p)
                sa_low = sqrt(price_low_a)
                sb_low = sqrt(price_low_b)
                sa_high = sqrt(price_high_a)
                sb_high = sqrt(price_high_b)

                if p < price_low_a:
                    # amounts before rebalancing
                    x_low = v3_math.calculate_x(L_low, sp, sa_low, sb_low)
                    y_low = v3_math.calculate_y(L_low, sp, sa_low, sb_low)
                    v_old_low = v3_math.position_value(x_low, y_low, p)

                    # amounts after rebalancing
                    y_low = v_old_low / 2
                    x_low = y_low / p

                    price_low_a = p / range_factor
                    price_low_b = p * range_factor
                    sa_low = sqrt(price_low_a)
                    sb_low = sqrt(price_low_b)
                    L_low = v3_math.get_liquidity(x_low, y_low, sp, sa_low, sb_low)

                    x_high = v3_math.calculate_x(L_high, sp, sa_high, sb_high)
                    price_high_a = p * (range_factor ** 2)
                    price_high_b = p * (range_factor ** 3)
                    sa_high = sqrt(price_high_a)
                    sb_high = sqrt(price_high_b)
                    L_high = v3_math.get_liquidity(x_high, 0, sp, sa_high, sb_high)

                else:
                    # amounts before rebalancing
                    x_high = v3_math.calculate_x(L_high, sp, sa_high, sb_high)
                    y_high = v3_math.calculate_y(L_high, sp, sa_high, sb_high)
                    v_old_high = v3_math.position_value(x_high, y_high, p)

                    # amounts after rebalancing
                    y_high = v_old_high / 2
                    x_high = y_high / p

                    price_high_a = p / range_factor
                    price_high_b = p * range_factor
                    sa_high = sqrt(price_high_a)
                    sb_high = sqrt(price_high_b)
                    L_high = v3_math.get_liquidity(x_high, y_high, sp, sa_high, sb_high)

                    y_low = v3_math.calculate_y(L_low, sp, sa_low, sb_low)
                    price_low_a = p / (range_factor ** 3)
                    price_low_b = p / (range_factor ** 2)
                    sa_low = sqrt(price_low_a)
                    sb_low = sqrt(price_low_b)
                    L_low = v3_math.get_liquidity(0, y_low, sp, sa_low, sb_low)


        Vfinal_low = v3_math.position_value_from_liquidity(L_low, prices[-1], price_low_a, price_low_b)
        Vfinal_high = v3_math.position_value_from_liquidity(L_high, prices[-1], price_high_a, price_high_b)
        Vfinal = Vfinal_low + Vfinal_high
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def evaluate_fast_rebalancing(all_prices, range_factor):
    all_divloss = []
    sqrt_range_factor = sqrt(sqrt(range_factor))
    n = 0
    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        price_a = INITIAL_PRICE / range_factor
        price_b = INITIAL_PRICE * range_factor

        price_a_trigger = INITIAL_PRICE / sqrt_range_factor
        price_b_trigger = INITIAL_PRICE * sqrt_range_factor

        L = v3_math.get_liquidity(INITIAL_X, INITIAL_Y,
                                  sqrt(INITIAL_PRICE),
                                  sqrt(price_a), sqrt(price_b))
        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y

        for p in prices:
            if not (price_a_trigger <= p <= price_b_trigger):
                n += 1
                sp = sqrt(p)
                sa = sqrt(price_a)
                sb = sqrt(price_b)

                # amounts before rebalancing
                x = v3_math.calculate_x(L, sp, sa, sb)
                y = v3_math.calculate_y(L, sp, sa, sb)
                v_old = v3_math.position_value(x, y, p)

                # amounts after rebalancing
                y = v_old / 2
                x = y / p

                # sanity check
                v_new = x * p + y
                assert v_new - 1e-8 < v_old < v_new + 1e-8

                price_a = p / range_factor
                price_b = p * range_factor
                price_a_trigger = p / sqrt_range_factor
                price_b_trigger = p * sqrt_range_factor
                sa = sqrt(price_a)
                sb = sqrt(price_b)
                old_L = L
                L = v3_math.get_liquidity(x, y, sp, sa, sb)

                # sanity check
                v_new = v3_math.position_value_from_liquidity(L, p, price_a, price_b)
                assert v_new - 1e-8 < v_old < v_new + 1e-8

        Vfinal = v3_math.position_value_from_liquidity(L, prices[-1], price_a, price_b)
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def evaluate_partial_rebalancing(all_prices, range_factor):
    all_divloss = []
    sqrt_range_factor = sqrt(range_factor)
    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        price_a = INITIAL_PRICE / range_factor
        price_b = INITIAL_PRICE * range_factor

        L = v3_math.get_liquidity(INITIAL_X, INITIAL_Y,
                                  sqrt(INITIAL_PRICE),
                                  sqrt(price_a), sqrt(price_b))
        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y

        for p in prices:
            if not (price_a <= p <= price_b):
                sp = sqrt(p)
                sa = sqrt(price_a)
                sb = sqrt(price_b)

                # amounts before rebalancing
                x = v3_math.calculate_x(L, sp, sa, sb)
                y = v3_math.calculate_y(L, sp, sa, sb)
                v_old = v3_math.position_value(x, y, p)

                # amounts after rebalancing
                if p < price_a:
                    price_a /= sqrt_range_factor
                    price_b /= sqrt_range_factor
                else:
                    price_a *= sqrt_range_factor
                    price_b *= sqrt_range_factor
                sa = sqrt(price_a)
                sb = sqrt(price_b)
                x_per_unit = v3_math.calculate_x(1, sp, sa, sb) * p
                y_per_unit = v3_math.calculate_y(1, sp, sa, sb)
                total = y_per_unit + x_per_unit
                x_prop = x_per_unit / total
                y_prop = 1 - x_prop

                x = v_old * x_prop / p
                y = v_old * y_prop

                # sanity check
                v_new = x * p + y
                assert v_new - 1e-8 < v_old < v_new + 1e-8

                L = v3_math.get_liquidity(x, y, sp, sa, sb)

                # sanity check
                v_new = v3_math.position_value_from_liquidity(L, p, price_a, price_b)
                assert v_new - 1e-8 < v_old < v_new + 1e-8


        Vfinal = v3_math.position_value_from_liquidity(L, prices[-1], price_a, price_b)
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def evaluate_width_rebalancing(all_prices, range_factor):
    all_divloss = []

    initial_range_factor = range_factor

    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        range_factor = initial_range_factor

        price_a = INITIAL_PRICE / range_factor
        price_b = INITIAL_PRICE * range_factor

        L = v3_math.get_liquidity(INITIAL_X, INITIAL_Y,
                                  sqrt(INITIAL_PRICE),
                                  sqrt(price_a), sqrt(price_b))
        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y

        for p in prices:
            if not (price_a <= p <= price_b):
                v_old = v3_math.position_value_from_liquidity(L, p, price_a, price_b)

                if False:
                    # both directions
                    range_factor = range_factor ** 2
                    price_a = INITIAL_PRICE / range_factor
                    price_b = INITIAL_PRICE * range_factor
                else:
                    # towards the direction where the price has moved
                    if p < price_a:
                        price_a /= range_factor
                    else:
                        price_b *= range_factor

                sa = sqrt(price_a)
                sb = sqrt(price_b)
                sp = sqrt(p)
                x_per_unit = v3_math.calculate_x(1, sp, sa, sb) * p
                y_per_unit = v3_math.calculate_y(1, sp, sa, sb)
                total = y_per_unit + x_per_unit
                x_prop = x_per_unit / total
                y_prop = 1 - x_prop

                x = v_old * x_prop / p
                y = v_old * y_prop

                # sanity check
                v_new = x * p + y
                assert v_new - 1e-8 < v_old < v_new + 1e-8

                L = v3_math.get_liquidity(x, y, sp, sa, sb)

                # sanity check
                v_new = v3_math.position_value_from_liquidity(L, p, price_a, price_b)
                assert v_new - 1e-8 < v_old < v_new + 1e-8

        Vfinal = v3_math.position_value_from_liquidity(L, prices[-1], price_a, price_b)
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def gaussian_liquidity_distribution(n):
    n //= 2

    # use STD such that it's halfway in the open positions
    # e.g. num_pos = 7 -> sigma=1.5 -> the std is between the 1st and 2nd side positions
    sigma = n / 2   
    mu = 0

    liquidities = [0] * n
    for i in range(n):
        x = i + 1
        value = np.exp(-1/2 * (x - mu)**2 / sigma ** 2)
        liquidities[i] = value

    # prepend with itself, reversed, and add 1.0 in the center
    return liquidities[::-1] + [1] + liquidities


def gaussian_liq_from_values(distr, range_factor, total_value, center_price):
    n = len(distr)
    n = n // 2
    half_distr = distr[n:]

    price_a = center_price / range_factor
    price_b = center_price * range_factor

    scp = sqrt(center_price)
    range_factor_2 = range_factor ** 2

    value_0 = 2 * v3_math.calculate_y(half_distr[0], scp, sqrt(price_a), scp)
    values = [0] * n
    for i in range(n):
        price_a /= range_factor_2
        price_b /= range_factor_2
        values[i] = v3_math.calculate_y(half_distr[i+1], scp, sqrt(price_a), sqrt(price_b))

    unit_values = values[::-1] + [value_0] + values
    total_unit_liquidity_value = sum(unit_values)
    factor = total_value / total_unit_liquidity_value

    return [u * factor for u in distr]


def gaussian_values_from_liq(liquidities, range_factor, center_price, current_price):
    n = len(liquidities)
    p = current_price
    sp = sqrt(current_price)
    range_factor_2 = range_factor ** 2

    price_a = center_price / (range_factor ** n)
    price_b = price_a * range_factor_2

    values = [0] * n
    for i in range(n):
        if p < price_a:
            # x only
            values[i] = p * v3_math.calculate_x(liquidities[i], sp, sqrt(price_a), sqrt(price_b))
        elif p > price_b:
            # y only
            values[i] = v3_math.calculate_y(liquidities[i], sp, sqrt(price_a), sqrt(price_b))
        else:
            # both
            values[i] = p * v3_math.calculate_x(liquidities[i], sp, sp, sqrt(price_b)) \
                + v3_math.calculate_y(liquidities[i], sp, sqrt(price_a), sp)
        price_b *= range_factor_2
        price_a *= range_factor_2

    return values


def evaluate_gaussian(all_prices, range_factor, do_rebalance):
    assert NUM_GAUSSIAN_POSITIONS % 2 == 1

    full_range_factor = range_factor
    range_factor = full_range_factor ** (1/NUM_GAUSSIAN_POSITIONS)

    all_divloss = []
    for sim in range(NUM_SIMULATIONS):
        prices = all_prices[:,sim]

        trigger_low = INITIAL_PRICE / full_range_factor
        trigger_high = INITIAL_PRICE * full_range_factor

        distr = gaussian_liquidity_distribution(NUM_GAUSSIAN_POSITIONS)
        liquidities = gaussian_liq_from_values(distr, range_factor, INITIAL_VALUE, INITIAL_PRICE)
        center_price = INITIAL_PRICE

        new_value = sum(gaussian_values_from_liq(liquidities, range_factor, INITIAL_PRICE, INITIAL_PRICE))
        assert INITIAL_VALUE - 1e-8 < new_value < INITIAL_VALUE + 1e-8

        if do_rebalance:
          for p in prices:
            if not (trigger_low <= p <= trigger_high):
                value = sum(gaussian_values_from_liq(liquidities, range_factor, center_price, p))
                old_total_liq = sum(liquidities)

                center_price = p
                liquidities = gaussian_liq_from_values(distr, range_factor, value, center_price)
                # sanity check
                new_total_liq = sum(liquidities)
                assert new_total_liq < old_total_liq

                # sanity check
                new_value = sum(gaussian_values_from_liq(liquidities, range_factor, p, p))
                assert value - 1e-8 < new_value < value + 1e-8

                trigger_low = p / full_range_factor
                trigger_high = p * full_range_factor

        Vfinal = sum(gaussian_values_from_liq(liquidities, range_factor, center_price, prices[-1]))
        Vhodl = prices[-1] * INITIAL_X + INITIAL_Y
        divloss = (Vfinal - Vhodl) / Vhodl

        all_divloss.append(divloss)

    return np.mean(all_divloss)

############################################################

def compute_expected_divloss(sigma, mu):
    initial_x = INITIAL_X
    initial_y = INITIAL_Y

    all_prices = get_price_path(sigma, mu, blocks_per_day=12)

    final_prices = all_prices[-1,:]
    median = sorted(final_prices)[NUM_SIMULATIONS // 2]
    returns = final_prices / INITIAL_PRICE
    print(f"sigma={sigma:.2f} mean={np.mean(final_prices):.4f} median={median:.4f} std={np.std(np.log(returns)):.4f}")

    divloss_static = evaluate_static(all_prices, RANGE_FACTOR)
    divloss_with_rebal = evaluate_full_rebalancing(all_prices, RANGE_FACTOR)
    divloss_gaussian_static = evaluate_gaussian(all_prices, RANGE_FACTOR, False)
    divloss_gaussian_rebal = evaluate_gaussian(all_prices, RANGE_FACTOR, True)
    divloss_fast_rebal = evaluate_fast_rebalancing(all_prices, RANGE_FACTOR)
    divloss_partial_rebal = evaluate_partial_rebalancing(all_prices, RANGE_FACTOR)
    divloss_change_width = evaluate_width_rebalancing(all_prices, RANGE_FACTOR)
    divloss_2sided_static = evaluate_two_sided(all_prices, RANGE_FACTOR, False)
    divloss_2sided_rebal = evaluate_two_sided(all_prices, RANGE_FACTOR, True)

    if False:
        print("sigma=", sigma)
        print(f"static LP:                {divloss_static*100:.2f}")
        print(f"with rebalancing:          {divloss_with_rebal*100:.2f}")
        print(f"with gaussian, static:     {divloss_gaussian_static*100:.2f}")
        print(f"with gaussian, rebal.:     {divloss_gaussian_rebal*100:.2f}")
        print(f"with width change:        {divloss_change_width*100:.2f}")
        print(f"with partial rebalancing: {divloss_partial_rebal*100:.2f}")
        print(f"with fast rebalancing:    {divloss_fast_rebal*100:.2f}")
        print(f"two-sided static LP:       {divloss_2sided_static*100:.2f}")
        print(f"two-sided rebalancing LP:  {divloss_2sided_rebal*100:.2f}")
        print("****")

    return {"static": divloss_static,
            "full rebalancing": divloss_with_rebal,
            "gaussian static": divloss_gaussian_static,
            "gaussian rebalancing": divloss_gaussian_rebal,
            "fast rebalancing": divloss_fast_rebal,
            "partial rebalancing": divloss_partial_rebal,
            "width change only": divloss_change_width,
            "two-sided static": divloss_2sided_static,
            "two-sided rebalancing": divloss_2sided_rebal
            }
############################################################

# example with USDC/USD depeg, very narrow position
def depeg_example():
    print("simulating 2% depeg with price reversion")
    initial_price = 1.0
    price_a = initial_price / 1.001
    price_b = initial_price * 1.001
    initial_value = 1000
    x = initial_value / 2
    y = initial_value / 2
    L0 = v3_math.get_liquidity(x, y,
                               sqrt(initial_price),
                               sqrt(price_a), sqrt(price_b))

    new_price = 0.98
    x = v3_math.calculate_x(L0, sqrt(new_price), sqrt(price_a), sqrt(price_b))
    new_y = new_price * x / 2
    new_x = x / 2
    price_a = new_price / 1.001
    price_b = new_price * 1.001
    L1 = v3_math.get_liquidity(new_x, new_y,
                               sqrt(new_price),
                               sqrt(price_a), sqrt(price_b))

    new_price = initial_price
    y = v3_math.calculate_y(L1, sqrt(new_price), sqrt(price_a), sqrt(price_b))
    final_value = y
    print(f"final_value={final_value:.2f} loss={100 * (1 - final_value / initial_value):.2f}%")

############################################################

# shows that 4x higher price -> 2x deeper liquidity
def plot_liquidity_from_price():

    initial_price = 100

    prices = np.arange(initial_price, initial_price * 16, 0.01)
    liquidities = []
    price_a = initial_price / 1.01
    unit_x = 1.0
    unit_y = initial_price
    L0 = v3_math.get_liquidity_1(unit_y / 2, sqrt(price_a), sqrt(initial_price))
    print("L0=", L0)
    for price in prices:
        price_a = price / 1.01
        price_b = price * 1.01

        if price < initial_price:
            L = v3_math.get_liquidity_1(unit_y, sqrt(price_a), sqrt(price_b))
        else:
            L = v3_math.get_liquidity_0(unit_x, sqrt(price_a), sqrt(price_b))
        liquidities.append(L / L0)

    pl.figure(figsize=(5, 3))
    pl.plot(prices / initial_price, liquidities)
    pl.ylabel("Liquidity multiplier")
    pl.xlabel("Price multiplier")
    pl.savefig("article_6_price_vs_liquidity.png", bbox_inches='tight')
    pl.close()

############################################################

def plot_values(sigmas, values, expected_value_hodl, selector, filename):
    pl.figure()

    # convert to yearly sigma to improve x axis appearance
    sigmas = [s * sqrt(365) for s in sigmas]

    for label in selector:
        divloss = [experiment[label] for experiment in values]
        v = [expected_value_hodl * (1.0 + d) for d in divloss]
        pl.plot(sigmas, v, label=label, marker="o", linestyle="--")

    pl.legend()
    pl.xscale("log")
    x = [0.1, 0.2, 0.4, 0.8, 0.6, 1.0, 1.4, 1.8]
    pl.xticks(x, [str(u) for u in x])
    pl.ylabel("Expected final value, $")
    pl.xlabel("$\sigma$")
    pl.savefig(f"article_6_{filename}.png", bbox_inches='tight')
    pl.close()

############################################################
    
def main():
    mpl_style(True)

    depeg_example()

    plot_liquidity_from_price()

    sigmas = [SIGMA / 8, SIGMA / 4, SIGMA / 2, SIGMA, SIGMA * 2]
    values = [compute_expected_divloss(sigma, mu=ZERO_MU) for sigma in sigmas]

    plot_values(sigmas, values, 100.0, ("static", "full rebalancing"), "static_vs_full")
    plot_values(sigmas, values, 100.0,
                ("static", "full rebalancing", "partial rebalancing", "fast rebalancing", "width change only"),
                "full_vs_partial")
    plot_values(sigmas, values, 100.0,
                ("static", "full rebalancing", "two-sided static", "two-sided rebalancing"),
                "full_vs_twosided")
    plot_values(sigmas, values, 100.0,
                ("static", "full rebalancing", "gaussian static", "gaussian rebalancing"),
                "full_vs_gaussian")

    # experiment with positive price drift
    values = [compute_expected_divloss(sigma, mu=POSITIVE_MU) for sigma in sigmas]

    plot_values(sigmas, values, 199,
                ("static", "full rebalancing", "partial rebalancing", "fast rebalancing", "width change only"),
                "directional_full_vs_partial")
    

if __name__ == '__main__':
    main()
    print("all done!")
