#!/usr/bin/env python

#
# This script computes the power perpetual coefficients required to hedge
# a v3 concentrated liquidity coeffcient.
# The inspiration comes from the article "Spanning with Power Perpetuals" by J. Clarke.
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
import v2_math
import v3_math

# max order to use for the power perp
MAX_ORDER = 2

# the higher, the more accurate the fit
NUM_STEPS = 1000

INITIAL_PRICE = 100
INITIAL_VALUE = 200

#
# This uses hedging formula from the article "Spanning with Power Perpetuals" by Joseph Clark,
# where the AMM is approximated using the Taylor expansion around r=0.
#
# This assumes zero funding rate! (as time is not given as a parameter to this function)
#
def power_perp_v2(initial_value, initial_price, price, order, include_first_order=True):
    r = price / initial_price - 1.0 # r is the price return
    perp_return = 0
    if order >= 1 and include_first_order:
        perp_return += -0.5 * r
    if order >= 2:
        perp_return += 0.125 * (r ** 2)
    if order >= 3:
        perp_return += -3 / 48 * (r ** 3)
    if order >= 4:
        perp_return += 15 / 384 * (r ** 4)
    perp_value = initial_value * (perp_return + 1.0)
    #    print("power perp value at price", order, include_first_order, price, perp_value)
    return perp_value

#
# This uses numerically found coefficient array for a narrow-range v3 positions.
#
def power_perp_v3(initial_value, initial_price, price, price_a, price_b, coefficients, order, include_first_order=True):
    # assume that when the price crosses the LP range boundaries, the owner sells the power perp,
    # effectively implying that the price can never go out of boundaries.
    price = min(price, price_b)
    price = max(price, price_a)

    r = price / initial_price - 1.0 # r is the price return
    perp_return = 0
    if order >= 1 and include_first_order:
        perp_return += coefficients[1] * r
    if order >= 2:
        perp_return += coefficients[2] * (r ** 2)
    if order >= 3:
        perp_return += coefficients[3] * (r ** 3)
    if order >= 4:
        perp_return += coefficients[4] * (r ** 4)

    perp_value = initial_value * (perp_return + 1.0)
    #    print("power perp value at price", order, include_first_order, price, perp_value)
    return perp_value


#
# Returns coefficients of the power perpetual needed to hedge a given position.
#
# Negative coefficient means that the perp should be short, positive: long.
# The coefficient are in terms of the initial value of the LP position.
#
# For instance, if the function returns [0, -0.5, 0.125, -0.0625, 0.0390625],
# (these are the coefficients for a full-range position)
# the ETH/USD LP should:
# 1. Buy short ETH perp worth 50% of the position.
# 2. Buy long ETH^2 perp worth 12.5% of the position.
# 3. (Optionally) Buy short ETH^3 perp worth 6.25% of the position.
# 4. (Optionally) Buy long ETH^4 perp worth ~3.9% of the position.
#
#
# Example n-th order reconstructions from the result:
#
# coefficients = fit_power_hedging_coefficients(L, p_a, p_b)
# reconstruction1 = [power_perp_v3(price, coefficients, 1) for price in x]
# reconstruction2 = [power_perp_v3(price, coefficients, 2) for price in x]
# reconstruction3 = [power_perp_v3(price, coefficients, 3) for price in x]
#
def fit_power_hedging_coefficients(liquidity, price_a, price_b, max_order):
    # the max value is bounded by the amount of tokens `y` at price `price_b`
    v_max = liquidity * (price_b ** 0.5 - price_a ** 0.5)

    step = (price_b - price_a) / NUM_STEPS

    prices = np.arange(price_a, price_b + step, step)
    values = [v3_math.position_value_from_max_value(v_max, price, price_a, price_b) for price in prices]

    # convert from f(x) to f(r) before doing the fit
    price_returns = [price / INITIAL_PRICE - 1.0 for price in prices]
    value_returns = [v / INITIAL_VALUE - 1.0 for v in values]

    # do a polynomial fit, with a high-order polynomial
    assert max_order < 20
    polyfit_many = np.polyfit(price_returns, value_returns, 20)

    # use just the first N coefficients from the result
    coefficients = [0] * (max_order + 1)
    coefficients[0] = 0 # always zero for the 0-th order hedge
    for i in range(1, max_order + 1):
        coefficients[i] = -polyfit_many[-(i + 1)] # i-th order power perp

    return coefficients

    
def main():
    mpl_style(True)

    # max order of the power perp
    order = MAX_ORDER

    # Full range positions
    min_price = INITIAL_PRICE / 4
    max_price = INITIAL_PRICE * 4
    step = (max_price - min_price) / NUM_STEPS
    prices = np.arange(min_price, max_price + step, step)
    liquidity = v2_math.get_liquidity(INITIAL_VALUE / INITIAL_PRICE / 2, INITIAL_VALUE / 2)
    pl.figure()
    profit_pos = [v2_math.position_value_from_liquidity(liquidity, price) - INITIAL_VALUE for price in prices]
    pl.plot(prices, profit_pos, label="LP position")
    profit_perp = [power_perp_v2(INITIAL_VALUE, INITIAL_PRICE, price, order) - INITIAL_VALUE for price in prices]
    pl.plot(prices, profit_perp, label=f"Power perperpetual, order={order}")
    pl.plot(prices, [a+b for a,b in zip(profit_perp, profit_pos)], label="Hedged position")

    pl.xlabel("Volatile asset price, $")
    pl.ylabel("Profit, $")
    pl.title("Full range")
    pl.legend()
    pl.show()
    pl.close()


    # Concentrated liquidity positions with different range r
    for r in [1.01, 1.1, 1.5, 2.0]:
        for skew in [-1, 0, +1]:
            r_low = r_high = r
            if skew < 0:
                r_low = r_low ** 2
            elif skew > 0:
                r_high = r_high ** 2
            price_a = INITIAL_PRICE / r_low
            price_b = INITIAL_PRICE * r_high
            liquidity = v3_math.position_liquidity_from_value(
                INITIAL_VALUE, INITIAL_PRICE, price_a, price_b)

            # check that the liquidity/value math is correct
            v_test = v3_math.position_value_from_liquidity(
                liquidity, INITIAL_PRICE, price_a, price_b)
            assert v_test - 1e-8 < INITIAL_VALUE < v_test + 1e-8

            # get the hedging power perp coefficients
            coefficients = fit_power_hedging_coefficients(liquidity, price_a, price_b, order)

            min_price = price_a / (r ** 0.5)
            max_price = price_b * (r ** 0.5)
            step = (max_price - min_price) / NUM_STEPS
            prices = np.arange(min_price, max_price + step, step)
            pl.figure()

            profit_pos = [v3_math.position_value_from_liquidity(
                liquidity, price, price_a, price_b) - INITIAL_VALUE for price in prices]
            pl.plot(prices, profit_pos, label="LP position")
            profit_perp = [power_perp_v3(INITIAL_VALUE, INITIAL_PRICE, price, price_a, price_b,
                                         coefficients, order) - INITIAL_VALUE for price in prices]
            pl.plot(prices, profit_perp, label=f"Power perp, order={order}")
            pl.plot(prices, [a+b for a,b in zip(profit_perp, profit_pos)], label="Hedged position")

            pl.title(f"range=[{price_a:.1f} {price_b:.1f}]")
            pl.legend()
            pl.show()
            pl.close()

if __name__ == '__main__':
    main()
    print("all done!")
