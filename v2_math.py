#
# This file implements utility functions for the based x*y=k AMM, such as Uniswap v2.
#

from math import sqrt


#
# Check if two floating point numbers are equal, with up to `epsilon` error
#
def float_equal(f1, f2, epsilon=1e-10):
    return f1 - epsilon <= f2 <= f1 + epsilon


#
# Price is defined as P = y / x
#
def price(x, y):
    assert not float_equal(x, 0.0)
    return y / x

#
# Liquidity is defined as the positive sqrt(k), where:
#   x * y = k
#
def get_liquidity(x, y):
    return sqrt(x * y)

#
# Since sqrt(P) = sqrt(y / x) by the definition of price,
# and L = sqrt(x * y) by the basic AMM formula x * y = k (=L^2),
# it follows that:
#
#    sqrt(P) = L / x = y / L
#
# Solving for x and y respectively:
#
#    x = L / sqrt(P)
#    y = L * sqrt(P)
#
def calculate_x(L, price):
    if float_equal(price, 0.0):
        return float("inf")
    return L / sqrt(price)


def calculate_y(L, price):
    return L * sqrt(price)


#
# Compute the value of a position given amounts and price
#
def position_value(x, y, P):
    return x * P + y


#
# Compute the value of a position given liquidity and price
#
def position_value_from_liquidity(L, price):
    return 2 * L * sqrt(price)


# swap token0 for token1; assumes no fees
def get_amount1_out(reserve0, reserve1, amount0):
    k = reserve0 * reserve1
    return reserve1 - k / (reserve0 + amount0)


# returns amount x obtained
def buy_x(reserve_x, reserve_y, amount_y, fee_tier):
    amount_y *= (1 - fee_tier)
    return get_amount1_out(reserve_y, reserve_x, amount_y)


# returns amount y obtained
def sell_x(reserve_x, reserve_y, amount_x, fee_tier):
    amount_x *= (1 - fee_tier)
    return get_amount1_out(reserve_x, reserve_y, amount_x)
