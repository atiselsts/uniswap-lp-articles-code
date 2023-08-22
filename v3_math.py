#
# This file implements utility functions for Uniswap v3 AMM.
#

from math import log, sqrt
from v2_math import position_value

#
# Liquidity math adapted from https://github.com/Uniswap/uniswap-v3-periphery/blob/main/contracts/libraries/LiquidityAmounts.sol
#
def get_liquidity_0(x, sa, sb):
    return x * sa * sb / (sb - sa)

def get_liquidity_1(y, sa, sb):
    return y / (sb - sa)

def get_liquidity(x, y, sp, sa, sb):
    if sp <= sa:
        liquidity = get_liquidity_0(x, sa, sb)
    elif sp < sb:
        liquidity0 = get_liquidity_0(x, sp, sb)
        liquidity1 = get_liquidity_1(y, sa, sp)
        liquidity = min(liquidity0, liquidity1)
    else:
        liquidity = get_liquidity_1(y, sa, sb)
    return liquidity

#
# Calculate x and y given liquidity and price range
#
def calculate_x(L, sp, sa, sb):
    sp = max(min(sp, sb), sa)     # if the price is outside the range, use the range endpoints instead
    return L * (sb - sp) / (sp * sb)

def calculate_y(L, sp, sa, sb):
    sp = max(min(sp, sb), sa)     # if the price is outside the range, use the range endpoints instead
    return L * (sp - sa)


# Convert Uniswap v3 tick to price
def to_price(tick):
    return 1.0001 ** tick

# Convert Uniswap v3 tick to sqrt(price)
def to_sqrt_price(tick):
    return 1.0001 ** (tick // 2)

# Convert price to Uniswap v3 tick
def to_tick(price):
    return int(round(log(price, 1.0001)))


#
# Compute the amount of assets in a position, and value from amounts
#
def position_value_from_liquidity(liquidity, price_current, price_low, price_high):
    sp = sqrt(price_current)
    sa = sqrt(price_low)
    sb = sqrt(price_high)
    x = calculate_x(liquidity, sp, sa, sb)
    y = calculate_y(liquidity, sp, sa, sb)
    return position_value(x, y, price_current)

#
# Compute the amount of assets in a position, and value from amounts, return both L and amounts
#
def position_value_from_liquidity_with_amounts(liquidity, price_current, price_low, price_high):
    sp = sqrt(price_current)
    sa = sqrt(price_low)
    sb = sqrt(price_high)
    x = calculate_x(liquidity, sp, sa, sb)
    y = calculate_y(liquidity, sp, sa, sb)
    return position_value(x, y, price_current), x, y


#
# Compute the value of the position from its initial value and price.
# Formula inspired by the pool value equation in:
# https://lambert-guillaume.medium.com/gamma-transforms-how-to-hedge-squeeth-using-uni-v3-da785cb8b378
#
def position_value_from_max_value(value_max, price, price_a, price_b):
    sa = sqrt(price_a)
    sb = sqrt(price_b)
    K = sa * sb # strike price
    r = sb / sa # range
    numerator = 2 * sqrt(price) * sb - K - price
    denominator = (r - 1) * K
    value_return = numerator / denominator
    return value_max * value_return


#
# Compute the liquidity of a position from its value and price range.
# Formula derived from the value formula above.
#
def position_liquidity_from_value(value, price, price_a, price_b):
    sa = sqrt(price_a)
    sb = sqrt(price_b)
    K = sa * sb # strike price
    r = sb / sa # range
    numerator = (r - 1) * K
    denominator = 2 * (price_b - K) * sqrt(price) - (K + price) * (sb - sa)
    value_return = numerator / denominator
    return value * value_return
