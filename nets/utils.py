import math

def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def calc_block_num(n,gd):
    '''
    :param n: 重复的bottleneck模块个数
    :param gd: 深度控制因子
    :return:
    '''
    n = max(round(n * gd), 1) if n > 1 else n  # depth gain, note: 深度因子只对number!=1的模块起作用即此处的csp模块
    return n