"""
compare_and_count now replaces `softmax` and `softmin`

Usage:
- compare_and_count(..., by='lower') replace `softmax`
- compare_and_count(..., by='higher') replace `softmin`
"""


# 取代softmax和softmin（输出值太小了）
def compare_and_count(x, nums, by='lower'):
    """
    if by == 'lower', then count -> [count] numbers lower or equal than x
    else, count -> [count] numbers higher than x
    """
    if by == 'lower':   # 对标softmax
        return len([num for num in nums if num <= x])
    else:   # 对标softmin
        return len([num for num in nums if num > x])
