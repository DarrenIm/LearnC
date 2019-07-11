#!/usr/bin/env python
# coding: utf-8
import numpy as np

'''Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].'''

class Solution_01(object):
    def two_sum(self, nums, target):
        lookup = {}
        for i, num in enumerate(nums):
            if target - num in lookup:
                return [lookup[target - num], i]
            else:
                lookup[num] = i


'''You are given two non-empty linked lists representing two non-negative integers. 

The digits are stored in reverse order and each of their nodes contain a single digit. 

Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
'''
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution_02:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 and not l2: # l1, l2 都为 0
            return
        elif not (l1 and l2): # 若 l1 and l2 其中之一为 0
            return l1 or l2   # 即返回 l1, l2 的或
        else:
            if l1.val + l2.val < 10:
                l3 = ListNode(l1.val+l2.val)
                l3.next = self.addTwoNumbers(l1.next, l2.next)
            else:
                l3 = ListNode(l1.val+l2.val-10)
                l3.next = self.addTwoNumbers(l1.next, self.addTwoNumbers(l2.next, ListNode(1)))
        return l3

'''Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.'''

class Solution_03(object):
    def longest_sub_str(self, str):
        '''res 代表目前最大子串的长度
        start 是这一轮未重复子串首字母的 index
        maps 放置每一个字符的 index，如果 maps.get(s[i], -1) 大于等于 start 的话，就说明字符重复了，此时就要重置 res 和 start 的值了'''
        res, start, n = 0, 0, len(str)
        maps = {}
        for i in range(n):
            start = max(start, maps.get(str[i], -1)+1)
            print(maps.get(str[i], -1))
            res = max(res, i - start + 1)
            maps[str[i]] = i
        # return res


'''——————04 There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.

Example 1:

nums1 = [1, 3]
nums2 = [2]

The median is 2.0
Example 2:

nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5'''

# 此题不明

'''——————05 Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
Example 2:

Input: "cbbd"
Output: "bb"'''


class Solution_05:
    def longestPalindrome(self, s):
        res = ""
        for i in range(len(s)):
            odd = self.palindromeAt(s, i, i)
            even = self.palindromeAt(s, i, i + 1)

            res = max(res, odd, even, key=len)
        return res

        # starting at l,r expand outwards to find the biggest palindrome

    def palindromeAt(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]


'''*此*题*有*意*思* The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to 
display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:

string convert(string s, int numRows);
Example 1:

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
Example 2:

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:

P     I    N
A   L S  I G
Y A   H R
P     I'''


class Solution_06(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1 or numRows >= len(s):
            return s
        res = [''] * numRows
        idx, step = 0, 1

        for x in s:
            res[idx] += x
            if idx == 0:  ## 第一行，一直向下走
                step = 1
            elif idx == numRows - 1:  ## 最后一行了，向上走
                step = -1
            idx += step
        return ''.join(res)

# EXAMPLE: s = “abcdefghijklmn”, numRows = 4, print(res[idx] += x)
# (0, 1, ['a', '', '', ''])
# (1, 1, ['a', 'b', '', ''])
# (2, 1, ['a', 'b', 'c', ''])
# (3, 1, ['a', 'b', 'c', 'd'])
# (2, -1, ['a', 'b', 'ce', 'd'])
# (1, -1, ['a', 'bf', 'ce', 'd'])
# (0, -1, ['ag', 'bf', 'ce', 'd'])
# (1, 1, ['ag', 'bfh', 'ce', 'd'])
# (2, 1, ['ag', 'bfh', 'cei', 'd'])
# (3, 1, ['ag', 'bfh', 'cei', 'dj'])
# (2, -1, ['ag', 'bfh', 'ceik', 'dj'])
# (1, -1, ['ag', 'bfhl', 'ceik', 'dj'])
# (0, -1, ['agm', 'bfhl', 'ceik', 'dj'])
# (1, 1, ['agm', 'bfhln', 'ceik', 'dj'])


'''***inverse语句需记住！***Given a 32-bit signed integer, reverse digits of an integer.

Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
Note:
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.'''


class Solution_07(object):
    def inverse_int(self, x):
        if x < 0:
            return -self.inverse_int(-x)
        res = 0
        while x:
            res = res * 10 + x % 10 # do it the hard way (using mod % and integer division // to find each digit
            x //= 10                 # and construct the reverse number):
        return res if res <= 0x7fffffff else 0

# method 2:
class Solution_07_02(object):
    def __inverse_int(self, x):
        ''':type x: int
        :rtype: int '''
        # turning the number into a string, using slice notation to reverse the string and turning it back to an integer
        x = -int(str(x)[:,:,-1][:,-1]) if x < 0 else int(str(x)[:,:,-1])
        x = 0 if abs(x) > 0x7FFFFFFF else x
        return x

'''Input: "42"
Output: 42
Example 2:

Input: "   -42"
Output: -42
Explanation: The first non-whitespace character is '-', which is the minus sign.
             Then take as many numerical digits as possible, which gets 42.
Example 3:

Input: "4193 with words"
Output: 4193
Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.
Example 4:

Input: "words and 987"
Output: 0
Explanation: The first non-whitespace character is 'w', which is not a numerical 
             digit or a +/- sign. Therefore no valid conversion could be performed.
Example 5:

Input: "-91283472332"
Output: -2147483648
Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer.
             Thefore INT_MIN (−231) is returned.'''
# 需要考虑比较多的边界条件&特殊情况
#
# 首先输入可能会有空格，所以先去掉空格
# 去掉空格后要考虑空字符串情况
# 字符串首位可能会有正负号，要考虑
# 开始转换成数字，题目说只要遇到非数字就可以break了
# 结果太大或者太小超过int限制就要返回特定数字 2147483647 或者 -2147483648
# 根据之前的正负号结果返回对应数值

class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()
        strNum = 0
        if len(str) == 0:
            return strNum

        positive = True
        if str[0] == '+' or str[0] == '-':
            if str[0] == '-':
                positive = False
            str = str[1:]

        for char in str:
            if char >= '0' and char <= '9':
                strNum = strNum * 10 + ord(char) - ord('0')
            if char < '0' or char > '9':
                break

        if strNum > 2147483647:
            if positive == False:
                return -2147483648
            else:
                return 2147483647
        if not positive:
            strNum = 0 - strNum
        return strNum

'''Q9: 
Input: 121
Output: true
Example 2:

Input: -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
Follow up:'''
class Solution_09(object):
    def def_Palindrome_int(self, x):
        """
                :type x: int
                :rtype: bool
                """
        if x < 0 or (x != 0 and x % 10 == 0):
            return False
        rev, y = 0, x
        while x > 0:
            rev = rev * 10 + x % 10
            x //= 10   #整除！
        return y == rev

'''Q10, Unknown
https://leetcode.com/problems/regular-expression-matching/description/
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
Input:
s = "aa"
p = "a*"
Output: true
Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".

Example 3:
Input:
s = "ab"
p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".

Example 4:
Input:
s = "aab"
p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore it matches "aab".

Example 5:
Input:
s = "mississippi"
p = "mis*is*p*."
Output: false'''
# 先尝试暴力解法，难点就在 * 身上， * 不会单独出现，它一定是和前面一个字母或"."配成一对。看成一对后"X*"，它的性质就是：要不匹配
# 0个，要不匹配连续的“X”.所以尝试暴力解法的时候一个trick是从后往前匹配.
#
# 暴力解法居然也能AC?
#
# 是这样来分情况看得:
#
# 如果s[i] = p[j] 或者 p[j]= . ： 往前匹配一位
# 如果p[j] = ' * ', 检查一下，如果这个时候p[j-1] = . 或者p[j-1] = s[i] ，那么就往前匹配，如果这样能匹配过，就return True（注
# 意如果这样不能最终匹配成功的话我们不能直接返回False，因为还可以直接忽略' X* '进行一下匹配试试是否可行）， 否则我们忽略 ' X* '
# 这里注意里面的递推关系
# 再处理一下边界状况：
# s已经匹配完了， 如果此时p还有，那么如果剩下的是 X* 这种可以过，所以检查
# p匹配完毕，如果s还有那么报错
class Solution_10(object):
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        def helper(s, i, p, j):
            if j == -1:
                return i == -1
            if i == -1:
                if p[j] != '*':
                    return False
                return helper(s, i, p, j-2)
            if p[j] == '*':
                if p[j-1] == '.' or p[j-1] == s[i]:
                    if helper(s, i-1, p, j):
                        return True
                return helper(s, i, p, j-2)
            if p[j] == '.' or p[j] == s[i]:
                return helper(s, i-1, p, j-1)
            return False

        return helper(s, len(s)-1, p, len(p)-1)

'''背包问题——Dynamic Programming **************
假设我们有n件物品，分别编号为1, 2...n。其中编号为i的物品价值为vi，它的重量为wi。为了简化问题，假定价值和重量都是整数值。
现在，假设我们有一个背包，它能够承载的重量是W。现在，我们希望往包里装这些物品，
使得包里装的物品价值最大化，那么我们该如何来选择装的东西呢？'''

# 假定我们定义一个函数c[i, w]表示到第i个元素为止，在限制总重量为w的情况下我们所能选择到的最优解。那么这个最优解要么包含有i这个
# 物品，要么不包含，肯定是这两种情况中的一种。
# 如果我们选择了第i个物品，那么实际上这个最优解是c[i - 1, w-wi] + vi。
# 而如果我们没有选择第i个物品，这个最优解是c[i-1, w]。这样，实际上对于到底要不要取第i个物品，我们只要比较这两种情况，哪个的结果
# 值更大不就是最优
# 这一组物品分别有价值和重量，我们可以定义两个数组int[] v, int[] w。v[i]表示第i个物品的价值，w[i]表示第i个物品的重量。为了表示
# c[i, w]，我们可以使用一个int[i][w]的矩阵。
# 其中i的最大值为物品的数量，而w表示最大的重量限制。按照前面的递推关系，c[i][0]和c[0][w]都是0。而我们所要求的最终结果是c[n][w]。
# 所以我们实际中创建的矩阵是(n + 1) x (w + 1)的规格
# c[i][j] = max{c[i-1][j], c[i-1][j-value[i]] + value[i]}


def solve(vlist,wlist,totalWeight,totalLength):
    resArr = np.zeros((totalLength+1,totalWeight+1),dtype=np.int32)
    for i in range(1,totalLength+1):
        for j in range(1,totalWeight+1):
            if wlist[i] <= j:
                resArr[i,j] = max(resArr[i-1, j-wlist[i]]+vlist[i], resArr[i-1,j])
            else:
                resArr[i,j] = resArr[i-1,j]
    return resArr[-1,-1]

if __name__ == '__main__':
    v = [0,60,100,120]
    w = [0,10,20,30]
    weight = 50
    n = 3
    result = solve(v,w,weight,n)
    print(result)




