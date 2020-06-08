'''Leetcode #11 - Contain most water
ai和aj (i<j) 组成的container的面积：S(i,j) = min(ai, aj) * (j-i)
当height[left] < height[right]时，对任何left < j < right来说
   1.min(height[left], height[j]) <= height[left] = min(height[left], height[right])
   2.j - left < right - left
所以S(left, right) = min(height[left], height[right]) * (right-left) > S(left, j) = min(height[left], height[j]) * (j-left)

这就排除了所有以left为左边界的组合，因此需要右移left。

因此：

height[left] < height[right]，需要右移left.

同理，当height[left] > height[right]时，需要左移right。

而当height[left] = height[right]时，需要同时移动left和right。

思路整理： left = 0, right = n-1

height[left] < height[right], left++
height[left] > height[right], right--
height[left] = height[right], left++, right--
终止条件：left > right
'''

class Solution_11(object):
    def contains_most_water(self, height):
        """
               :type height: List[int]
               :rtype: int
               """
        left, right, most_water = 0, len(height) - 1, 0
        while left <= right:
            water = (right - left) * min(height[left], height[right])
            most_water = max(water, most_water)
            if height[left] < height[right]:
                left += 1
            elif height[left] > height[right]:
                right -= 1
            else:
                left += 1
                right -= 1
        return most_water

'''Leetcode #12 - Integer to Roman'''

class Solution_12(object):
    def int2Roman(self, number):
        lookup = {
            'M': 1000,
            'CM': 900,
            'D': 500,
            'CD': 400,
            'C': 100,
            'XC': 90,
            'L': 50,
            'XL': 40,
            'X': 10,
            'IX': 9,
            'V': 5,
            'IV': 4,
            'I': 1
        }
        roman = ''
        for symbol, val in sorted(lookup.items(), key=lambda t: t[1])[::-1]:
            while number >= val:
                roman += symbol
                number -= val
        return roman

''' 13.Roman to Integer:
从前往后扫描，用一个临时变量记录分段数字。
如果当前比前一个大，说明这一段的值应当是这个值减去上一个值。比如IV = 5-1 =4; 否则，将当前值加入到结果中，然后开始下一段记录，
比如VI = 5 + 1, II = 1 +1'''

class Solution_13(object):
    def roman2int(self, string):
        lookup = {
            'M': 1000,
            'CM': 900,
            'D': 500,
            'CD': 400,
            'C': 100,
            'XC': 90,
            'L': 50,
            'XL': 40,
            'X': 10,
            'IX': 9,
            'V': 5,
            'IV': 4,
            'I': 1
        }
        res = 0
        for i in range(len(string)):
            if i > 0 and lookup[string[i]] > lookup[string[i-1]]:
                res += lookup[string[i]] - 2 * lookup[string[i-1]]
            else:
                res += lookup[string[i]]
        return res

'''14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
Note:'''
# dp[i]代表前i+1个字符串的最大前缀串，
# 如果第i+2个字符串不以dp[i]为前缀，就去掉dp[i]的最后一个字符再试一次
# 都去完了那么output肯定就是空串了，也就等于这时候的dp[i]，因为dp[i]的每个字符已经被去完了

class Solution_14(object):
    def longest_common_prefix(self, strs):
        if not strs:
            return ''
        dp = strs[0] * len(strs)
        for i in range(1, len(strs)):
            while not strs[i].startswith(dp[i-1]):
                dp[i-1] = dp[i-1][:-1]
            dp[i] = dp[i-1]
        return dp[-1]

'''15. 3 sum
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets 
in the array which gives the sum of zero.

Note:

The solution set must not contain duplicate triplets.

Example:

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
'''
class Solution_15_01(object):
    def three_sums(self, nums):
        nums.sort()
        res = []
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    if nums[i] + nums[j] + nums[k] == 0 and j != i and k != j and i != k:
                        curres = [nums[i], nums[j], nums[k]]
                        if curres not in res:
                            res.append(curres)
        return res

class Solution_15_02(object):
    def threesum(self, nums):
        n, res = len(nums), []
        nums.sort()
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, n-1
            while l < r:
                tmp = nums[i] + nums[l] + nums[r]
                if tmp == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                elif tmp > 0:
                    r -= 1
                else:
                    l += 1
        return res

'''16. 3Sum Closest
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. 
Return the sum of the three integers. You may assume that each input would have exactly one solution.

Example:

Given array nums = [-1, 2, 1, -4], and target = 1.

The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).'''

class Solution_16(object):
    def closest_sum(self, num, target):
        num.sort()
        n, res, diff = len(num), None, float('inf')
        for i in range(n):
            if i > 0 and num[i] == num[i-1]:
                continue
            l, r = i+1, n-1
            while l < r:
                tmp = num[i] + num[l] + num[r]
                if tmp - target == 0:
                    return target
                elif tmp > target:
                    r -= 1
                    if abs(tmp - target) < diff:
                        diff = abs(tmp - target)
                        res = tmp
                    while l < r and num[r] == num[r-1]:
                        r -= 1
                else:
                    l += 1
                    if abs(tmp - target) < diff:
                        diff = abs(tmp - target)
                        res = tmp
                    while l < r and num[l] == num[l+1]:
                        l += 1
        return res

class Solution_17(object):

    # hash table一个，用来对应digit -> letter
    # s用来记录结果，每次从digits里面去一个，然后寻找其可能的char，加到s中，digits长度减小
    # digits长度为0时候，把它加入结果
    def lettercombinations(self, digits):
        """
        https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
        :type digits: str
        :return: List[str]
        """
        lookup = {
            '2':['a','b','c'],
            '3':['d','e','f'],
            '4':['g','h','i'],
            '5':['j','k','l'],
            '6':['m','n','o'],
            '7':['p','q','r','s'],
            '8':['t','u','v'],
            '9':['w','x','y','z']
        }
        res = []
        def helper(s, digits):
            if len(digits) == 0:
                res.append(s)
            else:
                current_digit = digits[0]
                for char in lookup[current_digit]:
                    helper(s+char, digits[1:])
        if not digits or len(digits) == 0:
            return res
        helper('', digits)
        return res

'''Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such 

that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

Note:

The solution set must not contain duplicate quadruplets.

Example:

Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.

A solution set is:
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]'''

class Solution_18:
    def four_sum(self, nums, target):
        n, res = len(nums), []
        nums.sort()
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            for j in range(i+1, n):
                if j > 0 and nums[j] == nums[j-1]:
                    continue
                r, l = j+1, n-1
                while r < l:
                    tmp = nums[i] + nums[j] + nums[r] + nums[l]
                    if tmp == target:
                        if tmp not in res:
                            res.append([nums[i], nums[j], nums[r], nums[l]])
                            r +=1
                            l -= 1
                    elif tmp < target:
                        r += 1
                    else:
                        l -= 1






