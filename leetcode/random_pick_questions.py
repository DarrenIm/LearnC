import math

import numpy as np

'''
Examples:
Input: answers = [1, 1, 2]
Output: 5
Explanation:
The two rabbits that answered "1" could both be the same color, say red.
The rabbit than answered "2" can't be red or the answers would be inconsistent.
Say the rabbit that answered "2" was blue.
Then there should be 2 other blue rabbits in the forest that didn't answer into the array.
The smallest possible number of rabbits in the forest is therefore 5: 3 that answered plus 2 that didn't.

Input: answers = [10, 10, 10]
Output: 11

Input: answers = []
Output: 0

[0,0,1,1,1]
6
[00010]
6
Solution 1
If x+1 rabbits have same color, then we get x+1 rabbits who all answer x.
now n rabbits answer x.
If n % (x + 1) == 0, we need n / (x + 1) groups of x + 1 rabbits.
If n % (x + 1) != 0, we need n / (x + 1) + 1 groups of x + 1 rabbits.
'''
class Solution:
    def numRabbits(self, answers) -> int:
        if len(answers) == 0:
            return 0
        else:
            un, count = np.unique(answers, return_counts=True)
            total = 0
            for i,c in zip(un, count):
                total += math.ceil(c / (i + 1)) * (i + 1)
            return total

'''
1184. Distance Between Bus Stops
A bus has n stops numbered from 0 to n - 1 that form a circle. We know the distance between all pairs of neighboring stops where distance[i] is the distance between the stops number i and (i + 1) % n.

The bus goes along both directions i.e. clockwise and counterclockwise.

Return the shortest distance between the given start and destination stops.

e.x.
Input: distance = [1,2,3,4], start = 0, destination = 1
Output: 1
Explanation: Distance between 0 and 1 is 1 or 9, minimum is 1.

'''
class Solution1184:
    def shortestDist(self, l, start, end):
        if start == end:
            return 0
        elif start > end:
            return min(sum(l[end:start]), (sum(l[0:end]) + sum(l[start:])))
        else:
            return min(sum(l[start:end]), (sum(l[0:start]) + sum(l[end:])))

    def betterAnother(self, l, start, end):
        a = min(start, end)
        b = max(start, end)
        return min(sum(l[a:b]), sum(l) - sum(l[a:b]))

'''
60. Permutation Sequence
The set [1,2,3,...,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for n = 3:

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.

Note:

Given n will be between 1 and 9 inclusive.
Given k will be between 1 and n! inclusive.
Example 1:

Input: n = 3, k = 3
Output: "213"
Example 2:

Input: n = 4, k = 9
Output: "2314"
'''
class Solution60:

    def permutations(self, arr, position, end):

        if position == end:
            print(arr)

        else:
            for index in range(position, end):
                if index == position:
                    print(index, position)
                    print(True)
                else:
                    print(index, position)
                    print(False)
                arr[index], arr[position] = arr[position], arr[index]
                self.permutations(arr, position + 1, end)
                # 回溯, 继续执行递F(x)时未完成的语句
                arr[position], arr[index] = arr[index], arr[position]


"""全排列 用递归方法
全排列：
1、列表只有一个元素[a]，它的全排列只有a。
2、列表有两个元素[a, b]，它的全排列为[a, b], [b, a]：
    { 将第一个元素a固定，对b进行全排列得到[a, b]。
    将第一个元素与第二个元素交换得到[b, a]。
    将b固定，对a进行全排列，得到[b, a] }
3、列表有三个元素[a, b, c]
    { 将a固定，对bc进行全排列{ 将b固定，对c全排列[abc]。交换bc，将c固定对b进行全排列[acb] }
     交换ab，[b, a, c] 对ac进行全排列{ ... }
     ... ...}
4、列表有n个元素，将第一个元素固定，对剩下n - 1个元素进行全排列。
    将第一个元素依此与其他元素交换，对每次交换后剩下的n-1个元素进行全排列。
5、对剩下的n - 1个元素全排列，同上，固定后对n - 2排列。
6、直到数组数量为1，全排列就是它自己，完成一次排列。
j = begin
for i in range(begin, end):  # 对begin到end的数组进行i次交换。
    data[i], data[j] = data[j], data[i]  # 交换。
    perm(data, begin + 1, end)  # 交换后对剩下数组进行全排列。[begin + 1, end]
    data[i], data[j] = data[j], data[i]  # 全排列完成后，换回原来的顺序，回到for进行下一次交换。
    """


def perm(data, begin, end):
    if begin == end:  # 递归结束条件，当交换到最后一个元素的时候不需要交换，1的全排列还是1。
        print(data)  # 打印一次排列完成后的数组。
    else:
        j = begin
        for i in range(begin, end):  # 从begin到end全排列。
            data[i], data[j] = data[j], data[i]
            perm(data, begin + 1, end)
            data[i], data[j] = data[j], data[i]  # 递归完成后，交换回原来的位置。

#
# arr = [1, 2, 3, 4, 5]
# perm(arr, 0, len(arr))


class Solution_perm:
    # 应该可以用回溯法
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 定义保存最后结果的列表集合
        final_answer = []
        nums_length = len(nums)

        def back(answer=[]):
            # 如果临时结果集中的元素和给定数组nums长度相等，说明拿到了结果
            if len(answer) == nums_length:
                final_answer.append(answer)
                return
            for index in nums:
                if index not in answer:
                    back(answer+[index])
        back()
        return final_answer





if __name__ == '__main__':
    # s = Solution1184()
    # print(s.betterAnother([1,2,3,4], 0, 2))

    # s = Solution60()
    # s.perm_generator([1,2,3,4], [])

    s = Solution_perm()
    print(s.permute([1, 2, 3]))









