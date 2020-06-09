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



if __name__ == '__main__':
    s = Solution1184()
    print(s.betterAnother([1,2,3,4], 0, 2))