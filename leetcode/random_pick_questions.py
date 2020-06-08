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

if __name__ == '__main__':
    s = Solution()
    print(s.numRabbits([4,0,0,2,4]))