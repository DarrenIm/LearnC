class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

'''Q.19
Unknown
'''


'''Q.20
https://leetcode.com/problems/valid-parentheses/
Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true
'''
# 因为一共只有三种状况"(" -> ")", "[" -> "]", "{" -> "}".
#
# 一遇到左括号就入栈，右括号出栈，这样来寻找对应
#
# 需要检查几件事：
#
# 出现右括号时stack里还有没有东西
# 出stack时是否对应
# 最终stack是否为空

class Solution_20(object):
    def valid_parentheses(self, s):
        left = '({['
        right = ')}]'
        stack = []
        for char in s:
            if char in left:
                stack.append(char)
            if char in right:
                if not stack:
                    return False
                tmp = stack.pop()
                if char == ')' and tmp != '(':
                    return False
                if char == ']' and tmp != '[':
                    return False
                if char == '}' and tmp != '{':
                    return False
        return stack == []


'''Q.21 [Unknown]: https://leetcode.com/problems/merge-two-sorted-lists/description/
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing 
together the nodes of the first two lists.

Example:

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
'''


class Solution_21(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1:
            return l2
        if not l2:
            return l1

        dummy = cur = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next

        cur.next = l1 if l1 else l2
        return dummy.next
'''Q.22  Backtracking!!! 忘记看解析： 
https://github.com/apachecn/awesome-algorithm/blob/master/docs/Leetcode_Solutions/Python/0022._generate_parentheses.md
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
'''
class Solution:
    def generate_Parenthesis(self, n):
        self.res = []
        self.singlestr('', 0, 0, n)
    def singlestr(self, s, left, right, n):
        if left == n and right == n:
            self.res.append(s)
        if left < n:
            self.singlestr(s+'(', left+1, right, n)
        if left > right:
            self.singlestr(s+')', left, right+1, n)


'''23. Merge k Sorted Lists 
难度: Hard

刷题内容
原题连接

https://leetcode.com/problems/merge-k-sorted-lists/description/
内容描述

Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6'''
#Brute force
# Definition for singly-linked list.
class Solution_23(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        self.nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next

'''24. Swap Nodes in Pairs
https://leetcode.com/problems/swap-nodes-in-pairs/
内容描述

Given a linked list, swap every two adjacent nodes and return its head.

Example:

Given 1->2->3->4, you should return the list as 2->1->4->3.
Note:

Your algorithm should use only constant extra space.
You may not modify the values in the list's nodes, only nodes itself may be changed.'''
#递归
class Solution_24(object):
    def swapnodes(self, head):
        if not head or not head.next:
            return head
        tmp = head.next
        head.next = self.swapnodes(head.next.next)
        tmp.next = head
        return tmp



'''25. Reverse Nodes in k-Group
https://leetcode.com/problems/reverse-nodes-in-k-group/
内容描述

Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.

k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.

Example:

Given this linked list: 1->2->3->4->5

For k = 2, you should return: 2->1->4->3->5

For k = 3, you should return: 3->2->1->4->5

Note:

Only constant extra memory is allowed.
You may not alter the values in the list's nodes, only nodes itself may be changed.'''

#TODO

'''26. Remove Duplicates from Sorted Array'''
def remove_duplicates(l):
    idx = 0
    while idx < (len(l) - 1):
        if l[idx] == l[idx+1]:
            l.pop(idx)
        else:
            idx += 1
        return len(l)
# ****注意 for 和 while 的区别:*****
n = 10
i = 6
while i < n:
    n = n - 1
    i += 1
    print(i) # return 7, 8
for j in range(n):
    n = n - 1
    print(n) # return 9 ~ 0