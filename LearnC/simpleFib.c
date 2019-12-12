#include<stdio.h>
#include<stdlib.h>
int main()
{
    char sp = " ";
    char c = "*";
    float line = 7;
    int l = 7;
    int zeroline = ceil(line/2);
    if (sizeof(l/2) == sizeof(int))
    {
        printf("int\n");
    }
    printf("%d\n", zeroline);
    for (int i = 1; i <= line; i++)
    {
        for (int j = 0; j < abs(zeroline - i); j++)
        {
            /* code */
            printf(" ");
        }
        for (int k = 0; k < line - 2*abs(zeroline - i); k++)
        {
            printf("*");
        }
        for (int j = 0; j < abs(zeroline - i); j++)
        {
            /* code */
            printf(" ");
        }
        printf("\n");
    }
}