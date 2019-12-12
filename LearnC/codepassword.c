#include<stdio.h>
int main()
{
    char init[9] = {'S','h','i','Y','a','n','L','o','u'};
    for (int i = 0; i < sizeof(init); i++)
    {
        /* code */
        init[i] += 1;
    }
    for (int i = 0; i < sizeof(init); i++)
    {
        /* code */
        printf("%c", init[i]);
    }
    printf("\n");
    for (int i = 0; i < sizeof(init); i++)
    {
        putchar(init[i]);
    }
    printf("\n");
    return 0;    
}