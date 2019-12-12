#include<stdio.h>
#include<stdlib.h>

int main(int argc, char *argv[])
{   
    int sum = 0;
    if (argc < 2)
    {   
        printf("you need input at least one arg \n");
        return 1;
    }
    for (char **i = argv + 1; *i != NULL; i++)
    {   
        printf("str is : %s \n", *i);
        sum += atoi(*i);
        for (char *p = *i; *p != '\0'; p++)
        {
            printf("num is : %c \n", *p);
        }
        
    }
    printf("sum is %d\n", sum);
    getchar();
    return 0;
    
}