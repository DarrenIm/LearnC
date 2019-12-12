#include<stdio.h>
#include<string.h>
int main(int argc, char *argv[]){ 
    if (argc < 2)
    {
        printf("Error, enter more number please! \n");
        return 1;
    }
    int inputarray[argc-1]; // 此处需考虑指针问题：argv[0]默认没有数据(值为0), pass
    // printf("%d\n", atoi(argv[0]));
    for (int i = 1; i < argc; i++) 
    {
        inputarray[i-1] = atoi(argv[i]);
    }
    int array_size = sizeof(inputarray)/sizeof(inputarray[0]);
    printf("%d\n", array_size);
    printf("the Unsorted array is: \n");
    for (int i = 0; i < array_size; i++)
    {
        printf("%d\t", inputarray[i]);
    }
    printf("\n");
    // sort
    for (int i = 0; i < array_size-1; i++)
    {
        for (int j = 0; j<array_size-1-i; j++)
        {
            if (inputarray[j] <= inputarray[j+1])
            {
                continue;
            } else
            {   
                // switch
                int tmp = inputarray[j];
                inputarray[j] = inputarray[j+1];
                inputarray[j+1] = tmp;
            }
        }
    }
    printf("the sorted array is: \n");
    for (int i = 0; i < array_size; i++)
    {
        printf("%d\t", inputarray[i]);
    }
    getchar();
    return 0;
}