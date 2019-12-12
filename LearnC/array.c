#include<stdio.h>
int main(){
    int array[10];
    for (int i = 0; i < 10; i++)
    {
        array[i] = i + 1;
    }
    for (int i = 9; i >= 0; i--)
    {
        printf("%d", array[i]);
    }
    getchar();
    return 0;
}