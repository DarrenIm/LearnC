#include<stdio.h>
#include<stdlib.h>

int main(){
int *ar, i, s = 1;
int sum = 0;

ar = (int*)malloc(sizeof(int));
do{
scanf("%d", &i);
if(i == -1) break;
sum += i;
ar[s-1] = i;

realloc(ar, ++s);
}while(1);
float aver = sum/(s-2);
printf("the average is: %f\n", aver);
int below_average;
for(i = 0; i < s - 1; i++) 
{
    printf("%d\n", ar[i]);
    if (ar[i]<aver)
    {
        below_average += 1;
    }
}
printf("below average is: %d\n", below_average);
free(ar);
return 0;
}
