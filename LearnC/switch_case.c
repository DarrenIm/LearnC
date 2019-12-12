#include<stdio.h>
int main()
{   
    char grade;
    printf("please enter your score\n");
    scanf("%c", &grade);
    switch (grade)
    {
    case 'a':
        /* code */
        printf("90-100\n");
        break;
    case 'b':
        /* code */
        printf("80-90\n");
        break;
    case 'c':
        /* code */
        printf("70-80\n");
        break;
    default:
        printf("input grade error\n");
        break;
    }
    getchar();
    return 0;
}