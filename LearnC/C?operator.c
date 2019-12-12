#include<stdio.h>
int main(){
    float dis1 = 0.15;
    float dis2 = 0.1;
    float dis3 = 0.05;
    int number1 = 50;
    int number2 = 20;
    int number3 = 10;
    int input;
    printf("how much staff you want?\n");
    scanf("%d", &input);
    printf("discount is %f\n", (input>number1?dis1:(
                                    input>number2?dis2:(
                                        input>number3?dis3:1))));
    getchar();
    return 0;
}