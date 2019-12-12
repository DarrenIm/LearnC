#include<stdio.h>
#include<math.h>
#include<stdlib.h>

int main()
{
    double a;
    double b;
    double c;
    double k;
    printf("please\n");
    scanf("%lf%lf%lf",&a,&b,&c);
    printf("%lf", a);
    // printf("Please enter a value\n");
    // scanf("%d", &a);
    // printf("Please enter b value\n");
    // scanf("%d", &b);
    // printf("Please enter c value\n");
    // scanf("%d", &c);
    k = b*b - 4*a*c;
    printf("%d\n", k);
    if (k >= 0)
    {
        /* code */
        printf("x1: %f\n", (-b + sqrt(k))/(2*a));
        printf("x1: %f\n", (-b - sqrt(k))/(2*a));
        
    } else{
        printf("No shishu ans\n");
    }
    getchar();
    return 0;
}