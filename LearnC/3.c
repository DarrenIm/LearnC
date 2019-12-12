/* book management system */
#include<stdio.h>
#include<stdlib.h>

int main()
{
    int cardnum;
    char name[20];
    printf("******************************************\n");
    printf("welcome\n");
    printf("~~~~~\t\t\t~~~~~~\n");
    printf("please enter your card number\n");
    // scanf("%d", &cardnum);
    printf("please enter your name\n");
    if (fgets(name, sizeof(name), stdin) == NULL) {
    // TODO: Read failed: handle this.
        printf("read failed\n");
    }
    printf("\nWelcome, %s! Your card number is %d", name, cardnum);
    return 0;
}