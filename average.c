#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]){

        FILE* fp1 = fopen(argv[1], "r");
        FILE* fp2 = fopen(argv[2], "w");

        int curr;
        int numbers = 0;
        int total = 0;

        while(fscanf(fp1, "%d", &curr) == 1)
        {
            numbers++;
            total += curr;
        }

        fprintf(fp2, "%d average: %d", numbers, total / numbers);

        exit(0);


}