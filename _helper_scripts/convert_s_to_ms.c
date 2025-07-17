#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]){

        FILE* fp1 = fopen(argv[1], "r");
        FILE* fp2 = fopen(argv[2], "w");

        double in_seconds;

        while(fscanf(fp1, "%lf", &in_seconds) == 1)
        {
            fprintf(fp2, "%.0lf\n", in_seconds * 1000);

        }

        exit(0);


}