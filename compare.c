#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){

        FILE* fp1 = fopen(argv[1], "r");
        FILE* fp2 = fopen(argv[2], "r");

        float cuda_number;
        float polyhok_number;
        int index = 0;

        while(fscanf(fp1, "%f", &cuda_number) == 1 && fscanf(fp2, "%f", &polyhok_number) == 1)
        {
                if(abs(cuda_number - polyhok_number) > 0.0000005)
                {
                        printf("Error! expected %f, got %f at index %d", cuda_number, polyhok_number, index);
                        exit(1);
                }
                index++;

        }

        printf("Success! All numbers are equal.\n");
        exit(0);


}