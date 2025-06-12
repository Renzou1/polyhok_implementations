#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]){

        FILE* fp1 = fopen(argv[1], "r");
        FILE* fp2 = fopen(argv[2], "r");

        float cuda_number;
        float polyhok_number;

        while(fscanf(fp1, "%f", &cuda_number) == 1 && fscanf(fp2, "%f", &polyhok_number) == 1)
        {
                if(cuda_number != polyhok_number)
                {
                        printf("Error! expected %f, got %f", cuda_number, polyhok_number);
                        exit(1);
                }

        }

        printf("Success! All numbers are equal.\n");
        exit(0);


}