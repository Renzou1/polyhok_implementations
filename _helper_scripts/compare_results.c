// checks if all of the numbers of two files are the same

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]){

        FILE* fp1 = fopen(argv[1], "r");
        FILE* fp2 = fopen(argv[2], "r");

        float cuda_number;
        float polyhok_number;
        int index = 0;
        float TOL = 0.0005;
        int ret1, ret2;

        while (1)
        {
                int r1 = fscanf(fp1, "%f", &cuda_number);
                int r2 = fscanf(fp2, "%f", &polyhok_number);
                if (r1 == 1 && r2 == 1) {
                        if(fabs(cuda_number - polyhok_number) > TOL)
                        {
                                printf("Error! expected %f, got %f at index %d", cuda_number, polyhok_number, index);
                                exit(1);
                        }
                }  else if (r1 == EOF && r2 == EOF) {
                // Both files ended correctly
                printf("Files are identical in size\n");
                break;
                } else if (r1 == EOF) {
                printf("File 1 ended early at index %d\n", index);
                break;
                } else if (r2 == EOF) {
                printf("File 2 ended early at index %d\n", index);
                break;
                } else {
                // One of them failed to parse a float (bad format)
                printf("Format mismatch or read error at index %d\n", index);
                break;
                }
                index++;

        }

        printf("Success! All numbers are equal.\n");
        exit(0);


}