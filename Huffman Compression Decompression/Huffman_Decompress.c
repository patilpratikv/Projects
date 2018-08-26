//#include <stdio.h>
//#include <conio.h>
//#include <stdlib.h>
//#include <windows.h>
//#include <commdlg.h>
//#include <math.h>
//#include <stdbool.h>
//
//// Preprocessors
//#define DEBUG 0
////-------------------------------------------------------
//
//// Structures
//struct temp{
//    char letter;
//    char code[10];
//};
////-------------------------------------------------------
//
//// Function Declaration
//void decode_file(char*);
//char *decodeBuffer(char, char, unsigned char, struct temp*);
//char *int2string(long int);
//int match(char[], char[], int);
//char* intTostring(int);
////-------------------------------------------------------
//
////int2string
//char* intTostring(int i){
//    //char returnstring[50] = {0};
//    char* str = NULL;
//    char* dest = NULL;
//    str = malloc(sizeof(char)*64);
//    dest = malloc(sizeof(char)*64);
//    memset(str,'\0',sizeof(char)*64);
//    memset(dest,'\0',sizeof(char)*64);
//    itoa (i, str, 2);
//    //printf("binary: %s\n", str);
//    if (strlen(str) < 16){
//        sprintf(dest, "%.*d%s", (int)(16-strlen(str)), 0, str);
//        free(str);
//        //printf("binary: %s\n", dest);
//        return dest;
//    }
//    else if (strlen(str) > 16){
//        strncpy(dest, str+(strlen(str)-16), 16);
//        //printf("binary: %s\n", dest);
//        free(str);
//        return dest;
//    }
//    else{
//        //printf("binary: %s\n", str);
//        free(dest);
//        return str;
//    }
//}
//// Match with character limit
//int match(char a[],char b[],int limit){
//
//	b[strlen(a)]='\0';
//	b[limit]='\0';
//
//	return strcmp(a,b);
//}
//
//// Integer to string Conversion
//char *int2string(long int n){
//
//	long int i = 0, k = 0, and = 0, j = 0;
//	char *temp = NULL;
//	temp = malloc(16*sizeof(char));
//
//	for(i = 15; i >= 0; i--){
//		and = 1<< i;
//		k = n & and;
//		if(k==0){
//			temp[j++]='0';
//		}
//		else{
//			temp[j++]='1';
//		}
//	}
//	temp[j] = '\0';
//
//	return temp;
//}
//
//// Decode String back to character
//char *decodeBuffer(char b, char padding, unsigned char n, struct temp* code){
//
//    int i = 0, j = 0, t = 0;
//    static int k, FirstTime;
//    static int buffer;
//    char* decoded = malloc(sizeof(char)*256);
//    char* tempstring = NULL;
//
//    memset(decoded, '\0', sizeof(char)*256);
//
//    t = (int)b;
//    #if DEBUG
//    printf("t = %s\tk = %d\n",int2string(t),k);
//    printf("t = %s\tk = %d\n",intTostring(t),k);
//    #endif // DEBUG
//    t = t & 0x00FF;		//mask high byte
//    #if DEBUG
//    printf("t = %s\tk = %d\n",int2string(t),k);
//    printf("t = %s\tk = %d\n",intTostring(t),k);
//    #endif // DEBUG
//    t = t << (8 - k);
//    #if DEBUG		//shift bits keeping zeroes for old buffer
//    printf("t = %s\tk = %d\n",int2string(t),k);
//    printf("t = %s\tk = %d\n",intTostring(t),k);
//    #endif // DEBUG
//    buffer = buffer | t;	//joined b to buffer
//    k = k + 8;			//first useless bit index +8 , new byte added
//    if(!FirstTime){
//        buffer = buffer<<padding;
//        k = 8 - padding;	//k points to first useless bit index
//        padding = 0;
//        FirstTime = 1;
//    }
//    #if DEBUG
//    printf("buffer = %s\tk = %d\n",int2string(buffer),k);
//    printf("buffer = %s\tk = %d\n",intTostring(buffer),k);
//    #endif // DEBUG
////    //loop to find matching codewords
//    while(i < n){
//        tempstring = intTostring(buffer);
//        if(!match(code[i].code, tempstring,k)){
//            decoded[j++] = code[i].letter;	//match found inserted decoded
//            t = strlen(code[i].code);	//matched bits
//            buffer = buffer<<t;		//throw out matched bits
//            k = k - t;				//k will be less
//            i = 0;				//match from initial record
//            #if DEBUG
//            printf("Buffer = %s,removed = %c,k = %d\n",int2string(buffer),decoded[j-1],k);
//            #endif // DEBUG
//            if(k == 0) break;
//            continue;
//        }
//        free(tempstring);
//        i++;
//    }
////    decoded[j] = '\0';
//
//    return decoded;
//}
//
//// Read .dat File
//void decode_file(char* filename){
//
//    unsigned char N;
//    char buffer, paddingbits;
//    int i = 0;
//    char* decoded;
//    FILE* fp;
//    FILE* outfile;
//    struct temp* code;
//
//    fp = fopen(filename,"rb");
//    outfile = fopen("Decompress_Text.txt", "wt+");
//    fread(&buffer,sizeof(unsigned char),1,fp);
//    #if DEBUG
//    printf("\nDetected: %u different characters.",buffer);
//    #endif // DEBUG
//    N = buffer;
//    //allocate memory for mapping table
//    code = malloc(sizeof(struct temp)*(N+1));
//    memset(code, '\0', sizeof(struct temp)*(N+1));
//    fread(code, sizeof(struct temp), N, fp);
//    #if DEBUG
//    for(int i = 0; i < N; i++)
//        printf("[%c|%s] ", code[i].letter, code[i].code);
//    #endif // DEBUG
//    fread(&buffer, sizeof(char), 1, fp);
//    paddingbits = buffer;
//    #if DEBUG
//    printf("\nDetected: %u bit padding.",paddingbits);
//    #endif // DEBUG
//    while(fread(&buffer, sizeof(char), 1, fp) != 0){
//        #if DEBUG
//        printf("\nReading: %u",buffer);
//        #endif // DEBUG
//        decoded = decodeBuffer(buffer, paddingbits, N, code);	//decoded is pointer to array of characters read from buffer byte
//        i = 0;
//        while(decoded[i++]!='\0');	//i-1 characters read into decoded array
//        //#if DEBUG
//        printf("message: %s\n",decoded);
//        //#endif // DEBUG
//        fwrite(decoded,sizeof(char),i-1,outfile);
//        free(decoded);
//    }
//    fclose(fp);
//    fclose(outfile);
//    free(code);
//}
//
//int main(void)
//{
//
//    decode_file("a_text_file_compressed.dat");
//    getch();
//    return 0;
//}
