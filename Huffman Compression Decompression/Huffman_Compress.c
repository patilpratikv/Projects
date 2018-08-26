#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <windows.h>
#include <commdlg.h>
#include <math.h>
#include <stdbool.h>

// Preprocessors
#define DEBUG 0
//-------------------------------------------------------

// Global Variables
char* FileTitle = NULL;
// Structures
// Structure for letter array and respective frequency
struct LetterFreqData{
    int count;
    char* letter;
    int* freq;
    float* probab;

};

// Tree Node structure
struct node{
    char letter;
    int node_count;
    int frequency;
    int status;
    int left;
    int right;
    float probab;
};

// Huffman Code
struct HuffCode{
    char letter;
    char* code;
};

struct temp{
    char letter;
    char code[10];
};
//-------------------------------------------------------
// Function Declaration
char* getFileName();
struct LetterFreqData letter_freq(char*);
float twoPrecision(float, float);
int compare (const void*, const void*);
bool isvalueinarray(int, char*, int);
struct node* construct_huffman_tree(struct node*, int);
int get_low_probab(struct node*, int, int);
void show_tree(struct node*, int);
struct HuffCode* construct_huffman_code(struct node*, int);
char* return_code(struct node*, int, int, char*, int);
char* ReverseConstString(char*);
void write_encoded_file( char*, struct HuffCode*, struct LetterFreqData);
void writeBit(int, FILE*);
void decode_file(char*);
char *decodeBuffer(char, char, unsigned char, struct temp*);
char *int2string(int);
int match(char[], char[], int);
//-------------------------------------------------------

// Match with character limit
int match(char a[],char b[],int limit){

	b[strlen(a)]='\0';
	b[limit]='\0';

	return strcmp(a,b);
}

// Integer to string Conversion
char *int2string(int n){

	int i, k, and, j = 0;
	char *temp = malloc(16*sizeof(char));

	for(i = 15; i >= 0; i--){
		and = 1<< i;
		k = n & and;
		if(k==0){
			temp[j++]='0';
		}
		else{
			temp[j++]='1';
		}
	}
	temp[j] = '\0';

	return temp;
}

// Decode String back to character
char *decodeBuffer(char b, char padding, unsigned char n, struct temp* code){

    int i = 0, j = 0, t;
    static int k, FirstTime;
    static int buffer;
    char *decoded = malloc(sizeof(char)*256);

    t = (int)b;
    #if DEBUG
    printf("t = %sk = %d\n",int2string(t),k);
    #endif // DEBUG
    t = t & 0x00FF;		//mask high byte
    #if DEBUG
    printf("t = %sk = %d\n",int2string(t),k);
    #endif // DEBUG
    t = t << (8 - k);
    #if DEBUG		//shift bits keeping zeroes for old buffer
    printf("t = %sk = %d\n",int2string(t),k);
    #endif // DEBUG
    buffer = buffer | t;	//joined b to buffer
    k = k + 8;			//first useless bit index +8 , new byte added
    if(!FirstTime){
        buffer = buffer<<padding;
        k = 8 - padding;	//k points to first useless bit index
        padding = 0;
        FirstTime = 1;
    }
    #if DEBUG
    printf("buffer = %s, k = %d\n",int2string(buffer),k);
    #endif // DEBUG
    //loop to find matching codewords
    while(i < n){
        if(!match(code[i].code, int2string(buffer),k)){
            decoded[j++] = code[i].letter;	//match found inserted decoded
            t = strlen(code[i].code);	//matched bits
            buffer = buffer<<t;		//throw out matched bits
            k = k - t;				//k will be less
            i = 0;				//match from initial record
            #if DEBUG
            printf("Buffer = %s,removed = %c,k = %d\n",int2string(buffer),decoded[j-1],k);
            #endif // DEBUG
            if(k == 0) break;
            continue;
        }
    i++;
    }
    decoded[j] = '\0';

    return decoded;
}
// Read .dat File
void decode_file(char* filename){

    unsigned char N;
    char buffer, paddingbits;
    int i = 0;
    char* decoded;
    FILE* fp;
    FILE* outfile;
    struct temp* code;

    fp = fopen(filename,"rb");
    outfile = fopen("Decompress_Text.txt", "wt+");
    fread(&buffer,sizeof(unsigned char),1,fp);
    #if DEBUG
    printf("\nDetected: %u different characters.",buffer);
    #endif // DEBUG
    N = buffer;
    //allocate memory for mapping table
    code = malloc(sizeof(struct temp)*(N+1));
    memset(code, '\0', sizeof(struct temp)*(N+1));
    fread(code, sizeof(struct temp), N, fp);
    #if DEBUG
    for(int i = 0; i < N; i++)
        printf("[%c|%s] ", code[i].letter, code[i].code);
    #endif // DEBUG
    fread(&buffer, sizeof(char), 1, fp);
    paddingbits = buffer;
    #if DEBUG
    printf("\nDetected: %u bit padding.",paddingbits);
    #endif // DEBUG
    while(fread(&buffer, sizeof(char), 1, fp) != 0){
        #if DEBUG
        printf("\nReading: %u",buffer);
        #endif // DEBUG
        decoded = decodeBuffer(buffer, paddingbits, N, code);	//decoded is pointer to array of characters read from buffer byte
        i = 0;
        while(decoded[i++]!='\0');	//i-1 characters read into decoded array
        //#if DEBUG
        printf("message: %s\n",decoded);
        //#endif // DEBUG
        fwrite(decoded,sizeof(char),i-1,outfile);
    }
    fclose(fp);
    fclose(outfile);
}

// Write individual bit to .dat file
void writeBit(int b,FILE *f){

	static char byte;
	static int cnt;
	char temp;

	if(b == 1){
        temp=1;
		temp=temp<<(7-cnt);		//right shift bits
        byte=byte | temp;
    }

	cnt++;
	if(cnt==8){
		fwrite(&byte,sizeof(char),1,f);
		cnt=0; byte=0;	//reset buffer
		return;// buffer written to file
	}

	return;
}

// Write encoded file
void write_encoded_file(char* filename, struct HuffCode* code, struct LetterFreqData letter_frq){

    unsigned char uniquechar = letter_frq.count;
    char padding, ch;
    int temp = 0;
    char* compressedFileName;
    char charcode[256];

    FILE* fout;
    FILE* txt_file;
    struct temp* tempstruct;

    compressedFileName = malloc(sizeof(char)*30);
    memset(compressedFileName, '\0', sizeof(char)*30);
    compressedFileName = strtok(FileTitle, ".");
    strcat(compressedFileName, "_compressed.dat");

    txt_file = fopen (filename, "rt");

    if ((fout = fopen(compressedFileName, "wb")) == NULL) {
        perror("Failed to open output file");
        fclose(txt_file);
        return;
    }

    fwrite(&uniquechar, sizeof(unsigned char), 1, fout);	//read these many structures while reading
    #if DEBUG
    printf("\nuniquechar=%u\n",uniquechar);
    #endif // DEBUG
    tempstruct = malloc(sizeof(struct temp)*uniquechar);
    for (int i = 0; i < uniquechar ; i++){
        memset(tempstruct[i].code, '\0', 10);
        strncpy(tempstruct[i].code, code[i].code, 9);
        tempstruct[i].letter = code[i].letter;
    }
    for (int i = 0; i < uniquechar ; i++){
        fwrite(&tempstruct[i], sizeof(struct temp), 1, fout);
        //fwrite(&code[i].letter, sizeof(char), 1, fout);
        //fwrite(code[i].code, sizeof(char), 1, fout);
        temp += strlen(tempstruct[i].code) * letter_frq.freq[i];
        temp %= 8;
        #if DEBUG
        printf("Code : %s\n", tempstruct[i].code);
        printf("Letter Freq : %d\n", letter_frq.freq[i]);
        printf("Padding bits : %d\n", temp);
        #endif // DEBUG
    }
    padding = 8 - (char)temp;	//int to char & padding = 8-bitsExtra
    fwrite(&padding, sizeof(char), 1, fout);
    #if DEBUG
    printf("Padding = %d\n",padding);
    #endif // DEBUG
    for(int i = 0; i < padding; i++)
        writeBit(0, fout);

    while(fread(&ch,sizeof(char),1,txt_file)!=0){
        #if DEBUG
        printf("Character from file is %c\t", ch);
        printf("Character from file is %d\n", (int)ch);
        #endif // DEBUG
        for(int i = 0; i < uniquechar; i++){
            #if DEBUG
            printf("Character is %c\t", code[i].letter);
            printf("Character is %d\n", (int)code[i].letter);
            #endif // DEBUG
            if ((int)ch == (int)code[i].letter){
                #if DEBUG
                printf("Code is %s\n", code[i].code);
                printf("Writing Code to file :");
                #endif
                memset(charcode, '\0', 256);
                strncpy(charcode, code[i].code, sizeof(charcode) - 1);
                for(int j = 0; j < strlen(code[i].code); j++){
                    if(charcode[j] == 49){
                        #if DEBUG
                        printf("1");
                        #endif // DEBUG
                        writeBit(1, fout);
                    }
                    else if(charcode[j] == 48){
                        #if DEBUG
                        printf("0");
                        #endif // DEBUG
                        writeBit(0, fout);
                    }
                }
                break;
            }
        }
    }
    fclose(fout);
    fclose(txt_file);
    return;
}
// Reverse String
char* ReverseConstString(char *str){
    int start, end, len;
    char temp, *ptr = NULL;

    // find length of string
    len = strlen(str);

    // create dynamic pointer char array
    ptr = malloc(sizeof(char)*(len+1));

    // copy of string to ptr array
    ptr = strcpy(ptr,str);

    // swapping of the characters
    for (start=0,end=len-1; start<=end; start++,end--)
    {
        temp = ptr[start];
        ptr[start] = ptr[end];
        ptr[end] = temp;
    }

    // return pointer of reversed string
    return ptr;
}

// return code for individual letter
char* return_code(struct node* tree, int node_number, int itr_val, char* psudo_code, int FirstTime){

        int letter_count = itr_val;
        static int LocalFirstTime;

        if (FirstTime){
            LocalFirstTime = 0;
        }

        while(true){
            if(tree[itr_val].status != 0){
                if (tree[itr_val].left == node_number){
                    strcat(psudo_code, "0");
                    node_number = tree[itr_val].node_count;
                    psudo_code = return_code(tree, node_number, letter_count, psudo_code, 0);
                    break;
                }
                else if (tree[itr_val].right == node_number){
                    strcat(psudo_code, "1");
                    node_number = tree[itr_val].node_count;
                    psudo_code = return_code(tree, node_number, letter_count, psudo_code, 0);
                    break;
                }
                itr_val++;
            }
            else{
                if (!LocalFirstTime){
                    if (tree[itr_val].left == node_number)
                        strcat(psudo_code, "0");
                    else if (tree[itr_val].right == node_number)
                        strcat(psudo_code, "1");
                    LocalFirstTime = 1;
                }
                break;
            }
        }

        return psudo_code;
}
//Construct Huffman Code
struct HuffCode* construct_huffman_code(struct node* tree, int letter_count){

    int node_number = -1;
    char* returned_code;
    struct HuffCode* code;
    code = malloc(sizeof(struct HuffCode)*(letter_count + 1));
    for(int ch = 0 ; ch < letter_count ; ch++){
        #if DEBUG
        printf("Letter is %c and node is %d\n", tree[ch].letter, tree[ch].node_count);
        #endif // DEBUG
        code[ch].code = malloc(sizeof(char)*26);
        returned_code = malloc(sizeof(char)*26);
        memset(code[ch].code, '\0', sizeof(char)*26);
        memset(returned_code, '\0', sizeof(char)*26);
        code[ch].letter = tree[ch].letter;
        node_number = tree[ch].node_count;
        returned_code = return_code(tree, node_number, letter_count, code[ch].code, 1);
        code[ch].code = ReverseConstString(returned_code);
    }
    #if DEBUG
    //for (int i = 0; i < letter_count ; i++)
    //    printf("%c\t\t%s\n", code[i].letter, code[i].code);
    #endif // DEBUG
    return code;
}
// Print Tree
void show_tree(struct node* tree, int count){

	int i = 0;

	printf("char\tProbab\t\tleft\tright\tflag\tcount\n");
	for(; i <= count; i++)
		printf("%c\t%f\t%d\t%d\t%d\t%d\n", tree[i].letter, tree[i].probab, tree[i].left, tree[i].right, tree[i].status, tree[i].node_count);

    return;
}

// Get the node with lowest probability
int get_low_probab(struct node* tree, int node_count, int element){

    static int first_node_index;
    int min_index;
    int First_time = 0;
    float min;

    if (element)
        first_node_index = node_count + 1;

    for(int i = 0; i < node_count; i++ ){

        if ((first_node_index == i) || (tree[i].status == 1)){
            continue;
        }
        else{
            if (!First_time){
                min = tree[i].probab;
                min_index = i;
                First_time = 1;
            }
        }

        if(tree[i].probab < min){
            min = tree[i].probab;
            min_index = i;
        }
    }

    if (element)
        first_node_index = min_index;

    tree[min_index].status = 1;
    return min_index;
}

// Construct huffman tree
struct node* construct_huffman_tree(struct node* tree, int node_count){

        int small_1, small_2;

        for (int i = 0; i < node_count; i++){
            small_1 = get_low_probab(tree, node_count, 1);
            small_2 = get_low_probab(tree, node_count, 0);
            tree[node_count].letter = '#';
            tree[node_count].node_count = node_count;
            tree[node_count].probab = tree[small_1].probab + tree[small_2].probab;
            tree[node_count].status = 0;
            tree[node_count].left = small_1;
            tree[node_count].right = small_2;
            if (tree[node_count].probab > 0.9999)
                break;
            node_count++;
        }
        #if DEBUG
        show_tree(tree, node_count);
        #endif // DEBUG
        return tree;
}
// TO find if value is already present in array or not
bool isvalueinarray(int val, char *arr, int size){
    int i;
    for (i=0; i < size; i++) {
        if (arr[i] == val)
            return true;
    }
    return false;
}

// For Binary search method
int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

// To reduce float numbers to two decimal points
float twoPrecision(float n, float i)
{
    float val = floor(pow(10,i)*n)/pow(10,i);
    return val;
}

struct LetterFreqData letter_freq(char* path){

    int frequency[128 - 32]; // ASCII Values for alphanumeric
    int count = 1;
    int noOfletters = 0;
    int ch, ch1;
    float float_error = 0.00001;
    char* unsortedletter;
    int* unsortedFreq;
    float* unsortedProbab;
    struct LetterFreqData letterFreqDt;
    #if DEBUG
    printf("File : %s\n", path);
    #endif // DEBUG
    FILE* txt_file = fopen (path, "rt");

    /* init the freq table: */
    for (ch = 0; ch < 128-32; ch++)
        frequency[ch] = 0;

    // Counting letter frequency
    while (1){
        ch = fgetc(txt_file);
        noOfletters++;
        if (ch == EOF) break; /* end of file or read error.  EOF is typically -1 */
            frequency[ch - 32]++;
    }
    fclose(txt_file);
    // Allocating maximum memory for the array pointers
    letterFreqDt.freq = malloc(sizeof(int)*(128-32));
    letterFreqDt.letter = malloc(sizeof(char)*(128-32));
    letterFreqDt.probab = malloc(sizeof(float)*(128-32));
    // Find letter, freq of letter and total count of letters
    for (ch = 0; ch < 128-32; ch++){
        // Discarding zero frequency letters
        if (frequency[ch] != 0){
            letterFreqDt.freq[count - 1] = frequency[ch];
            letterFreqDt.letter[count - 1] = (char)(ch+32);
            count++;
            #if DEBUG
            printf("Frequency of %c is %d\n", letterFreqDt.letter[count - 2], letterFreqDt.freq[count - 2]);
            #endif // DEBUG
        }
    }

    letterFreqDt.count = count - 1;
    // Find probability of each letter
    for (ch = 0; ch < count - 1; ch++){
        letterFreqDt.probab[ch] = twoPrecision((float)(letterFreqDt.freq[ch] / (float)(noOfletters - 1)), 5);
    }
    // Allocate memory for duplicate array
    unsortedProbab = malloc(sizeof(float)*letterFreqDt.count);
    unsortedFreq = malloc(sizeof(int)*letterFreqDt.count);
    unsortedletter = malloc(sizeof(char)*letterFreqDt.count);
    // Copy array
    memcpy(unsortedProbab, letterFreqDt.probab, sizeof(float)*letterFreqDt.count);
    memcpy(unsortedFreq, letterFreqDt.freq, sizeof(int)*letterFreqDt.count);
    memcpy(unsortedletter, letterFreqDt.letter, sizeof(char)*letterFreqDt.count);
    #if DEBUG
    printf("Before sorting\n");
    for (ch=0; ch<letterFreqDt.count; ch++)
     printf ("Frequency of %c is %d with probability %f\n",letterFreqDt.letter[ch], letterFreqDt.freq[ch], letterFreqDt.probab[ch]);
    #endif // DEBUG
    // Clearing previous order
    memset(letterFreqDt.freq, '\0', sizeof(int)*letterFreqDt.count);
    memset(letterFreqDt.letter, '\0', sizeof(char)*letterFreqDt.count);
    // Sort probabilities

    qsort (letterFreqDt.probab, letterFreqDt.count, sizeof(float), compare);
    for (ch = 0; ch < letterFreqDt.count; ch++){
        for (ch1 = 0; ch1 < letterFreqDt.count; ch1++){
            if (((letterFreqDt.probab[ch] - float_error) < unsortedProbab[ch1]) && ((letterFreqDt.probab[ch] + float_error) > unsortedProbab[ch1])){
                if (!(isvalueinarray(unsortedletter[ch1], letterFreqDt.letter, letterFreqDt.count))){
                    letterFreqDt.letter[ch] = unsortedletter[ch1];
                    letterFreqDt.freq[ch] = unsortedFreq[ch1];
                    break;
                }
            }
        }
    }
    #if DEBUG
    printf("After sorting\n");
    for (ch=0; ch<letterFreqDt.count; ch++)
     printf ("Frequency of %c is %d with probability %f\n",letterFreqDt.letter[ch], letterFreqDt.freq[ch], letterFreqDt.probab[ch]);
    #endif // DEBUG
    return letterFreqDt;
}
char* getFileName(){

    OPENFILENAME ofn;
    char *FilterSpec = "Text Files(.txt;.doc;.docx)\0*.txt;*.doc;*.docx\0All Files(.)\0*.*\0";
    char *Title = "Please select text file you want to compress";
    char* FileName = malloc(sizeof(char)*MAX_PATH);
    char szFileName[MAX_PATH];
    char szFileTitle[MAX_PATH];

    *szFileName = 0; *szFileTitle = 0;

    /* fill in non-variant fields of OPENFILENAME struct. */
    ofn.lStructSize       = sizeof(OPENFILENAME);
    ofn.hwndOwner         = GetFocus();
    ofn.lpstrFilter       = FilterSpec;
    ofn.lpstrCustomFilter = NULL;
    ofn.nMaxCustFilter    = 0;
    ofn.nFilterIndex      = 0;
    ofn.lpstrFile         = szFileName;
    ofn.nMaxFile          = MAX_PATH;
    ofn.lpstrInitialDir   = "."; // Initial directory.
    ofn.lpstrFileTitle    = szFileTitle;
    ofn.nMaxFileTitle     = MAX_PATH;
    ofn.lpstrTitle        = Title;
    ofn.lpstrDefExt   = 0;//I've set to null for demonstration

    ofn.Flags             = OFN_FILEMUSTEXIST|OFN_HIDEREADONLY;

    if (!GetOpenFileName((LPOPENFILENAME)&ofn))
    {
        return ("NULL"); // Failed or cancelled
    }
    else
    {
        FileTitle = malloc(sizeof(char)*MAX_PATH);
        strcpy(FileTitle, szFileTitle);
        strcpy(FileName, szFileName);
        return FileName;
    }
}

int main(void)
{

    char* filename;
    struct LetterFreqData letterFreqDt;
    struct node* tree = NULL;
    struct HuffCode* code = NULL;

    filename = getFileName();
    #if DEBUG
    printf("File location is %s\n", filename);
    #endif // DEBUG
    if (filename != NULL){
        printf("Reading File...\n");
        letterFreqDt = letter_freq(filename);
        tree = malloc(sizeof(struct node)*(2*letterFreqDt.count));
        code = malloc(sizeof(struct HuffCode)*(letterFreqDt.count + 1));

        for (int ch = 0; ch < letterFreqDt.count; ch++){
            tree[ch].letter = letterFreqDt.letter[ch];
            tree[ch].probab = letterFreqDt.probab[ch];
            tree[ch].frequency = letterFreqDt.freq[ch];
            tree[ch].node_count = ch;
            tree[ch].left = -1;
            tree[ch].right = -1;
            tree[ch].status = 0;

            #if DEBUG
            //printf("Frequency of %c is %d with probability %f\n", tree[ch].letter, letterFreqDt.freq[ch], tree[ch].probab);
            #endif // DEBUG
        }
        printf("Constructing Huffman Tree...\n");
        tree = construct_huffman_tree(tree, letterFreqDt.count);
        printf("Constructing Huffman Code...\n");
        code = construct_huffman_code(tree, letterFreqDt.count);
        #if DEBUG
        printf("Character\tCode\n");
        for (int i = 0; i < letterFreqDt.count ; i++)
            printf("%c\t\t%s\n", code[i].letter, code[i].code);
        #endif // DEBUG
        printf("Writing Encoded file...\n");
        write_encoded_file(filename, code, letterFreqDt);
        printf("Done!!!\n");
        // Deconde Compressed File
        //decode_file("a_text_file_compressed.dat");
    }
    else{
        printf("Failed to open the file!!!");
    }
    getch();
    return 1;
}
