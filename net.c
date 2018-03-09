/*
AI Assignment 2
Arvind Deshraj - 201601007
Rutvik Vijjali - 201601105
Sum of Sqaures Deviation Loss
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define hiddenLayerSize 5 // can vary from 5 to 8 for the assignment
#define inputLayerSize 16
#define outputLayerSize 10
#define trainingSetSize 2216
#define testSetSize 998
#define epsilon 0.01 

int inputLayer[trainingSetSize][17] = {1}; // 16 plus 1 bias unit
float eta = 0.001; // eta is the learning rate
float outputLayer[trainingSetSize][10] = {0.0};
float hiddenLayer[trainingSetSize][hiddenLayerSize + 1]  ={0.0}; // hidden layer neurons plus 1 bias unit
float weightsHiddenLayer[inputLayerSize + 1][hiddenLayerSize] ;
float weightsOutputLayer[hiddenLayerSize + 1][outputLayerSize] ;
int target[trainingSetSize][outputLayerSize] ;
int testTarget[testSetSize][outputLayerSize] ;
int testInput[testSetSize][17] = {1}; // 16 plus 1 bias unit
int flag ;


float rn()
{
        return ((float)(rand()) / (float)(RAND_MAX)) * 0.6 - 0.3   ;
}

float sigmoid(float x)
{
    return (float)(1.0/(1.0 + exp(-1.0 * x)));
}

void convertToOneHotEncoding(int vect[][outputLayerSize], int vectsize)
{
    int temp;
    for(int i =0 ; i < vectsize; i++)
        {
            temp = vect[i][0];
            for(int j = 0; j < outputLayerSize; j++)
            {
                if( j+1 == temp)
                {
                    vect[i][j] = 1;
                }
                else
                {
                    vect[i][j] = 0;
                }
            }
        }
}

int deltaWStopping(float weightsOutputDiff[][outputLayerSize], float weightsHiddenDiff[][hiddenLayerSize])
{
    int count = 0;
    for(int i =0 ; i < outputLayerSize; i++)
    {
        for(int j = 0 ;j < hiddenLayerSize + 1; j++)
        {
            if(weightsOutputDiff[j][i] < epsilon)
                count++;
        }
    }

    for(int i = 0; i < hiddenLayerSize + 1; i++)
    {
        for(int j = 0;j < inputLayerSize + 1; j++)
        {
            if(weightsHiddenDiff[j][i] < epsilon)
                count++;
        }
    }


    if(count > ((inputLayerSize + 1) * hiddenLayerSize) + ((hiddenLayerSize + 1) * outputLayerSize) - 23)
        return 1;
    else 
        return 0;
}

void readTrainingDatafromCSV()
{
    FILE *fp;
    char data[60];
    fp = fopen("trainingdata.csv", "r");
    if(fp == NULL)
        printf("Could not open the file!");
    for(int i =0 ; i < trainingSetSize; i++)
    {
        if(i == 0) 
        {
            fgets(data, 60, fp); //skips the header names
            fgets(data, 60, fp);
        }
        fgets(data, 60, fp);
        target[i][0] = atoi(strtok(data, ","));
        for(int j = 1; j <= 16; j++)
        {
            inputLayer[i][j] = atoi(strtok(NULL,","));
        }
    }
    fclose(fp);
    convertToOneHotEncoding(target, (int)trainingSetSize);
}


void readTestDataFromCSV()
{
    FILE *fp = fopen("testdata.csv","r");
    char data[60];
    if(fp == NULL)
    {
        printf("Could not open the file!");
    }
    for(int i =0 ; i < testSetSize; i++)
    {
        if(i == 0) 
        {
            fgets(data, 60, fp); //skips the header names
            fgets(data, 60, fp);
        }
        fgets(data, 60, fp);

        testTarget[i][0] = atoi(strtok(data, ","));
        for(int j = 1; j <= 16; j++)
        {
            testInput[i][j] = atoi(strtok(NULL,","));
        }
    }
    fclose(fp);
    convertToOneHotEncoding(testTarget, (int)testSetSize);    

}
void initialise()
{
    for(int i = 0; i < trainingSetSize ; i++ )
    {    
        inputLayer[i][0] = 1; // bias unit
        hiddenLayer[i][0] = 1; // bias unit
    }
    
    for(int i = 0 ; i < inputLayerSize + 1; i++)
    {
        for(int j = 0 ; j < hiddenLayerSize; j++)
        {
            weightsHiddenLayer[i][j] =  rn() ;
        }
    }

    for(int i =0 ; i < hiddenLayerSize + 1; i++)
    {
        for(int j = 0 ; j < outputLayerSize; j++)
        {
            weightsOutputLayer[i][j] = rn() ;
        }
    }
}


void train(int opt)
{
    long int limit;
    flag = opt;
    if(opt == 0)
        limit = 1000000;
    else
        limit = 100;
    for(long int epoch = 0; epoch < limit; epoch++ )
    {
        float temperror = 0.0;
        float outputError[outputLayerSize] = {0.0};
        //float outputDelta[outputLayerSize] = {0.0};
        float hiddenDelta[hiddenLayerSize + 1] = {0.0};
        float hiddenError[hiddenLayerSize + 1] = {0.0};
        float weightsOutputDiff[hiddenLayerSize + 1][outputLayerSize] ;
        float weightsHiddenDiff[inputLayerSize + 1][hiddenLayerSize] ;
        float error ;

        /*if(error < 100)
            break;*/

        error = 0.0;
        memset(weightsHiddenDiff, 0, sizeof(weightsHiddenDiff[0][0]) * (inputLayerSize + 1) * hiddenLayerSize);
        memset(weightsOutputDiff, 0, sizeof(weightsOutputDiff[0][0]) *(hiddenLayerSize + 1) * outputLayerSize );
        
        for(int m = 0; m < trainingSetSize; m++ )
        {
            for(int j = 1; j < hiddenLayerSize + 1; j++ ) // except for the bias unit
            {
                for(int i = 0; i < inputLayerSize + 1; i++) // starts from 0 to include contribution from bias unit
                {
                    hiddenLayer[m][j] += (inputLayer[m][i] * weightsHiddenLayer[i][j]);
                }
                hiddenLayer[m][j] = sigmoid(hiddenLayer[m][j]);
            }
            // BY this point all neurons in hidden layer are populated with values for that particular training data

            for(int k = 0; k < outputLayerSize; k++)
            {
                for(int j = 0; j <  hiddenLayerSize + 1; j++ ) // = for bias unit
                {
                    outputLayer[m][k] += hiddenLayer[m][j]*weightsOutputLayer[j][k];
                }
                outputLayer[m][k] = sigmoid(outputLayer[m][k]);
                error += 0.5* (target[m][k] - outputLayer[m][k]) * (target[m][k] - outputLayer[m][k]) ;
            }

            for(int i = 0 ;i < outputLayerSize; i++)
            {
                outputError[i] = (float)(target[m][i] - outputLayer[m][i] );    
            }

            for(int i = 0; i <= hiddenLayerSize ; i++)
            {
                for(int j = 0; j < outputLayerSize; j++)
                {
                    hiddenDelta[i] += (float)(weightsOutputLayer[i][j] * outputError[j]); //outputDelta
                }
                //hiddenDelta[i] *= hiddenLayer[m][i] * (1 - hiddenLayer[m][i]);
            }

            for(int i = 0; i < outputLayerSize; i++)
            {
                weightsOutputDiff[0][i] += eta * outputError[i]/(trainingSetSize);
                for(int j = 1; j < hiddenLayerSize + 1; j++)
                {
                    weightsOutputDiff[j][i] +=   (float)eta * outputError[i] * hiddenLayer[m][j] /(trainingSetSize); //outputDelta
                }
            }

            for(int i = 0 ; i <= hiddenLayerSize ; i++)
            {
                for(int j = 0 ; j < inputLayerSize + 1; j++)
                {
                    weightsHiddenDiff[j][i] += (float)eta*hiddenDelta[i]*inputLayer[m][j]/(trainingSetSize) ;
                }
            }  
        }

        for(int i = 0 ; i < outputLayerSize; i++)
        {
            for(int j = 0 ; j < hiddenLayerSize + 1 ; j++)
            {
                weightsOutputLayer[j][i] += weightsOutputDiff[j][i];
            }
        }

        for(int i = 0 ; i< hiddenLayerSize; i++)
        {
            for(int j = 0 ; j < inputLayerSize; j++)
            {
                weightsHiddenLayer[j][i] += weightsHiddenDiff[j][i];
		printf("%f ", weightsHiddenDiff[j][i]);
            }
	    printf("\n");
        }
        //printf("%f\n",error );
        if(opt == 0)
        {
            if(deltaWStopping(weightsOutputDiff,weightsHiddenDiff));
            break; // delta w stopping condition
        }
    }
    
}

void test()
{
    int sum =0;
    for(int count = 0 ; count < testSetSize; count++)
    {
        for(int j = 1; j <= hiddenLayerSize; j++ )
        {
            for(int i = 0; i <= inputLayerSize; i++) // starts from 0 to include contribution from bias unit
            {
                hiddenLayer[count][j] += testInput[count][i]*weightsHiddenLayer[i][j];
            }
            hiddenLayer[count][j] = sigmoid(hiddenLayer[count][j]);
        }

        for(int k = 0; k < outputLayerSize; k++)
        {
            for(int j = 0; j <= hiddenLayerSize; j++ )
            {
                outputLayer[count][k] += hiddenLayer[count][j]*weightsOutputLayer[j][k];
            }
            outputLayer[count][k] = sigmoid(outputLayer[count][k]);
        }

           // printf("\nTest Output: ");
        float max = outputLayer[count][0];
        int index  =0;
        for(int i = 0; i < outputLayerSize; i++)
        {
            if(outputLayer[count][i] > max)
                index = i;
        } 
        int fl = 1;
        fl =1;
        for (int i = 0; i < outputLayerSize; ++i)
        {
            if(testTarget[count][i] == 1)
                fl = i;    
        }
        if(fl == index)
        sum++;
   }
    float pct = (float)sum/testSetSize;
    printf("Using Sum of Squared Deviation Loss.\nNo of hidden layer neurons: %d\n", hiddenLayerSize);
    if(flag == 0)
        printf("Using Delta W stopping condition\n");
    else
        printf("Using fixed number of epochs\n");
    printf("Test set Accuracy:%lf\n", pct*100);
    
}

int main()
{
    srand((unsigned int)time(NULL));
    initialise();
    readTrainingDatafromCSV();
    train(rand() % 2);
    readTestDataFromCSV();
    test();
}
