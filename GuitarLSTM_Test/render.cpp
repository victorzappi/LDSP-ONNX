#include "LDSP.h"
#include "libraries/OrtModel/OrtModel.h"

OrtModel model;
std::string modelType = "onnx";
std::string modelName = "GuitarLSTM";

const int inputSize = 5;

float input[inputSize] = {0};
float output[1] = {0};

const int circBuffLength = 48000; // a reasonably large buffer, to limit end-of-buffer overhead
int writePointer;
int readPointer;
float circBuff[circBuffLength];


bool setup(LDSPcontext *context, void *userData)
{

    std::string modelPath = "./"+modelName+"."+modelType;
    if (!model.setup("session1", modelPath))
        printf("unable to setup ortModel");

    writePointer = inputSize-1; // the first intputSize-1 samples must be zeros
    readPointer = 0;

    return true;
}

void render(LDSPcontext *context, void *userData)
{
    for(int n=0; n<context->audioFrames; n++)
	{
        circBuff[writePointer] = audioRead(context,n,0);

        if(readPointer<=circBuffLength-inputSize)
            std::copy(circBuff + readPointer, circBuff + readPointer + inputSize, input);
        else 
        {
            int firstPartSize = circBuffLength - readPointer;
            std::copy(circBuff + readPointer, circBuff + circBuffLength, input);
            std::copy(circBuff, circBuff + (inputSize - firstPartSize), input + firstPartSize);
        }

        model.run(input, output);
    
        // passthrough test, because the model may not be trained
        audioWrite(context, n, 0, input[inputSize-1]);
        audioWrite(context, n, 1, input[inputSize-1]);

        if(++readPointer >= circBuffLength)
            readPointer = 0;
        if(++writePointer >= circBuffLength)
            writePointer = 0;	
    }
}

void cleanup(LDSPcontext *context, void *userData)
{
}