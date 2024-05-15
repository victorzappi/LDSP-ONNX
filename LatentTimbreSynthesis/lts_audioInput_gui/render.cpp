#include "LDSP.h"
#include <libraries/OrtModel/OrtModel.h>
#include <libraries/AudioFile/AudioFile.h>
#include <libraries/Gui/Gui.h>
#include <libraries/GuiController/GuiController.h>


OrtModel model(true);
std::string modelType = "onnx";
std::string modelName = "audioInput_rawvae";

const int segment_size = 1024;

float* inputs[3];
float output[segment_size] = {0};

float interpolation = 0.5;

std::string filename[2] = {"472451__erokia__msfxp-sound-399.wav", "472454__erokia__msfxp-sound-402.wav"};	// name of the sound files (in project folder)
std::vector<float> fileSamples[2];
int readPointer[2] = {0};
std::vector<float> audioInput[2];

int outputSampleCnt = 0;

bool liveInput = true;

Gui gui;
GuiController controller;

bool setup(LDSPcontext *context, void *userData)
{
    std::string modelPath = "./"+modelName+"."+modelType;
    if (!model.setup("session1", modelPath))
        printf("unable to setup ortModel");


    fileSamples[0] = AudioFileUtilities::loadMono(filename[0]);	
	if(fileSamples[0].size() == 0) 
	{
    	printf("Error loading audio file '%s'\n", filename[0].c_str());
    	return false;
	}

    fileSamples[1] = AudioFileUtilities::loadMono(filename[1]);	
	if(fileSamples[1].size() == 0) 
	{
    	printf("Error loading audio file '%s'\n", filename[1].c_str());
    	return false;
	}

    audioInput[0].resize(segment_size);
    audioInput[1].resize(segment_size);

    // Set up the GUI
	gui.setup(context->projectName);
	controller.setup(&gui, "RawVAE");
	controller.addSlider("Interpolation", 0.5, 0, 1, 0); 

    return true;
}


inline void fillAudioInput(const std::vector<float>& file_samples, std::vector<float>& audio_input, int& read_pointer) 
{
    size_t remaining = file_samples.size() - read_pointer;

    // no wrapping needed, copy directly
    if (remaining >= audio_input.size()) 
        std::copy(file_samples.begin() + read_pointer, file_samples.begin() + read_pointer + audio_input.size(), audio_input.begin());
    else // wrapping needed
    {
        // fill part of the audio_input until the end of the file_samples vector
        std::copy(file_samples.begin() + read_pointer, file_samples.end(), audio_input.begin());
        // wrap around and continue filling from the beginning of the file_samples vector
        std::copy(file_samples.begin(), file_samples.begin() + audio_input.size() - remaining, audio_input.begin() + remaining);
    }

    read_pointer = (read_pointer + audio_input.size()) % file_samples.size();
}

void render(LDSPcontext *context, void *userData)
{
    interpolation = controller.getSliderValue(0);

    for(int n=0; n<context->audioFrames; n++)
	{
        // generate new output samples when we run out of them
        if(outputSampleCnt >= segment_size)
        {
            // if not live input, combine two audio files
            if(!liveInput)
                fillAudioInput(fileSamples[0], audioInput[0], readPointer[0]);
            fillAudioInput(fileSamples[1], audioInput[1], readPointer[1]);
            
            // combine audio inputs and interpolation into single input data structure
            inputs[0] = audioInput[0].data();
            inputs[1] = audioInput[1].data();
            inputs[2] = &interpolation;

            // generate a new segment of output samples
            model.run(inputs, output);
            
            outputSampleCnt = 0;
        }
    
        audioWrite(context, n, 0, output[outputSampleCnt]);
        audioWrite(context, n, 1, output[outputSampleCnt]);

        // if live input, combine live input with the second audio file
        if(liveInput)
            audioInput[0][outputSampleCnt] = audioRead(context, n, 0);

        outputSampleCnt++;
	}
}

void cleanup(LDSPcontext *context, void *userData)
{

}