#include "LDSP.h"
#include <libraries/OrtModel/OrtModel.h>
#include <libraries/AudioFile/AudioFile.h>
#include <fstream>
#include <iostream>

OrtModel model(true);
std::string modelType = "onnx";
std::string modelName = "mixedInput_rawvae";

const int segment_size = 1024;
const int latent_dim = 256;

float* inputs[4];
float output[segment_size] = {0};

float interpolation = 0.5;

std::string filename_mu = {"mu1.lts"};	// name of the mu bin file (in project folder)
std::string filename_logvar = {"logvar1.lts"}; // name of the logvar bin file (in project folder)
std::string filename_audio = {"472454__erokia__msfxp-sound-402.wav"};	// name of the sound file (in project folder)
std::vector<float> muFileSamples;
std::vector<float> logvarFileSamples;
std::vector<float> audioFileSamples;
int readPointer_mu = 0;
int readPointer_logvar = 0;
int readPointer_audio = 0;
std::vector<float> muInput;
std::vector<float> logvarInput;
std::vector<float> audioInput;

int outputSampleCnt = 0;

bool liveInput = false;





std::vector<float> read_binary_file(const std::string& filename) 
{
    std::ifstream input(filename, std::ios::binary);
    if (!input) 
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }
    // Get the size of the file
    input.seekg(0, std::ios::end);
    std::size_t size = input.tellg();
    input.seekg(0, std::ios::beg);

    // Read the content of the file into a vector
    std::vector<float> buffer(size / sizeof(float));
    input.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

bool setup(LDSPcontext *context, void *userData)
{
    std::string modelPath = "./"+modelName+"."+modelType;
    if (!model.setup("session1", modelPath))
        printf("unable to setup ortModel");

    muFileSamples = read_binary_file(filename_mu);
    logvarFileSamples = read_binary_file(filename_logvar);

    muInput.resize(latent_dim);
    logvarInput.resize(latent_dim);
    

    audioFileSamples = AudioFileUtilities::loadMono(filename_audio);	
	if(audioFileSamples.size() == 0) 
	{
    	printf("Error loading audio file '%s'\n", filename_audio.c_str());
    	return false;
	}

    audioInput.resize(segment_size);

    return true;
}


inline void fillInput(const std::vector<float>& file_samples, std::vector<float>& input, int& read_pointer) 
{
    size_t remaining = file_samples.size() - read_pointer;

    // no wrapping needed, copy directly
    if (remaining >= input.size()) 
        std::copy(file_samples.begin() + read_pointer, file_samples.begin() + read_pointer + input.size(), input.begin());
    else // wrapping needed
    {
        // fill part of the input until the end of the file_samples vector
        std::copy(file_samples.begin() + read_pointer, file_samples.end(), input.begin());
        // wrap around and continue filling from the beginning of the file_samples vector
        std::copy(file_samples.begin(), file_samples.begin() + input.size() - remaining, input.begin() + remaining);
    }

    read_pointer = (read_pointer + input.size()) % file_samples.size();
}


inline void fillLatentInput(const std::vector<float>& mu_samples, const std::vector<float>& logvar_samples, 
                       std::vector<float>& mu_input, std::vector<float>& logvar_input, int& mu_read_pointer, int& logvar_read_pointer) 
{
    fillInput(mu_samples, mu_input, mu_read_pointer);
    fillInput(logvar_samples, logvar_input, logvar_read_pointer);
}

inline void fillAudioInput(const std::vector<float>& audio_samples, std::vector<float>& audio_input, int& read_pointer) 
{
    fillInput(audio_samples, audio_input, read_pointer);
}

void render(LDSPcontext *context, void *userData)
{

    for(int n=0; n<context->audioFrames; n++)
	{
        // generate new output samples when we run out of them
        if(outputSampleCnt >= segment_size)
        {
            fillLatentInput(muFileSamples, logvarFileSamples, muInput, logvarInput, readPointer_mu, readPointer_logvar);
            // if not live input, combine latent files with audio file
            if(!liveInput)
                fillAudioInput(audioFileSamples, audioInput, readPointer_audio);
            
            
            // combine letent inputs, audio input and interpolation into single input data structure
            inputs[0] = muInput.data();
            inputs[1] = logvarInput.data();
            inputs[2] = audioInput.data();
            inputs[3] = &interpolation;

            // generate a new segment of output samples
            model.run(inputs, output);
            
            outputSampleCnt = 0;
        }
    
        audioWrite(context, n, 0, output[outputSampleCnt]);
        audioWrite(context, n, 1, output[outputSampleCnt]);

        // if live input, combine latent files with live input
        if(liveInput)
            audioInput[outputSampleCnt] = audioRead(context, n, 0);

        outputSampleCnt++;
	}
}

void cleanup(LDSPcontext *context, void *userData)
{

}