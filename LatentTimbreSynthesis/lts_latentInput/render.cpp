#include "LDSP.h"
#include <libraries/OrtModel/OrtModel.h>
#include <libraries/AudioFile/AudioFile.h>
#include <fstream>
#include <iostream>

OrtModel model(true);
std::string modelType = "onnx";
std::string modelName = "latentInput_rawvae";

const int segment_size = 1024;
const int latent_dim = 256;

float* inputs[5];
float output[segment_size] = {0};

float interpolation = 0.5;

std::string filename_mu[2] = {"472451__erokia__msfxp-sound-399_mu.lts", "472454__erokia__msfxp-sound-402_mu.lts"};	// name of the mu bin files (in project folder)
std::string filename_logvar[2] = {"472451__erokia__msfxp-sound-399_logvar.lts", "472454__erokia__msfxp-sound-402_logvar.lts"};	// name of the logvar bin files (in project folder)
std::vector<float> muFileSamples[2];
std::vector<float> logvarFileSamples[2];
int readPointer[2] = {0};
std::vector<float> muInput[2];
std::vector<float> logvarInput[2];

int outputSampleCnt = 0;






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
    {
        printf("unable to setup ortModel");
        return false;
    }


    muFileSamples[0] = read_binary_file(filename_mu[0]);
    if(muFileSamples[0].empty())
    {
    	printf("Error loading binary file '%s'\n", filename_mu[0].c_str());
    	return false;
	}
    muFileSamples[1] = read_binary_file(filename_mu[1]);
    if(muFileSamples[1].empty())
    {
    	printf("Error loading binary file '%s'\n", filename_mu[1].c_str());
    	return false;
	}
    
    logvarFileSamples[0] = read_binary_file(filename_logvar[0]);
    if(logvarFileSamples[0].empty())
    {
    	printf("Error loading binary file '%s'\n", filename_logvar[0].c_str());
    	return false;
	}
    logvarFileSamples[1] = read_binary_file(filename_logvar[1]);
    if(logvarFileSamples[1].empty())
    {
    	printf("Error loading binary file '%s'\n", filename_logvar[1].c_str());
    	return false;
	}

    muInput[0].resize(latent_dim);
    muInput[1].resize(latent_dim);
    logvarInput[0].resize(latent_dim);
    logvarInput[1].resize(latent_dim);

    return true;
}


inline void fillLatentInput(const std::vector<float>& mu_samples, const std::vector<float>& logvar_samples, 
                       std::vector<float>& mu_input, std::vector<float>& logvar_input, int& read_pointer) 
{
    size_t input_size = mu_input.size(); // mu and logvar inputs are expected to have the same size
    size_t source_size = mu_samples.size(); // mu and logvar sources are expected to have the same size
    size_t remaining = source_size - read_pointer;

    // no wrapping needed, copy directly
    if (remaining >= input_size) 
    {
        std::copy(mu_samples.begin() + read_pointer, mu_samples.begin() + read_pointer + input_size, mu_input.begin());
        std::copy(logvar_samples.begin() + read_pointer, logvar_samples.begin() + read_pointer + logvar_input.size(), logvar_input.begin());
    }
    else // wrapping needed
    {
        // fill part of the mu_input until the end of the mu_samples vector
        std::copy(mu_samples.begin() + read_pointer, mu_samples.end(), mu_input.begin());
        std::copy(logvar_samples.begin() + read_pointer, logvar_samples.end(), logvar_input.begin());
        // wrap around and continue filling from the beginning of the mu_samples vector
        std::copy(mu_samples.begin(), mu_samples.begin() + input_size - remaining, mu_input.begin() + remaining);
        std::copy(logvar_samples.begin(), logvar_samples.begin() + logvar_input.size() - remaining, logvar_input.begin() + remaining);
    }
    
    read_pointer = (read_pointer + input_size) % source_size;
}


void render(LDSPcontext *context, void *userData)
{

    for(int n=0; n<context->audioFrames; n++)
	{
        // generate new output samples when we run out of them
        if(outputSampleCnt >= segment_size)
        {            
            fillLatentInput(muFileSamples[0], logvarFileSamples[0], muInput[0], logvarInput[0], readPointer[0]);
            fillLatentInput(muFileSamples[1], logvarFileSamples[1], muInput[1], logvarInput[1], readPointer[1]);
            
            // combine latent inputs and interpolation into single input data structure
            inputs[0] = muInput[0].data();
            inputs[1] = logvarInput[0].data();
            inputs[2] = muInput[1].data();
            inputs[3] = logvarInput[1].data();
            inputs[4] = &interpolation;

            // generate a new segment of output samples
            model.run(inputs, output);
            
            outputSampleCnt = 0;
        }
    
        audioWrite(context, n, 0, output[outputSampleCnt]);
        audioWrite(context, n, 1, output[outputSampleCnt]);

        outputSampleCnt++;
	}
}

void cleanup(LDSPcontext *context, void *userData)
{

}