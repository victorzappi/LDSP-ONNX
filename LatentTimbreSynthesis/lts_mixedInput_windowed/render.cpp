#include "LDSP.h"
#include <libraries/OrtModel/OrtModel.h>
#include <libraries/AudioFile/AudioFile.h>
#include <fstream>
#include <iostream>

OrtModel model(true);
std::string modelType = "onnx";
std::string modelName = "mixedInput_windowed_rawvae";

const int segment_size = 1024;
const int hop_size = segment_size/2;
const int latent_dim = 256;

float* inputs[4];
float outputSegment[2][segment_size] = {0};
float *output;
int outputSegmentIdx = 0;

float interpolation = 0.5;

std::string filename_mu = "472451__erokia__msfxp-sound-399_mu_windowed.lts";	// name of the mu bin file (in project folder)
std::string filename_logvar = "472451__erokia__msfxp-sound-399_logvar_windowed.lts"; // name of the logvar bin file (in project folder)
std::string filename_audio = "472454__erokia__msfxp-sound-402.wav";	// name of the sound file (in project folder)
std::vector<float> muFileSamples;
std::vector<float> logvarFileSamples;
std::vector<float> audioFileSamples;
int readPointer_mu = 0;
int readPointer_logvar = 0;
int readPointer_audioFile = 0;
std::vector<float> muInput;
std::vector<float> logvarInput;
std::vector<float> audioInput;

int outputSampleCnt = 0;
int overlap_size;
int overlap_start;

bool liveInput = false;
std::vector<float> liveInputSamples; // circular buffer
int writePointer_liveIn;
int readPointer_liveIn;




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

    muFileSamples = read_binary_file(filename_mu);
    if(muFileSamples.empty())
    {
    	printf("Error loading binary file '%s'\n", filename_mu.c_str());
    	return false;
	}
    logvarFileSamples = read_binary_file(filename_logvar);
    if(logvarFileSamples.empty())
    {
    	printf("Error loading binary file '%s'\n", filename_logvar.c_str());
    	return false;
	}

    muInput.resize(latent_dim);
    logvarInput.resize(latent_dim);
    

    audioFileSamples = AudioFileUtilities::loadMono(filename_audio);	
	if(audioFileSamples.size() == 0) 
	{
    	printf("Error loading audio file '%s'\n", filename_audio.c_str());
    	return false;
	}

    audioInput.resize(segment_size);


    if(liveInput) 
    {
        liveInputSamples.resize(hop_size*10);
        std::fill(liveInputSamples.begin(), liveInputSamples.end(), 0.0f); // we need zeros for proper initial overlap and add
        writePointer_liveIn = hop_size; // the first hop_size samples will be left as zeros
        readPointer_liveIn = 0;
    }

    
    // for overlap and add mechanism
    output = outputSegment[outputSegmentIdx];
    overlap_size = segment_size - hop_size;
    overlap_start = segment_size - overlap_size;

    return true;
}


inline void fillInput(const std::vector<float>& file_samples, std::vector<float>& input, int& read_pointer, int hopSize=-1) 
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

    // supports advancement of hop size only, to allow for overlapping behavior at next call
    if(hopSize == -1)
        hopSize = input.size();
    read_pointer = (read_pointer + hopSize) % file_samples.size();
}


inline void fillLatentInput(const std::vector<float>& mu_samples, const std::vector<float>& logvar_samples, 
                       std::vector<float>& mu_input, std::vector<float>& logvar_input, int& mu_read_pointer, int& logvar_read_pointer) 
{
    // windowed latent bin files are composed of spread out overlapping segments already, so no need for hop mechanism
    fillInput(mu_samples, mu_input, mu_read_pointer);
    fillInput(logvar_samples, logvar_input, logvar_read_pointer);
}

inline void fillAudioInput(const std::vector<float>& audio_samples, std::vector<float>& audio_input, int& read_pointer, int hopSize) 
{
    fillInput(audio_samples, audio_input, read_pointer, hopSize);
}

void render(LDSPcontext *context, void *userData)
{

    for(int n=0; n<context->audioFrames; n++)
	{
        // generate new output samples when we run out of them
        if(outputSampleCnt >= hop_size)
        {
            fillLatentInput(muFileSamples, logvarFileSamples, muInput, logvarInput, readPointer_mu, readPointer_logvar);
            // if live input, combine latent files with live input
            if(liveInput)
                fillAudioInput(liveInputSamples, audioInput, readPointer_liveIn, hop_size);
            else // otherwise, combine latent files with audio file
                fillAudioInput(audioFileSamples, audioInput, readPointer_audioFile, hop_size);
            
            // combine letent inputs, audio input and interpolation into single input data structure
            inputs[0] = muInput.data();
            inputs[1] = logvarInput.data();
            inputs[2] = audioInput.data();
            inputs[3] = &interpolation;
            
            outputSegmentIdx = 1-outputSegmentIdx; // update current segment
            output = outputSegment[outputSegmentIdx]; // point to current output segment to fill

            // generate a new segment of output samples
            model.run(inputs, output);
            
            // add first samples of current output segment with overlapping samples of previous output segment
            for(int i=0; i<overlap_size; i++)
                output[i] = outputSegment[outputSegmentIdx][i] + outputSegment[1-outputSegmentIdx][overlap_start+i];

            outputSampleCnt = 0;
        }
    
        audioWrite(context, n, 0, output[outputSampleCnt]);
        audioWrite(context, n, 1, output[outputSampleCnt]);

        // if live input, fill live input circular buffer 
        if(liveInput)
        {
            liveInputSamples[writePointer_liveIn] = audioRead(context, n, 0); 
            writePointer_liveIn = (writePointer_liveIn + 1) % liveInputSamples.size();
        }

        outputSampleCnt++;
	}
}

void cleanup(LDSPcontext *context, void *userData)
{

}