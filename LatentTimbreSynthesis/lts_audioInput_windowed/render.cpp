#include "LDSP.h"
#include <libraries/OrtModel/OrtModel.h>
#include <libraries/AudioFile/AudioFile.h>
#include <algorithm>

OrtModel model(true);
std::string modelType = "onnx";
std::string modelName = "audioInput_windowed_rawvae";

const int segment_size = 1024;
const int hop_size = segment_size/2;

float* inputs[3];
float outputSegment[2][segment_size] = {0};
float *output;
int outputSegmentIdx = 0;

float interpolation = 0.5;

std::string filename[2] = {"472451__erokia__msfxp-sound-399.wav", "472454__erokia__msfxp-sound-402.wav"};	// name of the sound files (in project folder)
std::vector<float> audioFileSamples[2];
int readPointer_audioFile[2] = {0};
std::vector<float> audioInput[2];

int outputSampleCnt = 0;
int overlap_size;
int overlap_start;

bool liveInput = false;
std::vector<float> liveInputSamples; // circular buffer
int writePointer_liveIn;
int readPointer_liveIn;

bool setup(LDSPcontext *context, void *userData)
{
    std::string modelPath = "./"+modelName+"."+modelType;
    if (!model.setup("session1", modelPath))
    {
        printf("unable to setup ortModel");
        return false;
    }


    audioFileSamples[0] = AudioFileUtilities::loadMono(filename[0]);	
	if(audioFileSamples[0].size() == 0) 
	{
    	printf("Error loading audio file '%s'\n", filename[0].c_str());
    	return false;
	}

    audioFileSamples[1] = AudioFileUtilities::loadMono(filename[1]);	
	if(audioFileSamples[1].size() == 0) 
	{
    	printf("Error loading audio file '%s'\n", filename[1].c_str());
    	return false;
	}

    audioInput[0].resize(segment_size);
    audioInput[1].resize(segment_size);

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


inline void fillAudioInput(const std::vector<float>& audio_samples, std::vector<float>& audio_input, int& read_pointer, int hopSize) 
{
    size_t remaining = audio_samples.size() - read_pointer;

    // no wrapping needed, copy directly
    if (remaining >= audio_input.size()) 
        std::copy(audio_samples.begin() + read_pointer, audio_samples.begin() + read_pointer + audio_input.size(), audio_input.begin());
    else // wrapping needed
    {
        // fill part of the audio_input until the end of the audio_samples vector
        std::copy(audio_samples.begin() + read_pointer, audio_samples.end(), audio_input.begin());
        // wrap around and continue filling from the beginning of the audio_samples vector
        std::copy(audio_samples.begin(), audio_samples.begin() + audio_input.size() - remaining, audio_input.begin() + remaining);
    }

    read_pointer = (read_pointer + hopSize) % audio_samples.size(); // advance of hop size only, to obtain overlapping behavior at next call
}

void render(LDSPcontext *context, void *userData)
{

    for(int n=0; n<context->audioFrames; n++)
	{
        // generate new output samples when we run out of them
        if(outputSampleCnt >= hop_size)
        {
            // if live input, combine live input with the second audio file
            if(liveInput)
                fillAudioInput(liveInputSamples, audioInput[0], readPointer_liveIn, hop_size);
            else // otherwise, combine two audio files
                fillAudioInput(audioFileSamples[0], audioInput[0], readPointer_audioFile[0], hop_size);
            fillAudioInput(audioFileSamples[1], audioInput[1], readPointer_audioFile[1], hop_size);
            
            // combine audio inputs and interpolation into single input data structure
            inputs[0] = audioInput[0].data();
            inputs[1] = audioInput[1].data();
            inputs[2] = &interpolation;

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