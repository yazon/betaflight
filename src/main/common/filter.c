/*
 * This file is part of Cleanflight.
 *
 * Cleanflight is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Cleanflight is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Cleanflight.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "arm_math.h"

#include "build/debug.h"
#include "common/filter.h"
#include "common/maths.h"
#include "common/utils.h"

#define M_LN2_FLOAT 0.69314718055994530942f
#define M_PI_FLOAT  3.14159265358979323846f
#define BIQUAD_BANDWIDTH 1.9f     /* bandwidth in octaves */
#define BIQUAD_Q 1.0f / sqrtf(2.0f)     /* quality factor - butterworth*/

void arm_fir_init_f32(arm_fir_instance_f32 * S, uint16_t numTaps, float32_t * pCoeffs,float32_t * pState, uint32_t blockSize);
void arm_fir_f32(const arm_fir_instance_f32 * S, float32_t * pSrc,float32_t * pDst, uint32_t blockSize);
void arm_lms_norm_init_f32(arm_lms_norm_instance_f32 * S, uint16_t numTaps, float32_t * pCoeffs, float32_t * pState, float32_t mu, uint32_t blockSize);
void arm_lms_norm_f32(arm_lms_norm_instance_f32 * S, float32_t * pSrc, float32_t * pRef, float32_t * pOut, float32_t * pErr, uint32_t blockSize);

#if USE_ADAPTIVE_FILTER

uint16_t adaptiveFilterFsAndTapsSize[ADAPTIVE_FILTER__MAX][3] = {
	{ADAPTIVE_FILTER_FS_1KHZ_VALUE, ADAPTIVE_FILTER_FS_1KHZ_TAPS_SIZE, LPF_GYRO_FILTER_FS_1KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_2KHZ_VALUE, ADAPTIVE_FILTER_FS_2KHZ_TAPS_SIZE, LPF_GYRO_FILTER_FS_2KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_4KHZ_VALUE, ADAPTIVE_FILTER_FS_4KHZ_TAPS_SIZE, LPF_GYRO_FILTER_FS_4KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_8KHZ_VALUE, ADAPTIVE_FILTER_FS_8KHZ_TAPS_SIZE, LPF_GYRO_FILTER_FS_8KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_16KHZ_VALUE, ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE, LPF_GYRO_FILTER_FS_16KHZ_TAPS_SIZE}};

float lpfGyro1kHzCoeffs[LPF_GYRO_COEFFS_1KHZ_LENGTH] = {
	0.3224316835,   0.3551366329,   0.3224316835
};

float lpfGyro2kHzCoeffs[LPF_GYRO_COEFFS_2KHZ_LENGTH] = {
	0.1532828212,   0.1692033708,    0.177513808,    0.177513808,   0.1692033708,
	0.1532828212
};

float lpfGyro4kHzCoeffs[LPF_GYRO_COEFFS_4KHZ_LENGTH] = {
	0.07423233241,  0.07906711102,  0.08306333423,  0.08613695949,  0.08822301775,
	0.0892772451,   0.0892772451,  0.08822301775,  0.08613695949,  0.08306333423,
	0.07906711102,  0.07423233241
};

float lpfGyro8kHzCoeffs[LPF_GYRO_COEFFS_8KHZ_LENGTH] = {
	0.03646478057,  0.03777036816,  0.03898027539,  0.04008816928,  0.04108823463,
	0.04197520018,  0.04274438322,  0.04339171201,  0.04391375557,  0.04430773482,
	0.04457155988,  0.04470382631,  0.04470382631,  0.04457155988,  0.04430773482,
	0.04391375557,  0.04339171201,  0.04274438322,  0.04197520018,  0.04108823463,
	0.04008816928,  0.03898027539,  0.03777036816,  0.03646478057
};

float lpfGyro16kHzCoeffs[LPF_GYRO_COEFFS_16KHZ_LENGTH] = {
	0.01806368493,  0.01840149611,  0.01872797497,  0.01904269494,  0.01934524812,
	0.01963523589,  0.01991228014,  0.02017601393,  0.02042609267,  0.02066218667,
	0.02088398486,  0.02109119296,  0.02128353715,  0.02146076411,  0.02162263729,
	0.02176894248,  0.02189948596,  0.02201409452,  0.02211261541,  0.02219491638,
	0.02226088941,  0.02231044509,  0.02234352008,  0.02236006781,  0.02236006781,
	0.02234352008,  0.02231044509,  0.02226088941,  0.02219491638,  0.02211261541,
	0.02201409452,  0.02189948596,  0.02176894248,  0.02162263729,  0.02146076411,
	0.02128353715,  0.02109119296,  0.02088398486,  0.02066218667,  0.02042609267,
	0.02017601393,  0.01991228014,  0.01963523589,  0.01934524812,  0.01904269494,
	0.01872797497,  0.01840149611,  0.01806368493
};

float lpfGyro32kHzCoeffs[LPF_GYRO_COEFFS_32KHZ_LENGTH] = {
	0.008988952264, 0.009074785747, 0.009159243666, 0.009242299013, 0.009323922917,
	0.009404091164, 0.009482776746, 0.009559953585, 0.009635596536, 0.009709680453,
	0.009782182053, 0.009853077121, 0.009922342375, 0.009989955463,   0.0100558931,
	0.01012013387,  0.01018265821,  0.01024344377,    0.010302471,  0.01035972033,
	0.01041517314,   0.0104688108,  0.01052061655,  0.01057057176,   0.0106186606,
	0.01066486817,  0.01070917677,   0.0107515743,  0.01079204492,  0.01083057653,
	0.01086715516,  0.01090176962,  0.01093440689,  0.01096505858,  0.01099371165,
	0.01102035958,   0.0110449912,  0.01106759906,   0.0110881757,  0.01110671367,
	0.01112320833,  0.01113765314,  0.01115004253,  0.01116037369,   0.0111686429,
	0.01117484737,  0.01117898524,  0.01118105371,  0.01118105371,  0.01117898524,
	0.01117484737,   0.0111686429,  0.01116037369,  0.01115004253,  0.01113765314,
	0.01112320833,  0.01110671367,   0.0110881757,  0.01106759906,   0.0110449912,
	0.01102035958,  0.01099371165,  0.01096505858,  0.01093440689,  0.01090176962,
	0.01086715516,  0.01083057653,  0.01079204492,   0.0107515743,  0.01070917677,
	0.01066486817,   0.0106186606,  0.01057057176,  0.01052061655,   0.0104688108,
	0.01041517314,  0.01035972033,    0.010302471,  0.01024344377,  0.01018265821,
	0.01012013387,   0.0100558931, 0.009989955463, 0.009922342375, 0.009853077121,
	0.009782182053, 0.009709680453, 0.009635596536, 0.009559953585, 0.009482776746,
	0.009404091164, 0.009323922917, 0.009242299013, 0.009159243666, 0.009074785747,
	0.008988952264
};

/* PID filter */
float lpfPid1kHzCoeffs[LPF_PID_COEFFS_1KHZ_LENGTH] = {
	0.5,            0.5
};

float lpfPid2kHzCoeffs[LPF_PID_COEFFS_2KHZ_LENGTH] = {
	0.2330026776,   0.2669973075,   0.2669973075,   0.2330026776
};

float lpfPid4kHzCoeffs[LPF_PID_COEFFS_4KHZ_LENGTH] = {
	0.1104012877,   0.1226609424,    0.131255284,   0.1356824785,   0.1356824785,
	0.131255284,   0.1226609424,   0.1104012877
};

float lpfPid8kHzCoeffs[LPF_PID_COEFFS_8KHZ_LENGTH] = {
	0.05347249657,   0.0569414869,  0.06001491845,   0.0626481548,  0.06480275095,
	0.06644710153,  0.06755700707,  0.06811608374,  0.06811608374,  0.06755700707,
	0.06644710153,  0.06480275095,   0.0626481548,  0.06001491845,   0.0569414869,
	0.05347249657
};

float lpfPid16kHzCoeffs[LPF_PID_COEFFS_16KHZ_LENGTH] = {
	0.02628087252,  0.02719357982,  0.02806142531,  0.02888127975,  0.02965016849,
	0.0303653013,  0.03102406859,  0.03162406385,    0.032163091,  0.03263916448,
	0.03305054083,  0.03339570016,  0.03367336839,  0.03388252482,  0.03402239457,
	0.0340924561,   0.0340924561,  0.03402239457,  0.03388252482,  0.03367336839,
	0.03339570016,  0.03305054083,  0.03263916448,    0.032163091,  0.03162406385,
	0.03102406859,   0.0303653013,  0.02965016849,  0.02888127975,  0.02806142531,
	0.02719357982,  0.02628087252
};

float lpfPid32kHzCoeffs[LPF_PID_COEFFS_32KHZ_LENGTH] = {
	0.01302381046,  0.01325732935,  0.01348554622,  0.01370825339,  0.01392525248,
	0.01413634699,  0.01434134599,  0.01454006322,  0.01473231893,  0.01491793897,
	0.0150967529,  0.01526859868,  0.01543331891,  0.01559076365,  0.01574078947,
	0.01588325575,  0.01601803675,  0.01614500396,  0.01626404375,  0.01637504622,
	0.01647790708,  0.01657253504,  0.01665883884,  0.01673674211,  0.01680617221,
	0.01686706394,  0.01691936329,   0.0169630181,  0.01699799113,   0.0170242507,
	0.01704176888,  0.01705053262,  0.01705053262,  0.01704176888,   0.0170242507,
	0.01699799113,   0.0169630181,  0.01691936329,  0.01686706394,  0.01680617221,
	0.01673674211,  0.01665883884,  0.01657253504,  0.01647790708,  0.01637504622,
	0.01626404375,  0.01614500396,  0.01601803675,  0.01588325575,  0.01574078947,
	0.01559076365,  0.01543331891,  0.01526859868,   0.0150967529,  0.01491793897,
	0.01473231893,  0.01454006322,  0.01434134599,  0.01413634699,  0.01392525248,
	0.01370825339,  0.01348554622,  0.01325732935,  0.01302381046
};

#endif

// NULL filter

float nullFilterApply(void *filter, float input)
{
    UNUSED(filter);
    return input;
}


// PT1 Low Pass filter

void pt1FilterInit(pt1Filter_t *filter, uint8_t f_cut, float dT)
{
    filter->RC = 1.0f / ( 2.0f * M_PI_FLOAT * f_cut );
    filter->dT = dT;
    filter->k = filter->dT / (filter->RC + filter->dT);
}

float pt1FilterApply(pt1Filter_t *filter, float input)
{
    filter->state = filter->state + filter->k * (input - filter->state);
    return filter->state;
}

float pt1FilterApply4(pt1Filter_t *filter, float input, uint8_t f_cut, float dT)
{
    // Pre calculate and store RC
    if (!filter->RC) {
        filter->RC = 1.0f / ( 2.0f * M_PI_FLOAT * f_cut );
        filter->dT = dT;
        filter->k = filter->dT / (filter->RC + filter->dT);
    }

    filter->state = filter->state + filter->k * (input - filter->state);

    return filter->state;
}

// Slew filter with limit

void slewFilterInit(slewFilter_t *filter, float slewLimit, float threshold)
{
    filter->state = 0.0f;
    filter->slewLimit = slewLimit;
    filter->threshold = threshold;
}

float slewFilterApply(slewFilter_t *filter, float input)
{
    if (filter->state >= filter->threshold) {
        if (input >= filter->state - filter->slewLimit) {
            filter->state = input;
        }
    } else if (filter->state <= -filter->threshold) {
        if (input <= filter->state + filter->slewLimit) {
            filter->state = input;
        }
    } else {
        filter->state = input;
    }
    return filter->state;
}


float filterGetNotchQ(uint16_t centerFreq, uint16_t cutoff) {
    float octaves = log2f((float) centerFreq  / (float) cutoff) * 2;
    return sqrtf(powf(2, octaves)) / (powf(2, octaves) - 1);
}

/* sets up a biquad Filter */
void biquadFilterInitLPF(biquadFilter_t *filter, float filterFreq, uint32_t refreshRate)
{
    biquadFilterInit(filter, filterFreq, refreshRate, BIQUAD_Q, FILTER_LPF);
}

void biquadFilterInit(biquadFilter_t *filter, float filterFreq, uint32_t refreshRate, float Q, biquadFilterType_e filterType)
{
    // setup variables
    const float omega = 2.0f * M_PI_FLOAT * filterFreq * refreshRate * 0.000001f;
    const float sn = sin_approx(omega);
    const float cs = cos_approx(omega);
    const float alpha = sn / (2.0f * Q);

    float b0 = 0, b1 = 0, b2 = 0, a0 = 0, a1 = 0, a2 = 0;

    switch (filterType) {
    case FILTER_LPF:
        b0 = (1 - cs) * 0.5f;
        b1 = 1 - cs;
        b2 = (1 - cs) * 0.5f;
        a0 = 1 + alpha;
        a1 = -2 * cs;
        a2 = 1 - alpha;
        break;
    case FILTER_NOTCH:
        b0 =  1;
        b1 = -2 * cs;
        b2 =  1;
        a0 =  1 + alpha;
        a1 = -2 * cs;
        a2 =  1 - alpha;
        break;
    case FILTER_BPF:
        b0 = alpha;
        b1 = 0;
        b2 = -alpha;
        a0 = 1 + alpha;
        a1 = -2 * cs;
        a2 = 1 - alpha;
        break;
    }

    // precompute the coefficients
    filter->b0 = b0 / a0;
    filter->b1 = b1 / a0;
    filter->b2 = b2 / a0;
    filter->a1 = a1 / a0;
    filter->a2 = a2 / a0;

    // zero initial samples
    filter->x1 = filter->x2 = 0;
    filter->y1 = filter->y2 = 0;
}

void biquadFilterUpdate(biquadFilter_t *filter, float filterFreq, uint32_t refreshRate, float Q, biquadFilterType_e filterType)
{
    // backup state
    float x1 = filter->x1;
    float x2 = filter->x2;
    float y1 = filter->y1;
    float y2 = filter->y2;

    biquadFilterInit(filter, filterFreq, refreshRate, Q, filterType);

    // restore state
    filter->x1 = x1;
    filter->x2 = x2;
    filter->y1 = y1;
    filter->y2 = y2;
}

/* Computes a biquadFilter_t filter on a sample (slightly less precise than df2 but works in dynamic mode) */
float biquadFilterApplyDF1(biquadFilter_t *filter, float input)
{
    /* compute result */
    const float result = filter->b0 * input + filter->b1 * filter->x1 + filter->b2 * filter->x2 - filter->a1 * filter->y1 - filter->a2 * filter->y2;

    /* shift x1 to x2, input to x1 */
    filter->x2 = filter->x1;
    filter->x1 = input;

    /* shift y1 to y2, result to y1 */
    filter->y2 = filter->y1;
    filter->y1 = result;

    return result;
}

/* Computes a biquadFilter_t filter in direct form 2 on a sample (higher precision but can't handle changes in coefficients */
float biquadFilterApply(biquadFilter_t *filter, float input)
{
    const float result = filter->b0 * input + filter->x1;
    filter->x1 = filter->b1 * input - filter->a1 * result + filter->x2;
    filter->x2 = filter->b2 * input - filter->a2 * result;
    return result;
}

/*
 * FIR filter
 */
void firFilterInit2(firFilter_t *filter, float *buf, uint8_t bufLength, const float *coeffs, uint8_t coeffsLength)
{
    filter->buf = buf;
    filter->bufLength = bufLength;
    filter->coeffs = coeffs;
    filter->coeffsLength = coeffsLength;
    filter->movingSum = 0.0f;
    filter->index = 0;
    filter->count = 0;
    memset(filter->buf, 0, sizeof(float) * filter->bufLength);
}

/*
 * FIR filter initialisation
 * If the FIR filter is just to be used for averaging, then coeffs can be set to NULL
 */
void firFilterInit(firFilter_t *filter, float *buf, uint8_t bufLength, const float *coeffs)
{
    firFilterInit2(filter, buf, bufLength, coeffs, bufLength);
}

void firFilterUpdate(firFilter_t *filter, float input)
{
    filter->buf[filter->index++] = input; // index is at the first empty buffer positon
    if (filter->index >= filter->bufLength) {
        filter->index = 0;
    }
}

/*
 * Update FIR filter maintaining a moving sum for quick moving average computation
 */
void firFilterUpdateAverage(firFilter_t *filter, float input)
{
    filter->movingSum += input; // sum of the last <count> items, to allow quick moving average computation
    filter->movingSum -=  filter->buf[filter->index]; // subtract the value that "drops off" the end of the moving sum
    filter->buf[filter->index++] = input; // index is at the first empty buffer positon
    if (filter->index >= filter->bufLength) {
        filter->index = 0;
    }
    if (filter->count < filter->bufLength) {
        ++filter->count;
    }
}

float firFilterApply(const firFilter_t *filter)
{
    float ret = 0.0f;
    int ii = 0;
    int index;
    for (index = filter->index - 1; index >= 0; ++ii, --index) {
        ret += filter->coeffs[ii] * filter->buf[index];
    }
    for (index = filter->bufLength - 1; ii < filter->coeffsLength; ++ii, --index) {
        ret += filter->coeffs[ii] * filter->buf[index];
    }
    return ret;
}

float firFilterUpdateAndApply(firFilter_t *filter, float input)
{
    firFilterUpdate(filter, input);
    return firFilterApply(filter);
}

/*
 * Returns average of the last <count> items.
 */
float firFilterCalcPartialAverage(const firFilter_t *filter, uint8_t count)
{
    float ret = 0.0f;
    int index = filter->index;
    for (int ii = 0; ii < filter->coeffsLength; ++ii) {
        --index;
        if (index < 0) {
            index = filter->bufLength - 1;
        }
        ret += filter->buf[index];
    }
    return ret / count;
}

float firFilterCalcMovingAverage(const firFilter_t *filter)
{
    return filter->movingSum / filter->count;
}

float firFilterLastInput(const firFilter_t *filter)
{
    // filter->index points to next empty item in buffer
    const int index = filter->index == 0 ? filter->bufLength - 1 : filter->index - 1;
    return filter->buf[index];
}

void firFilterDenoiseInit(firFilterDenoise_t *filter, uint8_t gyroSoftLpfHz, uint16_t targetLooptime)
{
    memset(filter, 0, sizeof(firFilterDenoise_t));
    filter->targetCount = constrain(lrintf((1.0f / (0.000001f * (float)targetLooptime)) / gyroSoftLpfHz), 1, MAX_FIR_DENOISE_WINDOW_SIZE);
}

// prototype function for denoising of signal by dynamic moving average. Mainly for test purposes
float firFilterDenoiseUpdate(firFilterDenoise_t *filter, float input)
{
    filter->state[filter->index] = input;
    filter->movingSum += filter->state[filter->index++];
    if (filter->index == filter->targetCount) {
        filter->index = 0;
    }
    filter->movingSum -= filter->state[filter->index];

    if (filter->targetCount >= filter->filledCount) {
        return filter->movingSum / filter->targetCount;
    } else {
        return filter->movingSum / ++filter->filledCount + 1;
    }
}

#if USE_ADAPTIVE_FILTER

void adaptiveFilterInit(adaptiveFilter_t *filter, uint32_t refreshRate, uint8_t axis, uint8_t pidFilter)
{
	uint8_t i, fs_idx;
	uint16_t lpfCoeffsSize;

//	if (pidFilter > 0)
//	{
		if(refreshRate == 1000) {
			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_1KHZ;
			filter->noiseFilter.lpfFirTaps = (float *)&lpfPid1kHzCoeffs[0];
			lpfCoeffsSize = LPF_PID_FILTER_FS_1KHZ_TAPS_SIZE;
		} else if(refreshRate == 500) {
			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_2KHZ;
			filter->noiseFilter.lpfFirTaps = (float *)&lpfPid2kHzCoeffs[0];
			lpfCoeffsSize = LPF_PID_FILTER_FS_2KHZ_TAPS_SIZE;
		} else if(refreshRate == 250) {
			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_4KHZ;
			filter->noiseFilter.lpfFirTaps = (float *)&lpfPid4kHzCoeffs[0];
			lpfCoeffsSize = LPF_PID_FILTER_FS_4KHZ_TAPS_SIZE;
		} else if(refreshRate == 125) {
			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_8KHZ;
			filter->noiseFilter.lpfFirTaps = (float *)&lpfPid8kHzCoeffs[0];
			lpfCoeffsSize = LPF_PID_FILTER_FS_8KHZ_TAPS_SIZE;
		} else if(refreshRate == 63) {
			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_16KHZ;
			filter->noiseFilter.lpfFirTaps = (float *)&lpfPid16kHzCoeffs[0];
			lpfCoeffsSize = LPF_PID_FILTER_FS_16KHZ_TAPS_SIZE;
		} else {
			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_32KHZ;
			filter->noiseFilter.lpfFirTaps = (float *)&lpfPid32kHzCoeffs[0];
			lpfCoeffsSize = LPF_PID_FILTER_FS_32KHZ_TAPS_SIZE;
		}
//	} else {
//		if(refreshRate == 1000) {
//			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_1KHZ;
//			filter->noiseFilter.lpfFirTaps = (float *)&lpfGyro1kHzCoeffs[0];
//			lpfCoeffsSize = LPF_GYRO_FILTER_FS_1KHZ_TAPS_SIZE;
//		} else if(refreshRate == 500) {
//			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_2KHZ;
//			filter->noiseFilter.lpfFirTaps = (float *)&lpfGyro2kHzCoeffs[0];
//			lpfCoeffsSize = LPF_GYRO_FILTER_FS_2KHZ_TAPS_SIZE;
//		} else if(refreshRate == 250) {
//			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_4KHZ;
//			filter->noiseFilter.lpfFirTaps = (float *)&lpfGyro4kHzCoeffs[0];
//			lpfCoeffsSize = LPF_GYRO_FILTER_FS_4KHZ_TAPS_SIZE;
//		} else if(refreshRate == 125) {
//			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_8KHZ;
//			filter->noiseFilter.lpfFirTaps = (float *)&lpfGyro8kHzCoeffs[0];
//			lpfCoeffsSize = LPF_GYRO_FILTER_FS_8KHZ_TAPS_SIZE;
//		} else if(refreshRate == 63) {
//			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_16KHZ;
//			filter->noiseFilter.lpfFirTaps = (float *)&lpfGyro16kHzCoeffs[0];
//			lpfCoeffsSize = LPF_GYRO_FILTER_FS_16KHZ_TAPS_SIZE;
//		} else {
//			fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_32KHZ;
//			filter->noiseFilter.lpfFirTaps = (float *)&lpfGyro32kHzCoeffs[0];
//			lpfCoeffsSize = LPF_GYRO_FILTER_FS_32KHZ_TAPS_SIZE;
//		}
//	}

	/* Set LPF filter sampling freq. and taps size. */
	filter->axis = axis;
	filter->lpfFsTaps.fs = adaptiveFilterFsAndTapsSize[fs_idx][0];
	filter->lpfFsTaps.taps = adaptiveFilterFsAndTapsSize[fs_idx][2];

	/* Block size is set to 1, we process only one sample at a time. */
	arm_fir_init_f32(&filter->lpf_instance, lpfCoeffsSize, (float32_t *)filter->noiseFilter.lpfFirTaps,
			(float32_t *)&filter->noiseFilter.lpfFirState[0], ADATIVE_FILTER_BS);

	for (i = 0; i < lpfCoeffsSize; i++) {
		filter->noiseFilter.lpfFirState[i] = 0.0;
	}
}

float adaptiveFilterApply(adaptiveFilter_t *filter, float input)
{
	float noise = 0.0;

	arm_fir_f32(&filter->lpf_instance, &input, &noise, ADATIVE_FILTER_BS);
	//DEBUG_SET(DEBUG_GYRO_RAW, filter->axis, output);

	return noise;
}

#endif
