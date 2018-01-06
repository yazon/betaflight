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
//#include "platform.h"
#include "arm_math.h"

#include "common/filter.h"
#include "common/maths.h"
#include "common/utils.h"

#define M_LN2_FLOAT 0.69314718055994530942f
#define M_PI_FLOAT  3.14159265358979323846f
#define BIQUAD_BANDWIDTH 1.9f     /* bandwidth in octaves */
#define BIQUAD_Q 1.0f / sqrtf(2.0f)     /* quality factor - butterworth*/

#if USE_STATIC_IIR_FILTER
//#define A_LENGTH 		13
#define B_LENGTH 		91

/* a coefficients (numerator), a0 is 1, we skip it */
//double aIirCoeffs[A_LENGTH] = {1.0, -10.9932247187354, 55.4329790485894, -169.536483253792, 350.265265970947,
//							  -514.995208299233, 552.548751876674, -435.891944304471, 250.927570674114,
//							  -102.799294531210, 28.4492510419751, -4.77532989250449, 0.367666387654882};
//
///* b coefficients (denominator) */
//double bIirCoeffs[B_LENGTH] = {0.000431593183932006, -0.00398642651320452, 0.0162123852450434, -0.0373605256407417,
//							  0.0509807835975414, -0.0346460500463756, -0.00851441217836203, 0.0465753359589142,
//							  -0.0591101242679946, 0.0588348785639115, -0.0591101124459702, 0.0465753173287827,
//							  -0.00851440706971594, -0.0346460223295452, 0.0509807326167808, -0.0373604808081356,
//							  0.0162123625477188, -0.00398642013492689, 0.000431592407064935};
float bFirCoeffs[B_LENGTH] = {
		-2.613850484e-05,2.168760511e-05,7.498973719e-05,0.0001380539907,0.0002154018439,
		  0.0003117166343,0.0004317650164,0.0005803143722,0.0007620477118,0.0009814773221,
		   0.001242858358, 0.001550104353, 0.001906705322, 0.002315649996, 0.002779355273,
		   0.003299599746, 0.003877468407, 0.004513305612, 0.005206675269, 0.005956338719,
		   0.006760234945, 0.007615482435,  0.00851838477, 0.009464453906,  0.01044844463,
		    0.01146439929,  0.01250570361,  0.01356515847,  0.01463505067,  0.01570724696,
		    0.01677327789,  0.01782444865,  0.01885193959,  0.01984690502,  0.02080060355,
		    0.02170448564,  0.02255031839,  0.02333028801,  0.02403709479,  0.02466405369,
		    0.02520518005,  0.02565527335,    0.026009975,  0.02626583725,  0.02642036229,
		    0.02647203952,  0.02642036229,  0.02626583725,    0.026009975,  0.02565527335,
		    0.02520518005,  0.02466405369,  0.02403709479,  0.02333028801,  0.02255031839,
		    0.02170448564,  0.02080060355,  0.01984690502,  0.01885193959,  0.01782444865,
		    0.01677327789,  0.01570724696,  0.01463505067,  0.01356515847,  0.01250570361,
		    0.01146439929,  0.01044844463, 0.009464453906,  0.00851838477, 0.007615482435,
		   0.006760234945, 0.005956338719, 0.005206675269, 0.004513305612, 0.003877468407,
		   0.003299599746, 0.002779355273, 0.002315649996, 0.001906705322, 0.001550104353,
		   0.001242858358,0.0009814773221,0.0007620477118,0.0005803143722,0.0004317650164,
		  0.0003117166343,0.0002154018439,0.0001380539907,7.498973719e-05,2.168760511e-05,
		  -2.613850484e-05};

//float firState[B_LENGTH+1];

//float iirInputBuf[XYZ_AXIS_COUNT][B_LENGTH];
//float iirOutputBuf[XYZ_AXIS_COUNT][A_LENGTH];

#endif

void arm_fir_init_f32(arm_fir_instance_f32 * S, uint16_t numTaps, float32_t * pCoeffs,float32_t * pState, uint32_t blockSize);
void arm_fir_f32(const arm_fir_instance_f32 * S, float32_t * pSrc,float32_t * pDst, uint32_t blockSize);


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

#if USE_STATIC_IIR_FILTER

void iirFilterInit(lpfIIRFitler_t *filter, axis_e axis)
{
	uint8_t i;
	/* We don't support other sampling frequencies yet. */
	filter->fs = IIR_FS_8KHZ;
//	filter->aLength = A_LENGTH;
	filter->bLength = B_LENGTH;
	//filter->aCoeffs = (float *) aIirCoeffs;
	//filter->bCoeffs = (float *) bIirCoeffs;
	//filter->inputBuf = (float *) iirInputBuf[axis % XYZ_AXIS_COUNT];
	//filter->outputBuf = (float *) iirOutputBuf[axis % XYZ_AXIS_COUNT];

	for (i = 0; i < B_LENGTH; i++) {
		filter->firState[i] = 0.0;
	}

	axis = axis;

	arm_fir_init_f32((arm_fir_instance_f32 *)&filter->fir_instance, B_LENGTH, (float32_t *)&bFirCoeffs[0], (float32_t *)&filter->firState[0], 1);

//	for (i = 0; i < A_LENGTH; i++) {
//		filter->outputBuf[i] = 0.0;
//	}
}

//float iirFilterApply(lpfIIRFitler_t *filter, float input)
//{
////	uint8_t i;
//	float result = 0.0;
//
//    /* Write new data to buffer */
////	filter->inputBuf[0] = (double) input;
////
////	/* Calculate new ouput */
////	for (i = 0; i < B_LENGTH; i++) {
////		result += (filter->inputBuf[i] * bIirCoeffs[i]);
////	}
////
////	for (i = 1; i < A_LENGTH; i++) {
////		result -= (filter->outputBuf[i-1] * aIirCoeffs[i]);
////	}
////
////	/* Move input and output samples by 1 */
////	for (i = B_LENGTH-1; i > 0; i--) {
////		filter->inputBuf[i] = filter->inputBuf[i-1];
////	}
////
////	for (i = A_LENGTH-1; i > 0; i--) {
////		filter->outputBuf[i] = filter->outputBuf[i-1];
////	}
////
////	/* Store result */
////	filter->outputBuf[0] = result;
//	//filter->state = result;
//
//	return result;
//}


float firFilterApplyMy(lpfIIRFitler_t *filter, float input)
{
//	uint8_t i;
	float result = 0.0;
//
//    /* Write new data to buffer */
//	filter->inputBuf[0] = input;
//
//	/* Calculate new ouput */
//	for (i = 0; i < B_LENGTH; i++) {
//		result += (filter->inputBuf[i] * bFirCoeffs[i]);
//	}
//
//	/* Move input and output samples by 1 */
//	for (i = B_LENGTH-1; i > 0; i--) {
//		filter->inputBuf[i] = filter->inputBuf[i-1];
//	}
//
//	/* Return result */
//	return (float) result;

	arm_fir_f32((arm_fir_instance_f32 *)&filter->fir_instance, &input, &result, 1);

	return result;
}

#endif
