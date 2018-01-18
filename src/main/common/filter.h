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

#include <stdint.h>
#include <math.h>
#include "arm_math.h"

#pragma once

#include "common/axis.h"
#include "common/hpf_coeffs.h"

// Don't use it on F1 and F3 to lower RAM usage
// FIR/Denoise filter can be cleaned up in the future as it is rarely used and used to be experimental
#if (defined(STM32F1) || defined(STM32F3))
#define MAX_FIR_DENOISE_WINDOW_SIZE 1
#else
#define MAX_FIR_DENOISE_WINDOW_SIZE 120
#endif

#if (defined(STM32F1) || defined(STM32F3))
#define USE_ADAPTIVE_FILTER 0
#else
#define USE_ADAPTIVE_FILTER 1
#endif

#if USE_ADAPTIVE_FILTER

#define ADATIVE_FILTER_BS					(1)
#define ADATIVE_FILTER_STEP_SIZE			(0.04)

#define ADAPTIVE_FILTER_FS_1KHZ_VALUE		(1000)
#define ADAPTIVE_FILTER_FS_2KHZ_VALUE		(2000)
#define ADAPTIVE_FILTER_FS_4KHZ_VALUE		(4000)
#define ADAPTIVE_FILTER_FS_8KHZ_VALUE		(8000)
#define ADAPTIVE_FILTER_FS_16KHZ_VALUE		(16000)

/* Filter delay is 2.5ms */
#define ADAPTIVE_FILTER_FS_1KHZ_TAPS_SIZE	(6)
#define ADAPTIVE_FILTER_FS_2KHZ_TAPS_SIZE	(11)
#define ADAPTIVE_FILTER_FS_4KHZ_TAPS_SIZE	(21)
#define ADAPTIVE_FILTER_FS_8KHZ_TAPS_SIZE	(41)
#define ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE	(81)

/* Filter delay is 3ms */
#define HPF_FILTER_FS_1KHZ_TAPS_SIZE 		(HPF_COEFFS_1KHZ_LENGTH)
#define HPF_FILTER_FS_2KHZ_TAPS_SIZE 		(HPF_COEFFS_2KHZ_LENGTH)
#define HPF_FILTER_FS_4KHZ_TAPS_SIZE 		(HPF_COEFFS_4KHZ_LENGTH)
#define HPF_FILTER_FS_8KHZ_TAPS_SIZE 		(HPF_COEFFS_8KHZ_LENGTH)
#define HPF_FILTER_FS_16KHZ_TAPS_SIZE 		(HPF_COEFFS_16KHZ_LENGTH)
#define HPF_FILTER_FS_32KHZ_TAPS_SIZE 		(HPF_COEFFS_32KHZ_LENGTH)

typedef enum {
	ADAPTIVE_FILTER_FS_1KHZ = 0,
	ADAPTIVE_FILTER_FS_2KHZ,
    ADAPTIVE_FILTER_FS_4KHZ,
    ADAPTIVE_FILTER_FS_8KHZ,
    ADAPTIVE_FILTER_FS_16KHZ,
    ADAPTIVE_FILTER_FS_32KHZ,
    ADAPTIVE_FILTER__MAX
} adaptiveFilterFs_e;

typedef struct filterFsToTaps_s {
	uint16_t fs;
	uint16_t taps;
} filterFsToTaps_t;

typedef struct hpfNoiseFitler_s {
	float *hpfFirTaps; /* Pointer to HPF FIR filter taps to get noise signal */
	float hpfFirState[HPF_FILTER_FS_32KHZ_TAPS_SIZE]; /* HPF FIR filter state */
} hpfNoiseFitler_t;

typedef struct adaptiveFilter_s {
	uint8_t axis;
	float state;
	float nlmsFirTaps[ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE];
	float nlmsFirState[ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE];
	hpfNoiseFitler_t noiseFilter;
	filterFsToTaps_t nlmsFsTaps;
	filterFsToTaps_t hpfFsTaps;
	arm_lms_norm_instance_f32 lms_instance;
	arm_fir_instance_f32 hpf_instance;
} adaptiveFilter_t;

#endif

typedef struct pt1Filter_s {
    float state;
    float k;
    float RC;
    float dT;
} pt1Filter_t;

typedef struct slewFilter_s {
    float state;
    float slewLimit;
    float threshold;
} slewFilter_t;

/* this holds the data required to update samples thru a filter */
typedef struct biquadFilter_s {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
} biquadFilter_t;

typedef struct firFilterDenoise_s {
    int filledCount;
    int targetCount;
    int index;
    float movingSum;
    float state[MAX_FIR_DENOISE_WINDOW_SIZE];
} firFilterDenoise_t;

typedef enum {
    FILTER_PT1 = 0,
    FILTER_BIQUAD,
    FILTER_FIR,
    FILTER_SLEW
} filterType_e;

typedef enum {
    FILTER_LPF,
    FILTER_NOTCH,
    FILTER_BPF,
} biquadFilterType_e;

typedef struct firFilter_s {
    float *buf;
    const float *coeffs;
    float movingSum;
    uint8_t index;
    uint8_t count;
    uint8_t bufLength;
    uint8_t coeffsLength;
} firFilter_t;

typedef float (*filterApplyFnPtr)(void *filter, float input);

float nullFilterApply(void *filter, float input);

void biquadFilterInitLPF(biquadFilter_t *filter, float filterFreq, uint32_t refreshRate);
void biquadFilterInit(biquadFilter_t *filter, float filterFreq, uint32_t refreshRate, float Q, biquadFilterType_e filterType);
void biquadFilterUpdate(biquadFilter_t *filter, float filterFreq, uint32_t refreshRate, float Q, biquadFilterType_e filterType);
float biquadFilterApplyDF1(biquadFilter_t *filter, float input);
float biquadFilterApply(biquadFilter_t *filter, float input);
float filterGetNotchQ(uint16_t centerFreq, uint16_t cutoff);

// not exactly correct, but very very close and much much faster
#define filterGetNotchQApprox(centerFreq, cutoff)   ((float)(cutoff * centerFreq) / ((float)(centerFreq - cutoff) * (float)(centerFreq + cutoff)))

void pt1FilterInit(pt1Filter_t *filter, uint8_t f_cut, float dT);
float pt1FilterApply(pt1Filter_t *filter, float input);
float pt1FilterApply4(pt1Filter_t *filter, float input, uint8_t f_cut, float dT);

void slewFilterInit(slewFilter_t *filter, float slewLimit, float threshold);
float slewFilterApply(slewFilter_t *filter, float input);

void firFilterInit(firFilter_t *filter, float *buf, uint8_t bufLength, const float *coeffs);
void firFilterInit2(firFilter_t *filter, float *buf, uint8_t bufLength, const float *coeffs, uint8_t coeffsLength);
void firFilterUpdate(firFilter_t *filter, float input);
void firFilterUpdateAverage(firFilter_t *filter, float input);
float firFilterApply(const firFilter_t *filter);
float firFilterUpdateAndApply(firFilter_t *filter, float input);
float firFilterCalcPartialAverage(const firFilter_t *filter, uint8_t count);
float firFilterCalcMovingAverage(const firFilter_t *filter);
float firFilterLastInput(const firFilter_t *filter);

void firFilterDenoiseInit(firFilterDenoise_t *filter, uint8_t gyroSoftLpfHz, uint16_t targetLooptime);
float firFilterDenoiseUpdate(firFilterDenoise_t *filter, float input);

#if USE_ADAPTIVE_FILTER

void adaptiveFilterInit(adaptiveFilter_t *filter, uint32_t refreshRate, uint8_t axis);
float adaptiveFilterApply(adaptiveFilter_t *filter, float input);

#endif
