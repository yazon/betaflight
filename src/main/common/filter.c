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
	{ADAPTIVE_FILTER_FS_1KHZ_VALUE, ADAPTIVE_FILTER_FS_1KHZ_TAPS_SIZE, HPF_FILTER_FS_1KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_2KHZ_VALUE, ADAPTIVE_FILTER_FS_2KHZ_TAPS_SIZE, HPF_FILTER_FS_2KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_4KHZ_VALUE, ADAPTIVE_FILTER_FS_4KHZ_TAPS_SIZE, HPF_FILTER_FS_4KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_8KHZ_VALUE, ADAPTIVE_FILTER_FS_8KHZ_TAPS_SIZE, HPF_FILTER_FS_8KHZ_TAPS_SIZE},
	{ADAPTIVE_FILTER_FS_16KHZ_VALUE, ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE, HPF_FILTER_FS_16KHZ_TAPS_SIZE}};

float hpf1kHzCoeffs[HPF_COEFFS_1KHZ_LENGTH] = {
	0.1177067384,   0.1866018176,   0.2388461232,   0.2388461232,   0.1866018176,
	0.1177067384
};

float hpf2kHzCoeffs[HPF_COEFFS_2KHZ_LENGTH] = {
	0.05949703977,  0.07504294813,  0.09160571545,   0.1078507453,   0.1202100739,
	0.1248726249,   0.1202100739,   0.1078507453,  0.09160571545,  0.07504294813,
	0.05949703977
};

float hpf4kHzCoeffs[HPF_COEFFS_4KHZ_LENGTH] = {
//   0.000374071009,0.0004622917331,0.0005667991354,0.0006925721536,0.0008424897096,
//    0.00101677503, 0.001212538453, 0.001423445763, 0.001639535651, 0.001847203588,
//   0.002029361902, 0.002165780636, 0.002233604668, 0.002208032412,  0.00206314004,
//   0.001772822929, 0.001311819186,0.0006567826495,-0.0002126396284,-0.001312766224,
//  -0.002654920332,-0.004244588781,-0.006080741063, -0.00815534126, -0.01045307238,
//   -0.01295131166, -0.01562033687, -0.01842379756, -0.02131941915,  -0.0242599342,
//   -0.02719421312, -0.03006855957, -0.03282812983, -0.03541843221, -0.03778684512,
//    -0.0398841314, -0.04166585952, -0.04309371114, -0.04413662851, -0.04477172717,
//     0.9546816945, -0.04477172717, -0.04413662851, -0.04309371114, -0.04166585952,
//    -0.0398841314, -0.03778684512, -0.03541843221, -0.03282812983, -0.03006855957,
//   -0.02719421312,  -0.0242599342, -0.02131941915, -0.01842379756, -0.01562033687,
//   -0.01295131166, -0.01045307238, -0.00815534126,-0.006080741063,-0.004244588781,
//  -0.002654920332,-0.001312766224,-0.0002126396284,0.0006567826495, 0.001311819186,
//   0.001772822929,  0.00206314004, 0.002208032412, 0.002233604668, 0.002165780636,
//   0.002029361902, 0.001847203588, 0.001639535651, 0.001423445763, 0.001212538453,
//    0.00101677503,0.0008424897096,0.0006925721536,0.0005667991354,0.0004622917331,
//   0.000374071009
	0.03014974482,  0.03371924907,  0.03738739714,  0.04126530513,  0.04536224902,
	0.04956695065,  0.05365596712,  0.05732771009,  0.06025525555,  0.06214680523,
	0.06280121952,  0.06214680523,  0.06025525555,  0.05732771009,  0.05365596712,
	0.04956695065,  0.04536224902,  0.04126530513,  0.03738739714,  0.03371924907,
	0.03014974482
};

float hpf8kHzCoeffs[HPF_COEFFS_8KHZ_LENGTH] = {
//  0.0001870647684,0.0002082936262,0.0002311820572,0.0002561182191,0.0002834439219,
//  0.0003134447325,0.0003463402973,0.0003822752042,0.0004213108041,0.0004634172656,
//  0.0005084670847,0.0005562291481,0.0006063641049,0.0006584209041, 0.000711834291,
//  0.0007659242838,0.0008198961732,0.0008728425601,0.0009237463819,0.0009714857442,
//   0.001014839741,  0.00105249614, 0.001083059818, 0.001105063246,  0.00111697719,
//   0.001117223641, 0.001104188967, 0.001076238579, 0.001031731605,0.0009690370061,
//  0.0008865502314,0.0007827093359,0.0006560122711,0.0005050338805,0.0003284427221,
//   0.000125017541,-0.0001063364616,-0.0003665721742,-0.0006564858486,-0.000976703479,
//  -0.001327667967, -0.00170962757, -0.00212262664, -0.00256649591,-0.003040846437,
//  -0.003545064945,-0.004078308586,-0.004639505874,-0.005227354355,-0.005840326194,
//  -0.006476669572,-0.007134416606, -0.00781139126,-0.008505219594,-0.009213340469,
//  -0.009933023714, -0.01066137757, -0.01139537524, -0.01213186514,   -0.012867596,
//   -0.01359923463, -0.01432339009, -0.01503663324, -0.01573552564, -0.01641663536,
//   -0.01707656868, -0.01771198958, -0.01831964217, -0.01889638044, -0.01943918504,
//   -0.01994518749,  -0.0204116907, -0.02083619125, -0.02121639252, -0.02155022882,
//   -0.02183587663, -0.02207176946, -0.02225660719, -0.02238936909, -0.02246932127,
//      0.977327168, -0.02246932127, -0.02238936909, -0.02225660719, -0.02207176946,
//   -0.02183587663, -0.02155022882, -0.02121639252, -0.02083619125,  -0.0204116907,
//   -0.01994518749, -0.01943918504, -0.01889638044, -0.01831964217, -0.01771198958,
//   -0.01707656868, -0.01641663536, -0.01573552564, -0.01503663324, -0.01432339009,
//   -0.01359923463,   -0.012867596, -0.01213186514, -0.01139537524, -0.01066137757,
//  -0.009933023714,-0.009213340469,-0.008505219594, -0.00781139126,-0.007134416606,
//  -0.006476669572,-0.005840326194,-0.005227354355,-0.004639505874,-0.004078308586,
//  -0.003545064945,-0.003040846437, -0.00256649591, -0.00212262664, -0.00170962757,
//  -0.001327667967,-0.000976703479,-0.0006564858486,-0.0003665721742,-0.0001063364616,
//   0.000125017541,0.0003284427221,0.0005050338805,0.0006560122711,0.0007827093359,
//  0.0008865502314,0.0009690370061, 0.001031731605, 0.001076238579, 0.001104188967,
//   0.001117223641,  0.00111697719, 0.001105063246, 0.001083059818,  0.00105249614,
//   0.001014839741,0.0009714857442,0.0009237463819,0.0008728425601,0.0008198961732,
//  0.0007659242838, 0.000711834291,0.0006584209041,0.0006063641049,0.0005562291481,
//  0.0005084670847,0.0004634172656,0.0004213108041,0.0003822752042,0.0003463402973,
//  0.0003134447325,0.0002834439219,0.0002561182191,0.0002311820572,0.0002082936262,
//  0.0001870647684
	0.015785506,  0.01659414917,  0.01740231179,  0.01822611317,  0.01907925121,
	0.01997187547,  0.02090970427,  0.02189338952,  0.02291820385,  0.02397404239,
	0.02504577115,  0.02611387335,  0.02715536393,  0.02814494818,  0.02905631065,
	0.02986350469,  0.03054232709,   0.0310716182,  0.03143441305,  0.03161885589,
	0.03161885589,  0.03143441305,   0.0310716182,  0.03054232709,  0.02986350469,
	0.02905631065,  0.02814494818,  0.02715536393,  0.02611387335,  0.02504577115,
	0.02397404239,  0.02291820385,  0.02189338952,  0.02090970427,  0.01997187547,
	0.01907925121,  0.01822611317,  0.01740231179,  0.01659414917,    0.015785506
};

float hpf16kHzCoeffs[HPF_COEFFS_16KHZ_LENGTH] = {
	0.00892344676, 0.009077665396, 0.009230666794, 0.009384483099, 0.009541069157,
	0.009702256881, 0.009869719855,  0.01004493795,   0.0102291666,  0.01042340696,
	0.01062838361,  0.01084452868,  0.01107196696,  0.01131050754,  0.01155964565,
	0.0118185645,  0.01208614651,  0.01236098632,    0.012641415,  0.01292551961,
	0.01321117952,  0.01349609531,  0.01377782971,  0.01405384205,  0.01432153396,
	0.01457829215,  0.01482152753,  0.01504872553,  0.01525748055,  0.01544554066,
	0.01561084483,  0.01575155556,   0.0158660952,  0.01595316455,  0.01601177081,
	0.01604124345,  0.01604124345,  0.01601177081,  0.01595316455,   0.0158660952,
	0.01575155556,  0.01561084483,  0.01544554066,  0.01525748055,  0.01504872553,
	0.01482152753,  0.01457829215,  0.01432153396,  0.01405384205,  0.01377782971,
	0.01349609531,  0.01321117952,  0.01292551961,    0.012641415,  0.01236098632,
	0.01208614651,   0.0118185645,  0.01155964565,  0.01131050754,  0.01107196696,
	0.01084452868,  0.01062838361,  0.01042340696,   0.0102291666,  0.01004493795,
	0.009869719855, 0.009702256881, 0.009541069157, 0.009384483099, 0.009230666794,
	0.009077665396,  0.00892344676
};

float hpf32kHzCoeffs[HPF_COEFFS_32KHZ_LENGTH] = {
	0.005295984913, 0.005364173092, 0.005431761034, 0.005499001127, 0.005566145293,
	0.005633440334, 0.005701126996,  0.00576943811, 0.005838596262, 0.005908812396,
	0.005980284419, 0.006053195335, 0.006127711385, 0.006203981116, 0.006282134913,
	0.00636228174, 0.006444509607, 0.006528885569, 0.006615452934, 0.006704231258,
	0.006795217749, 0.006888383999, 0.006983679254, 0.007081027608, 0.007180328947,
	0.007281460799, 0.007384276949, 0.007488609757,   0.0075942697, 0.007701046299,
	0.007808710914, 0.007917015813, 0.008025696501, 0.008134474978, 0.008243057877,
	0.008351139724, 0.008458406664, 0.008564534597, 0.008669192903, 0.008772049099,
	0.008872764185, 0.008971001022, 0.009066424333, 0.009158700705, 0.009247503243,
	0.009332511574, 0.009413416497, 0.009489917196, 0.009561727755, 0.009628577158,
	0.009690210223, 0.009746389464, 0.009796896949, 0.009841536172, 0.009880132042,
	0.009912531823, 0.009938607924, 0.009958256036,  0.00997139886, 0.009977983311,
	0.009977983311,  0.00997139886, 0.009958256036, 0.009938607924, 0.009912531823,
	0.009880132042, 0.009841536172, 0.009796896949, 0.009746389464, 0.009690210223,
	0.009628577158, 0.009561727755, 0.009489917196, 0.009413416497, 0.009332511574,
	0.009247503243, 0.009158700705, 0.009066424333, 0.008971001022, 0.008872764185,
	0.008772049099, 0.008669192903, 0.008564534597, 0.008458406664, 0.008351139724,
	0.008243057877, 0.008134474978, 0.008025696501, 0.007917015813, 0.007808710914,
	0.007701046299,   0.0075942697, 0.007488609757, 0.007384276949, 0.007281460799,
	0.007180328947, 0.007081027608, 0.006983679254, 0.006888383999, 0.006795217749,
	0.006704231258, 0.006615452934, 0.006528885569, 0.006444509607,  0.00636228174,
	0.006282134913, 0.006203981116, 0.006127711385, 0.006053195335, 0.005980284419,
	0.005908812396, 0.005838596262,  0.00576943811, 0.005701126996, 0.005633440334,
	0.005566145293, 0.005499001127, 0.005431761034, 0.005364173092, 0.005295984913
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

void adaptiveFilterInit(adaptiveFilter_t *filter, uint32_t refreshRate, uint8_t axis)
{
	uint8_t i, fs_idx;
	uint16_t hpfCoeffsSize, nlmsCoeffsSize;

	if(refreshRate == 1000) {
		fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_1KHZ;
		filter->noiseFilter.hpfFirTaps = (float *)&hpf1kHzCoeffs[0];
		hpfCoeffsSize = HPF_FILTER_FS_1KHZ_TAPS_SIZE;
		nlmsCoeffsSize = ADAPTIVE_FILTER_FS_1KHZ_TAPS_SIZE;
	} else if(refreshRate == 500) {
		fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_2KHZ;
		filter->noiseFilter.hpfFirTaps = (float *)&hpf2kHzCoeffs[0];
		hpfCoeffsSize = HPF_FILTER_FS_2KHZ_TAPS_SIZE;
		nlmsCoeffsSize = ADAPTIVE_FILTER_FS_2KHZ_TAPS_SIZE;
	} else if(refreshRate == 250) {
		fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_4KHZ;
		filter->noiseFilter.hpfFirTaps = (float *)&hpf4kHzCoeffs[0];
		hpfCoeffsSize = HPF_FILTER_FS_4KHZ_TAPS_SIZE;
		nlmsCoeffsSize = ADAPTIVE_FILTER_FS_4KHZ_TAPS_SIZE;
	} else if(refreshRate == 125) {
		fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_8KHZ;
		filter->noiseFilter.hpfFirTaps = (float *)&hpf8kHzCoeffs[0];
		hpfCoeffsSize = HPF_FILTER_FS_8KHZ_TAPS_SIZE;
		nlmsCoeffsSize = ADAPTIVE_FILTER_FS_8KHZ_TAPS_SIZE;
	} else if(refreshRate == 63) {
		fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_16KHZ;
		filter->noiseFilter.hpfFirTaps = (float *)&hpf16kHzCoeffs[0];
		hpfCoeffsSize = HPF_FILTER_FS_16KHZ_TAPS_SIZE;
		nlmsCoeffsSize = ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE;
	} else {
		fs_idx = (uint8_t) ADAPTIVE_FILTER_FS_32KHZ;
		filter->noiseFilter.hpfFirTaps = (float *)&hpf32kHzCoeffs[0];
		hpfCoeffsSize = HPF_FILTER_FS_32KHZ_TAPS_SIZE;
		nlmsCoeffsSize = ADAPTIVE_FILTER_FS_16KHZ_TAPS_SIZE;
	}


	/* Set HPF and NLMS filter sampling freq. and taps size. */
	filter->axis = axis;
//	filter->nlmsFsTaps.fs = adaptiveFilterFsAndTapsSize[fs_idx][0];
//	filter->nlmsFsTaps.taps = adaptiveFilterFsAndTapsSize[fs_idx][1];
	filter->hpfFsTaps.fs = adaptiveFilterFsAndTapsSize[fs_idx][0];
	filter->hpfFsTaps.taps = adaptiveFilterFsAndTapsSize[fs_idx][2];

	/* Block size is set to 1, we process only one sample at a time. */
	arm_fir_init_f32(&filter->hpf_instance, hpfCoeffsSize, (float32_t *)filter->noiseFilter.hpfFirTaps,
			(float32_t *)&filter->noiseFilter.hpfFirState[0], ADATIVE_FILTER_BS);
//	arm_lms_norm_init_f32(&filter->lms_instance, nlmsCoeffsSize, (float32_t *)&filter->nlmsFirTaps[0],
//			(float32_t *)&filter->nlmsFirState[0], (float32_t)ADATIVE_FILTER_STEP_SIZE,  ADATIVE_FILTER_BS);

	for (i = 0; i < hpfCoeffsSize; i++) {
		filter->noiseFilter.hpfFirState[i] = 0.0;
	}
//	for (i = 0; i < nlmsCoeffsSize; i++) {
//		filter->nlmsFirState[i] = 0.0;
//	}
}

float adaptiveFilterApply(adaptiveFilter_t *filter, float input)
{
	float noise = 0.0;
	//float error = 0.0;
	//float output = 0.0;

	arm_fir_f32(&filter->hpf_instance, &input, &noise, ADATIVE_FILTER_BS);
	//arm_lms_norm_f32(&filter->lms_instance, &input, &noise, &output, &error, ADATIVE_FILTER_BS);
	//DEBUG_SET(DEBUG_GYRO_RAW, filter->axis, output);

	return noise;
}

#endif
