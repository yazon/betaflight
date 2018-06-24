![Important Notice: This is custom FW with added a low pass FIR static filter.](https://raw.githubusercontent.com/wiki/betaflight/betaflight/images/betaflight/stm32f1_support_notice.png)


![Betaflight](https://raw.githubusercontent.com/wiki/betaflight/betaflight/images/betaflight/bf_logo.png)

Betaflight is flight controller software (firmware) used to fly multi-rotor craft and fixed wing craft.

This fork differs from Baseflight and Cleanflight in that it focuses on flight performance, leading-edge feature additions, and wide target support.

## Installation & Documentation

See: https://github.com/betaflight/betaflight/wiki

Filter coefficients were generated and exported in Matlab fdatool. FIR filter group delay is 1ms and 1.5ms respectievely for PID and Gyro filter.

To enable FIR static filter please use CLI and type:

set gyro_lowpass_type = FIR
set dterm_lowpass_type = FIR

To enable filter on Yaw axis just set Yaw frequency > 0.

That's all! Please remember the Gyro update frequency must be equal to PID loop time, i.e. 8kHz-8kHz, 16kHz-16kHz.
Combinations like 8kHZ-4kHz will NOT work now! If you want to enable it, please mody filter.c.


