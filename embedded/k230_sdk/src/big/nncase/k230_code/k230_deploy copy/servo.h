/*
 * Simple PWM servo helper for K230.
 *
 * Maps a PWM channel to a standard hobby-servo friendly API.
 */
#pragma once

#include <sys/ioctl.h>
#include <cstdint>

// PWM ioctl definitions copied from SDK driver headers.
#ifndef KD_PWM_CMD_ENABLE
#define KD_PWM_CMD_ENABLE _IOW('P', 0, int)
#endif
#ifndef KD_PWM_CMD_DISABLE
#define KD_PWM_CMD_DISABLE _IOW('P', 1, int)
#endif
#ifndef KD_PWM_CMD_SET
#define KD_PWM_CMD_SET _IOW('P', 2, int)
#endif
#ifndef KD_PWM_CMD_GET
#define KD_PWM_CMD_GET _IOW('P', 3, int)
#endif

typedef struct
{
    unsigned int channel; /* 0-5 */
    unsigned int period;  /* unit: ns */
    unsigned int pulse;   /* unit: ns (pulse <= period) */
} pwm_config_t;

class ServoController
{
public:
    ServoController(int channel = 0, unsigned int frequency_hz = 50);
    ~ServoController();

    // Open /dev/pwm, configure the initial duty cycle, and enable the channel.
    bool init(float duty_percent = 7.5f);

    // Update duty cycle directly (0-100%).
    bool set_duty_cycle(float duty_percent);

    // Write an angle in degrees (0-180 mapped to 5-10% duty).
    bool write_angle(float degrees);

    // Disable PWM output; init() must be called again before re-use.
    void disable();

private:
    bool configure_pwm(float duty_percent);
    bool enable();

    int fd_;
    int channel_;
    unsigned int period_ns_;
};
