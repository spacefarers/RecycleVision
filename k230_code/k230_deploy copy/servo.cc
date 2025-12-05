#include "servo.h"
#include <fcntl.h>
#include <unistd.h>

ServoController::ServoController(int channel, unsigned int frequency_hz)
    : fd_(-1), channel_(channel), period_ns_(frequency_hz ? 1000000000u / frequency_hz : 0)
{
}

ServoController::~ServoController()
{
    disable();
    if (fd_ >= 0)
        close(fd_);
}

bool ServoController::init(float duty_percent)
{
    if (fd_ < 0)
        fd_ = open("/dev/pwm", O_RDWR);
    return configure_pwm(duty_percent) && enable();
}

bool ServoController::set_duty_cycle(float duty_percent)
{
    return configure_pwm(duty_percent) && enable();
}

bool ServoController::write_angle(float degrees)
{
    float duty = 5 + (degrees / 180) * 5;
    return set_duty_cycle(duty);
}

void ServoController::disable()
{
    if (fd_ < 0)
        return;
    pwm_config_t config{};
    config.channel = channel_;
    ioctl(fd_, KD_PWM_CMD_DISABLE, &config);
}

bool ServoController::configure_pwm(float duty_percent)
{
    pwm_config_t config{};
    config.channel = channel_;
    config.period = period_ns_;
    config.pulse = (duty_percent * period_ns_) / 100.0f;
    if (config.pulse > config.period)
        config.pulse = config.period;
    return ioctl(fd_, KD_PWM_CMD_SET, &config) == 0;
}

bool ServoController::enable()
{
    if (fd_ < 0)
        return false;
    pwm_config_t config{};
    config.channel = channel_;
    return ioctl(fd_, KD_PWM_CMD_ENABLE, &config) == 0;
}
