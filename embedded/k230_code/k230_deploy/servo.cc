#include "servo.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

namespace
{
constexpr const char *PWM_DEVICE_NAME = "/dev/pwm";
constexpr float kMinServoDuty = 5.0f;
constexpr float kMaxServoDuty = 10.0f;
}

ServoController::ServoController(int channel, unsigned int frequency_hz)
    : fd_(-1), channel_(channel), period_ns_(frequency_hz ? 1000000000u / frequency_hz : 0)
{
}

ServoController::~ServoController()
{
    disable();
    if (fd_ >= 0)
    {
        close(fd_);
        fd_ = -1;
    }
}

bool ServoController::init(float duty_percent)
{
    if (period_ns_ == 0)
    {
        std::cerr << "ServoController: invalid PWM period\n";
        return false;
    }

    if (fd_ < 0)
    {
        fd_ = open(PWM_DEVICE_NAME, O_RDWR);
        if (fd_ < 0)
        {
            std::cerr << "ServoController: failed to open " << PWM_DEVICE_NAME << ": " << strerror(errno) << "\n";
            return false;
        }
    }

    if (!configure_pwm(duty_percent))
    {
        return false;
    }

    return enable();
}

bool ServoController::set_duty_cycle(float duty_percent)
{
    if (fd_ < 0)
    {
        std::cerr << "ServoController: device not opened\n";
        return false;
    }

    return configure_pwm(duty_percent) && enable();
}

bool ServoController::write_angle(float degrees)
{
    // 0-180 deg -> 5-10% duty (roughly 1-2 ms at 50 Hz)
    float clamped = std::max(0.0f, std::min(180.0f, degrees));
    float duty = kMinServoDuty + (clamped / 180.0f) * (kMaxServoDuty - kMinServoDuty);
    return set_duty_cycle(duty);
}

void ServoController::disable()
{
    if (fd_ < 0)
        return;

    pwm_config_t config{};
    config.channel = static_cast<unsigned int>(channel_);
    ioctl(fd_, KD_PWM_CMD_DISABLE, &config);
}

bool ServoController::configure_pwm(float duty_percent)
{
    if (fd_ < 0)
        return false;

    float clamped_duty = std::max(0.0f, std::min(100.0f, duty_percent));
    pwm_config_t config{};
    config.channel = static_cast<unsigned int>(channel_);
    config.period = period_ns_;
    config.pulse = static_cast<unsigned int>((clamped_duty * period_ns_) / 100.0f);

    if (config.pulse > config.period)
    {
        config.pulse = config.period;
    }

    if (ioctl(fd_, KD_PWM_CMD_SET, &config) != 0)
    {
        std::cerr << "ServoController: ioctl(KD_PWM_CMD_SET) failed: " << strerror(errno) << "\n";
        return false;
    }

    return true;
}

bool ServoController::enable()
{
    if (fd_ < 0)
        return false;

    pwm_config_t config{};
    config.channel = static_cast<unsigned int>(channel_);
    if (ioctl(fd_, KD_PWM_CMD_ENABLE, &config) != 0)
    {
        std::cerr << "ServoController: ioctl(KD_PWM_CMD_ENABLE) failed: " << strerror(errno) << "\n";
        return false;
    }

    return true;
}
