/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd */

#include <chrono>
#include <iostream>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include "servo.h"

extern "C" {
#include "mpi_pm_api.h"
}

using std::cout;
using std::endl;

// IOMUX configuration constants
constexpr uint32_t IOMUX_BASE_ADDR = 0x91105000;
constexpr uint32_t IOMUX_PAGE_SIZE = 4096;
constexpr uint32_t GPIO60_PIN = 60;

typedef struct {
    uint32_t st : 1;
    uint32_t ds : 4;
    uint32_t pd : 1;
    uint32_t pu : 1;
    uint32_t oe_en : 1;
    uint32_t ie_en : 1;
    uint32_t msc : 1;
    uint32_t sl : 1;
    uint32_t io_sel : 3;
    uint32_t resv0 : 17;
    uint32_t pad_di : 1;
} mux_config_t;

void enter_deep_sleep(void) {
    cout << "Entering deep sleep mode..." << endl;

    // Set CPU to lowest frequency
    kd_mpi_pm_set_governor(PM_DOMAIN_CPU, PM_GOVERNOR_ENERGYSAVING);
    kd_mpi_pm_set_profile(PM_DOMAIN_CPU, -1);

    // Power down KPU
    kd_mpi_pm_set_clock(PM_DOMAIN_KPU, false);
    kd_mpi_pm_set_power(PM_DOMAIN_KPU, false);

    // Power down DPU
    kd_mpi_pm_set_clock(PM_DOMAIN_DPU, false);
    kd_mpi_pm_set_power(PM_DOMAIN_DPU, false);

    // Power down VPU
    kd_mpi_pm_set_clock(PM_DOMAIN_VPU, false);
    kd_mpi_pm_set_power(PM_DOMAIN_VPU, false);

    // Power down Display
    kd_mpi_pm_set_clock(PM_DOMAIN_DISPLAY, false);
    kd_mpi_pm_set_power(PM_DOMAIN_DISPLAY, false);

    cout << "Deep sleep mode activated" << endl;
}

bool configure_iomux_for_pwm()
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        cout << "Error: Failed to open /dev/mem" << endl;
        return false;
    }

    void* iomux_base = mmap(nullptr, IOMUX_PAGE_SIZE, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, IOMUX_BASE_ADDR);
    if (iomux_base == MAP_FAILED) {
        cout << "Error: Failed to mmap IOMUX registers" << endl;
        close(fd);
        return false;
    }

    volatile uint32_t* iomux_regs = static_cast<volatile uint32_t*>(iomux_base);

    mux_config_t pwm_config = {
        .st = 1, .ds = 0x7, .pd = 0, .pu = 0, .oe_en = 1, .ie_en = 0,
        .msc = 1, .sl = 0, .io_sel = 1, .resv0 = 0, .pad_di = 0
    };

    uint32_t pwm_value;
    memcpy(&pwm_value, &pwm_config, sizeof(uint32_t));
    iomux_regs[GPIO60_PIN] = pwm_value;

    cout << "GPIO60 configured for PWM" << endl;

    munmap(iomux_base, IOMUX_PAGE_SIZE);
    close(fd);
    return true;
}

int main()
{
    cout << "Servo demo" << endl;

    if (!configure_iomux_for_pwm())
        return -1;

    ServoController servo(0, 50);
    if (!servo.init(0))
        return -1;

    auto pause = []() { std::this_thread::sleep_for(std::chrono::seconds(1)); };

    int center = 150;
    int sweep = 100;

    servo.write_angle(center);
    pause();
    servo.write_angle(center - sweep);
    pause();
    servo.write_angle(center);
    pause();
    servo.write_angle(center + sweep);
    pause();
    servo.write_angle(center);
    pause();

    cout << "Servo demo complete" << endl;

    enter_deep_sleep();

    while (true) {
        std::this_thread::sleep_for(std::chrono::hours(24));
    }

    return 0;
}