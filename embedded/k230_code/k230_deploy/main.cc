/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <iostream>
#include <thread>
#include <map>
#include "utils.h"
#include "vi_vo.h"
#include "classification.h"
#include "servo.h"

using std::cerr;
using std::cout;
using std::endl;
using namespace std;


// Classification pipeline temporarily disabled while exercising PWM/servo output.
#if 0
std::atomic<bool> isp_stop(false);

std::map<string, int> modeltype;

void print_usage()
{
    cout << "模型推理时传参说明："
         << "<kmodel_path> <image_path> <debug_mode>" << endl
         << "Options:" << endl
         << "  kmodel_path     Kmodel的路径\n"
         << "  image_path      待推理图片路径/摄像头(None)\n"
         << "  labels_txt      类别标签文件路径\n"
         << "  debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;
}

vector<string> read_labels_txt(string &labels_txt){
    std::vector<std::string> labels;
    std::ifstream file(labels_txt);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return labels;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[line.length() - 1] == '\n') {
            line.erase(line.length() - 1);
        }
        labels.push_back(line);
    }
    file.close();
    return labels;
}


void image_proc_cls(string &kmodel_path, string &image_path,vector<string> labels,float cls_thresh ,int debug_mode)
{
    cv::Mat ori_img = cv::imread(image_path);
    Classification cls(kmodel_path,image_path,labels,cls_thresh,debug_mode);
    cls.pre_process(ori_img);
    cls.inference();
    vector<cls_res> results;
    cls.post_process(results);
    Utils::draw_cls_res(ori_img,results);
    cv::imwrite("result_cls.jpg", ori_img);
}

void video_proc_cls(string &kmodel_path, string &image_path,vector<string> labels,float cls_thresh , int debug_mode)
{
    // ... original video classification body ...
}

int video_proc(char *argv[])
{
    string kmodel_path = argv[1];
    string image_path = argv[2];
    string labels_txt=argv[3];
    int debug_mode = std::stoi(argv[4]);
    vector<string> labels=read_labels_txt(labels_txt);
    float cls_thresh=0.5;
    video_proc_cls(kmodel_path,image_path,labels,cls_thresh,debug_mode);
    return -1;
}

int image_proc(char *argv[])
{
    string kmodel_path = argv[1];
    string image_path = argv[2];
    string labels_txt=argv[3];
    int debug_mode = std::stoi(argv[4]);
    vector<string> labels=read_labels_txt(labels_txt);
    float cls_thresh=0.5;
    image_proc_cls(kmodel_path,image_path,labels,cls_thresh,debug_mode);
    return -1;
}
#endif

int main()
{
    std::cout << "Servo demo built at " << __DATE__ << " " << __TIME__ << std::endl;

    // Route PWM0 to the desired servo line (board DTS should map PWM0 -> IO60).
    ServoController servo(/*channel=*/0, /*frequency_hz=*/50);
    if (!servo.init(7.5f))
    {
        std::cerr << "Failed to initialize PWM servo channel\n";
        return -1;
    }

    auto pause_one_second = []() { std::this_thread::sleep_for(std::chrono::seconds(1)); };

    servo.write_angle(150);
    pause_one_second();
    servo.write_angle(50);
    pause_one_second();
    servo.write_angle(150);
    pause_one_second();
    servo.write_angle(250); // will clamp to the servo's supported duty window
    pause_one_second();
    servo.write_angle(150);

    return 0;
}
