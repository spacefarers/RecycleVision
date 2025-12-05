[TOC]

## 1. 环境搭建

### 1.1 启动docker

```shell
$ cd /path/to/k230_sdk
$ docker run -u root -it --rm -v $(pwd):/mnt -v $(pwd)/toolchain:/opt/toolchain -w /mnt ghcr.io/kendryte/k230_sdk:latest /bin/bash
```

默认挂载了两个目录

- 将k230_sdk目录挂载到docker的/mnt
- 将交叉编译工具链k230_sdk/toolchain目录挂载到docker的/opt/toolchain



### 1.2 环境搭建

```shell
root@c08eb1760d50:/mnt# cd src/big/nncase/examples
root@c08eb1760d50:/mnt/src/big/nncase/examples# source setup_py39.sh
```



查看安装的nncase版本信息

```shell
(python39_venv) root@c08eb1760d50:/mnt/src/big/nncase/examples# pip list|grep nncase
nncase            2.10.0  
nncase-kpu        2.10.0  
```



> - **linux**平台支持在线安装nncase和nncase-kpu. 
> - **windows平台**只支持nncase在线安装， nncase-kpu需要到[nncase github release](https://github.com/kendryte/nncase/releases)单独下载并安装



## 2. 编译模型

### 2.1 编译

```shell
(python39_venv) root@c08eb1760d50:/mnt/src/big/nncase/examples# ./build_model.sh
```



### 2.2 查看编译结果

```shell
(python39_venv) root@c08eb1760d50:/mnt/src/big/nncase/examples# ls -l tmp/
total 12
drwxr-xr-x 11 root root 4096 Jul 18 08:45 mbv2_tflite
drwxr-xr-x 11 root root 4096 Jul 18 08:46 mobile_retinaface
drwxr-xr-x 11 root root 4096 Jul 18 08:45 yolov5s_onnx
```



## 3. 编译App

###  3.1 App介绍

| App               | 备注                                                         |
| ----------------- | ------------------------------------------------------------ |
| image_classify    | 图片分类demo, 输入是RGB图片, 推理结果打印到串口              |
| object_detect     | 目标检测demo, 输入是RGB图片, 推理结果打印到串口              |
| image_face_detect | 人脸检测demo, 输入是RGB图片, 推理结果会输出到画了人脸box和landmark的图片 |



### 3.2 编译

```shell
(python39_venv) root@c08eb1760d50:/mnt/src/big/nncase/examples# ./build_app.sh
```



### 3.3 查看编译结果

编译app结束后, 默认会将demo及其运行所需文件拷贝到当前目录下的k230_bin子目录

```shell
(python39_venv) root@c08eb1760d50:/mnt/src/big/nncase/examples# tree k230_bin
k230_bin
├── image_classify
│   ├── cat.png
│   ├── cpp.sh
│   ├── image_classify.elf
│   ├── labels_1001.txt
│   └── test.kmodel
├── image_face_detect
│   ├── cpp.sh
│   ├── face_1280x720.jpg
│   ├── face_500x500.jpg
│   ├── face_720x1280.jpg
│   ├── image_face_detect.elf
│   └── test.kmodel
└── object_detect
    ├── 320x256.jpg
    ├── cpp.sh
    ├── object_detect.elf
    └── test.kmodel

3 directories, 15 files
```



### 3.4 传送demo

- 将k230_bin目录拷贝到本地PC的nfsroot目录并重命名为nncase_k230_v2.10.0_rtos
- 在小核串口搭建sharefs环境, 并将本地PC的nfsroot目录mount到/sharefs
- 在大核串口下进入/sharefs执行相应demo



## 4. 上板运行


### 4.1 图片分类demo

```shell
msh /sharefs/nncase_k230_v2.10.0_rtos>cd image_classify/
msh /sharefs/nncase_k230_v2.10.0_rtos/image_classify>./cpp.sh
case ./image_classify.elf built at Jul 18 2025 08:48:51
interp.run() took: 2.18722 ms
image classify result: tabby(0.332828)
```



### 4.2 目标检测demo

```shell
msh /sharefs/nncase_k230_v2.10.0_rtos>cd object_detect/
msh /sharefs/nncase_k230_v2.10.0_rtos/object_detect>./cpp.sh
case ./object_detect.elf built at Jul 18 2025 08:48:51
od set_input took 0.166111 ms
od run took 16.4519 ms
od get output took 12.826 ms
post process took 57.8055 ms
text = bicycle:0.680000
text = truck:0.640000
text = dog:0.590000
draw result took 34.3927 ms
```

推理结果(画框)会生成到od_result.jpg



### 4.3 人脸检测demo

```shell
msh /sharefs/nncase_k230_v2.10.0_rtos>cd image_face_detect/
msh /sharefs/nncase_k230_v2.10.0_rtos/image_face_detect>./cpp.sh
case ./image_face_detect.elf built at Jul 18 2025 08:48:51
Press 'q + enter' to exit!!!
```

推理结果(人脸box和landmark)会生成到face_500x500_result_x.jpg
