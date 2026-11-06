// 编译: g++ -std=c++17 -Wall -o stream stream.cpp
// 运行: 先在当前目录准备 input.txt，再执行 ./stream
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

int main() {
    // TODO 1: 创建文件输入流对象，打开 input.txt
    std::ifstream inputFile("input.txt");

    // TODO 2: 创建文件输出流对象，打开 output.txt
    std::ofstream outputFile("output.txt");

    // TODO 3: 检查文件是否成功打开
    if (inputFile.is_open() == false) {
        printf("Failed to open input.txt\n");
        return 1;
    }
    if (outputFile.is_open() == false) {
        printf("Failed to open output.txt\n");
        return 1;
    }

    // TODO 4: 使用 std::getline 逐行读取文件
    std::string line;
    while (std::getline(inputFile, line)) {

        // TODO 5: 将读取的内容输出到终端
        std::cout << line;

        // TODO 6: 将相同内容写入 output.txt
        outputFile << line;
    }

    // TODO 7: 显式刷新文件输出流的缓冲区
    outputFile.flush();
    inputFile.close();
    outputFile.close();

    return 0;
}
