(1)
./codec/HM/  HM加速前后的编码器，以及解码器
(2)
./codec/x265 x265加速前后的编码器，以及解码器（分为8/10bit、anchor和origin等不同情况）

origin: 原始x265
anchor: 在origin的基础上，增加了针对CTU的划分判断

dpfast: 快速块划分

anchor_output_depth_x265_encoder.exe/origin_output_depth_x265_encoder.exe: 单帧线程、关闭wpp后顺序输出块划分的深度信息，保存在 ./Sequences，用于制备块划分数据集
anchor_output_depth_x265_encoder_10bit.exe/origin_output_depth_x265_encoder_10bit.exe: 同上，但是只针对10bit序列

origin_dpfast_x265_encoder.exe: 原始x265 + 快速块划分
anchor_dpfast_x265_encoder.exe: 完备x265 + 快速块划分

anchor_x265_encoder.exe: 完备x265编码器 (未加速)
origin_x265_encoder.exe: 原始x265编码器 (未加速)

注意：在代码文件dp_total_test.py中，默认并推荐使用origin x265，因为通过分析CTU而增加的性能十分有限。

(3)
./codec/Release 可以输出亮度分量、块划分信息和率失真损失的编码器，用于制备数据集

Note:
x265 encoder 参考调用命令
(1) 8bit, QP=22
.\x265_encoder.exe --preset medium --keyint 0  --input E:\iDepthMap\sequences\BasketballPass_4500_50_8.yuv --fps 50 --input-res 416x240 --output E:\iDepthMap\output\medium\BasketballPass_416x240_500_50_8_anchor_Q22.bin --tune psnr --psnr --qpmin 22 --qpmax 22 --csv E:\iDepthMap\log\encoder\medium\enc_BasketballPass_416x240_500_50_8_anchor_Q22.log --csv-log-level 2 --frames 500 --profile main
(2) 10bit, QP=22
.\x265_encoder_10bit.exe --preset medium --keyint 0  --input E:\iDepthMap\sequences\Nebuta_2560x1600_300_60_10.yuv --fps 60 --input-res 2560x1600 --output E:\iDepthMap\output\mediumNebuta_2560x1600_300_60_10_anchor_Q22.bin  --tune psnr --psnr --qpmin 22 --qpmax 22 --csv E:\iDepthMap\log\encoder\medium\enc_Nebuta_2560x1600_300_60_10_anchor_Q22.log --csv-log-level 2 --frames 300 --profile main10 --input-depth 10 --output-depth 10
