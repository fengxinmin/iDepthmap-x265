## 测试命令

需要说明的是，x265不同于HM，相当一部分的命令必须使用命令行传入而不能用cfg配置文件传入，为了让控制方式更加统一，所有的命令都是通过命令行的方式传入的。因此，每个YUV序列的名称需要被精心设计，命名格式如下
`SeqName_WidthxHeight_FrameNum_FrameRate_bitdepth.yuv`，例如`BasketballPass_416x240_500_50_8`。

用户需要在`./Test_Sequence_List.txt`中指定需要编码的YUV序列，序列名称需满足以上要求。`./HEVC_Test_Sequence.txt`提供了HEVC CTC 所涉及的序列名称。

检查`./Test_Sequence_List.txt`中的视频序列保存在`./sequences`之后，cd到项目目录路径下，输入以下命令
```
python dp_total_test.py --qp 22 --batchSize 200 --encoder_type origin
```
即可自动编解码，编码得到的码流存放在`./output`中，解码得到的重建序列在`./rebuild`中。

**说明**

(1) x265编码器分为origin和anchor两类，后者相比前者增加了针对CTU是否划分的判断，但是因为该判断在Y域对编解码性能的提升十分有限，所以更加建议使用origin

(2) 增加`--save_depth_flag`的命令，可以保存划分深度文件（保存在`./DepthFlag`中），但是一般出于减少内存消耗的考量，划分深度文件会作为中间文件阅后即焚

## 其他文件

(1) `./dataset.py`搭配`./codec/Release`下的编码器，用于制作基于DIV2K的块划分数据集

(2) `./dp_total_train.py`，训练代码

(3) `./log2xls.py`，读取encoder的日志文件，写入计算BD-Rate的Excel脚本。但是保存时excel脚本会丢失宏，所以需要手动将数据再复制到没有丢失宏的excel脚本，该问题目前还没有较好的解决方案。

## 其他问题

程序会识别YUV比特深度，如果是10bit，会调用`./codec/x265/anchor_dpfast_x265_encoder_10bit.exe`或者`./codec/x265/origin_dpfast_x265_encoder_10bit.exe`，因为x265编码高比特深度的YUV序列时会存在某些限制，导致编码可能不成功，一个已经确定的限制是 编码10bit序列时，系统必须为64位。

更多关于x265 10bit的配置和讨论，可以参考文章 https://mp.weixin.qq.com/s/BmLCGH3F8LDhrPO7Zz9SLw