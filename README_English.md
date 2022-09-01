## Test Commands

It should be noted that x265 is different from HM in that manny of commands must be passed through the command line rather than the configuration file. To make the control mode more uniform, all commands are passed through the command line. Therefore, the name of each YUV sequence needs to be carefully designed in the following format.

`SeqName_WidthxHeight_FrameNum_FrameRate_bitdepth.yuv`

for example, `BasketballPass_416x240_500_50_8`.

The user needs to specify the YUV sequence to be encoded in the `./Test_Sequence_List.txt` firstly. The sequence name must meet the above requirements. Moreover, `./HEVC_Test_Sequence.txt` provides the sequence names involved in the HEVC CTC.

Check that the sequences in `./Test_Sequence_List.txt` is saved in `./sequences/`, then *cd* to the project directory, and run the following command

```
python dp_total_test.py --qp 22 --batchSize 200 --encoder_type origin
```

Then, the encoded bit-stream is stored in `./output`, and the decoded reconstructed sequence is in `./rebuild`.


**Supplementary Instruction**

(1) x265 encoders are divided into two categories in this repo: Origin and Anchor. Compared with the Origin, the Anchor adds judgment on whether CTU is divided. However, since the improvement of codecs performance in Y domain by this judgment is very limited in our experiments, the Origin is more recommended in practice.

(2) You can add the '--save_depth_flag' command to save the partition depth file (save in `./DepthFlag`). But generally for the sake of reducing memory consumption, the partition depth file will be burned as an intermediate file after using.


## Other Files

(1) `./dataset.py` could be used to make a block partition dataset based on DIV2K

(2) `./dp_total_train.py` is used to train the model

(3) `./log2xls.py` is used to read Encoder's log file and write Excel script to calculate BD-rate in the HEVC CTC. However, the Excel script will lose macros when saving, so you need to manually copy the data to the Excel script that does not lose macros. As far as now, there is no good solution to this problem.


## Other Problems

The program will recognize bit-depth of encoded YUV sequence, if it's 10 bits, `./codec/x265/anchor_dpfast_x265_encoder_10bit.exe` or `./codec/x265/origin_dpfast_x265_encoder_10bit.exe` will be called, because x265 encodes YUV sequences at high bit depths with certain limitations, therefore the encoding may not be successful. One established limitation is that the system must be 64 bits when encoding 10bit sequences.

More configuration and discuss about x265 10 bit, you can refer to https://mp.weixin.qq.com/s/BmLCGH3F8LDhrPO7Zz9SLw

