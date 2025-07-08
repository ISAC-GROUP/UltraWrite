
# 模型
## seq2seq 版本
<center>
    <img style="border-radius: 0.3125em; transform:rotate(-90deg);
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="img/System/seq2seq_model.jpg" width = "60%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      seq2seq model
  	</div>
</center>

* ENCODER
    * Input：切分好的时频图序列([BatchSize, TimeStep, Channels, Width, Heigh])
    * Label：单词（[BatchSize, WordLength]）
    * Output：每个时间步的维度27，其中最后一个维度表示black([BatchSize, TimeStep, 27])
    zcl075411
* DECODER
    * Input：输入为单词，但在单词前加入了SOS字符，表示开始([BatchSize, 1 + WordLength])
    * Label：标签为单词，但在单词结尾加入EOS字符，表示结束([BatchSize, WordLength + 1])
    * Output：每个时间步的维度28，其中后两个维度表示SOS和EOS([BatchSize, WordLength + 1, 28])

注意：在Decoding时，训练阶段可以使用pack之后，以批量进行计算，而测试阶段，由于下一个时间片的输入由上一个时间片的输出代替，故而只能够一样本为单位，并且按时间片逐个计算，直到输出为代表 $<EOS>$ 才停止计算。

注意：对于lstm，若是以batch进行计算，由于sample的长度不对等，需要对batch进行padding和pack，如若是已经按照长度递减，则在使用pack_padded_sequence时，参数enforce_sorted使用True，否则使用False。也可以统一用False。本次中，encoder的batch是已经按照递减排序好的，而对应的target并没有按照递减排序，故而decoder的时候，enforce_sorted一定要使用False。

注：

&emsp;&emsp;在训练和测试阶段都以时间步为单位进行计算，这样能够规避掉$<pad>$的影响。在embedding阶段，词库不用包括$<pad>$。从而提高性能。

&emsp;&emsp;在测试阶段，decoder的输出也可以使用search方法。可以使用beam serach或者best search？（不过search方法需要调整，ctc的search需要将重复的字符合并，而这里不需要。）


## spell correct模型
