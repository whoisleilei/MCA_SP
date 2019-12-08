# MCA_SP
A Method of Sememe Prediction using Dictionary Definitions Based on Deep Learning<br>
基于深度学习和词典定义的义原预测方法<br>
2019.07《信息工程大学学报》录用。

* +TW 是在编码定义后的向量concat词向量（TW）一起做预测

* +Pr 是在定义的头部加 `word :` 

* +Se 是定义中的词向量加上 义原向量的平均

做十折交叉验证:

```bash
nohup python -u sp_lstm.py -e 31 -g 0 -f 0 > out.log 2>&1 &
```
