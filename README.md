# CATS
辛苦double check的同学！由于实验数量比较多，我提供了general.sh和robust.sh分别来跑论文中Table3和Table6的实验。由于Huggingface改了他们的API(可能是某些处理token的方式变了)，所以有些得不到一模一样的结果，我重新写了新的代码并实验了一遍，具体可以参考https://docs.google.com/spreadsheets/d/1M7eQdRxob4mpCxtneQskFrdmnbzSxszUHULz10fiZGo/edit?usp=sharing \
请注意数字的些许变动并不影响论文中所得到的结论。
具体的复现细节请参考下面：

Commonsense Ability Tests

Dataset and script for paper [Evaluating Commonsense in Pre-trained Language Models](https://arxiv.org/abs/1911.11931)

Use `making_sense.py` to run the experiments:\
For ordinary tests:\
`python making_sense.py ca bert nr` 

For robust tests:\
`python making_sense.py ca bert r`

Note that `ca` is the name of the task and `bert` is the model we are using. The default model is `bert-base-uncased`. To use `bert-large`, just modify the `from_pretrained('bert-base-uncased')` in the code. For more details, see [Huggingface Transformers](https://huggingface.co/transformers/index.html).

*Due to the updating of Huggingface models and some of our datasets, some numbers we showed in the paper may not exactly match the what you might get by rerunning the experiments. However, the conclusion should be the same.*   
