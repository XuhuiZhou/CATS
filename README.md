# CATS

Commonsense Ability Tests

Dataset and script for paper [Evaluating Commonsense in Pre-trained Language Models](https://arxiv.org/abs/1911.11931)

Use `making_sense.py` to run the experiments:\
For ordinary tests:\
`python making_sense.py ca bert nr` 

For robust tests:\
`python making_sense.py ca bert r`

Note that `ca` is the name of the task and `bert` is the model we are using. The default model is `bert-base-uncased`. To use `bert-large`, just modify the `from_pretrained('bert-base-uncased')` in the code. For more details, see [Huggingface Transformers](https://huggingface.co/transformers/index.html).

*Due to the updating of Huggingface models and some of our datasets, some numbers we showed in the paper may not exactly match the what you might get by rerunning the experiments. However, the conclusion should be the same.*   
