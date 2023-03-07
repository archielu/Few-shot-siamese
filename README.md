# Few-shot-siamese


A reimplementation of [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Dataset (Omniglot)

### Training data
Select 10500000 positive pairs and 10500000 negative pairs, each pair contains two picture and one label. If two images in one pair are from same class -> label 1, else -> label 0.

### Testing data
Prepare sevaral groups of data, each group contains 1 positive pair and [group_size-1] neagtive pairs.

## Model Architecture
![img](https://github.com/archielu/Few-shot-siamese/blob/main/model.png)

## Loss Function
[BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

## Result

