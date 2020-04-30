import network as nn
import data as dt

train_inputs, train_labels, test_inputs, test_labels = dt.prepare_data()

nn.train(train_inputs, train_labels, test_inputs, test_labels)

# Rodar o tensorboard com o comando:
# tensorboard --logdir logs/fit
# no prompt
