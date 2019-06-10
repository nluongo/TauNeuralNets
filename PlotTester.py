import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from sklearn.model_selection import KFold

with PdfPages('PO5LayerRecoEt.pdf') as pdf:

    plt.plot(range(epochs+1), cost_history)
    plt.axis([0, epochs, 0, np.max(cost_history)])
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Cost vs. Epochs (With Hadronic, Without Bias PO)')
    pdf.savefig()
    plt.close()

    plt.plot(range(epochs+1), resolution_history)
    plt.axis([0, epochs, 0, np.max(resolution_history)])
    plt.xlabel('Epochs')
    plt.ylabel('Resolution')
    plt.title('Resolution vs. Epochs (With Hadronic, Without Bias PO)')
    pdf.savefig()
    plt.close()

    plt.bar([0, 1, 2, 3, 4], [new_W[0][0], new_W[1][0], new_W[2][0], new_W[3][0], new_W[4][0]], align='center')
    plt.xlabel('Layer')
    plt.xticks(range(5), ('0', '1', '2', '3', 'Hadronic'))
    plt.ylabel('Weight')
    plt.title('Layer Weights (With Hadronic, Without Bias PO)')
    pdf.savefig()
    plt.close()