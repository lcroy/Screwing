import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # dataset path
        self.data_folder_path = os.path.join(self.project_path, 'data')
        self.raw_aursad_path = os.path.join(self.data_folder_path, 'aursad_tabular.dat')

        # parameters
        self.lr = 0.0001
        self.loss = 'sparse_categorical_crossentropy'
        self.epochs = 500
        self.batch_size = 32
        self.num_class = 4

        # model path
        self.model_path = os.path.join(self.project_path, 'models')
        self.model_Conv1D_path = os.path.join(self.model_path, 'Conv1D', 'model.h5')

        # figures of loss and acc
        self.fig_path = os.path.join(self.project_path, 'loss_acc')
        self.loss_fig_path = os.path.join(self.fig_path, 'loss.png')
        self.acc_fig_path = os.path.join(self.fig_path, 'acc.png')

        # json file for saving scores
        self.scores_path = os.path.join(self.project_path, 'scores')
        self.scores_file_path = os.path.join(self.scores_path, 'scores.json')


