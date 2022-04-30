import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # dataset path
        self.data_folder_path = os.path.join(self.project_path, 'data')
        self.raw_aursad_path = os.path.join(self.data_folder_path, 'aursad_tabular.dat')

        # parmeters == common
        self.num_class = 4
        self.lr = 0.0001
        self.loss = 'sparse_categorical_crossentropy'  
        self.epochs = 200
        self.batch_size = 32

        # parameters == Transformer
        self.head_size=256
        self.num_heads=4
        self.ff_dim=4
        self.num_transformer_blocks=6
        self.mlp_units=[128]
        self.mlp_dropout=0.3
        self.dropout=0.2

        # parameters == DNN
        self.units_h1 = 938
        self.units_h2 = 1876
        self.units_h3 = 938
        self.units_h4 = 4

        # model path
        self.model_path = os.path.join(self.project_path, 'models')
        self.model_Conv1D_path = os.path.join(self.model_path, 'Conv1D', 'model.h5')
        self.model_TRM_path = os.path.join(self.model_path, 'TRM', 'model.h5')
        self.model_DNN_path = os.path.join(self.model_path, 'DNN', 'model.h5')

        # figures of loss and acc
        self.fig_path = os.path.join(self.project_path, 'loss_acc')
        self.conv1D_loss_fig_path = os.path.join(self.fig_path, 'Conv1D', 'loss.png')
        self.conv1D_acc_fig_path = os.path.join(self.fig_path, 'Conv1D', 'acc.png')
        self.TRM_loss_fig_path = os.path.join(self.fig_path, 'TRM', 'loss.png')
        self.TRM_acc_fig_path = os.path.join(self.fig_path, 'TRM', 'acc.png')
        self.DNN_loss_fig_path = os.path.join(self.fig_path, 'DNN', 'loss.png')
        self.DNN_acc_fig_path = os.path.join(self.fig_path, 'DNN', 'acc.png')

        # json file for saving scores
        self.scores_path = os.path.join(self.project_path, 'scores')
        self.scores_file_path = os.path.join(self.scores_path, 'scores.json')


