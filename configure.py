import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # dataset path
        self.data_folder_path = os.path.join(self.project_path, 'data')
        self.feature_aursad_path = os.path.join(self.data_folder_path, 'aursad_tabular.dat')
        self.raw_aursad_path = os.path.join(self.data_folder_path, 'aursad_D_t.dat')
        # default raw aursad D
        self.raw_data_source = True

        # parmeters == common
        self.num_class = 4
        self.lr = 0.0001
        self.loss = 'sparse_categorical_crossentropy'  
        self.epochs = 200
        self.batch_size = 32
        self.patience = 20

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

        # model checkpoint path
        self.model_path = os.path.join(self.project_path, 'checkpoints')
        self.model_Conv1D_path = os.path.join(self.model_path, 'Conv1D')
        self.model_TRM_path = os.path.join(self.model_path, 'TRM')
        self.model_DNN_path = os.path.join(self.model_path, 'DNN')

        # figures of loss and acc
        self.fig_path = os.path.join(self.project_path, 'loss_acc')
        self.Conv1D_loss_acc_fig_path = os.path.join(self.fig_path, 'Conv1D')
        self.TRM_loss_acc_fig_path = os.path.join(self.fig_path, 'TRM')
        self.DNN_loss_acc_fig_path = os.path.join(self.fig_path, 'DNN')

        # json file for saving scores
        self.scores_path = os.path.join(self.project_path, 'scores')
        self.scores_file_path = os.path.join(self.scores_path, 'scores.json')


