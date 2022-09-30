import os

class Config:
    def __init__(self):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))

        # dataset path
        self.data_folder_path = os.path.join(self.project_path, 'data')
        self.feature_aursad_path = os.path.join(self.data_folder_path, 'aursad_tabular.dat')
        self.raw_aursad_path = os.path.join(self.data_folder_path, 'aursad_D_t.dat')
        self.org_aursad_cln_path = os.path.join(self.data_folder_path, 'aursad_cln.dat')
        self.org_aursad_flt_path = os.path.join(self.data_folder_path, 'aursad_flt.dat')
        # raw aursad D
        self.raw_data_source = True
        # original aursad (process + task)
        self.org_data_only_process = True

        # parmeters == common
        self.num_class = 4
        self.lr = 0.0001
        self.loss = 'sparse_categorical_crossentropy'  
        self.epochs = 200
        self.batch_size = 8
        self.patience = 30

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

        # parameters == ConvLSTM2D
        self.steps = 15
        self.length = 113
        self.verbose = 1
        self.filters = 64

        # model checkpoint path
        self.model_path = os.path.join(self.project_path, 'checkpoints')
        self.model_Conv1D_path = os.path.join(self.model_path, 'Conv1D')
        self.model_TRM_path = os.path.join(self.model_path, 'TRM')
        self.model_DNN_path = os.path.join(self.model_path, 'DNN')

        self.model_TRM_org_data_path = os.path.join(self.model_path, 'TRM_org_data')
        self.model_Conv1D_org_data_path = os.path.join(self.model_path, 'Conv1D_org_data')
        self.model_ConvLSTM2D_org_data_path = os.path.join(self.model_path, 'ConvLSTM2D_org_data')
        self.model_LSTM_org_data_path = os.path.join(self.model_path, 'LSTM_org_data')

        # figures of loss and acc
        self.fig_path = os.path.join(self.project_path, 'loss_acc')
        self.Conv1D_loss_acc_fig_path = os.path.join(self.fig_path, 'Conv1D')
        self.TRM_loss_acc_fig_path = os.path.join(self.fig_path, 'TRM')
        self.DNN_loss_acc_fig_path = os.path.join(self.fig_path, 'DNN')

        self.TRM_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'TRM_org_data')
        self.Conv1D_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'Conv1D_org_data')
        self.ConvLSTM2D_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'ConvLSTM2D_org_data')
        self.LSTM_org_data_loss_acc_fig_path = os.path.join(self.fig_path, 'LSTM_org_data')

        # json file for saving scores
        self.scores_path = os.path.join(self.project_path, 'scores')
        self.scores_file_path = os.path.join(self.scores_path, 'scores.json')


    def model_parameters_set(self, model_name, raw_data_source):

        if model_name == "DNN":
            if raw_data_source == True:
                model_path = os.path.join(self.model_DNN_path, 'raw_model.h5')
                loss_img = os.path.join(self.DNN_loss_acc_fig_path, 'raw_loss.png')
                acc_img = os.path.join(self.DNN_loss_acc_fig_path, 'raw_acc.png')
                precision = "raw_DNN_precision"
                recall = "raw_DNN_recall"
                f1 = "raw_DNN_f1"
            else:
                model_path = os.path.join(self.model_DNN_path, 'feature_model.h5')
                loss_img = os.path.join(self.DNN_loss_acc_fig_path, 'feature_loss.png')
                acc_img = os.path.join(self.DNN_loss_acc_fig_path, 'feature_acc.png')
                precision = "feature_DNN_precision"
                recall = "feature_DNN_recall"
                f1 = "feature_DNN_f1"

        elif model_name == "Conv1D":
            if raw_data_source == True:
                model_path = os.path.join(self.model_Conv1D_path, 'raw_model.h5')
                loss_img = os.path.join(self.Conv1D_loss_acc_fig_path, 'raw_loss.png')
                acc_img = os.path.join(self.Conv1D_loss_acc_fig_path, 'raw_acc.png')
                precision = "raw_Conv1D_precision"
                recall = "raw_Conv1D_recall"
                f1 = "raw_Conv1D_f1"
            else:
                model_path = os.path.join(self.model_Conv1D_path, 'feature_model.h5')
                loss_img = os.path.join(self.Conv1D_loss_acc_fig_path, 'feature_loss.png')
                acc_img = os.path.join(self.Conv1D_loss_acc_fig_path, 'feature_acc.png')
                precision = "feature_Conv1D_precision"
                recall = "feature_Conv1D_recall"
                f1 = "feature_Conv1D_f1"

        elif model_name == "TRM":
            if raw_data_source == True:
                model_path = os.path.join(self.model_TRM_path, 'raw_model.h5')
                loss_img = os.path.join(self.TRM_loss_acc_fig_path, 'raw_loss.png')
                acc_img = os.path.join(self.TRM_loss_acc_fig_path, 'raw_acc.png')
                precision = "raw_TRM_precision"
                recall = "raw_TRM_recall"
                f1 = "raw_TRM_f1"
            else:
                model_path = os.path.join(self.model_TRM_path, 'feature_model.h5')
                loss_img = os.path.join(self.TRM_loss_acc_fig_path, 'feature_loss.png')
                acc_img = os.path.join(self.TRM_loss_acc_fig_path, 'feature_acc.png')
                precision = "feature_TRM_precision"
                recall = "feature_TRM_recall"
                f1 = "feature_TRM_f1"

        return model_path, loss_img, acc_img, precision, recall, f1


    def model_parameters_set_process_task(self, model_name, org_data_only_process, is_flt):

        # check if it is the filtered data                    
        if is_flt == 'Yes':
            flt_path = "flt"
        else:
            flt_path = "unflt"

        if model_name == "TRM_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_TRM_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_TRM_precision"
                recall = flt_path + "process_TRM_recall"
                f1 = flt_path + "process_TRM_f1"
            else:
                model_path = os.path.join(self.model_TRM_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.TRM_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_TRM_precision"
                recall = flt_path + "process_task_TRM_recall"
                f1 = flt_path + "process_task_TRM_f1"
        elif model_name == "Conv1D_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_Conv1D_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_Conv1D_precision"
                recall = flt_path + "process_Conv1D_recall"
                f1 = flt_path + "process_Conv1D_f1"
            else:
                model_path = os.path.join(self.model_Conv1D_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.Conv1D_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_Conv1D_precision"
                recall = flt_path + "process_task_Conv1D_recall"
                f1 = flt_path + "process_task_Conv1D_f1"
        elif model_name == "ConvLSTM2D_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_ConvLSTM2D_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_ConvLSTM2D_precision"
                recall = flt_path + "process_ConvLSTM2D_recall"
                f1 = flt_path + "process_ConvLSTM2D_f1"
            else:
                model_path = os.path.join(self.model_ConvLSTM2D_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.ConvLSTM2D_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_ConvLSTM2D_precision"
                recall = flt_path + "process_task_ConvLSTM2D_recall"
                f1 = flt_path + "process_task_ConvLSTM2D_f1"
        elif model_name == "LSTM_org_data":
            if org_data_only_process == 'Yes':
                model_path = os.path.join(self.model_LSTM_org_data_path, flt_path, 'process_model.h5')
                loss_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_loss.png')
                acc_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_acc.png')
                precision = flt_path + "process_LSTM_precision"
                recall = flt_path + "process_LSTM_recall"
                f1 = flt_path + "process_LSTM_f1"
            else:
                model_path = os.path.join(self.model_LSTM_org_data_path, flt_path, 'process_task_model.h5')
                loss_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_task_loss.png')
                acc_img = os.path.join(self.LSTM_org_data_loss_acc_fig_path, flt_path, 'process_task_acc.png')
                precision = flt_path + "process_task_LSTM_precision"
                recall = flt_path + "process_task_LSTM_recall"
                f1 = flt_path + "process_task_LSTM_f1"


        return model_path, loss_img, acc_img, precision, recall, f1
