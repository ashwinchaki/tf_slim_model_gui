import Tkinter as tk
import tkFileDialog
import tkMessageBox
import pygubu
import urllib
import os
from run_training import run_model_training
from aux_functions import get_current_optimizer

class Container:
    pass

class Application:
    def __init__(self, master):
        # Create main window and build buttons/lists/etc.
        self.master = master
        self.builder = builder = pygubu.Builder()
        builder.add_from_file('model_wrapper_gui.ui')
        self.mainwindow = builder.get_object('main_window', master)
        self.datasets = builder.get_object('dataset_list')
        self.model_name = builder.get_object('model_list')
        self.batch_size = builder.get_object('batch_size_spinbox')
        self.num_steps = builder.get_object('num_steps_spinbox')
        self.directory_label = builder.get_object('dataset_directory_label')
        self.init_ckpt_label = builder.get_object('initial_ckpt_dir_label')
        self.out_ckpt_label = builder.get_object('final_ckpt_dir_label')
        self.use_exist_ckpt = builder.get_object('use_ckpt')
        self.current_optimizer = builder.get_object('optimizer_list')
        self.current_learning_rate = builder.get_object('learning_rate_box')
        self.eval_dataset = builder.get_object('eval_dataset')

        builder.connect_callbacks(self)
        self.container = Container

    def open_browse_dir(self):
        custom_data_dir = tkFileDialog.askdirectory()
        self.directory_label['text'] = custom_data_dir
        self.datasets.current(3)

    def on_ckpt_browse(self):
        custom_data_dir = tkFileDialog.askopenfilename()
        self.init_ckpt_label['text'] = custom_data_dir

    def on_ckpt_out_browse(self):
        custom_data_dir = tkFileDialog.askdirectory()
        self.out_ckpt_label['text'] = custom_data_dir

    def on_eval_press(self):
        if self.eval_dataset.get() == 'Validation':
            path_to_val_file = data_directory + '/validation-00000-of-00001'
        elif self.eval_dataset.get() == 'Test':
            path_to_val_file = data_directory + '/test-00000-of-00001'
        curr_model_name = self.model_name.get()
        custom_dir = self.directory_label['text']

    def on_run_button_click(self): # RUN BUTTON BLOCK
        # Get current variables
        curr_dataset = self.datasets.get()
        curr_num_steps = self.num_steps.get()
        curr_model_name = self.model_name.get()
        curr_batch_size = self.batch_size.get()
        custom_dir = self.directory_label['text']
        curr_check = self.use_exist_ckpt.state()
        if isinstance(curr_check, tuple):
            init_ckpt = self.init_ckpt_label['text']
        else:
            init_ckpt = ''
        curr_optimizer = self.current_optimizer.get()
        out_dir = self.out_ckpt_label['text']
        curr_learning_rate = self.current_learning_rate.get()
        os.system('mkdir -p /tmp/checkpoints/')

        # Get dataset if Flowers/MNIST/cifar10 dataset (using prebuilt scripts)
        if curr_dataset == 'Flowers':
            dataset_name = 'flowers'
            checkpoint_dir = '/tmp/flowers/'
        elif curr_dataset == 'MNIST':
            dataset_name = 'mnist'
            checkpoint_dir = '/tmp/mnist/'
        elif curr_dataset == 'cifar10':
            dataset_name = 'cifar10'
            checkpoint_dir = '/tmp/cifar10/'
        elif curr_dataset == 'Custom':
            dataset_name = os.path.basename(os.path.normpath(custom_dir))
            if not init_ckpt == '':
                checkpoint_dir = init_ckpt
            else:
                checkpoint_dir = '/tmp/checkpoints/'

        # If checkpoint selected but none loaded, throw error
        if init_ckpt == '' and curr_check == 'selected':
            if not tkMessageBox.askokcancel('Checkpoint Error!', 'No initial checkpoint selected\nUse pre-trained checkpoint?'):
                return
            curr_check = ''


        # Download model checkpoints
        if curr_check == '' and init_ckpt == '':
            if curr_model_name == 'inception_v1':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/inception_v1_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/inception_v1_2016_08_28.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'inception_v2':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/inception_v2_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/inception_v2_2016_08_28.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'inception_v3':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/inception_v3_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/inception_v3_2016_08_28.tar.gz -C /tmp/checkpoints')
            elif curr_model_name == 'inception_v4':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
                                       '/tmp/checkpoints/inception_v4_2016_09_09.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/inception_v4_2016_09_09.tar.gz -C /tmp/checkpoints')
            elif curr_model_name == 'inception_resnet_v2':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
                                       '/tmp/checkpoints/inception_resnet_v2_2016_08_30.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/inception_resnet_v2_2016_08_30.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'resnet_v1_50':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/resnet_v1_50_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/resnet_v1_50_2016_08_28.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'resnet_v1_101':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/resnet_v1_101_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/resnet_v1_101_2016_08_28.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'resnet_v1_152':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/resnet_v1_152_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/resnet_v1_152_2016_08_28.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'resnet_v2_50':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
                                       '/tmp/checkpoints/resnet_v2_50_2017_04_14.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/resnet_v2_50_2017_04_14.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'resnet_v2_101':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
                                       '/tmp/checkpoints/resnet_v2_101_2017_04_14.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/resnet_v2_101_2017_04_14.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'resnet_v2_152':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
                                       '/tmp/checkpoints/resnet_v2_152_2017_04_14.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/resnet_v2_152_2017_04_14.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'vgg_16':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/vgg_16_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/vgg_16_2016_08_28.tar.gz -C /tmp/checkpoints/')
            elif curr_model_name == 'vgg_19':
                if not os.path.isfile('/tmp/checkpoints/%s.ckpt' % curr_model_name):
                    urllib.urlretrieve('http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
                                       '/tmp/checkpoints/vgg_19_2016_08_28.tar.gz')
                    os.system('tar -xvf /tmp/checkpoints/vgg_19_2016_08_28.tar.gz -C /tmp/checkpoints/')

        #If initial checkpoint not specified, use default downloaded checkpoint
        if init_ckpt == '':
            init_ckpt = '/tmp/checkpoints/%s.ckpt' % curr_model_name

        # Get correct optimizer using user defined options (optimizer and learning rate)
        optimizer = get_current_optimizer(curr_optimizer, float(curr_learning_rate))


## MORE ARE NEEDED FOR EACH MODEL TYPE ##
        # If output directory not specified, create simple one in /tmp/dataset/model
        if out_dir == '':
            out_dir = '/tmp/%s/%s' % (dataset_name, curr_model_name)

        # Begin training function
        run_model_training(curr_model_name,
                           int(curr_batch_size),
                           out_dir,
                           custom_dir,
                           init_ckpt,
                           int(curr_num_steps),
                           optimizer)








if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    root.wm_title("Model Trainer")

    root.mainloop()

