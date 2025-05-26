import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import lib.utils.misc as misc
import lib.train.data_recorder as data_recorder


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        # ----- Modification Start: Define and Create Checkpoint Directory -----
        print("--- Modifying ltr_trainer: Defining checkpoint directory ---")
        self.checkpoint_dir = os.path.join(self.settings.env.workspace_dir, self.settings.project_path, "checkpoints")
        if self.settings.local_rank in [-1, 0]:  # Only main process creates directory
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
                print(f"--- ltr_trainer: Created checkpoint directory at: {self.checkpoint_dir} ---")
            else:
                print(f"--- ltr_trainer: Checkpoint directory already exists at: {self.checkpoint_dir} ---")
        # ----- Modification End: Define and Create Checkpoint Directory -----

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # ----- NEW: Initialize iteration counter for Excel logging frequency -----
        self.iteration_counter = 0

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        print('start tracking...')
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()
        print('epoch no.= ', self.epoch)

        # ----- NEW: Initialize timing variables for manual control -----
        self.last_time_print = time.time()

        for i, data in enumerate(loader, 1):
            self.iteration_counter += 1  # NEW: Increment global iteration counter

            data_info = data[1]
            sample_index = data[2]
            data = data[0]

            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)
            data_recorder.log_data(sample_index, data_info, stats)
            # ----- MODIFIED: Excel data logging with frequency control -----

            log_freq = getattr(self.settings, 'log_sample_stats_interval', 10)
            if i % log_freq == 0 or i == len(loader):
                print(f"Excel data logged at iteration {self.iteration_counter} (every {self.settings.log_sample_stats_interval} iterations)")

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            torch.cuda.synchronize()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

            # ----- Modification Start: Save Checkpoint Conditionally -----
            # Save checkpoint only for the first 10 epochs as requested for the initial stage
            if self.epoch <= 10:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pt")
                print(f"--- ltr_trainer: Attempting to save checkpoint for epoch {self.epoch} to {checkpoint_path} ---")
                # Save the network's state_dict (all parameters)
                torch.save(self.actor.net.state_dict(), checkpoint_path)
                print(f"--- ltr_trainer: Successfully saved checkpoint for epoch {self.epoch} to {checkpoint_path} ---")
            # ----- Modification End: Save Checkpoint Conditionally -----

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time

        # ----- MODIFIED: Manual time printing frequency control -----
        should_print_stats = i % self.settings.parameters_printing_interval == 0 or i == loader.__len__()
        should_print_timing = (current_time - self.last_time_print) >= (
                    self.settings.parameters_printing_interval * 60) or i == loader.__len__()

        if should_print_stats:
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            if misc.is_main_process():
                # Ensure log file path is correctly handled
                log_file_path = getattr(self.settings, 'log_file', None)
                if log_file_path:
                    try:
                        with open(log_file_path, 'a') as f:
                            f.write(log_str)
                    except Exception as e:
                        print(f"Error writing to log file {log_file_path}: {e}")
                else:
                    print("Log file path not configured in settings.")

        # ----- NEW: Manual timing information printing -----
        if should_print_timing:
            elapsed_time = current_time - self.start_time
            remaining_samples = (loader.__len__() - i) * batch_size
            total_samples = loader.__len__() * batch_size
            samples_processed = total_samples - remaining_samples

            if samples_processed > 0:
                estimated_total_time = elapsed_time * total_samples / samples_processed
                remaining_time = estimated_total_time - elapsed_time
                progress_percent = (samples_processed / total_samples) * 100

                # Calculate epoch time (this is the time for current epoch)
                epoch_time = elapsed_time

                timing_str = f"[Epoch {self.epoch}, Iter {i}/{loader.__len__()}] "
                timing_str += f"Samples: {remaining_samples} left ({progress_percent:.1f}%), "
                timing_str += f"Time: {elapsed_time / 3600:.2f}h used, {remaining_time / 3600:.2f}h left, "
                timing_str += f"Last epoch: {epoch_time / 3600:.2f}h, Total: {elapsed_time / 3600:.2f}h"

                print(timing_str)

                # Log timing information to file as well
                if misc.is_main_process():
                    log_file_path = getattr(self.settings, 'log_file', None)
                    if log_file_path:
                        try:
                            with open(log_file_path, 'a') as f:
                                f.write(timing_str + '\n')
                        except Exception as e:
                            print(f"Error writing timing to log file {log_file_path}: {e}")

            self.last_time_print = current_time

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    # Correct way to get LR in newer PyTorch versions
                    lr_list = [param_group['lr'] for param_group in self.optimizer.param_groups]
                except AttributeError:
                    # Fallback for older versions or different schedulers
                    try:
                        lr_list = self.lr_scheduler.get_lr()
                    except AttributeError:
                        # Handle cases where scheduler might not have get_lr or _get_lr
                        try:
                            lr_list = self.lr_scheduler._get_lr(self.epoch)
                        except Exception as e:
                            print(f"Could not retrieve learning rate: {e}")
                            lr_list = [self.optimizer.param_groups[0]['lr']]  # Default to first group LR

                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if loader.name not in self.stats or self.stats[loader.name] is None:
                        self.stats[loader.name] = OrderedDict()
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

# import os
# from collections import OrderedDict
# from lib.train.trainers import BaseTrainer
# from lib.train.admin import AverageMeter, StatValue
# from lib.train.admin import TensorboardWriter
# import torch
# import time
# from torch.utils.data.distributed import DistributedSampler
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# import lib.utils.misc as misc
# #from lib.train.data_recorder import log_data
# import lib.train.data_recorder as data_recorder
#
# class LTRTrainer(BaseTrainer):
#     def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
#         """
#         args:
#             actor - The actor for training the network
#             loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
#                         epoch for each loader.
#             optimizer - The optimizer used for training, e.g. Adam
#             settings - Training settings
#             lr_scheduler - Learning rate scheduler
#         """
#         super().__init__(actor, loaders, optimizer, settings, lr_scheduler)
#         self._set_default_settings()
#
#         # Initialize statistics variables
#         self.stats = OrderedDict({loader.name: None for loader in self.loaders})
#
#         # Initialize tensorboard
#         if settings.local_rank in [-1, 0]:
#             tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
#             if not os.path.exists(tensorboard_writer_dir):
#                 os.makedirs(tensorboard_writer_dir)
#             self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])
#
#         self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
#         self.settings = settings
#         self.use_amp = use_amp
#         if use_amp:
#             self.scaler = GradScaler()
#
#     def _set_default_settings(self):
#         # Dict of all default values
#         default = {'print_interval': 10,
#                    'print_stats': None,
#                    'description': ''}
#
#         for param, default_value in default.items():
#             if getattr(self.settings, param, None) is None:
#                 setattr(self.settings, param, default_value)
#
#     def cycle_dataset(self, loader):
#         """Do a cycle of training or validation."""
#         print('start tracking...')
#         self.actor.train(loader.training)
#         torch.set_grad_enabled(loader.training)
#         self._init_timing()
#         print('epoch no.= ',self.epoch)
#         for i, data in enumerate(loader, 1):
#             data_info=data[1]
#             sample_index = data[2]
#             data = data[0]
#
#             if self.move_data_to_gpu:
#                 data = data.to(self.device)
#
#             data['epoch'] = self.epoch
#             data['settings'] = self.settings
#             # forward pass
#             if not self.use_amp:
#                 loss, stats = self.actor(data)
#             else:
#                 with autocast():
#                     loss, stats = self.actor(data)
#             data_recorder.log_data(sample_index, data_info, stats)
#             # backward pass and update weights
#             if loader.training:
#                 self.optimizer.zero_grad()
#                 if not self.use_amp:
#                     loss.backward()
#                     if self.settings.grad_clip_norm > 0:
#                         torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
#                     self.optimizer.step()
#                 else:
#                     self.scaler.scale(loss).backward()
#                     self.scaler.step(self.optimizer)
#                     self.scaler.update()
#
#             torch.cuda.synchronize()
#
#             # update statistics
#             batch_size = data['template_images'].shape[loader.stack_dim]
#             self._update_stats(stats, batch_size, loader)
#
#             # print statistics
#             self._print_stats(i, loader, batch_size)
#
#
#     def train_epoch(self):
#         """Do one epoch for each loader."""
#         for loader in self.loaders:
#             if self.epoch % loader.epoch_interval == 0:
#                 # 2021.1.10 Set epoch
#                 if isinstance(loader.sampler, DistributedSampler):
#                     loader.sampler.set_epoch(self.epoch)
#                 self.cycle_dataset(loader)
#
#         self._stats_new_epoch()
#         if self.settings.local_rank in [-1, 0]:
#             self._write_tensorboard()
#
#     def _init_timing(self):
#         self.num_frames = 0
#         self.start_time = time.time()
#         self.prev_time = self.start_time
#
#     def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
#         # Initialize stats if not initialized yet
#         if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
#             self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})
#
#         for name, val in new_stats.items():
#             if name not in self.stats[loader.name].keys():
#                 self.stats[loader.name][name] = AverageMeter()
#             self.stats[loader.name][name].update(val, batch_size)
#
#     def _print_stats(self, i, loader, batch_size):
#         self.num_frames += batch_size
#         current_time = time.time()
#         batch_fps = batch_size / (current_time - self.prev_time)
#         average_fps = self.num_frames / (current_time - self.start_time)
#         self.prev_time = current_time
#         if i % self.settings.print_interval == 0 or i == loader.__len__():
#             print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
#             print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
#             for name, val in self.stats[loader.name].items():
#                 if (self.settings.print_stats is None or name in self.settings.print_stats):
#                     if hasattr(val, 'avg'):
#                         print_str += '%s: %.5f  ,  ' % (name, val.avg)
#                     # else:
#                     #     print_str += '%s: %r  ,  ' % (name, val)
#
#             print(print_str[:-5])
#             log_str = print_str[:-5] + '\n'
#             if misc.is_main_process():
#                 print(self.settings.log_file)
#                 with open(self.settings.log_file, 'a') as f:
#                     f.write(log_str)
#
#     def _stats_new_epoch(self):
#         # Record learning rate
#         for loader in self.loaders:
#             if loader.training:
#                 try:
#                     lr_list = self.lr_scheduler.get_lr()
#                 except:
#                     lr_list = self.lr_scheduler._get_lr(self.epoch)
#                 for i, lr in enumerate(lr_list):
#                     var_name = 'LearningRate/group{}'.format(i)
#                     if var_name not in self.stats[loader.name].keys():
#                         self.stats[loader.name][var_name] = StatValue()
#                     self.stats[loader.name][var_name].update(lr)
#
#         for loader_stats in self.stats.values():
#             if loader_stats is None:
#                 continue
#             for stat_value in loader_stats.values():
#                 if hasattr(stat_value, 'new_epoch'):
#                     stat_value.new_epoch()
#
#     def _write_tensorboard(self):
#         if self.epoch == 1:
#             self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)
#
#         self.tensorboard_writer.write_epoch(self.stats, self.epoch)
