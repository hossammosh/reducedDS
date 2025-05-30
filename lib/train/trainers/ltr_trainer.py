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
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False, log_save=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
            use_amp - Use Automatic Mixed Precision for faster training if True
            log_save - Whether to save data to data_recorder (default: False)
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

        # ----- NEW: Add log_save parameter -----
        self.log_save = log_save

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""
        print('start tracking...')
        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        self._init_timing()
        print('epoch no.= ', self.epoch)

        # Initialize gradient saving if enabled
        self._save_gradients = False
        if getattr(self.settings, 'save_gradients', False) and loader.training:
            try:
                self._grad_output_dir = os.path.join(self.settings.env.workspace_dir, 'gradients')
                print(f"Gradient saving is ENABLED. Gradients will be saved to: {self._grad_output_dir}")
                self._save_gradients = True
            except Exception as e:
                print(f"Error initializing gradient saving: {e}")

        # Initialize timing
        self.last_time_print = time.time()
        self.iteration_counter = 0

        for i, data in enumerate(loader, 1):
            self.iteration_counter += 1  # Increment global iteration counter

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

            # ----- MODIFIED: Only log data if log_save is True -----
            if self.log_save:
                data_recorder.log_data(sample_index, data_info, stats)
                # ----- Excel data logging with frequency control -----
                log_freq = getattr(self.settings, 'log_sample_stats_interval', 10)
                if i % log_freq == 0 or i == len(loader):
                    print(
                        f"Excel data logged at iteration {self.iteration_counter} (every {self.settings.log_sample_stats_interval} iterations)")

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    
                    # Save gradients if enabled (now on every iteration when _save_gradients is True)
                    if self._save_gradients:
                        try:
                            import lib.train.data_recorder as data_recorder
                            data_recorder.save_gradients(
                                model=self.actor.net,
                                sample_index=sample_index,
                                epoch=self.epoch,
                                output_dir=self._grad_output_dir
                            )
                            print(f"Saved gradients for sample {sample_index}")
                        except Exception as e:
                            print(f"Error saving gradients: {e}")
                    
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
        # Set the current epoch in the data recorder at the beginning of each epoch
        if self.log_save:
            data_recorder.set_epoch(self.epoch)

        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        # Finalize data recording for the current epoch (save remaining buffer, merge chunks)
        # Ensure this runs only on the main process to avoid race conditions during merge
        if self.log_save and self.settings.local_rank in [-1, 0]:
            print(f"--- ltr_trainer: Finalizing data recording for epoch {self.epoch} ---")
            data_recorder.finalize_epoch(self.epoch)
            print(f"--- ltr_trainer: Data recording finalized for epoch {self.epoch} ---")

        self._stats_new_epoch()  # Now reset stats for the next epoch
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

        # Define parameters_printing_interval once at the beginning
        parameters_printing_interval = getattr(self.settings, 'parameters_printing_interval', 50)

        # Then use it in the conditional check
        if i % parameters_printing_interval == 0 or i == loader.__len__():
            # ===== NEW CODE BLOCK START =====
            # Calculate progress information
            total_samples_per_epoch = loader.__len__()
            samples_completed = i
            samples_left = total_samples_per_epoch - i
            progress_ratio = samples_completed / total_samples_per_epoch
            samples_left_ratio = samples_left / total_samples_per_epoch

            # Time calculations for current epoch
            time_used_seconds = current_time - self.start_time
            time_used_hours = time_used_seconds / 3600

            # Estimate time left for current epoch
            if progress_ratio > 0:
                estimated_total_epoch_time = time_used_seconds / progress_ratio
                time_left_epoch_seconds = estimated_total_epoch_time - time_used_seconds
                time_left_epoch_hours = time_left_epoch_seconds / 3600
            else:
                time_left_epoch_hours = 0

            # Time for last completed epoch (if not first epoch)
            if hasattr(self, 'last_epoch_time'):
                last_epoch_time_hours = self.last_epoch_time / 3600
            else:
                last_epoch_time_hours = 0.0

            # Total time since training start
            if hasattr(self, 'training_start_time'):
                total_training_time_seconds = current_time - self.training_start_time
                total_training_time_hours = total_training_time_seconds / 3600
            else:
                # First epoch, initialize training start time
                self.training_start_time = self.start_time
                total_training_time_hours = time_used_hours

            # Comprehensive progress line
            progress_info = (f"[{loader.name}: Epoch {self.epoch}, {i}/{total_samples_per_epoch}] "
                             f"Samples Left: {samples_left} ({samples_left_ratio:.1%}) | "
                             f"Current Epoch: {time_used_hours:.2f}h used, {time_left_epoch_hours:.2f}h left | "
                             f"Last Epoch: {last_epoch_time_hours:.2f}h | "
                             f"Total Training: {total_training_time_hours:.2f}h | "
                             f"FPS: {average_fps:.1f} ({batch_fps:.1f})")

            # Add loss statistics to the same line
            stats_str = ""
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        stats_str += f'{name}: {val.avg:.5f}, '

            # Combine progress info with stats
            if stats_str:
                full_line = progress_info + " | " + stats_str[:-2]  # Remove last ", "
            else:
                full_line = progress_info

            print(full_line)

            # Log to file
            log_str = full_line + '\n'
            # ===== NEW CODE BLOCK END =====

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
