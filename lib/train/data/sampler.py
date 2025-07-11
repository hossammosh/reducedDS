import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
def no_processing(data):
    return data

class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - 'causal', 'interval', or 'order'.
            train_cls - this is for Stark-ST, should be False for SeqTrack.

        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        #data_info={}

    def __len__(self):
        return self.samples_per_epoch




    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        #breakpoint()
        #print("index =  ",index)
        if self.train_cls:
            return self.getitem_cls()
        else:
            v=self.getitem()
            return  (*v, index)


    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        count_valid = 0
        data_info={}
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()
            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            data_info['seq_id'] = seq_id
            data_info['seq_path'] = dataset.sequence_info['seq_path']
            data_info['seq_name'] = dataset.sequence_info['seq_name']
            data_info['class_name'] = dataset.sequence_info['class_name']
            data_info['vid_id'] = dataset.sequence_info['vid_id']

            if is_video_dataset:
                template_ids = None
                search_id = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_id > template_ids
                    while search_id is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1,
                                                                 min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            if gap_increase > 1000:
                                print("too large gap")
                                print(str(gap_increase))
                            continue
                        template_ids = base_frame_id + prev_frame_ids
                        search_id = self._sample_visible_ids(visible, min_id=template_ids[0] + 1,
                                                                    max_id=template_ids[
                                                                               0] + self.max_gap + gap_increase,
                                                                    num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5
                        if gap_increase > 1000:
                            print("too large gap")
                            print(str(gap_increase))

                elif self.frame_sample_mode == "order":
                    template_ids, search_id = self.get_frame_ids_order(visible)
                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_ids, search_id = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_ids, search_id = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_ids = [1] * self.num_template_frames
                search_id = [1] * self.num_search_frames
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_ids,seq_info_dict)
                data_info['template_ids'] = template_ids
                data_info['template_names'] = dataset.frames['frame_names']
                data_info['template_path'] = dataset.frames['frame_paths']

                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_id, seq_info_dict)

                data_info['search_id'] = search_id
                data_info['search_names'] = dataset.frames['frame_names']
                data_info['search_path'] = dataset.frames['frame_paths']





                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                    (H, W))] * self.num_search_frames
                data = TensorDict({'template_images': template_frames,
                                    'template_anno': template_anno['bbox'],
                                    'template_masks': template_masks,
                                    'search_images': search_frames,
                                    'search_anno': search_anno['bbox'],
                                    'search_masks': search_masks,
                                    'dataset': dataset.get_name(),
                                    'test_class': meta_obj_test.get('object_class_name')
                                   })
                data = self.processing(data)
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

            count_valid += 1
            if count_valid > 200:
                print("too large count_valid")
                print(str(count_valid))

        return data, data_info


    def show(self, data, strr, i):
        image = data[strr+'_images'][i]
        _, H, W = image.shape
        import cv2
        x1, y1, w, h = data[strr+'_anno'][i]
        x1, y1, w, h = int(x1*W), int(y1*H), int(w*W), int(h*H)
        image_show = image.permute(1,2,0).numpy()
        max = image_show.max()
        min = image_show.min()
        image_show = (image_show-min) * 255 / (max-min)
        image_show = np.ascontiguousarray(image_show.astype('uint8'))
        cv2.rectangle(image_show, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        cv2.imshow(strr+str(i), image_show)
        if cv2.waitKey() & 0xFF == ord('q'):
            pass

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            sequence_info=dataset.sequence_info
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_ids, search_id = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_ids, search_id = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_ids = [1] * self.num_template_frames
                search_id = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_id, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_id = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_id is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_id,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        #add by chenxin to debug
        count = 0
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
            count += 1
            if count > 200:
                print("too large count")
                print(str(count))
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_id = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_id = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_id = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_id, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_ids_extra = []
        while None in template_ids_extra or len(template_ids_extra) == 0:
            template_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_id = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_id[0]:
                    min_id, max_id = search_id[0], search_id[0] + max_gap
                else:
                    min_id, max_id = search_id[0] - max_gap, search_id[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_ids_extra += [None]
                else:
                    template_ids_extra += f_id

        template_ids = template_frame_id1 + template_ids_extra
        return template_ids, search_id

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_ids_extra = []
        while None in template_ids_extra or len(template_ids_extra) == 0:
            template_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_id = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_id[0]:
                    min_id, max_id = search_id[0], search_id[0] + max_gap
                else:
                    min_id, max_id = search_id[0] - max_gap, search_id[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_ids_extra += [None]
                else:
                    template_ids_extra += f_id

        template_ids = template_frame_id1 + template_ids_extra
        return template_ids, search_id

    def get_frame_ids_order(self, visible):
        # get template and search ids in an 'order' manner, the template and search regions are arranged in chronological order
        frame_ids = []
        gap_increase = 0
        while (None in frame_ids) or (len(frame_ids)==0):
            base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=0,
                                                     max_id=len(visible))
            frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames+self.num_search_frames,
                                                      min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                      max_id=base_frame_id[0] + self.max_gap + gap_increase)
            if (frame_ids is None) or (None in frame_ids):
                gap_increase += 5
                if gap_increase > 1000:
                    print("too large gap")
                    print(str(gap_increase))
                continue
            if torch.rand(1) < 0.5:
                frame_ids.sort(reverse=True)
                template_ids = frame_ids[0:self.num_template_frames]
                search_id = frame_ids[self.num_template_frames:]
            else:
                frame_ids.sort(reverse=False)
                template_ids = frame_ids[0:self.num_template_frames]
                search_id = frame_ids[self.num_template_frames:]
            # Increase gap until a frame is found
            gap_increase += 5
            if gap_increase > 1000:
                print("too large gap")
                print(str(gap_increase))
        return template_ids, search_id
