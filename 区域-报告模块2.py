from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from functools import partial
import tqdm


class RadGenomeDataset_Train(PersistentDataset):
    def __init__(self, text_tokenizer, data_folder, mask_folder, csv_file, cache_dir, max_region_size=10,
                 max_img_size=1, image_num=32, region_num=33, max_seq=2048, resize_dim=500, voc_size=32000,
                 force_num_frames=True):
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_tokenizer,
        )

        special_token = {
            "additional_special_tokens": ["<image>", "</image>", "<region>", "</region>"]}

        self.image_padding_tokens = []
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(
                    "<image" + str(i * image_num + j) + ">")
            self.image_padding_tokens.append(image_padding_token)

        self.region_padding_tokens = []
        for i in range(max_region_size):
            region_padding_tokens = ""
            for j in range(region_num):
                region_token = "<region" + str(i * region_num + j) + ">"
                region_padding_tokens = region_padding_tokens + region_token
                special_token["additional_special_tokens"].append(
                    "<region" + str(i * region_num + j) + ">")
            self.region_padding_tokens.append(region_padding_tokens)

        self.text_tokenizer.add_special_tokens(
            special_token
        )
        self.text_tokenizer.pad_token_id = 0
        self.text_tokenizer.bos_token_id = 1
        self.text_tokenizer.eos_token_id = 2

        self.voc_size = voc_size
        self.max_seq = max_seq
        self.data_folder = data_folder
        self.mask_folder = mask_folder

        self.accession_to_sentences = self.load_accession_sentences(csv_file)
        self.paths = []
        self.samples = self.prepare_samples()
        self.target_size = (256, 256, 64)

        def threshold(x):
            return x > -1000

        self.region_transform = transforms.Compose([
            transforms.CropForeground(select_fn=threshold),
            transforms.Resize(spatial_size=self.target_size),
            transforms.ToTensor()
        ])
        self.image_transform = transforms.Compose([
            transforms.CropForegroundd(keys=['img', 'seg'], source_key='img', select_fn=threshold),
            transforms.Resized(keys=['img', 'seg'], spatial_size=self.target_size),
            transforms.ToTensord(keys=['img', 'seg'])
        ])
        self.mask_img_to_tensor = partial(self.mask_nii_img_to_tensor, region_transform=self.region_transform,
                                          image_transform=self.image_transform)
        super().__init__(data=self.samples, transform=None, cache_dir=cache_dir)

    def load_accession_sentences(self, xlsx_file):
        df = pd.read_csv(xlsx_file)
        df_grouped = df.groupby('Volumename')

        accession_to_sentences = {}
        for accession, group in df_grouped:
            sentences = {}
            for i, row in group.iterrows():
                if pd.isna(row['Anatomy']):
                    anatomy_key = 'whole'
                else:
                    anatomy_key = row['Anatomy']
                sentences[anatomy_key] = row['Sentence']
            accession_to_sentences[accession] = sentences
        return accession_to_sentences