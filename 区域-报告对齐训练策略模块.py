from monai.data import PersistentDataset
import nibabel as nib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
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