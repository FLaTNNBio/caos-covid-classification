from pathlib import Path
from typing import List
import numpy as np
from .encoder_output import EncoderOutput

class InputOutputLoader:

    def __init__(self, order_output_model: List[str]):
        self.order_output_model = order_output_model
        self.encoder_output = EncoderOutput(order_output_model)

    def __call__(self, file_path: Path,):
        "given an image path, return the input-output for the model"
        file_path = str(file_path)
        #print("File_path:" + str(file_path).split('\\')[3])
        label = str(file_path).split('\\')[3]
        #label = Path(file_path).parent.stem #Commentato perché non restituisce la giusta cartella della variante.
        #print("Label:" + str(label))
        npy = np.load(file_path)
        return npy, self.encoder_output([label])