from fastapi import FastAPI
# import matplotlib.pyplot as plt
# import seaborn as sns
import torch
from rdkit import Chem
from models import Cigin

from fastapi.middleware.cors import CORSMiddleware

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_solv_free_energy(model, solute, solvent):
    """Get the solvation free energy"""
    solv_free_energy, interaction_map = model(solute, solvent)
    return solv_free_energy, interaction_map

def load_model():
    """Load the model"""
    model = Cigin().to(DEVICE)
    model.load_state_dict(torch.load("data/cigin.tar"))
    model.eval()
    return model

response = {}
def predictions(solute, solvent):

    mol = Chem.MolFromSmiles(solute)
    mol = Chem.AddHs(mol)
    solute = Chem.MolToSmiles(mol)

    mol = Chem.MolFromSmiles(solvent)
    mol = Chem.AddHs(mol)
    solvent = Chem.MolToSmiles(mol)

    model = load_model()

    delta_g, interaction_map =  model(solute, solvent)
    response["interaction_map"] = (interaction_map.detach().numpy()).tolist()
    response["solvation"] = delta_g.item()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/prediction")
def prediction(solute, solvent):
    results = predictions(solute, solvent)
    return {"prediction": response}


