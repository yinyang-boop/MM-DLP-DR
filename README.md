Multi-modal Deep Learning for Drug Repurposing: Integrating GNN and CNN Approaches

Overview

This repository hosts the DDLS 2025 final project, a student-led practical exploration of drug repurposing using multi-modal deep learning. The project integrates two complementary approaches: a Graph Neural Network (GNN) advanced model for structured data analysis and a Convolutional Neural Network (CNN) baseline model for sequence-based prediction. The work emphasizes reproducibility, modular design, and real-world applicability in peptide drug repurposing.

Note: This is a student practice project. While AI tools assisted in code writing and debugging, the project conception, methodology design, and validation pathways are original intellectual contributions of the author.

Creator: yinyang@kth.se

Features

• Multi-modal Data Integration: Combines chemical, biological, and pharmacological data from multiple sources.

• Dual-Model Architecture: 

  • GNN Advanced Model: Handles molecular graphs and structured interactions.

  • CNN Baseline Model: Processes sequence data for affinity prediction.

• Reproducible Pipelines: Prefect-based workflow orchestration for deterministic execution.

• Cheminformatics & Sequence Analysis: RDKit for molecular descriptors and similarity scoring; protein/peptide sequence alignment.

• Modular Design: Separated data loaders, model training, and evaluation scripts for flexibility.

Model Architectures

GNN Advanced Model

• Input: Molecular graphs (from DrugBank XML), binding affinities (BindingDB dump), and drug sensitivity profiles (GDSC CSVs).

• Algorithm/Logic: Graph convolutional layers to encode molecular structures, followed by multi-layer perceptrons for interaction prediction. Employs attention mechanisms to weight important substructures.

• Output: Drug-target affinity scores and repurposing probabilities.

• Key Reference: Inspired by graph-based approaches like DeepDrug (see References).

CNN Baseline Model (DeepDTA-based)

• Input: SMILES strings (via ChEMBL API) and protein sequences (via PubChem).

• Algorithm/Logic: 1D convolutional networks to extract features from sequences, combined in a twin-tower architecture for affinity regression.

• Output: Continuous binding affinity values (e.g., pChEMBL, IC50).

• Key Reference: Öztürk et al. (2018) DeepDTA: Deep Drug–Target Binding Affinity Prediction.

Datasets

• DrugBank: Full database XML (licensed; pre-downloaded to Google Drive: MyDrive/Colab_Projects/drugbank_all_full_database.xml.zip).

• BindingDB: MySQL dump (pre-downloaded: MyDrive/Colab_Projects/BDB-mySQL_All_202511_dmp.zip).

• GDSC: MOBEM CSV files (pre-downloaded: MyDrive/Colab_Projects/GDSC-mobem-csv.zip).

• ChEMBL: Bioactivity data (via chembl_webresource_client API).

• PubChem: Compound and peptide data (via API).

Installation

1. Clone the repositories:
   # GNN advanced model (with pre-downloaded data)
   git clone https://github.com/yinyang-boop/MM-DLP-DR.git
   # CNN baseline model (API-based)
   git clone https://github.com/yinyang-boop/DDLS_Drug_Repurposing.git
   

2. Install dependencies:
   pip install -r requirements.txt  # For each project
   

3. Set up data:
   • For GNN model: Place DrugBank, BindingDB, and GDSC files from Google Drive into data/raw/.

   • For CNN model: APIs will fetch data on-the-fly; no local files needed.

Usage

GNN Advanced Model

1. Preprocess data:
   python scripts/preprocess_gnn_data.py  # Handles XML/CSV loading
   
2. Train model:
   python src/train_gnn.py --config configs/gnn_config.yaml
   
3. Evaluate:
   python src/evaluate.py --model models/gnn_model.pt
   

CNN Baseline Model

1. Run molecular analysis:
   python mcp_pipeline/molecular_analyzer.py  # Generates descriptors and similarities
   
2. Train DeepDTA:
   python src/train_deepdta.py --epochs 100
   

Workflow Overview

• Data Flow: Raw data → preprocessing → feature extraction → model input.

• Training Flow: Model initialization → multi-modal fusion → loss optimization → validation.

• Output: Affinity predictions → repurposing candidates → visualization reports.

References

Model Architectures

• DeepDTA: Öztürk, H. et al. (2018). Bioinformatics, 34(17), i821–i829.  

• GNN for Molecules: Key references include Kipf & Welling (2017) and Duvenaud et al. (2015).

Datasets

• DrugBank: Wishart et al. (2018). Nucleic Acids Research, 46(D1), D1074–D1082.  

• BindingDB: Gilson et al. (2016). Nucleic Acids Research, 44(D1), D1045–D1053.  

• GDSC: Yang et al. (2013). Nucleic Acids Research, 41(D1), D955–D961.  

• ChEMBL: Gaulton et al. (2017). Nucleic Acids Research, 45(D1), D945–D954.  

• PubChem: Kim et al. (2023). Nucleic Acids Research, 51(D1), D1373–D1380.

Declaration

This project is part of academic coursework. AI tools were used for code assistance and debugging, but the core ideas, experimental design, and interpretations are the author's original work. Datasets are cited appropriately; models are reimplemented for educational purposes.

License

MIT License. See LICENSE files in each repository.
