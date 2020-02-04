### SINGLE-TASK TOX21 MODELS

# Train ST models on each task individually
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_ahr.cfg  
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_ar.cfg   
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_er.cfg      
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_mmp.cfg
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_p53.cfg
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_are.cfg  
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_atad5.cfg      
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_hse.cfg     
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/tox21/tox21_ppar-gamma.cfg

# Test ST models by pooling the 5 models from the 5 CV folds
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_ahr.cfg  
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_ar.cfg   
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_er.cfg      
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_mmp.cfg
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_p53.cfg
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_are.cfg  
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_atad5.cfg      
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_hse.cfg     
python conv_qsar_fast/scripts/consensus_test_from_CV.py conv_qsar_fast/inputs/tox21/tox21_ppar-gamma.cfg
