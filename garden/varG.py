
def get_case():
    case = "_FRA"             
    return case

def get_num_round():
    if nr != 0:
        return nr
    else:
        return None

def set_num_round(num_round):
    global nr
    nr = num_round

#
#   for DPA and PIA 15 round, for MUPA 100 round, for FRA 40 round
#
#	- case = "" --> No Attack
#
#	- case = "_DPA_U/T" --> Data Poisoned Attack tipo Untargeted / Targeted
#
#	- case = "_DPA_U/T_D(type of defence)" --> Defence from Data Poisoned Attack + Data Sanification/TrustFL
#                                              (krum, trimmed, median)      
#
#   - case = "_MUPA_BA" --> Model Update Poison Attack: Byzantine Attack
#
#	- case = "_MUPA_BA_DDP(type of defence)" --> Defence from Model Update Poisoned Attack + DP
#                                              (krum, trimmed, median)
#
#   - case = "_PIA" --> Property Inference Attack
#
#   - case = "_PIA_DLDP" --> Property Inference Attack + Local DP
#
#   - case = "_PIA_DI" --> Property Inference Attack + Quantizzation and Top-k Sparsification
#
#   - case = "_PIA_DM" --> Property Inference Attack + Masking
#
#   - case = "_FRA_S" --> Free Riding Attack (Same weights)
#
#   - case = "_FRA_D" --> Free Riding Attack + Detection
#
#   - case = "_DIF" --> Final Defence
#
#   - case = "*_DIF" *=(_DPA_U, _DPA_T, _MUPA_BA, _PIA, _FRA)
#
